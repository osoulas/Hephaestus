"""Capability phase: the agent proposes a missing tool, the user approves the plan and the code (Section 14).

It runs between goal parsing and the campaign, only when code finds a capability gap (AT-1).
The agent researches the method, proposes a plan (``propose_tool``), writes the tool once the
plan is accepted (``write_tool``), and the user reviews the code. Code runs the static check,
the tests and a trial run before the user sees the code (AT-11, AT-12).

Decisions reach the phase from a callback (Python API, CLI) or from ``HUB`` (HTTP). A tool call
never blocks on the user: ``propose_tool`` and ``write_tool`` return at once, and the decision
is the next user message. Waiting counts against no budget (AT-25).
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hephaestus.core.config import RunConfig, budget_limit
from hephaestus.core.events import EventBus
from hephaestus.core.store import atomic_write_text, load_model, save_json
from hephaestus.core.types import (
    Candidate,
    CustomTool,
    ErrorRecord,
    SearchSpec,
    ToolCheck,
    ToolDecision,
    ToolProposal,
    ToolStage,
    UserRequest,
)
from hephaestus.harness.agent import LITERATURE_TOOLS, shorten
from hephaestus.harness.prompts import CAPABILITY_SYSTEM, capability_kickoff
from hephaestus.llm.base import LLM, LLMError, UsageLimitError
from hephaestus.tools import custom
from hephaestus.tools.registry import TOOLS, CandidateState, Handler, ToolError, Workspace, execute

CAP_LITERATURE = ("search_literature", "related_papers", "get_paper", "note_evidence")
FIX_ATTEMPTS = 3
MAX_NUDGES = 3
MAX_IDLE_TURNS = 2  # turns ending with no tool call and nothing pending before the phase gives up
REQUEST_STAGES = {"package", "question", "give_up"}
WAIT_POLL_S = 0.5
TRIAL_ID = "trial"
DEFAULT_TRIAL = "SrTiO3"
Outcome = Literal["filled", "cancelled_tool", "cancelled_run", "budget", "failed"]
Decider = Callable[["ToolProposal | UserRequest"], "ToolDecision | dict[str, Any] | str | None"]


def _schema(name: str, description: str, props: dict[str, Any]) -> dict[str, Any]:
    """Tool definition with every property required (not strict: ``inputs`` is free-form).

    Parameters
    ----------
    name : str
        Tool name.
    description : str
        For the LLM.
    props : dict
        Input properties.

    Returns
    -------
    dict
        ``{name, description, input_schema}``.
    """
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": props,
            "required": list(props),
            "additionalProperties": False,
        },
    }


STR = {"type": "string"}
STR_LIST = {"type": "array", "items": {"type": "string"}}
CAPABILITY_TOOLS: list[dict[str, Any]] = [
    *(t for t in TOOLS if t["name"] in CAP_LITERATURE),
    _schema(
        "read_tool_source",
        "Read our own simulation and structure tools. function=null lists every public function with its "
        "signature and summary; a function name returns its full source.",
        {"function": {"type": ["string", "null"]}},
    ),
    _schema(
        "propose_tool",
        "Propose a tool for the missing property, for the user to accept, change or cancel. Returns at once; "
        "end your turn and the user's decision arrives as the next message. To revise after the user asks "
        "for changes (to the plan or to the code), call again with the same name.",
        {
            "name": {"type": "string", "description": "snake_case tool name"},
            "property": {"type": "string", "description": "the missing property's key, exactly"},
            "units": STR,
            "description": STR,
            "method": {"type": "string", "description": "how it is computed, citing the papers"},
            "citations": {"type": "array", "items": STR, "description": "retrieved paper IDs"},
            "inputs": {"type": "object", "description": "JSON schema of extra inputs, each with a default"},
            "building_blocks": {
                "type": "array",
                "items": STR,
                "description": "hp calls it uses; may be empty",
            },
            "expected_runtime_s": {"type": "number"},
            "supplies_target": {"type": "boolean"},
            "if_cancelled": {"type": "string", "description": "what the run loses without this tool"},
        },
    ),
    _schema(
        "write_tool",
        "Submit the tool file and its test file for an accepted plan. Code runs the static check, the tests and "
        "a trial run in a sandbox; the trial value must fall in TOOL['trial_range']. If a check fails, fix it and "
        "call again. When all pass, read the trial output; if it shows a bug, call write_tool again at once with "
        "the fix and a reason, which withdraws the code the user is reviewing. Otherwise end your turn: the user "
        "reviews the code.",
        {
            "proposal_id": STR,
            "source": STR,
            "tests": STR,
            "reason": {
                "type": ["string", "null"],
                "description": "why you replace code the user is already reviewing; null otherwise",
            },
        },
    ),
    _schema(
        "request_package",
        "Ask the user to allow a Python package outside the sandbox's allowlist (for example phonopy), and to "
        "install it if the server does not have it. Returns at once; end your turn and the user's answer "
        "arrives as the next message. Once allowed, list it in TOOL['packages'] and import it.",
        {
            "module": {"type": "string", "description": "top-level import name, e.g. phonopy"},
            "pip_name": {"type": ["string", "null"], "description": "pip name if it differs, may be pinned"},
            "reason": {"type": "string", "description": "why the tool needs it, in one or two sentences"},
        },
    ),
    _schema(
        "ask_user",
        "Ask the user a question when you need their choice to continue (the user never reads your plain text "
        "as a question). Returns at once; end your turn and the answer arrives as the next message.",
        {
            "question": STR,
            "options": {"type": "array", "items": STR, "description": "suggested answers; may be empty"},
        },
    ),
    _schema(
        "give_up",
        "Propose to the user to stop without a tool, when not even a useful proxy can be built. The user decides: "
        "they may let you stop (the run then cannot evaluate this property) or tell you what to build instead. "
        "Say why in one or two sentences, and name the closest proxy you considered.",
        {"reason": STR},
    ),
]


# ---------------------------------------------------------------------------
# Decisions from the user (AT-7, AT-13, AT-24)
# ---------------------------------------------------------------------------


@dataclass
class _Wait:
    """One proposal waiting for the user."""

    allowed: set[str]
    event: threading.Event = field(default_factory=threading.Event)
    decision: ToolDecision | None = None


class DecisionHub:
    """Where decisions from the HTTP API meet the phase waiting for them; thread-safe."""

    def __init__(self) -> None:
        """Create an empty hub."""
        self._lock = threading.Lock()
        self._waits: dict[tuple[str, str, str], _Wait] = {}

    def open(self, run_id: str, proposal_id: str, stage: str, allowed: set[str]) -> None:
        """Start waiting for a decision.

        Parameters
        ----------
        run_id : str
            Run.
        proposal_id : str
            Proposal.
        stage : {"plan", "code"}
            Stage.
        allowed : set of str
            Actions the user may take.
        """
        with self._lock:
            self._waits.setdefault((run_id, proposal_id, stage), _Wait(set(allowed)))

    def waiting(self, run_id: str) -> list[dict[str, Any]]:
        """Proposals of a run waiting for a decision.

        Parameters
        ----------
        run_id : str
            Run.

        Returns
        -------
        list of dict
            ``{proposal_id, stage, allowed}``.
        """
        with self._lock:
            return [
                {"proposal_id": p, "stage": s, "allowed": sorted(w.allowed)}
                for (r, p, s), w in self._waits.items()
                if r == run_id and w.decision is None
            ]

    def submit(self, run_id: str, decision: ToolDecision) -> None:
        """Deliver a decision.

        Parameters
        ----------
        run_id : str
            Run.
        decision : ToolDecision
            The user's decision.

        Raises
        ------
        ValueError
            If the proposal is not waiting at that stage, or the action is not offered.
        """
        with self._lock:
            w = self._waits.get((run_id, decision.proposal_id, decision.stage))
            if w is None or w.decision is not None:
                raise ValueError(
                    f"Proposal {decision.proposal_id} is not waiting for a {decision.stage} decision."
                )
            if decision.action not in w.allowed:
                raise ValueError(
                    f"{decision.action} is not offered here; choose one of {', '.join(sorted(w.allowed))}."
                )
            w.decision = decision
            w.event.set()

    def cancel_run(self, run_id: str) -> None:
        """Answer every wait of a run with ``cancel_run`` (a user stop).

        Parameters
        ----------
        run_id : str
            Run.
        """
        with self._lock:
            for (r, p, s), w in self._waits.items():
                if r == run_id and w.decision is None:
                    w.decision = ToolDecision(proposal_id=p, stage=s, action="cancel_run")  # type: ignore[arg-type]
                    w.event.set()

    def wait(
        self, run_id: str, proposal_id: str, stage: str, stop: threading.Event | None = None
    ) -> ToolDecision:
        """Block until the decision arrives, or the run is stopped.

        Parameters
        ----------
        run_id, proposal_id, stage : str
            What to wait for; ``open`` must have been called.
        stop : threading.Event, optional
            A user stop, answered as ``cancel_run``.

        Returns
        -------
        ToolDecision
            The decision.
        """
        with self._lock:
            w = self._waits[(run_id, proposal_id, stage)]
        while not w.event.wait(WAIT_POLL_S):
            if stop is not None and stop.is_set():
                self.cancel_run(run_id)
        with self._lock:
            self._waits.pop((run_id, proposal_id, stage), None)
        assert w.decision is not None
        return w.decision

    def withdraw(self, run_id: str, proposal_id: str, stage: str) -> bool:
        """Close a wait the user has not answered yet (the agent replaces its own code).

        Parameters
        ----------
        run_id, proposal_id, stage : str
            The wait.

        Returns
        -------
        bool
            False when the user already decided: the decision stands.
        """
        with self._lock:
            w = self._waits.get((run_id, proposal_id, stage))
            if w is not None and w.decision is not None:
                return False
            self._waits.pop((run_id, proposal_id, stage), None)
            return True

    def clear(self, run_id: str) -> None:
        """Forget a run's waits.

        Parameters
        ----------
        run_id : str
            Run.
        """
        with self._lock:
            for key in [k for k in self._waits if k[0] == run_id]:
                del self._waits[key]


HUB = DecisionHub()


# ---------------------------------------------------------------------------
# Phase state, saved in runs/<id>/capability/
# ---------------------------------------------------------------------------


class CapabilityState(BaseModel):
    """Everything the phase has done; ``capability/state.json``."""

    model_config = ConfigDict(extra="forbid")

    gap: str
    proposals: dict[str, ToolProposal] = Field(default_factory=dict)
    decisions: list[ToolDecision] = Field(default_factory=list)
    registered: list[CustomTool] = Field(default_factory=list)
    outcome: Outcome | None = None
    agent_s: float = 0.0
    fix_attempts: dict[str, int] = Field(default_factory=dict)
    trial_structure: dict[str, Any] | None = None
    trial_note: str = ""
    reviewed_files: list[str] = Field(default_factory=list)  # library files already shown (sha256)
    requests: dict[str, UserRequest] = Field(default_factory=dict)  # package requests and questions
    approved_packages: list[str] = Field(default_factory=list)  # imports the user allowed in this phase
    end_reason: str = ""  # why the phase ended without a tool (give_up, stuck)


def find_gap(spec: SearchSpec, cfg: RunConfig, library: list[CustomTool] | None = None) -> str | None:
    """Return the property no tool in this run can compute, if any (AT-1). Pure code.

    Parameters
    ----------
    spec : SearchSpec
        Parsed goal.
    cfg : RunConfig
        Disabled capabilities and the library folder.
    library : list of CustomTool, optional
        Approved tools; read from the library by default.

    Returns
    -------
    str or None
        The target property when it is a gap; ``None`` when a tool covers it or there is no
        named property.
    """
    if spec.target_property == "unsupported" or spec.target_tool:
        return None
    reg = custom.run_registry(cfg, library)
    return None if spec.target_property in reg else spec.target_property


class CapabilityPhase:
    """One capability phase; also the host the Claude Code backend drives (``CampaignHost``)."""

    def __init__(
        self,
        ws: Workspace,
        llm: LLM,
        handlers: Mapping[str, Handler],
        bus: EventBus,
        folder: Path,
        gap: str,
        decide: Decider | None = None,
        stop: threading.Event | None = None,
        state: CapabilityState | None = None,
        toolkit_factory: custom.ToolkitFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Set up a phase.

        Parameters
        ----------
        ws : Workspace
            Run workspace; approved tools are added to ``ws.custom_tools``.
        llm : LLM
            The orchestrator's model.
        handlers : mapping
            Tool handlers (literature, structures, simulation).
        bus : EventBus
            Events.
        folder : pathlib.Path
            ``runs/<id>/capability``.
        gap : str
            The missing property.
        decide : callable, optional
            ``f(proposal) -> ToolDecision | dict``; without it, decisions come through ``HUB``.
        stop : threading.Event, optional
            A user stop, treated as cancelling the run.
        state : CapabilityState, optional
            Saved state of an interrupted phase.
        toolkit_factory : callable, optional
            Toolkit factory (tests).
        clock : callable
            Monotonic clock.
        """
        self.ws, self.llm, self.handlers, self.bus = ws, llm, dict(handlers), bus
        self.folder, self.gap, self.decide = folder, gap, decide
        self.stop = stop or threading.Event()
        self.state = state or CapabilityState(gap=gap)
        self.toolkit_factory = toolkit_factory or custom.default_toolkit
        self.clock = clock
        self.lib = custom.library_dir(ws.cfg)
        self.usage: dict[str, int] = {}
        self.errors: list[ErrorRecord] = []
        self.pending: tuple[str, ToolStage] | None = None
        cfg = ws.cfg
        self.max_prompts = (
            2 * cfg.max_custom_tools * (cfg.max_tool_revisions + FIX_ATTEMPTS + 2) + MAX_NUDGES + 2
        )
        self._t0 = clock()
        self._wait_s = 0.0
        self._offset = self.state.agent_s
        self._seq = 0
        self._idle = 0  # agent turns in a row with no tool call
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ host protocol

    @property
    def agent_s(self) -> float:
        """Agent time in the phase, excluding time waiting for the user (AT-25).

        Returns
        -------
        float
            Seconds.
        """
        return self._offset + (self.clock() - self._t0) - self._wait_s

    def hard_limit_hit(self) -> bool:
        """Whether the phase must end: an outcome, a stop or the time budget.

        Returns
        -------
        bool
            True when done.
        """
        if self.state.outcome is not None or self.stop.is_set():
            return True
        return self.pending is None and self.agent_s >= budget_limit(self.ws.cfg.capability_time_budget_s)

    def emit(self, kind: str, payload: dict[str, Any], status: str = "finished") -> None:
        """Publish a ``capability`` event.

        Parameters
        ----------
        kind : str
            ``gap``, ``library``, ``plan``, ``checks``, ``code``, ``decision``, ``registered``,
            ``cancelled`` or ``log``.
        payload : dict
            Event payload.
        status : {"started", "finished", "failed"}
            Status.
        """
        self.bus.emit(self.ws.run_id, "capability", status, {"kind": kind, **payload})  # type: ignore[arg-type]

    def emit_agent(self, kind: str, text: str) -> None:
        """Publish the agent's text or thinking.

        Parameters
        ----------
        kind : {"text", "thinking"}
            Block kind.
        text : str
            Content.
        """
        if text:
            self.bus.emit(
                self.ws.run_id, "agent", "finished", {"kind": kind, "text": text, "phase": "capability"}
            )

    def nudge(self) -> str:
        """Next user message when the agent ends its turn: the pending decision, or a prompt.

        Returns
        -------
        str
            Message text.
        """
        if self.pending:
            return self.await_decision()
        self._idle += 1
        if self._idle > MAX_IDLE_TURNS:  # the agent is stuck: stop instead of nudging forever
            self._end(f"the agent made no progress for {self._idle} turns")
            return "The phase is over. Make no more calls."
        return (
            "You ended your turn without a tool call. The user does not read your text as a question and "
            "nothing is waiting for them. Continue with a tool call: ask_user to ask the user something, "
            "request_package for a package outside the allowlist, propose_tool or write_tool to make progress, "
            "or give_up if no tool can be built."
        )

    def _end(self, reason: str) -> None:
        """End the phase without a tool (give_up, or stuck).

        Parameters
        ----------
        reason : str
            Why, shown to the user.
        """
        self.state.outcome = "failed"
        self.state.end_reason = reason
        self.emit(
            "cancelled", {"proposal_id": None, "scope": "tool", "supplies_target": True, "reason": reason}
        )
        self.save()

    def save(self) -> None:
        """Write ``capability/state.json``."""
        self.state.agent_s = round(self.agent_s, 2)
        save_json(self.folder / "state.json", self.state)
        save_json(
            self.folder / "proposals.json", [p.model_dump(mode="json") for p in self.state.proposals.values()]
        )
        save_json(self.folder / "decisions.json", [d.model_dump(mode="json") for d in self.state.decisions])

    # ------------------------------------------------------------------ tools

    def call_tool(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Run one capability-phase tool call; never raises.

        Parameters
        ----------
        name : str
            Tool name.
        args : dict
            Tool input.

        Returns
        -------
        result : dict
            Tool result or ``{"error": ...}``.
        is_error : bool
            Whether it failed.
        """
        with self._lock:
            self._seq += 1
            self._idle = 0
            call_id = f"cap{self._seq}"
            shown = shorten(args, max_str=300)
            self.bus.emit(
                self.ws.run_id,
                "tool",
                "started",
                {"tool": name, "call_id": call_id, "input": shown, "phase": "capability"},
            )
            if self.hard_limit_hit():
                result, is_error = {"error": "The capability phase is over; make no more calls."}, True
            elif name in CAP_LITERATURE:
                result, is_error = execute(self.ws, name, args, self.handlers)
            else:
                try:
                    fn = {
                        "read_tool_source": self._read_source,
                        "propose_tool": self._propose,
                        "write_tool": self._write,
                        "request_package": self._request_package,
                        "ask_user": self._ask_user,
                        "give_up": self._give_up,
                    }.get(name)
                    if fn is None:
                        raise ToolError(f"{name} is not available in the capability phase.")
                    result, is_error = fn(**args), False
                except ToolError as exc:
                    result, is_error = {"error": str(exc)}, True
                except TypeError as exc:
                    result, is_error = {"error": f"Bad arguments for {name}: {exc}"}, True
                except Exception as exc:
                    result, is_error = {"error": f"{type(exc).__name__}: {exc}"[:500]}, True
            payload: dict[str, Any] = {
                "tool": name,
                "call_id": call_id,
                "input": shown,
                "result": shorten(result),
                "phase": "capability",
            }
            if name in LITERATURE_TOOLS and isinstance(result.get("papers"), list):
                payload["mode"] = args.get("mode") or args.get("relation")
                payload["papers"] = [
                    {k: p.get(k) for k in ("paper_id", "title", "year", "citations")}
                    for p in result["papers"]
                ]
            self.bus.emit(self.ws.run_id, "tool", "failed" if is_error else "finished", payload)
            self.save()
            return result, is_error

    def _read_source(self, function: str | None = None) -> dict[str, Any]:
        """``read_tool_source``.

        Parameters
        ----------
        function : str or None
            Function to read in full.

        Returns
        -------
        dict
            Listing or source.
        """
        return custom.read_tool_source(self.ws.cfg, function)

    def _allowed(self, prop: ToolProposal) -> set[str]:
        """Actions offered for a proposal (AT-7).

        Parameters
        ----------
        prop : ToolProposal
            Proposal.

        Returns
        -------
        set of str
            ``revise`` only while revisions are left.
        """
        out = {"accept", "cancel_tool", "cancel_run"}
        if prop.revisions_left > 0:
            out.add("revise")
        return out

    def _wait_for(self, prop: ToolProposal) -> None:
        """Mark a proposal as waiting and publish its card.

        Parameters
        ----------
        prop : ToolProposal
            Proposal at its current stage.
        """
        self.pending = (prop.proposal_id, prop.stage)
        HUB.open(self.ws.run_id, prop.proposal_id, prop.stage, self._allowed(prop))
        self.emit(
            prop.stage, {"proposal": prop.model_dump(mode="json"), "allowed": sorted(self._allowed(prop))}
        )

    def _propose(self, **plan: Any) -> dict[str, Any]:
        """``propose_tool``: register a plan for the user's decision (AT-6).

        Parameters
        ----------
        **plan
            The plan fields of ``ToolProposal``.

        Returns
        -------
        dict
            ``{proposal_id, status, instruction}``.

        Raises
        ------
        ToolError
            For a bad name or property, a revision nobody asked for, or too many tools.
        """
        if self.pending:
            raise ToolError("A proposal is already waiting for the user; end your turn.")
        name = str(plan.get("name", ""))
        if not custom.NAME_RE.match(name):
            raise ToolError("name must be snake_case, 2 to 41 characters.")
        if plan.get("property") != self.gap:
            raise ToolError(f"property must be the missing property {self.gap!r}.")
        if name in custom.read_manifest(self.lib):
            raise ToolError(f"The library already has a tool named {name}; choose another name.")
        existing = next((p for p in self.state.proposals.values() if p.name == name), None)
        if existing is not None:
            if existing.status != "revise":  # a change request at either stage may need a new plan
                raise ToolError(
                    f"{name} was already proposed; revise it only after the user asks for changes."
                )
            pid, revision, left = existing.proposal_id, existing.revision, existing.revisions_left
        else:
            names = {p.name for p in self.state.proposals.values()}
            if len(names) >= self.ws.cfg.max_custom_tools:
                raise ToolError(f"At most {self.ws.cfg.max_custom_tools} tools per run.")
            pid, revision, left = f"t{len(self.state.proposals) + 1}", 1, self.ws.cfg.max_tool_revisions
        prop = ToolProposal(
            proposal_id=pid,
            name=name,
            property=self.gap,
            units=str(plan.get("units", "")),
            description=str(plan.get("description", "")),
            method=str(plan.get("method", "")),
            citations=[str(c) for c in plan.get("citations") or []],
            inputs=dict(plan.get("inputs") or {}),
            building_blocks=[str(b) for b in plan.get("building_blocks") or []],
            expected_runtime_s=plan.get("expected_runtime_s"),
            supplies_target=self.gap == self.ws.spec.target_property,
            if_cancelled=str(plan.get("if_cancelled", "")),
            revision=revision,
            revisions_left=left,
        )
        if prop.supplies_target and not prop.if_cancelled:
            prop.if_cancelled = f"Without it the run cannot evaluate {self.gap}: it screens stability only."
        self.state.proposals[pid] = prop
        self._wait_for(prop)
        return {
            "proposal_id": pid,
            "status": "waiting_for_user",
            "instruction": "End your turn now; the user's decision arrives as the next message.",
        }

    def _trial(self) -> dict[str, Any] | None:
        """Relaxed structure of the reference material for trial runs, built once (AT-12).

        Returns
        -------
        dict or None
            pymatgen structure dict; ``None`` when it could not be built (``trial_note`` says why).
        """
        if self.state.trial_structure is not None or self.state.trial_note:
            return self.state.trial_structure
        material = self.ws.spec.reference_material or DEFAULT_TRIAL
        cand = Candidate(
            candidate_id=TRIAL_ID,
            iteration=0,
            composition=material,
            structure_type="perovskite_cubic",
            rationale="Trial structure for an agent-built tool.",
        )
        self.ws.candidates[TRIAL_ID] = CandidateState(candidate=cand, round=0)
        try:
            for name, args in [
                (
                    "build_structure",
                    {"candidate_id": TRIAL_ID, "source": "auto", "mp_id": None, "seed": self.ws.cfg.seed},
                ),
                ("relax", {"candidate_id": TRIAL_ID, "potential": self.ws.cfg.potential, "fmax": 0.02}),
            ]:
                result, is_error = execute(self.ws, name, args, self.handlers)
                if is_error:
                    self.state.trial_note = (
                        f"trial structure for {material} unavailable: {result.get('error')}"[:300]
                    )
                    return None
            self.state.trial_structure = self.ws.candidates[TRIAL_ID].structure or {}
        finally:
            self.ws.candidates.pop(TRIAL_ID, None)
        return self.state.trial_structure

    def _toolkit(self) -> custom.Toolkit:
        """Toolkit for checks, with the screening potential.

        Returns
        -------
        Toolkit
            Toolkit whose ``hp.log`` lines become events.
        """
        tk = self.toolkit_factory(self.ws, self.ws.cfg.potential, self.ws.cfg.dtype)
        tk.on_log = lambda m: self.emit("log", {"message": m})
        return tk

    def _write(self, proposal_id: str, source: str, tests: str, reason: str | None = None) -> dict[str, Any]:
        """``write_tool``: check the code, run its tests and a trial run (AT-10 to AT-12).

        Called while the user reviews this proposal's code, it withdraws that code first: the
        agent found a bug in its own work, and the user must not approve the old version.

        Parameters
        ----------
        proposal_id : str
            Accepted proposal.
        source : str
            Tool file.
        tests : str
            Test file.
        reason : str, optional
            Why the agent replaces code the user is reviewing; required in that case.

        Returns
        -------
        dict
            ``{passed, checks}``, plus the trial output and a wait instruction when all pass.

        Raises
        ------
        ToolError
            If the proposal is unknown or not ready for code.
        """
        prop = self.state.proposals.get(proposal_id)
        if prop is None:
            raise ToolError(f"Unknown proposal_id {proposal_id!r}.")
        if self.pending == (prop.proposal_id, "code") and prop.status == "waiting":
            self._withdraw(prop, reason)
        if self.pending:
            raise ToolError("Something is waiting for the user; end your turn.")
        ready = (prop.stage == "plan" and prop.status == "accepted") or (
            prop.stage == "code" and prop.status == "revise"
        )
        if not ready:
            raise ToolError("Write the tool only after the user accepts the plan (or asks for code changes).")
        checks: dict[str, ToolCheck] = {}
        try:
            declared = custom.header_packages(custom.read_header(source))
        except ValueError:
            declared = []
        extra = [p for p in declared if p in self.state.approved_packages]
        static = custom.static_check(source, "tool", extra)
        tstatic = custom.static_check(tests, "test", extra)
        problems = [] if static["passed"] else [static["report"]]
        for p in declared:
            if p not in self.state.approved_packages:
                problems.append(
                    f"TOOL['packages'] lists {p!r}, which the user has not allowed: call request_package"
                )
        if not tstatic["passed"]:
            problems.append("tests: " + tstatic["report"])
        if any("import" in x and "not allowed" in x for x in problems):
            problems.append(
                "To use a package outside the allowlist, call request_package and wait for the user."
            )
        trial_range = None
        if static["passed"]:
            header = custom.read_header(source)
            trial_range = custom.header_trial_range(header)
            if trial_range is None:
                problems.append(
                    "TOOL['trial_range'] must be [min, max]: the plausible value for the trial material "
                    f"({self.ws.spec.reference_material or DEFAULT_TRIAL}) in {prop.units}, from the literature"
                )
            for key in ("name", "property", "units"):
                if header.get(key) != getattr(prop, key):
                    problems.append(
                        f"TOOL[{key!r}] is {header.get(key)!r}; the plan says {getattr(prop, key)!r}"
                    )
        checks["static"] = ToolCheck(passed=not problems, report="\n".join(problems) or "passed")
        drafts = self.folder / "drafts"
        tool_file = drafts / f"{prop.name}-r{prop.revision}.py"
        test_file = drafts / f"test_{prop.name}-r{prop.revision}.py"
        atomic_write_text(tool_file, source)
        atomic_write_text(test_file, tests)
        trial: dict[str, Any] | None = None
        calls: list[str] = []
        if checks["static"].passed:
            structure = self._trial()
            tk = self._toolkit()
            out = custom.run_sandboxed(
                "test",
                tool_file,
                tk,
                structure=structure or {},
                test_file=test_file,
                timeout_s=self.ws.cfg.custom_tool_timeout_s,
                extra=extra,
            )
            if out["ok"]:
                results = list((out["result"] or {}).get("tests", []))
                failed = [f"{t['name']}: {t['error']}" for t in results if not t["passed"]]
                ok = bool(results) and not failed
                report = "\n".join(failed) or f"{len(results)} test(s) passed"
            else:
                ok, report = False, str(out["error"])
            checks["tests"] = ToolCheck(passed=ok, report=report[:2000])
            if structure is None:
                checks["trial"] = ToolCheck(passed=True, report=f"skipped: {self.state.trial_note}")
            else:
                out = custom.run_sandboxed(
                    "run",
                    tool_file,
                    tk,
                    structure=structure,
                    inputs={},
                    timeout_s=self.ws.cfg.custom_tool_timeout_s,
                    extra=extra,
                )
                if out["ok"]:
                    try:
                        value, details = custom.check_result(out["result"], prop.units)
                        trial = {
                            "material": self.ws.spec.reference_material or DEFAULT_TRIAL,
                            "value": value,
                            "units": prop.units,
                            "range": list(trial_range) if trial_range else None,
                            "details": shorten(details),
                        }
                        lo, hi = trial_range or (None, None)
                        if lo is not None and hi is not None and not lo <= value <= hi:
                            checks["trial"] = ToolCheck(
                                passed=False,
                                report=f"{value:.4g} {prop.units} is outside TOOL['trial_range'] [{lo:g}, {hi:g}] for "
                                f"{trial['material']}: find the bug, or correct the range only if the literature "
                                "supports it",
                            )
                        else:
                            checks["trial"] = ToolCheck(passed=True, report=f"{value:.4g} {prop.units}")
                    except ToolError as exc:
                        checks["trial"] = ToolCheck(passed=False, report=str(exc))
                else:
                    checks["trial"] = ToolCheck(passed=False, report=str(out["error"])[:2000])
            calls = sorted(set(tk.calls))
        save_json(
            self.folder / "checks" / f"{prop.name}-r{prop.revision}.json",
            {k: v.model_dump() for k, v in checks.items()},
        )
        passed = all(c.passed for c in checks.values()) and len(checks) == 3
        self.emit(
            "checks",
            {
                "proposal_id": prop.proposal_id,
                "revision": prop.revision,
                "checks": {k: v.model_dump() for k, v in checks.items()},
            },
        )
        if not passed:
            n = self.state.fix_attempts.get(prop.proposal_id, 0) + 1
            self.state.fix_attempts[prop.proposal_id] = n
            body: dict[str, Any] = {"passed": False, "checks": {k: v.model_dump() for k, v in checks.items()}}
            if n >= FIX_ATTEMPTS:
                self._cancel(prop, f"the checks failed {n} times")
                body["instruction"] = (
                    "The checks failed too many times; the tool is cancelled. Make no more calls."
                )
            else:
                body["attempts_left"] = FIX_ATTEMPTS - n
            return body
        self.state.fix_attempts.pop(prop.proposal_id, None)
        prop.stage, prop.status = "code", "waiting"
        prop.source, prop.tests, prop.checks, prop.trial, prop.toolkit_calls = (
            source,
            tests,
            checks,
            trial,
            calls,
        )
        self._wait_for(prop)
        return {
            "passed": True,
            "checks": {k: v.model_dump() for k, v in checks.items()},
            "trial": trial,
            "status": "waiting_for_user",
            "instruction": "End your turn now; the user reviews the code and the decision arrives as the next message.",
        }

    # ------------------------------------------------------------------ decisions

    def await_decision(self) -> str:
        """Wait for the pending decision, apply it and return the message for the agent.

        Returns
        -------
        str
            What the user decided, as an instruction.
        """
        assert self.pending is not None
        pid, stage = self.pending
        item: ToolProposal | UserRequest = (
            self.state.requests[pid] if stage in REQUEST_STAGES else self.state.proposals[pid]
        )
        t0 = self.clock()
        try:
            if self.decide is not None:
                decision = coerce_decision(self.decide(item.model_copy(deep=True)), item)
                HUB.clear(self.ws.run_id)
            else:
                decision = HUB.wait(self.ws.run_id, pid, stage, self.stop)
        finally:
            self._wait_s += self.clock() - t0
        self.pending = None
        if isinstance(item, UserRequest):
            return self._apply_request(item, decision)
        if decision.action not in self._allowed(item):  # a revision past the cap (AT-7)
            decision = decision.model_copy(update={"action": "cancel_tool"})
        return self._apply(item, decision)

    def _apply(self, prop: ToolProposal, decision: ToolDecision) -> str:
        """Apply a decision to a proposal.

        Parameters
        ----------
        prop : ToolProposal
            Proposal.
        decision : ToolDecision
            The user's decision.

        Returns
        -------
        str
            Message for the agent.
        """
        self.state.decisions.append(decision)
        self.emit("decision", {"decision": decision.model_dump(mode="json"), "name": prop.name})
        stage, action = decision.stage, decision.action
        msg: str
        if action == "accept" and stage == "plan":
            prop.status = "accepted"
            msg = f"The user accepted the plan for {prop.name}. Write it now with write_tool (proposal_id {prop.proposal_id})."
        elif action == "accept":
            assert prop.source is not None and prop.tests is not None
            entry = custom.save_tool(
                self.lib, prop.name, prop.source, prop.tests, prop.author, self.ws.run_id
            )
            prop.status = "registered"
            self.ws.custom_tools[prop.name] = entry
            self.state.registered.append(entry)
            self.emit("registered", {"tool": entry.model_dump(mode="json")})
            if prop.property == self.gap:
                self.state.outcome = "filled"
            msg = f"The user approved {prop.name}; it is saved and registered for the campaign. The phase is over."
        elif action == "revise":
            prop.status = "revise"
            prop.revision += 1
            prop.revisions_left -= 1
            what = decision.message or "(no message)"
            msg = (
                f"The user asks for changes to the plan: {what}\nRevise it and call propose_tool again with name {prop.name}."
                if stage == "plan"
                else f"The user asks for changes to the code: {what}\nFix it and call write_tool again with proposal_id "
                f"{prop.proposal_id}. If the change needs a different method, call propose_tool again with name {prop.name}."
            )
            msg += (
                " If it needs a package outside the allowlist, call request_package first. If you cannot do what is "
                "asked, call ask_user to say why and offer alternatives."
            )
        elif action == "cancel_tool":
            self._cancel(prop, "cancelled by the user")
            msg = "The user cancelled the tool. The phase is over; make no more calls."
        else:
            self.state.outcome = "cancelled_run"
            self.emit(
                "cancelled",
                {"proposal_id": prop.proposal_id, "scope": "run", "supplies_target": prop.supplies_target},
            )
            msg = "The user cancelled the run. Make no more calls."
        self.save()
        return msg

    def _withdraw(self, prop: ToolProposal, reason: str | None) -> None:
        """Withdraw code the user is reviewing, because the agent found a problem in it.

        Parameters
        ----------
        prop : ToolProposal
            Proposal waiting at the code stage.
        reason : str or None
            What is wrong; required.

        Raises
        ------
        ToolError
            Without a reason, when the user already decided, or after too many attempts.
        """
        if not (reason or "").strip():
            raise ToolError("The user is reviewing this code: give a reason to withdraw and replace it.")
        if not HUB.withdraw(self.ws.run_id, prop.proposal_id, "code"):
            raise ToolError("The user has already decided on this code; end your turn.")
        n = self.state.fix_attempts.get(prop.proposal_id, 0) + 1
        self.state.fix_attempts[prop.proposal_id] = n
        self.pending = None
        prop.status = "revise"
        prop.revision += 1  # the agent's own fix: the user's change requests are untouched
        self.emit(
            "withdrawn", {"proposal_id": prop.proposal_id, "revision": prop.revision - 1, "reason": reason}
        )
        if n > FIX_ATTEMPTS:
            self._cancel(prop, f"the agent replaced its code {n} times")
            raise ToolError("Too many attempts; the tool is cancelled. Make no more calls.")

    # ------------------------------------------------------------------ requests to the user

    def _open_request(self, req: UserRequest) -> dict[str, Any]:
        """Publish a request and wait for the user's answer.

        Parameters
        ----------
        req : UserRequest
            The request.

        Returns
        -------
        dict
            The wait instruction for the agent.

        Raises
        ------
        ToolError
            If something else is already waiting for the user.
        """
        if self.pending:
            raise ToolError("Something is already waiting for the user; end your turn.")
        self.state.requests[req.request_id] = req
        allowed = {
            "package": {"allow", "deny", "cancel_run"},
            "question": {"answer", "cancel_run"},
            "give_up": {"allow", "answer", "cancel_run"},
        }[req.kind]
        self.pending = (req.request_id, req.kind)
        HUB.open(self.ws.run_id, req.request_id, req.kind, allowed)
        self.emit(req.kind, {"request": req.model_dump(mode="json"), "allowed": sorted(allowed)})
        return {
            "request_id": req.request_id,
            "status": "waiting_for_user",
            "instruction": "End your turn now; the user's answer arrives as the next message.",
        }

    def _request_package(self, module: str, reason: str, pip_name: str | None = None) -> dict[str, Any]:
        """``request_package``: ask to allow (and install) a package outside the allowlist.

        Parameters
        ----------
        module : str
            Top-level import name.
        reason : str
            Why the tool needs it.
        pip_name : str, optional
            pip name, when it differs from the import name.

        Returns
        -------
        dict
            The wait instruction, or a note when it is already allowed.

        Raises
        ------
        ToolError
            For a bad name.
        """
        module = module.strip().split(".")[0]
        if not custom.MODULE_RE.match(module):
            raise ToolError(f"{module!r} is not a package name.")
        if custom.import_allowed(module) or module in self.state.approved_packages:
            return {
                "allowed": True,
                "note": f"{module} is already allowed; list it in TOOL['packages'] if needed.",
            }
        pip = (pip_name or module).strip()
        if not custom.PIP_RE.match(pip):
            raise ToolError(f"{pip!r} is not a plain pip package name.")
        req = UserRequest(
            request_id=f"q{len(self.state.requests) + 1}",
            kind="package",
            question=reason,
            package=module,
            pip_name=pip,
            installed=custom.importable(module),
        )
        return self._open_request(req)

    def _ask_user(self, question: str, options: list[str] | None = None) -> dict[str, Any]:
        """``ask_user``: put a question to the user.

        Parameters
        ----------
        question : str
            The question.
        options : list of str, optional
            Suggested answers.

        Returns
        -------
        dict
            The wait instruction.
        """
        req = UserRequest(
            request_id=f"q{len(self.state.requests) + 1}",
            kind="question",
            question=question,
            options=[str(o) for o in options or []][:6],
        )
        return self._open_request(req)

    def _give_up(self, reason: str) -> dict[str, Any]:
        """``give_up``: end the phase without a tool.

        Parameters
        ----------
        reason : str
            Why no tool can be built.

        Returns
        -------
        dict
            Confirmation.
        """
        req = UserRequest(request_id=f"q{len(self.state.requests) + 1}", kind="give_up", question=reason)
        return self._open_request(req)

    def _apply_request(self, req: UserRequest, decision: ToolDecision) -> str:
        """Apply the user's answer to a request.

        Parameters
        ----------
        req : UserRequest
            The request.
        decision : ToolDecision
            ``allow``, ``deny``, ``answer`` or ``cancel_run``.

        Returns
        -------
        str
            Message for the agent.
        """
        self.state.decisions.append(decision)
        self.emit(
            "decision", {"decision": decision.model_dump(mode="json"), "name": req.package or "question"}
        )
        req.status, req.answer = "answered", decision.message
        action = decision.action
        if action == "cancel_run":
            self.state.outcome = "cancelled_run"
            self.emit("cancelled", {"proposal_id": req.request_id, "scope": "run", "supplies_target": True})
            msg = "The user cancelled the run. Make no more calls."
        elif req.kind == "give_up" and action == "allow":
            self._end(f"the agent gave up, with the user's agreement: {req.question}"[:500])
            msg = "The user agreed to stop without a tool. The phase is over; make no more calls."
        elif req.kind == "give_up":
            msg = (
                f"The user does not want to stop. They say: {decision.message or '(no message)'}\n"
                "Follow that direction: propose a tool for it with propose_tool."
            )
        elif req.kind == "question":
            msg = (
                f"The user answered your question ({req.question[:200]}): {decision.message or '(no answer)'}"
            )
        elif action == "allow":
            assert req.package and req.pip_name
            note = ""
            if not custom.importable(req.package):
                self.emit("log", {"message": f"Installing {req.pip_name} into the server's Python…"})
                ok, log = custom.install_package(req.pip_name)
                req.installed = ok and custom.importable(req.package)
                self.emit(
                    "log", {"message": f"pip install {req.pip_name}: {'done' if req.installed else 'failed'}"}
                )
                if not req.installed:
                    self.save()
                    return (
                        f"The user allowed {req.package}, but installing {req.pip_name} failed:\n{log[-600:]}\n"
                        "Work within the allowlist, ask_user, or give_up."
                    )
                note = f" It was installed with pip ({req.pip_name})."
            self.state.approved_packages.append(req.package)
            msg = (
                f"The user allowed {req.package}.{note} List it in TOOL['packages'] (e.g. \"packages\": "
                f'["{req.package}"]) and import it in the tool. The user sees the package on the code review.'
            )
        else:
            msg = (
                f"The user did not allow {req.package}. Work within the allowlist, ask_user to discuss "
                "alternatives, or give_up."
            )
        self.save()
        return msg

    def _cancel(self, prop: ToolProposal, reason: str) -> None:
        """Cancel one tool; when it supplied the gap, the phase ends without it (AT-8).

        Parameters
        ----------
        prop : ToolProposal
            Proposal.
        reason : str
            Why.
        """
        prop.status = "cancelled"
        if prop.author == "agent" and prop.property == self.gap:
            self.state.outcome = "cancelled_tool"
        self.emit(
            "cancelled",
            {
                "proposal_id": prop.proposal_id,
                "scope": "tool",
                "supplies_target": prop.supplies_target,
                "reason": reason,
            },
        )

    # ------------------------------------------------------------------ library files (AT-21)

    def review_library(self) -> None:
        """Offer unapproved library files for the gap for code review before the agent starts."""
        _, unapproved = custom.scan_library(self.lib)
        for f in unapproved:
            header = f["header"]
            if f["error"] or header.get("property") != self.gap or f["sha256"] in self.state.reviewed_files:
                continue
            static = custom.static_check(f["source"], "tool", custom.header_packages(header))
            pid = f"lib-{f['name']}"
            prop = ToolProposal(
                proposal_id=pid,
                name=f["name"],
                property=self.gap,
                units=str(header.get("units", "")),
                description=str(header.get("description", "")),
                method=str(header.get("method", "")),
                citations=[str(c) for c in header.get("citations") or []],
                inputs=dict(header.get("inputs") or {}),
                building_blocks=[str(b) for b in header.get("building_blocks") or []],
                stage="code",
                author="user",
                source=f["source"],
                tests=f["tests"],
                checks={"static": ToolCheck(passed=static["passed"], report=static["report"])},
                if_cancelled="The agent proposes its own tool instead.",
            )
            self.state.proposals[pid] = prop
            self.state.reviewed_files.append(f["sha256"])
            if not static["passed"]:
                continue
            self._wait_for(prop)
            self.await_decision()
            if self.state.outcome is not None:
                return

    # ------------------------------------------------------------------ loop

    def run(self, resume: bool = False) -> Outcome:
        """Run the phase until the gap is filled, cancelled, or the budget runs out.

        Parameters
        ----------
        resume : bool
            Continue a saved phase: a waiting proposal is shown again, then a fresh
            conversation is told what was done (AT-24).

        Returns
        -------
        str
            The outcome.
        """
        self.folder.mkdir(parents=True, exist_ok=True)
        if not resume:
            self.emit(
                "gap",
                {
                    "property": self.gap,
                    "units": self.ws.spec.target_units,
                    "custom_tools": self.ws.cfg.custom_tools,
                },
            )
        self.save()
        done: dict[str, Any] | None = None
        first = ""
        if self.state.outcome is None:
            self.review_library()
        if resume and self.state.outcome is None:
            waiting = next((p for p in self.state.proposals.values() if p.status == "waiting"), None)
            asking = next((r for r in self.state.requests.values() if r.status == "waiting"), None)
            if waiting is not None:
                self._wait_for(waiting)
                first = self.await_decision()
            elif asking is not None:
                del self.state.requests[asking.request_id]
                self._open_request(asking)
                first = self.await_decision()
            done = {
                "proposals": [
                    p.model_dump(exclude={"source", "tests"}, mode="json")
                    for p in self.state.proposals.values()
                ],
                "decisions": [d.model_dump(mode="json") for d in self.state.decisions],
            }
            if first:
                done["latest_decision"] = first
        if self.state.outcome is None and not self.stop.is_set():
            library = [
                {"name": t.name, "property": t.property, "units": t.units, "description": t.description}
                for t in custom.scan_library(self.lib)[0]
            ]
            kickoff = capability_kickoff(self.ws.spec, self.ws.cfg, library, done)
            driver = getattr(self.llm, "run_campaign", None)
            try:
                if callable(driver):
                    driver(CAPABILITY_SYSTEM, kickoff, CAPABILITY_TOOLS, self)
                else:
                    self._loop(kickoff)
            except UsageLimitError:
                raise  # the runner pauses the run; the phase resumes from its saved state
            except LLMError as exc:
                self.errors.append(ErrorRecord(step="capability", code="llm_failed", message=str(exc)[:500]))
        if self.state.outcome is None:
            if self.stop.is_set():
                self.state.outcome = "cancelled_run"
            elif self.agent_s >= budget_limit(self.ws.cfg.capability_time_budget_s):
                self.state.outcome = "budget"
            else:
                self.state.outcome = "failed"
            self.emit(
                "cancelled",
                {"proposal_id": None, "scope": "tool", "supplies_target": True, "reason": self.state.outcome},
            )
        HUB.clear(self.ws.run_id)
        self.save()
        return self.state.outcome

    def _loop(self, kickoff: str) -> None:
        """Turn-by-turn loop for backends that return one turn at a time.

        Parameters
        ----------
        kickoff : str
            First user message.
        """
        messages: list[dict[str, Any]] = [{"role": "user", "content": kickoff}]
        failures = 0
        while not self.hard_limit_hit():
            try:
                turn = self.llm.step(CAPABILITY_SYSTEM, messages, CAPABILITY_TOOLS)
            except UsageLimitError:
                raise
            except LLMError as exc:
                failures += 1
                self.errors.append(ErrorRecord(step="capability", code="llm_failed", message=str(exc)[:500]))
                if failures >= 2:
                    return
                continue
            failures = 0
            for k, v in turn.usage.items():
                self.usage[k] = self.usage.get(k, 0) + int(v or 0)
            for t in turn.thinking:
                self.emit_agent("thinking", t)
            for t in turn.texts:
                self.emit_agent("text", t)
            messages.append({"role": "assistant", "content": turn.content})
            if turn.stop_reason == "refusal":
                self.errors.append(ErrorRecord(step="capability", code="refusal", message="Model declined."))
                return
            blocks = []
            for call in turn.tool_calls:
                result, is_error = self.call_tool(call.name, call.input)
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
                if is_error:
                    block["is_error"] = True
                blocks.append(block)
            text = self.await_decision() if self.pending else ""
            if self.hard_limit_hit():
                return
            if blocks:
                content: Any = blocks + ([{"type": "text", "text": text}] if text else [])
            elif text:
                content = text
            else:
                content = self.nudge()  # counts idle turns and ends the phase when stuck
                if self.hard_limit_hit():
                    return
            messages.append({"role": "user", "content": content})


def coerce_decision(raw: Any, prop: ToolProposal | UserRequest) -> ToolDecision:
    """Turn a callback's answer into a decision for a proposal or a request.

    Parameters
    ----------
    raw : ToolDecision, dict, str or None
        ``None`` cancels the tool (or denies a package); a string is an action, or the answer
        to a question.
    prop : ToolProposal or UserRequest
        What was decided on.

    Returns
    -------
    ToolDecision
        The decision, at the proposal's stage (or the request's kind).
    """
    if isinstance(prop, UserRequest):
        if isinstance(raw, ToolDecision):
            return raw.model_copy(update={"proposal_id": prop.request_id, "stage": prop.kind})
        if isinstance(raw, dict):
            action, message = raw.get("action"), raw.get("message")
        elif raw is None:
            action, message = {"package": "deny", "question": "answer", "give_up": "allow"}[prop.kind], None
        elif (prop.kind == "question" and raw != "cancel_run") or (
            prop.kind == "give_up" and raw not in ("allow", "cancel_run")
        ):
            action, message = "answer", str(raw)
        else:
            action, message = str(raw), None
        action = action or ("answer" if prop.kind == "question" else "deny")
        return ToolDecision(proposal_id=prop.request_id, stage=prop.kind, action=action, message=message)  # type: ignore[arg-type]
    if isinstance(raw, ToolDecision):
        return raw.model_copy(update={"proposal_id": prop.proposal_id, "stage": prop.stage})
    if raw is None:
        raw = {"action": "cancel_tool"}
    if isinstance(raw, str):
        raw = {"action": raw}
    return ToolDecision(
        proposal_id=prop.proposal_id,
        stage=prop.stage,
        action=raw.get("action", "cancel_tool"),
        message=raw.get("message"),
    )


def load_state(folder: Path) -> CapabilityState | None:
    """Load the saved phase state, if any.

    Parameters
    ----------
    folder : pathlib.Path
        ``runs/<id>/capability``.

    Returns
    -------
    CapabilityState or None
        The state.
    """
    path = folder / "state.json"
    return load_model(path, CapabilityState) if path.exists() else None
