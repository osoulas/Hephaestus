"""Outer order: parse, capability phase, campaign, final re-check, brief (OR-1, OR-6, OR-8).

``run`` is the single entry point the Python API, HTTP API and CLI call. Every
dependency (models, tool handlers, final check) can be injected so tests run offline.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hephaestus.core.config import RunConfig
from hephaestus.core.events import BUS, EventBus
from hephaestus.core.store import atomic_write_text, new_run_id, save_json
from hephaestus.core.types import (
    Candidate,
    ErrorRecord,
    Quantity,
    RunState,
    SearchSpec,
    Submission,
    ValidationReport,
)
from hephaestus.harness.agent import Campaign, NoveltyFn
from hephaestus.harness.brief import assemble_brief
from hephaestus.harness.brief_pdf import export_pdf
from hephaestus.harness.capability import HUB, CapabilityPhase, Decider, find_gap, load_state
from hephaestus.harness.compute_brief import assemble_compute_brief
from hephaestus.harness.goal import apply_stability_options, parse_goal, validate_for_run
from hephaestus.harness.state import RunFiles, load_for_resume, save_snapshot, sync_state, write_artifacts
from hephaestus.llm.base import LLM, LoggedLLM, make_llm
from hephaestus.tools import custom
from hephaestus.tools.registry import (
    CandidateState,
    Handler,
    Workspace,
    execute,
    load_default_handlers,
    target_step,
)

FinalCheck = Callable[[Workspace, Submission], list[ValidationReport]]
TOKEN_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


class UsageMeter:
    """Token totals for the run state: campaign turns plus parser, reviewer and brief calls."""

    def __init__(self, state: RunState, *llms: LoggedLLM) -> None:
        """Start from what the state already holds (a resumed run keeps its earlier tokens).

        Parameters
        ----------
        state : RunState
            Run state to update.
        *llms : LoggedLLM
            Wrappers whose structured calls count.
        """
        self.state = state
        self.llms = llms
        self.base = {k: int(state.usage.get(k, 0)) for k in TOKEN_KEYS}
        self.campaign: dict[str, int] = {}

    def update(self, campaign_usage: Mapping[str, int] | None = None) -> None:
        """Write the current totals into ``state.usage``.

        Parameters
        ----------
        campaign_usage : mapping, optional
            The campaign's orchestrator usage so far; kept from the last call when omitted.
        """
        if campaign_usage is not None:
            self.campaign = dict(campaign_usage)
        for k in TOKEN_KEYS:
            self.state.usage[k] = (
                self.base[k]
                + int(self.campaign.get(k, 0))
                + sum(int(m.structured_usage.get(k, 0)) for m in self.llms)
            )


ACTIVE: dict[str, Campaign] = {}
_STOPS: dict[str, threading.Event] = {}
_ABORTS: dict[str, threading.Event] = {}  # "stop entirely": no final re-check, no brief
_LOCK = threading.Lock()


class RunRefused(Exception):
    """A run cannot start or resume (finished, already running, or no spec)."""


@dataclass
class Deps:
    """Injectable dependencies of a run; defaults are the real ones."""

    llm: LLM | None = None
    reviewer_llm: LLM | None = None
    handlers: Mapping[str, Handler] | None = None
    novelty_fn: NoveltyFn | None = None
    final_check: FinalCheck | None = None
    bus: EventBus = field(default_factory=lambda: BUS)
    root: Path | None = None
    decide: Decider | None = None  # agent-built tool decisions; else they come through HUB (AT-29)
    toolkit_factory: custom.ToolkitFactory | None = None  # building blocks for agent-built tools


def default_final_check() -> Callable[..., list[ValidationReport]] | None:
    """Return B's ``validate_finalists`` when ``harness/final_check.py`` provides it.

    Returns
    -------
    callable or None
        ``f(ws, submission, handlers=...) -> list of ValidationReport``.
    """
    from hephaestus.harness import final_check

    return getattr(final_check, "validate_finalists", None)


def request_stop(run_id: str, mode: str = "campaign") -> bool:
    """Stop a running run (OR-8).

    Parameters
    ----------
    run_id : str
        Run ID.
    mode : {"campaign", "all"}
        ``"campaign"``: end the search; the best candidate still gets the final re-check and the
        brief. ``"all"``: end the run as soon as the current calculation returns, with no final
        re-check and no brief; it can be resumed later.

    Returns
    -------
    bool
        True if the run is active in this process.
    """
    with _LOCK:
        ev = _STOPS.get(run_id)
        abort = _ABORTS.get(run_id)
        camp = ACTIVE.get(run_id)
    if ev:
        ev.set()
        HUB.cancel_run(run_id)  # a stop during the capability phase cancels the run
    if abort and mode == "all":
        abort.set()
    if camp:
        camp.request_stop(abort=mode == "all")
    return bool(ev or camp)


def is_active(run_id: str) -> bool:
    """Whether a run is executing in this process.

    Parameters
    ----------
    run_id : str
        Run ID.

    Returns
    -------
    bool
        True while ``run`` or ``resume`` is executing it.
    """
    with _LOCK:
        return run_id in _STOPS


def new_run(
    raw_goal: str,
    cfg: RunConfig,
    spec: SearchSpec | None = None,
    root: Path | None = None,
    run_id: str | None = None,
) -> tuple[RunState, RunFiles]:
    """Create a run folder and its initial state (DL-1).

    Parameters
    ----------
    raw_goal : str
        The goal.
    cfg : RunConfig
        Options.
    spec : SearchSpec, optional
        Already-parsed spec.
    root : pathlib.Path, optional
        Runs folder.
    run_id : str, optional
        Chosen ID; a new one by default.

    Returns
    -------
    state : RunState
        Initial state.
    files : RunFiles
        Run paths.
    """
    rid = run_id or new_run_id()
    files = RunFiles.for_run(rid, root)
    state = RunState(run_id=rid, raw_goal=raw_goal, config=cfg, spec=spec)
    save_json(files.state, state)
    save_json(files.progress, cfg)
    return state, files


def _models(cfg: RunConfig, deps: Deps, files: RunFiles) -> tuple[LoggedLLM, LoggedLLM]:
    """Build logged orchestrator and reviewer models (OR-11, DL-4).

    Parameters
    ----------
    cfg : RunConfig
        Backend and model choice.
    deps : Deps
        Injected models, if any.
    files : RunFiles
        Run paths for ``llm_log.jsonl``.

    Returns
    -------
    tuple
        ``(orchestrator, reviewer)``.
    """
    llm = deps.llm or make_llm(cfg.llm_backend, cfg.llm_model)
    reviewer = deps.reviewer_llm or llm
    return LoggedLLM(llm, files.llm_log), LoggedLLM(reviewer, files.llm_log)


def resolve_reference(
    ws: Workspace, handlers: Mapping[str, Handler], potential: str | None = None, fmax: float = 0.02
) -> tuple[SearchSpec, list[ErrorRecord]]:
    """Compute every relative target's reference value with the candidates' model (GP-11).

    For each target with a reference material and no reference value yet: build the reference
    from Materials Project (ambient polymorph), relax it with the given potential and compute the
    target property, through the same tool handlers. The final check calls it again with the
    final potential (Section 9).

    Parameters
    ----------
    ws : Workspace
        Workspace; the temporary ``ref`` candidate is removed afterwards.
    handlers : mapping
        Tool handlers.
    potential : str, optional
        Potential; defaults to the screening potential ``ws.cfg.potential``.
    fmax : float
        Force threshold for the relaxation, in eV/Å.

    Returns
    -------
    spec : SearchSpec
        Spec with each target's ``reference_value`` and ``value`` set, where it worked.
    errors : list of ErrorRecord
        What failed.
    """
    spec = ws.spec
    errors: list[ErrorRecord] = []
    for i, t in enumerate(spec.targets()):
        if not t.reference_material or t.reference_value is not None or not t.evaluable:
            continue
        pred, err = _reference_value(ws, spec.target_view(i), handlers, potential, fmax)
        if err:
            errors.append(err)
            continue
        assert pred is not None
        spec = spec.with_target(i, t.model_copy(update={"reference_value": pred, "value": pred.value}))
        spec.assumptions = [
            *spec.assumptions,
            f"Target value for {t.property} is the model's own prediction for {t.reference_material} "
            f"({pred.value:.3g} {pred.units}), so candidates and reference sit on the same scale (GP-11).",
        ]
    return spec, errors


def _reference_value(
    ws: Workspace, view: SearchSpec, handlers: Mapping[str, Handler], potential: str | None, fmax: float
) -> tuple[Quantity | None, ErrorRecord | None]:
    """Compute one target's value for its reference material.

    Parameters
    ----------
    ws : Workspace
        Workspace; a temporary ``ref`` candidate is added and removed.
    view : SearchSpec
        Spec whose primary target is the one to compute.
    handlers : mapping
        Tool handlers.
    potential : str or None
        Potential; the screening one by default.
    fmax : float
        Force threshold, eV/Å.

    Returns
    -------
    value : Quantity or None
        The reference value.
    error : ErrorRecord or None
        What failed.
    """
    ref = view.reference_material
    assert ref
    cand = Candidate(
        candidate_id="ref",
        iteration=0,
        composition=ref,
        structure_type="perovskite_cubic",
        rationale="Reference material for a relative target.",
    )
    ws.candidates["ref"] = CandidateState(candidate=cand, round=0)
    calls: list[tuple[str, dict[str, Any]]] = [
        ("build_structure", {"candidate_id": "ref", "source": "auto", "mp_id": None, "seed": ws.cfg.seed}),
        ("relax", {"candidate_id": "ref", "potential": potential or ws.cfg.potential, "fmax": fmax}),
        target_step(view, "ref"),
    ]
    result: dict[str, Any] = {}
    try:
        for name, args in calls:
            result, is_error = execute(ws, name, args, handlers)
            if is_error:
                msg = f"{view.target_property}: {result.get('error')}"
                return None, ErrorRecord(step="reference", code=f"{name}_failed", message=msg[:300])
    finally:
        ws.candidates.pop("ref", None)
    return Quantity(
        value=float(result["value"]),
        uncertainty=result.get("uncertainty"),
        units=result.get("units", view.target_units),
        source=f"{result.get('model')} on {ref}",
    ), None


def run(
    raw_goal: str = "",
    cfg: RunConfig | None = None,
    spec: SearchSpec | None = None,
    answers: dict[str, str] | None = None,
    deps: Deps | None = None,
    run_id: str | None = None,
) -> RunState:
    """Run a goal end to end: parse, campaign, final re-check, brief (OR-1, SYS-1).

    Parameters
    ----------
    raw_goal : str
        The goal; may be empty when ``spec`` is given.
    cfg : RunConfig, optional
        Options; defaults to ``RunConfig()``.
    spec : SearchSpec, optional
        Parsed spec; skips the goal parser.
    answers : dict, optional
        Answers to the parser's clarifying questions.
    deps : Deps, optional
        Injected dependencies (tests, offline demos).
    run_id : str, optional
        Run ID to use.

    Returns
    -------
    RunState
        Final state; ``status`` is ``needs_input`` when the parser has questions.
    """
    cfg = cfg or RunConfig()
    deps = deps or Deps()
    state, files = new_run(raw_goal or (spec.raw_goal if spec else ""), cfg, spec, deps.root, run_id)
    return _drive(state, files, deps, answers, resume=False)


def resume(run_id: str, deps: Deps | None = None) -> RunState:
    """Resume an interrupted run in the same folder (OR-6).

    Parameters
    ----------
    run_id : str
        Run to resume.
    deps : Deps, optional
        Injected dependencies.

    Returns
    -------
    RunState
        Final state.

    Raises
    ------
    RunRefused
        If the run finished, is running, or does not exist.
    """
    deps = deps or Deps()
    files = RunFiles.for_run(run_id, deps.root)
    if not files.state.exists():
        raise RunRefused(f"No run {run_id}.")
    if is_active(run_id):
        raise RunRefused(f"Run {run_id} is already running.")
    state, _, _ = load_for_resume(files)
    if state.status == "finished":
        raise RunRefused(f"Run {run_id} already finished.")
    return _drive(state, files, deps, None, resume=True)


def _drive(
    state: RunState, files: RunFiles, deps: Deps, answers: dict[str, str] | None, resume: bool
) -> RunState:
    """Run the outer steps from wherever the state is (OR-1).

    Parameters
    ----------
    state : RunState
        Current state.
    files : RunFiles
        Run paths.
    deps : Deps
        Dependencies.
    answers : dict or None
        Parser answers.
    resume : bool
        Whether this resumes an interrupted run.

    Returns
    -------
    RunState
        Final state.
    """
    rid = state.run_id
    stop = threading.Event()
    abort = threading.Event()
    with _LOCK:
        if rid in _STOPS:
            raise RunRefused(f"Run {rid} is already running.")
        _STOPS[rid] = stop
        _ABORTS[rid] = abort
    bus = deps.bus
    try:
        llm, reviewer = _models(state.config, deps, files)
        meter = UsageMeter(state, llm, reviewer)
        bus.emit(
            rid,
            "run",
            "started",
            {"resume": resume, "goal": state.raw_goal, "config": state.config.model_dump()},
        )
        if state.spec is None and not _parse(state, files, llm, answers, bus):
            return state
        handlers = dict(deps.handlers) if deps.handlers is not None else load_default_handlers()
        ws, critiques = _workspace(state, files, resume)
        outcome = _capability(state, files, ws, llm, handlers, deps, stop)
        if outcome in ("cancelled_run", "off"):
            if outcome == "off":
                state.status = "failed"
            else:
                state.status, state.stop_reason = "stopped", "cancelled by user"
            meter.update()
            save_json(files.state, state)
            bus.emit(
                rid,
                "run",
                "failed" if outcome == "off" else "finished",
                {
                    "status": state.status,
                    "stop_reason": state.stop_reason,
                    "best": None,
                    "errors": [e.model_dump() for e in state.errors],
                },
            )
            return state
        handlers.update(custom.custom_handlers(ws, deps.toolkit_factory))
        needs_ref = any(
            t.reference_material and t.reference_value is None and t.evaluable for t in ws.spec.targets()
        )
        if needs_ref and not ws.candidates:
            spec, errors = resolve_reference(ws, handlers)
            state.errors += errors
            ws.spec = state.spec = spec
            if any(t.value is None and t.evaluable for t in spec.targets()):
                state.status = "failed"
                state.errors.append(
                    ErrorRecord(step="reference", code="no_target", message="Reference value unavailable.")
                )
                save_json(files.state, state)
                bus.emit(rid, "run", "failed", {"errors": [e.model_dump() for e in state.errors]})
                return state
        state.status = "running"
        if ws.submitted is None:
            _campaign(state, files, ws, critiques, llm, reviewer, handlers, deps, stop, resume, meter)
        else:
            state.stop_reason = state.stop_reason or "submitted"
        sync_state(state, ws, [it.critique for it in state.iterations if it.critique])
        write_artifacts(files, state)
        if not abort.is_set():
            _final(state, files, ws, deps, bus, handlers, stop)
        if not abort.is_set():
            _brief(state, files, ws, llm, bus)
        state.status = "stopped" if abort.is_set() else "finished"
        if abort.is_set():
            state.stop_reason = "user_abort"
        meter.update()
        save_json(files.state, state)
        bus.emit(
            rid,
            "run",
            "finished",
            {
                "status": state.status,
                "stop_reason": state.stop_reason,
                "best": state.best,
                "errors": len(state.errors),
            },
        )
        return state
    finally:
        with _LOCK:
            _STOPS.pop(rid, None)
            _ABORTS.pop(rid, None)
            ACTIVE.pop(rid, None)


def _parse(state: RunState, files: RunFiles, llm: LLM, answers: dict[str, str] | None, bus: EventBus) -> bool:
    """Parse the goal into ``state.spec``; stop with ``needs_input`` when questions remain (GP-5).

    Parameters
    ----------
    state : RunState
        State to update.
    files : RunFiles
        Run paths.
    llm : LLM
        Parser model.
    answers : dict or None
        Answers to clarifying questions.
    bus : EventBus
        Events.

    Returns
    -------
    bool
        True when the spec is ready.
    """
    bus.emit(state.run_id, "goal_parser", "started", {"goal": state.raw_goal})
    spec, questions, errors = parse_goal(llm, state.raw_goal, state.config, answers)
    state.errors += errors
    if spec is None or questions or validate_for_run(spec, state.config):
        state.status = "needs_input" if spec is not None else "failed"
        state.spec = spec
        save_json(files.state, state)
        bus.emit(
            state.run_id,
            "goal_parser",
            "failed",
            {
                "questions": questions,
                "errors": [e.message for e in errors],
                "spec": spec.model_dump() if spec else None,
            },
        )
        return False
    state.spec = spec
    save_json(files.spec, spec)
    save_json(files.state, state)
    bus.emit(state.run_id, "goal_parser", "finished", {"spec": spec.model_dump()})
    return True


def _capability(
    state: RunState,
    files: RunFiles,
    ws: Workspace,
    llm: LLM,
    handlers: Mapping[str, Handler],
    deps: Deps,
    stop: threading.Event,
) -> str | None:
    """Fill every capability gap before the campaign: reuse library tools or run phases (Section 14).

    Each target no built-in tool computes gets its own capability phase, one after another; a
    phase sees the spec with that target as its primary one.

    Parameters
    ----------
    state : RunState
        State; each gap's target gains ``tool``, or ``evaluable=False`` when it gets none.
    files : RunFiles
        Run paths.
    ws : Workspace
        Workspace; approved tools go into ``ws.custom_tools``.
    llm : LLM
        The orchestrator's model.
    handlers : mapping
        Tool handlers.
    deps : Deps
        Decision callback and toolkit factory.
    stop : threading.Event
        A user stop.

    Returns
    -------
    str or None
        ``None`` without a gap; ``"off"`` or ``"cancelled_run"`` to end the run; else the last
        outcome.
    """
    last: str | None = None
    for i in range(len(ws.spec.targets())):
        outcome = _capability_one(state, files, ws, llm, handlers, deps, stop, i)
        if outcome in ("off", "cancelled_run"):
            return outcome
        last = outcome or last
    return last


def _capability_one(
    state: RunState,
    files: RunFiles,
    ws: Workspace,
    llm: LLM,
    handlers: Mapping[str, Handler],
    deps: Deps,
    stop: threading.Event,
    i: int,
) -> str | None:
    """Fill one target's gap, if it has one.

    Parameters
    ----------
    state, files, ws, llm, handlers, deps, stop
        As for ``_capability``.
    i : int
        Index into ``spec.targets()``.

    Returns
    -------
    str or None
        ``None`` without a gap; ``"reused"``, ``"off"``, or the phase's outcome.
    """
    cfg = state.config
    t = ws.spec.targets()[i]
    if t.tool or not t.evaluable or t.property == "unsupported":
        return None
    if t.property in custom.run_registry(cfg, []):
        return None
    approved, _ = custom.scan_library(custom.library_dir(cfg))
    reused = next((x for x in approved if x.property == t.property), None)
    if reused is not None:  # AT-22: an approved library tool fills the gap without a proposal
        ws.custom_tools[reused.name] = reused
        _set_target_tool(
            state,
            ws,
            i,
            reused,
            f"The agent-built tool {reused.name} from the tool library computes {t.property}.",
        )
        deps.bus.emit(
            state.run_id,
            "capability",
            "finished",
            {"kind": "library", "reused": [reused.name], "property": t.property},
        )
        save_snapshot(files, state, ws, [])
        return "reused"
    view = ws.spec.target_view(i)
    gap = find_gap(view, cfg, approved)
    if gap is None:
        return None
    if cfg.custom_tools == "off":  # AT-3: a gap ends the run before the campaign
        state.errors.append(
            ErrorRecord(
                step="capability",
                code="capability_gap",
                message=f"No tool can compute {gap!r}, and agent-built tools are off for this run.",
            )
        )
        return "off"
    folder = files.root / ("capability" if i == 0 else f"capability_{gap}")
    saved = load_state(folder)
    state.status = "capability"
    save_json(files.state, state)
    phase_ws = ws.model_copy(update={"spec": view})  # shares candidates and custom_tools with ws
    phase = CapabilityPhase(
        phase_ws,
        llm,
        handlers,
        deps.bus,
        folder,
        gap,
        decide=deps.decide,
        stop=stop,
        state=saved,
        toolkit_factory=deps.toolkit_factory,
    )
    outcome = phase.run(resume=saved is not None)
    state.errors += phase.errors
    for k, v in phase.usage.items():
        state.usage[f"capability_{k}"] = state.usage.get(f"capability_{k}", 0) + int(v)
    if outcome == "cancelled_run":
        save_json(files.state, state)
        return outcome
    tool = next((x for x in phase.state.registered if x.property == gap), None)
    if outcome == "filled" and tool is not None:
        _set_target_tool(
            state, ws, i, tool, f"The agent-built tool {tool.name}, approved by the user, computes {gap}."
        )
    else:
        why = {
            "cancelled_tool": "the user cancelled the tool for it",
            "budget": "the capability phase ran out of time",
            "failed": phase.state.end_reason or "no tool could be built for it",
        }
        new = ws.spec.with_target(i, t.model_copy(update={"evaluable": False}))
        new.assumptions.append(
            f"The target property {gap} was not evaluated: {why.get(outcome, 'no tool could be built for it')}; "
            "the run did not check it (AT-8)."
        )
        ws.spec = state.spec = new
    state.status = "running"
    save_json(files.spec, state.spec)
    save_snapshot(files, state, ws, [])
    return outcome


def _set_target_tool(state: RunState, ws: Workspace, i: int, tool: Any, assumption: str) -> None:
    """Point target ``i`` at an agent-built tool (AT-16, AT-17).

    Parameters
    ----------
    state : RunState
        State whose spec is replaced.
    ws : Workspace
        Workspace whose spec is replaced.
    i : int
        Index into ``spec.targets()``.
    tool : CustomTool
        The tool.
    assumption : str
        Recorded for the brief.
    """
    t = ws.spec.targets()[i]
    new = ws.spec.with_target(i, t.model_copy(update={"tool": f"custom_{tool.name}", "units": tool.units}))
    new.assumptions.append(assumption + " Its values carry no uncertainty (agent-built tool).")
    ws.spec = state.spec = new


def _workspace(state: RunState, files: RunFiles, resume: bool) -> tuple[Workspace, list[Any]]:
    """Fresh workspace, or the restored one on resume.

    Parameters
    ----------
    state : RunState
        Run state with a spec.
    files : RunFiles
        Run paths.
    resume : bool
        Whether to restore.

    Returns
    -------
    tuple
        ``(workspace, critiques)``.
    """
    if resume and (files.workspace.exists() or state.iterations):
        _, ws, critiques = load_for_resume(files)
        return ws, critiques
    assert state.spec is not None
    state.spec = apply_stability_options(state.spec, state.config)
    return Workspace(run_id=state.run_id, spec=state.spec, cfg=state.config), []


def _campaign(
    state: RunState,
    files: RunFiles,
    ws: Workspace,
    critiques: list[Any],
    llm: LoggedLLM,
    reviewer: LoggedLLM,
    handlers: Mapping[str, Handler],
    deps: Deps,
    stop: threading.Event,
    resume: bool,
    meter: UsageMeter | None = None,
) -> None:
    """Run the agent campaign and record its outcome.

    Parameters
    ----------
    state : RunState
        State to update.
    files : RunFiles
        Run paths.
    ws : Workspace
        Workspace.
    critiques : list of Critique
        Reviews already done.
    llm, reviewer : LoggedLLM
        Models.
    handlers : mapping
        Tool handlers.
    deps : Deps
        Dependencies.
    stop : threading.Event
        Set by ``request_stop``.
    resume : bool
        Whether this resumes.
    meter : UsageMeter, optional
        Keeps ``state.usage`` token totals current.
    """
    offset = float(state.usage.get("campaign_s", 0))

    def snapshot(c: Campaign) -> None:
        """Persist workspace and state after a tool call (OR-6).

        Parameters
        ----------
        c : Campaign
            The campaign.
        """
        llm.iteration = reviewer.iteration = c.ws.round
        state.usage["campaign_s"] = int(c.elapsed_s)
        if meter:
            meter.update(c.usage)
        state.errors = [*base_errors, *c.errors]
        save_snapshot(files, state, c.ws, c.critiques)
        if c.critiques and len(c.critiques) != n_written[0]:
            n_written[0] = len(c.critiques)
            write_artifacts(files, state)

    base_errors = list(state.errors)
    n_written = [len(critiques)]
    camp = Campaign(
        ws,
        llm,
        reviewer_llm=reviewer,
        handlers=handlers,
        bus=deps.bus,
        critiques=critiques,
        novelty_fn=deps.novelty_fn,
        snapshot=snapshot,
        elapsed_offset_s=offset,
        tools=custom.campaign_tools(ws),
    )
    if stop.is_set():
        camp.request_stop()
    with _LOCK:
        ACTIVE[state.run_id] = camp
    save_snapshot(files, state, ws, camp.critiques)
    sub = camp.run(resume=resume)
    state.stop_reason = camp.stop_rule
    state.errors = [*base_errors, *camp.errors]
    state.usage["campaign_s"] = int(camp.elapsed_s)
    if meter:
        meter.update(camp.usage)
    save_snapshot(files, state, ws, camp.critiques)
    if sub is None:
        state.errors.append(
            ErrorRecord(step="orchestrator", code="no_candidates", message="No candidate was proposed.")
        )


def _final(
    state: RunState,
    files: RunFiles,
    ws: Workspace,
    deps: Deps,
    bus: EventBus,
    handlers: Mapping[str, Handler],
    stop: threading.Event | None = None,
) -> None:
    """Run the higher-fidelity re-check of the finalists (Section 9); failures degrade the run.

    Parameters
    ----------
    state : RunState
        State to update.
    files : RunFiles
        Run paths.
    ws : Workspace
        Workspace.
    deps : Deps
        Dependencies.
    bus : EventBus
        Events.
    handlers : mapping
        The run's tool handlers, used by the default final check.
    stop : threading.Event, optional
        Set by ``request_stop``; the default final check then skips its optional items.
    """
    if not ws.cfg.run_final or ws.submitted is None or state.validation:
        return
    fn = deps.final_check
    if fn is None and (default := default_final_check()) is not None:

        def on_step(status: str, tool: str, info: dict[str, Any]) -> None:
            """Publish one final-check tool call (``final_step`` events, for live progress).

            Parameters
            ----------
            status : {"started", "finished", "failed"}
                Step status.
            tool : str
                Tool name.
            info : dict
                ``candidate_id``, ``potential``, ``mode`` and, at the end, ``wall_s``.
            """
            bus.emit(state.run_id, "final_step", status, {"tool": tool, **info})  # type: ignore[arg-type]

        fn = lambda w, sub: default(  # noqa: E731
            w, sub, handlers=handlers, on_step=on_step, should_stop=stop.is_set if stop else None
        )
    if fn is None:
        state.errors.append(
            ErrorRecord(step="final_calculation", code="unavailable", message="Final check not implemented.")
        )
        return
    state.status = "final_check"
    save_json(files.state, state)
    bus.emit(
        state.run_id,
        "final_calculation",
        "started",
        {"candidates": [ws.submitted.candidate_id, ws.submitted.runner_up_id]},
    )
    try:
        state.validation = list(fn(ws, ws.submitted))
    except Exception as exc:
        state.errors.append(ErrorRecord(step="final_calculation", code="failed", message=str(exc)[:300]))
        bus.emit(state.run_id, "final_calculation", "failed", {"error": str(exc)[:300]})
        return
    for r in state.validation:
        save_json(files.root / "final" / f"validation_{r.candidate_id}.json", r)
    if state.validation:
        save_json(files.root / "final" / "validation.json", state.validation[0])
        w = state.validation[0]
        bus.emit(
            state.run_id,
            "final_calculation",
            "finished",
            {"verdict": w.verdict, "checks": [c.model_dump() for c in w.checks], "notes": w.notes},
        )


def _brief(state: RunState, files: RunFiles, ws: Workspace, llm: LLM, bus: EventBus) -> None:
    """Write the brief (Section 10).

    Parameters
    ----------
    state : RunState
        State to update.
    files : RunFiles
        Run paths.
    ws : Workspace
        Workspace.
    llm : LLM
        Writer model.
    bus : EventBus
        Events.
    """
    state.status = "writing_brief"
    save_json(files.state, state)
    bus.emit(state.run_id, "brief_generator", "started", {})
    brief, errors = assemble_brief(state, ws, llm)
    state.errors += errors
    atomic_write_text(files.brief_md, brief.markdown)
    try:  # a failed PDF degrades the run; the Markdown brief stands (EB-10)
        brief.pdf_path = str(export_pdf(brief.markdown, files.brief_pdf, state.run_id, state.raw_goal))
    except Exception as exc:
        state.errors.append(ErrorRecord(step="brief_pdf", code="export_failed", message=str(exc)[:300]))
    state.brief = brief
    compute = write_compute_brief(state, ws, files)
    bus.emit(
        state.run_id,
        "brief_generator",
        "finished",
        {"pdf": brief.pdf_path, "cited": brief.cited_paper_ids, "compute": compute},
    )


def write_compute_brief(state: RunState, ws: Workspace, files: RunFiles) -> bool:
    """Write the computational brief and its PDF; a failure degrades the run (Section 10).

    Parameters
    ----------
    state : RunState
        Run state with the submission; errors are added to it.
    ws : Workspace
        Workspace.
    files : RunFiles
        Run paths.

    Returns
    -------
    bool
        True when the Markdown was written.
    """
    try:
        md = assemble_compute_brief(state, ws).markdown
        atomic_write_text(files.compute_brief_md, md)
    except Exception as exc:
        state.errors.append(ErrorRecord(step="compute_brief", code="failed", message=str(exc)[:300]))
        return False
    try:
        export_pdf(md, files.compute_brief_pdf, state.run_id, state.raw_goal)
    except Exception as exc:
        state.errors.append(
            ErrorRecord(step="compute_brief_pdf", code="export_failed", message=str(exc)[:300])
        )
    return True
