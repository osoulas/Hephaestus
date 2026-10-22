"""Turn ``ProgressEvent``s into terminal output (OR-7, UI-2 to UI-5, UI-14, UI-15).

``Renderer`` is pure: it takes events and returns rich renderables, and keeps only the state a
display needs (calls in flight, the latest fields per candidate, round scores). It never decides
a verdict, a target check, a gate or a stop: every tick and cross comes from a field the harness
already computed (``thermo_stable``, ``dyn_stable``, ``verdict``, ``checks``, ``gate_note``).

A tool call is printed once, when it finishes, with its result underneath, so calls that run in
parallel never interleave; calls still running are shown in the live status line instead.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from hephaestus.core.types import ProgressEvent, RunState
from hephaestus.interfaces.cli.cards import (
    cancelled_text,
    code_card,
    decision_text,
    plan_card,
    request_card,
)
from hephaestus.interfaces.cli.theme import (
    DASH,
    GLYPH,
    Gutter,
    bar,
    elapsed,
    ember_frame,
    fmt_num,
    fmt_quantity,
    mark,
    panel,
    sparkline,
    strip_codes,
)

THINKING_LINES = 2  # lines of thinking shown unless verbose
RUBRIC_MAX = {"property": 30, "thermo": 20, "dynamic": 20, "reliability": 15, "novelty_synthesis": 15}
DIRECTION = {"minimise": "<", "maximise": ">", "hit": "≈"}

# Arguments shown in a call line, in order; the rest of the input is left out.
CALL_ARGS: dict[str, list[str]] = {
    "search_literature": ["query", "mode", "limit"],
    "related_papers": ["paper_id", "relation", "limit"],
    "get_paper": ["paper_id"],
    "publication_trend": ["query"],
    "note_evidence": ["paper_id", "finding_type", "composition"],
    "lookup_materials_project": ["composition"],
    "propose_candidate": ["composition", "structure_type"],
    "build_structure": ["candidate_id", "source", "mp_id"],
    "relax": ["candidate_id", "potential", "fmax"],
    "energy_above_hull": ["candidate_id"],
    "phonons": ["candidate_id", "mode"],
    "run_md": ["candidate_id", "duration_ps"],
    "predict_property": ["candidate_id", "property"],
    "candidate_status": [],
    "request_review": [],
    "submit_final": ["candidate_id", "runner_up_id"],
    # The tool builder's own tools (agent-built tools, Section 14)
    "read_tool_source": ["function"],
    "propose_tool": ["name"],
    "write_tool": ["proposal_id"],
    "request_package": ["module"],
    "ask_user": [],
    "give_up": [],
}

# What the status line says while a tool runs.
VERBS: dict[str, str] = {
    "search_literature": "Searching the literature",
    "related_papers": "Following related papers",
    "get_paper": "Reading a paper",
    "publication_trend": "Charting the field",
    "note_evidence": "Noting evidence",
    "lookup_materials_project": "Checking Materials Project",
    "propose_candidate": "Proposing a candidate",
    "build_structure": "Building {cid}",
    "relax": "Relaxing {cid}",
    "energy_above_hull": "Computing the hull for {cid}",
    "phonons": "Computing phonons for {cid}",
    "run_md": "Running MD on {cid}",
    "predict_property": "Predicting {prop} for {cid}",
    "candidate_status": "Checking candidates",
    "request_review": "Asking the reviewer",
    "submit_final": "Submitting",
    "read_tool_source": "Reading how our tools are written",
    "propose_tool": "Planning a tool",
    "write_tool": "Writing and testing the tool",
    "request_package": "Asking to use a package",
    "ask_user": "Asking you a question",
    "give_up": "Proposing to stop",
}

# Names of the final re-check's steps (``final_step`` events), as the web UI shows them.
FINAL_STEP_NAMES: dict[str, str] = {
    "relax": "Re-relax",
    "phonons": "Full phonons",
    "energy_above_hull": "Hull",
    "predict_property": "Target property",
    "build_structure": "Build structure",
    "run_md": "MD",
}

CANDIDATE_FIELDS = (
    "composition",
    "cand_round",
    "e_hull",
    "thermo_stable",
    "dyn_stable",
    "min_freq",
    "verdict",
    "space_group",
    "value",
    "uncertainty",
    "novelty",
    "targets",
)


@dataclass
class _Call:
    """A tool call that has started but not finished."""

    tool: str
    input: dict[str, Any]
    started: datetime
    local_start: float
    candidate_id: str | None = None


@dataclass
class Renderer:
    """Stateful event formatter; ``feed`` returns what to print, ``status`` the live line.

    Attributes
    ----------
    verbose : bool
        Show thinking in full and longer results.
    clock : callable
        Monotonic clock, injectable for tests.
    """

    verbose: bool = False
    clock: Callable[[], float] = time.monotonic
    calls: dict[Any, _Call] = field(default_factory=dict)
    candidates: dict[str, dict[str, Any]] = field(default_factory=dict)
    rounds: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    spec: dict[str, Any] | None = None
    tool_calls: int = 0
    round: int = 1
    run_id: str | None = None
    goal: str = ""
    backend: str | None = None
    model: str | None = None
    phase: str = "Starting"
    stopping: bool = False
    stop_all: bool = False  # the user stopped everything: no re-check or brief follows
    submitted: dict[str, Any] | None = None
    stop_rule: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] | None = None
    brief: dict[str, Any] | None = None
    end: dict[str, Any] | None = None
    first_time: datetime | None = None
    last_time: datetime | None = None
    _local_start: float | None = None
    live: bool = False  # True while watching a running run: open time spans run up to the clock
    replaying: bool = False  # Ctrl+C ends a replay rather than stopping a campaign
    menu: bool = False  # Ctrl+C opens a menu (someone can answer), so the hint names its choices
    campaign_span: list[datetime | None] = field(default_factory=lambda: [None, None])
    final_span: list[datetime | None] = field(default_factory=lambda: [None, None])
    in_campaign: bool = False  # draw the timeline gutter from the campaign's start to the run's end
    final_steps: dict[str, _Call] = field(default_factory=dict)  # open final re-check steps, by tool
    _pending: list[RenderableType] = field(default_factory=list)  # round line held until its review prints
    # Agent-built tools (Section 14): the property being built for, and every card by (phase, id, stage).
    cap_property: str | None = None
    agent_built: set[str] = field(default_factory=set)  # properties computed by an agent-built tool
    cards: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)
    answered: set[tuple[str, str, int]] = field(default_factory=set)  # sent from here, decision event pending

    # ------------------------------------------------------------------ dispatch

    def feed(self, event: ProgressEvent) -> list[RenderableType]:
        """Take one event and return what to print for it.

        Parameters
        ----------
        event : ProgressEvent
            The next event of the run.

        Returns
        -------
        list of RenderableType
            Zero or more renderables, in print order.
        """
        if self._local_start is None:
            self._local_start = self.clock()
        self.run_id = self.run_id or event.run_id
        self.first_time = self.first_time or event.time
        self.last_time = event.time
        self._track_spans(event)
        handler = getattr(self, f"_on_{event.section}", None)
        if handler is None:
            return [self._line(GLYPH["dot"], f"[muted]{event.section} {event.status}[/]", "muted")]
        out: list[RenderableType] = handler(event)
        if self.in_campaign:
            out = [x if isinstance(x, Rule) else Gutter(x) for x in out]
        if event.section == "run" and event.status != "started":
            self.in_campaign = False
        return out

    def _track_spans(self, e: ProgressEvent) -> None:
        """Keep the campaign and final re-check time spans from event timestamps, as the web UI does.

        The campaign runs from the first orchestrator start to the submission (else the final
        re-check's start, else the run's end); a resume reopens it. The re-check runs from its
        start to its end, or the run's end.

        Parameters
        ----------
        e : ProgressEvent
            The event being fed.
        """
        c, f = self.campaign_span, self.final_span
        if e.section == "orchestrator" and e.status == "started":
            c[0] = c[0] or e.time
            c[1] = None
        elif e.section == "orchestrator" and "submitted" in e.payload:
            c[1] = c[1] or e.time
        elif e.section == "final_calculation":
            if e.status == "started":
                c[1] = c[1] or (e.time if c[0] else None)
                f[0], f[1] = e.time, None
            else:
                f[1] = e.time
        elif e.section == "run" and e.status != "started":
            if c[0] and not c[1]:
                c[1] = e.time
            if f[0] and not f[1]:
                f[1] = e.time

    def _span_s(self, span: list[datetime | None]) -> float | None:
        """Seconds in a time span; an open one runs to now (live) or to the last event (replays).

        Parameters
        ----------
        span : list
            ``[start, end]`` timestamps.

        Returns
        -------
        float or None
            Duration, or ``None`` before the span starts.
        """
        start, end = span
        if start is None:
            return None
        if end is None:
            from hephaestus.core.types import utcnow

            end = utcnow() if self.live else (self.last_time or start)
        return max((end - start).total_seconds(), 0.0)

    def budget_text(self) -> Text:
        """Campaign and final re-check time against their budgets (OR-9, FC-9); display only.

        The harness enforces the budgets; this only colours the time amber above 80 % and red
        at 100 %.

        Returns
        -------
        rich.text.Text
            E.g. ``"campaign 4m 12s / 15m · re-check - / 10m"`` (a dash before it starts).
        """
        text = Text()
        parts = [("campaign", self.campaign_span, self.config.get("time_budget_s"))]
        if self.config.get("run_final", True):
            parts.append(("re-check", self.final_span, self.config.get("final_time_budget_s")))
        for i, (label, span, budget) in enumerate(parts):
            if i:
                text.append(f"  {GLYPH['sep']} ", style="gutter")
            used = self._span_s(span)
            style = "muted"
            if used is not None and budget:
                frac = used / float(budget)
                style = "bad" if frac >= 1 else ("warn" if frac > 0.8 else "muted")
            text.append(f"{label} ", style="faint")
            text.append(elapsed(used) if used is not None else DASH, style=style)
            if budget:
                text.append(f" / {elapsed(float(budget))}", style="faint")
            elif budget is not None:
                text.append(" · no limit", style="faint")  # a budget of 0 (OR-9)
        return text

    def candidate_facts(self, cid: str) -> str:
        """One-line facts for a candidate: composition, target value ± uncertainty, hull and phonon verdicts.

        Parameters
        ----------
        cid : str
            Candidate ID.

        Returns
        -------
        str
            Markup; the ticks and crosses are the harness's verdicts.
        """
        c = self.candidates.get(cid, {})
        facts = [str(c.get("composition") or "")]
        rows = c.get("targets") or []
        if rows:  # several targets: every one must be met (GP-1)
            for r in rows:
                facts.append(target_fact(r))
        elif c.get("value") is not None:
            prop_key = str((self.spec or {}).get("target_property", "value"))
            prop = prop_key.replace("_", " ")
            units = (self.spec or {}).get("target_units", "")
            if self.is_agent_built(prop_key):
                facts.append(f"{prop} {fmt_quantity(c.get('value'), None, units)} [accent](agent-built)[/]")
            else:
                facts.append(f"{prop} {fmt_quantity(c.get('value'), c.get('uncertainty'), units)}")
        if c.get("e_hull") is not None:
            facts.append(f"E_hull {fmt_num(c.get('e_hull'), 3)} eV/atom {mark(c.get('thermo_stable'))}")
        if c.get("verdict"):
            facts.append(f"phonons {c.get('verdict')} {mark(c.get('dyn_stable'))}")
        return " · ".join(f for f in facts if f)

    def is_agent_built(self, prop: str) -> bool:
        """Whether a target property is computed by an agent-built tool (AT-17).

        Parameters
        ----------
        prop : str
            Property key.

        Returns
        -------
        bool
            True when the spec names a tool for it or a result came from one.
        """
        spec = self.spec or {}
        if prop in self.agent_built:
            return True
        if prop == spec.get("target_property") and spec.get("target_tool"):
            return True
        return any(t.get("property") == prop and t.get("tool") for t in spec.get("extra_targets") or [])

    def _on_run(self, e: ProgressEvent) -> list[RenderableType]:
        """Run header, or the run's end.

        Parameters
        ----------
        e : ProgressEvent
            A ``run`` event.

        Returns
        -------
        list of RenderableType
            Output.
        """
        p = e.payload
        if e.status == "started":
            self.config = p.get("config") or {}
            self.goal = p.get("goal") or self.goal
            verb = "Resuming" if p.get("resume") else "Run"
            cfg = self.config
            opts = (
                GLYPH["dot"]
                .join(
                    f" {x} "
                    for x in (
                        cfg.get("llm_backend"),
                        cfg.get("llm_model"),
                        _limit(cfg.get("max_iterations"), "rounds"),
                        _limit(cfg.get("max_tool_calls"), "tool calls"),
                        "novel only" if cfg.get("require_novel") else None,
                    )
                    if x
                )
                .strip()
            )
            head = Text.assemble((f" {verb.upper()} ", "tab.accent"), " ", (e.run_id, "id"))
            title = (self.spec or {}).get("title")
            if title:
                head.append(f"  {strip_codes(str(title))}", style="bold")
            return [
                Text(),
                head,
                Text(f"  {self.goal}", style="muted"),
                Text(f"  {opts}", style="faint"),
                Text(),
            ]
        self.end = {**p, "status": p.get("status", e.status)}
        if e.status == "failed":
            errors = p.get("errors") or []
            lines = [
                f"[bad]{x.get('step', '?')}[/] {x.get('code', '')}: {x.get('message', '')}" for x in errors
            ]
            return [panel("\n".join(lines) or "The run failed.", "run failed", "bad")]
        if p.get("status") == "stopped":
            reason = _escape(str(p.get("stop_reason") or "stopped by the user"))
            return [self._line(GLYPH["mark"], f"[warn]Run stopped[/] · {reason}", "warn")]
        return []

    def _on_goal_parser(self, e: ProgressEvent) -> list[RenderableType]:
        """Show the parsed specification, or the parser's questions.

        Parameters
        ----------
        e : ProgressEvent
            A ``goal_parser`` event.

        Returns
        -------
        list of RenderableType
            Output.
        """
        if e.status == "started":
            self.phase = "Parsing the goal"
            return []
        spec = e.payload.get("spec")
        if e.status == "finished" and spec:
            self.spec = spec
            return [spec_panel(spec)]
        return [questions_panel(e.payload.get("questions") or [], e.payload.get("errors") or [])]

    def _on_orchestrator(self, e: ProgressEvent) -> list[RenderableType]:
        """Campaign start, round results and the submission.

        Parameters
        ----------
        e : ProgressEvent
            An ``orchestrator`` event.

        Returns
        -------
        list of RenderableType
            Output.
        """
        p = e.payload
        if e.status == "started":
            self.backend, self.model = p.get("backend"), p.get("model")
            self.phase = "Thinking"
            verb = "Campaign resumed" if p.get("resume") else "Campaign started"
            self.in_campaign = True
            return [
                round_rule(f"round {self.round}"),
                self._line(GLYPH["mark"], f"{verb} on [accent]{self.backend}[/] · {self.model}", "ember"),
            ]
        if "round" in p:
            self.rounds.append(p)
            if not p.get("stop"):
                self.round = int(p["round"]) + 1
            lines = self._round_lines(p)
            if any(c.tool == "request_review" for c in self.calls.values()):
                self._pending.extend(lines)  # the harness emits this before the review call finishes
                return []
            return lines
        if "submitted" in p:
            self.submitted = p.get("submitted")
            self.stop_rule = p.get("rule")
            self.usage = p.get("usage") or {}
            self.phase = "Re-checking at higher fidelity"
            return [self._submission_line(self.submitted, self.stop_rule)]
        return []

    def _on_agent(self, e: ProgressEvent) -> list[RenderableType]:
        """Orchestrator text (Markdown) and thinking (dim, shortened unless verbose).

        Parameters
        ----------
        e : ProgressEvent
            An ``agent`` event.

        Returns
        -------
        list of RenderableType
            Output.
        """
        text = str(e.payload.get("text") or "").strip()
        if not text:
            return []
        if e.payload.get("kind") == "thinking":
            lines = text.splitlines()
            if not self.verbose and len(lines) > THINKING_LINES:
                lines = [*lines[:THINKING_LINES], f"… ({len(text.splitlines()) - THINKING_LINES} more lines)"]
            self.phase = "Thinking"
            return [self._line(GLYPH["think"], Text("\n".join(lines), style="thinking"), "thinking")]
        return [self._line(GLYPH["say"], Markdown(text), "accent")]

    def _on_tool(self, e: ProgressEvent) -> list[RenderableType]:
        """One block per finished call: the call line and its one-line result.

        Parameters
        ----------
        e : ProgressEvent
            A ``tool`` event.

        Returns
        -------
        list of RenderableType
            Output; nothing for ``started``.
        """
        p = e.payload
        tool = str(p.get("tool"))
        key = p.get("call_id", f"{tool}:{p.get('input')}")
        cid = p.get("candidate_id")
        if cid:
            # Flat fields are the candidate's current state, so the latest event wins.
            self.candidates.setdefault(cid, {}).update({k: p[k] for k in CANDIDATE_FIELDS if k in p})
            self.candidates[cid]["candidate_id"] = cid
        if e.status == "started":
            self.calls[key] = _Call(tool, p.get("input") or {}, e.time, self.clock(), cid)
            return []
        call = self.calls.pop(key, None)
        self.tool_calls += 1
        inp = p.get("input") or (call.input if call else {})
        took = (e.time - call.started).total_seconds() if call else None
        raw = p.get("result")
        result: dict[str, Any] = raw if isinstance(raw, dict) else {}
        if str(result.get("model") or "").startswith("agent-built:") and result.get("property"):
            self.agent_built.add(str(result["property"]))
        if tool == "propose_candidate" and result.get("accepted") and result.get("candidate_id"):
            new = self.candidates.setdefault(result["candidate_id"], {"candidate_id": result["candidate_id"]})
            new.setdefault("composition", inp.get("composition"))
        head = call_line(tool, inp, took)
        if e.status == "failed" or "error" in result:
            msg = str(result.get("error") or p.get("error") or "failed")
            return [head, result_line(f"[bad]Error:[/] {_escape(msg)}")]
        if tool == "request_review":
            review: dict[str, Any] = p.get("review") or result
            held, self._pending = self._pending, []
            return [
                head,
                result_line(summarise(tool, inp, review, p)),
                review_panel(review, self.config, self.spec),
                *held,
            ]
        return [head, result_line(summarise(tool, inp, result, p, verbose=self.verbose))]

    def _on_final_calculation(self, e: ProgressEvent) -> list[RenderableType]:
        """Higher-fidelity re-check of the finalists (Section 9).

        Parameters
        ----------
        e : ProgressEvent
            A ``final_calculation`` event.

        Returns
        -------
        list of RenderableType
            Output.
        """
        p = e.payload
        if e.status == "started":
            self.phase = "Re-checking at higher fidelity"
            who = ", ".join(str(c) for c in p.get("candidates") or [] if c)
            return [
                round_rule("final re-check"),
                self._line(GLYPH["mark"], f"Re-checking [id]{who}[/] at higher fidelity", "ember"),
            ]
        if e.status == "failed":
            return [result_line(f"[bad]Final re-check failed:[/] {_escape(str(p.get('error', '')))}")]
        self.validation = p
        return [validation_panel(p)]

    def _on_final_step(self, e: ProgressEvent) -> list[RenderableType]:
        """One line per finished step of the final re-check (Section 9, FC-9).

        Steps are paired with their start by tool, as only one of each runs at a time.

        Parameters
        ----------
        e : ProgressEvent
            A ``final_step`` event: ``{tool, candidate_id, potential, mode}``, plus ``wall_s`` when done.

        Returns
        -------
        list of RenderableType
            Output; nothing for ``started``.
        """
        p = e.payload
        tool = str(p.get("tool"))
        cid = p.get("candidate_id")
        if e.status == "started":
            self.final_steps[tool] = _Call(tool, p, e.time, self.clock(), cid)
            return []
        start = self.final_steps.pop(tool, None)
        took = p.get("wall_s")
        if took is None and start is not None:
            took = (e.time - start.started).total_seconds()
        failed = e.status == "failed"
        line = final_step_line(p, took)
        if failed:
            return [line, result_line(f"[bad]failed[/] {_escape(str(p.get('error') or ''))}".rstrip())]
        return [line]

    def _on_capability(self, e: ProgressEvent) -> list[RenderableType]:
        """Agent-built tools: the gap, plans and code for review, requests, decisions and the outcome (AT-24, AT-28).

        Every card is kept by (phase property, id, stage), newest revision last, so the stream knows
        which card the run is waiting for. A card with no later decision or withdrawal is open.

        Parameters
        ----------
        e : ProgressEvent
            A ``capability`` event; ``payload.kind`` says which.

        Returns
        -------
        list of RenderableType
            Output.
        """
        p = e.payload
        kind = str(p.get("kind"))
        prop = str(p.get("property") or self.cap_property or "")
        if kind == "gap":
            self.cap_property = prop
            self.in_campaign = True
            self.phase = f"Researching a tool for {prop.replace('_', ' ')}"
            units = f" ({_escape(str(p['units']))})" if p.get("units") else ""
            return [
                round_rule(f"build tool {GLYPH['dot']} {prop.replace('_', ' ')}"),
                self._line(
                    GLYPH["mark"],
                    f"No tool computes {prop.replace('_', ' ')}{units}. The agent will read how papers compute "
                    "it and propose a tool; you review the plan, then the code, before the search starts.",
                    "ember",
                ),
            ]
        if kind == "library":
            name = ", ".join(p.get("reused") or [])
            return [
                self._line(
                    GLYPH["mark"],
                    f"Reusing [tool]{_escape(name)}[/] for {prop.replace('_', ' ')}: approved earlier.",
                    "ember",
                )
            ]
        if kind in ("plan", "code"):
            prop_d = p.get("proposal") or {}
            phase = str(prop_d.get("property") or self.cap_property or "")
            key = (phase, str(prop_d.get("proposal_id")), kind)
            self._remember(key, kind, prop_d, p.get("allowed") or [])
            waiting = self.live and not self.replaying
            if kind == "plan":
                return [plan_card(prop_d, waiting=waiting)]
            return [code_card(prop_d, waiting=waiting)]
        if kind in ("package", "question", "give_up"):
            req = p.get("request") or {}
            key = (str(self.cap_property or ""), str(req.get("request_id")), kind)
            self._remember(key, kind, req, p.get("allowed") or [])
            return [request_card(req, waiting=self.live and not self.replaying)]
        if kind == "decision":
            d = p.get("decision") or {}
            for card in reversed(list(self.cards.values())):
                if (
                    card["id"] == str(d.get("proposal_id"))
                    and card["stage"] == str(d.get("stage"))
                    and card["decision"] is None
                ):
                    card["decision"] = d
                    break
            self.phase = f"Working on the tool for {(self.cap_property or '').replace('_', ' ')}"
            return [self._line(GLYPH["mark"], decision_text(d, p.get("name")), "accent")]
        if kind == "withdrawn":
            for card in reversed(list(self.cards.values())):
                if (
                    card["id"] == str(p.get("proposal_id"))
                    and card["stage"] == "code"
                    and card["decision"] is None
                ):
                    card["withdrawn"] = str(p.get("reason") or "")
                    break
            return [
                result_line(f"[warn]The agent withdrew its code:[/] {_escape(str(p.get('reason') or ''))}")
            ]
        if kind == "checks":
            checks = p.get("checks") or {}
            failed = [k for k, v in checks.items() if not (v or {}).get("passed")]
            text = (
                f"[bad]{', '.join(failed)} failed[/]; the agent is fixing it"
                if failed
                else f"{mark(True)} all checks passed"
            )
            return [result_line(f"code checks: {text}")]
        if kind == "registered":
            tool = p.get("tool") or {}
            self.phase = "Thinking"
            return [
                self._line(
                    GLYPH["mark"],
                    f"[tool]{_escape(str(tool.get('name')))}[/] is saved to the tool library and available to "
                    "the campaign.",
                    "ok",
                )
            ]
        if kind == "cancelled":
            said = False
            for card in self.cards.values():  # nothing waits any more
                if card["decision"] is None and card["withdrawn"] is None:
                    card["withdrawn"] = "cancelled"
                said = said or (card["decision"] or {}).get("action") == "cancel_run"
            if p.get("scope") == "run" and said:
                return []  # the decision line already said so
            return [self._line(GLYPH["mark"], cancelled_text(p), "warn")]
        if kind == "log":
            return [result_line(f"[muted]{_escape(str(p.get('message') or ''))}[/]")]
        return [result_line(f"[muted]tool builder: {_escape(kind)}[/]")]

    def _remember(
        self, key: tuple[str, str, str], stage: str, payload: dict[str, Any], allowed: list[str]
    ) -> None:
        """Keep the newest card for a (phase, id, stage); a re-shown or revised card replaces the old one.

        Parameters
        ----------
        key : tuple
            ``(phase property, id, stage)``.
        stage : str
            ``plan``, ``code``, ``package``, ``question`` or ``give_up``.
        payload : dict
            The proposal or request.
        allowed : list of str
            Actions the engine offers.
        """
        self.cards.pop(key, None)  # move to the end: newest last
        self.cards[key] = {
            "key": key,
            "id": key[1],
            "stage": stage,
            "payload": payload,
            "allowed": list(allowed),
            "revision": int(payload.get("revision") or 1),
            "decision": None,
            "withdrawn": None,
        }

    def open_card(self) -> dict[str, Any] | None:
        """Return the newest card with no decision, not withdrawn and not already answered from here.

        Returns
        -------
        dict or None
            The card the run is most likely waiting for; the stream confirms with the engine.
        """
        for card in reversed(list(self.cards.values())):
            if card["decision"] is not None or card["withdrawn"] is not None:
                continue
            if (card["id"], card["stage"], card["revision"]) in self.answered:
                continue
            return card
        return None

    def _on_brief_generator(self, e: ProgressEvent) -> list[RenderableType]:
        """Brief progress.

        Parameters
        ----------
        e : ProgressEvent
            A ``brief_generator`` event.

        Returns
        -------
        list of RenderableType
            Output.
        """
        if e.status == "started":
            self.phase = "Writing the brief"
            return []
        if e.status == "failed":
            return [result_line(f"[bad]Brief failed:[/] {_escape(str(e.payload.get('error', '')))}")]
        self.brief = e.payload
        cited = len(e.payload.get("cited") or [])
        pdf = " · PDF written" if e.payload.get("pdf") else ""
        compute = " · computational brief too" if e.payload.get("compute") else ""
        return [self._line(GLYPH["mark"], f"Brief written · cites {cited} papers{pdf}{compute}", "ember")]

    # ------------------------------------------------------------------ pieces

    @staticmethod
    def _line(glyph: str, body: RenderableType | str, style: str) -> Table:
        """Lay out ``glyph  body`` so wrapped lines stay indented.

        Parameters
        ----------
        glyph : str
            Leading glyph.
        body : RenderableType or str
            Content; strings are rich markup.
        style : str
            Style of the glyph.

        Returns
        -------
        rich.table.Table
            A two-column grid.
        """
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=1, no_wrap=True)
        grid.add_column(ratio=1)
        grid.add_row(Text(glyph, style=style), Text.from_markup(body) if isinstance(body, str) else body)
        return grid

    def _round_lines(self, p: dict[str, Any]) -> list[RenderableType]:
        """Close a review round: its result with the confidence trend, then the next round's rule (UI-5, UI-15).

        Parameters
        ----------
        p : dict
            Per-round ``orchestrator`` payload.

        Returns
        -------
        list of RenderableType
            The result (with the gate note when a stop was blocked), and a rule opening the next
            round unless a stop rule fired.
        """
        scores = [float(x) for x in p.get("scores") or []]
        llm = [float(x) for x in p.get("llm_scores") or [] if x is not None]
        rubric = fmt_num(scores[-1], 0) if scores else DASH
        reviewer = fmt_num(llm[-1], 0) if llm else DASH
        outcome = f"[accent]stops · {p.get('rule')}[/]" if p.get("stop") else "continues"
        trend = f"  [muted]rubric[/] [accent]{sparkline(scores)}[/]  [muted]reviewer[/] {sparkline(llm)}"
        body = (
            f"[bold]Round {p.get('round')}[/] · best [id]{p.get('best') or DASH}[/] · rubric {rubric} · "
            f"reviewer {reviewer} · threshold {self.config.get('confidence_threshold', 72)} · {outcome}{trend}"
        )
        rows: list[RenderableType] = [self._line(GLYPH["mark"], body, "ember")]
        if p.get("best"):
            rows.append(
                result_line(f"best so far [accent]{p['best']}[/] {self.candidate_facts(str(p['best']))}")
            )
        if p.get("gate_note"):
            rows.append(result_line(f"[warn]{_escape(str(p['gate_note']))}[/]"))
        out: list[RenderableType] = [Group(*rows)]
        if not p.get("stop"):
            out.append(round_rule(f"round {self.round}"))
        return out

    def _submission_line(self, sub: dict[str, Any] | None, rule: str | None) -> RenderableType:
        """Show the accepted submission (OR-3).

        Parameters
        ----------
        sub : dict or None
            ``Submission`` as a dict.
        rule : str or None
            Stop rule that fired.

        Returns
        -------
        RenderableType
            The line.
        """
        if not sub:
            return self._line(
                GLYPH["mark"], f"Campaign ended without a submission · {rule or 'no rule'}", "warn"
            )
        cid = sub.get("candidate_id")
        comp = self.candidates.get(cid or "", {}).get("composition")
        runner = f" · runner-up [id]{sub['runner_up_id']}[/]" if sub.get("runner_up_id") else ""
        gate = f"gate {mark(sub.get('gate_passed'))}"
        not_met = " · [warn]reported as not meeting the target[/]" if sub.get("accept_not_met") else ""
        by = " · picked by code" if sub.get("picked_by") == "code" else ""
        body = f"[bold]Submitted[/] [id]{cid}[/] {comp or ''}{runner} · {gate} · stop rule {rule or DASH}{not_met}{by}"
        return self._line(GLYPH["mark"], body, "ember")

    # ------------------------------------------------------------------ live status

    def status(self) -> RenderableType:
        """Return the live status line: spinner, current activity, time, round and budget.

        Returns
        -------
        RenderableType
            Spinner line, then one line per call still running.
        """
        now = self.clock()
        start = self._local_start if self._local_start is not None else now
        waiting = self.live and not self.replaying and not self.calls and self.open_card() is not None
        if self.stopping:
            verb = (
                "Stopping after the current calculation"
                if self.stop_all
                else "Stopping: submitting the best candidate"
            )
        elif waiting:
            verb = "Waiting for your decision"
        elif self.calls:
            verb = _verb(next(reversed(self.calls.values())))
        elif self.final_steps:
            step = next(reversed(self.final_steps.values()))
            verb = f"Re-checking: {final_step_label(step.input)}"
        else:
            verb = self.phase
        max_rounds = self.config.get("max_iterations")
        max_calls = self.config.get("max_tool_calls")
        bits = [elapsed(now - start)]
        if max_rounds:
            bits.append(f"round {min(self.round, int(max_rounds))}/{max_rounds}")
        if max_calls:
            bits.append(f"{self.tool_calls}/{max_calls} tools")
        if self.menu:
            bits.append("^C speed, pause or end" if self.replaying else "^C stop or detach")
        else:
            bits.append(
                "^C ends the replay"
                if self.replaying
                else ("^C again to detach" if self.stopping else "^C stop")
            )
        line = ember_frame(now)
        line.append(f" {verb}", style="accent")
        for bit in bits:
            line.append(f"  {GLYPH['sep']} ", style="gutter")
            line.append(bit, style="muted")
        rows: list[RenderableType] = [line]
        if self.campaign_span[0] is not None and not waiting:
            rows.append(Text.assemble("   ", self.budget_text()))
        for call in list(self.calls.values())[-4:]:
            target = f" {call.candidate_id}" if call.candidate_id else ""
            rows.append(
                Text(
                    f"   {GLYPH['running']} {call.tool}{target} ({elapsed(now - call.local_start)})", "faint"
                )
            )
        return Group(*rows)

    # ------------------------------------------------------------------ tables

    def candidates_table(self) -> Table:
        """Every candidate's latest code-computed fields (UI-5).

        Returns
        -------
        rich.table.Table
            One row per candidate.
        """
        prop = str((self.spec or {}).get("target_property") or "value").replace("_", " ")
        units = (self.spec or {}).get("target_units") or ""
        t = Table(box=None, header_style="muted", pad_edge=False, padding=(0, 2))
        t.add_column("id", no_wrap=True)
        t.add_column("composition", no_wrap=True)
        for col in ("E_hull", "", "min THz", "phonons", ""):
            t.add_column(col, no_wrap=True)
        t.add_column(f"{prop} ({units})" if units else prop, no_wrap=True)
        extras = [x for x in (self.spec or {}).get("extra_targets") or [] if x.get("property")]
        for x in extras:
            label = str(x["property"]).replace("_", " ")
            t.add_column(f"{label} ({x['units']})" if x.get("units") else label, no_wrap=True)
        t.add_column("sg", no_wrap=True)
        best = (self.submitted or {}).get("candidate_id")
        for cid, c in self.candidates.items():
            name = f"[accent]{cid}[/]" if cid == best else f"[id]{cid}[/]"
            rows = {r.get("property"): r for r in c.get("targets") or []}
            extra_cells = []
            for x in extras:
                r = rows.get(x["property"])
                if r is None or r.get("value") is None:
                    extra_cells.append(
                        "not evaluated" if r is not None and r.get("evaluable") is False else DASH
                    )
                elif r.get("agent_built"):
                    extra_cells.append(f"{fmt_num(r['value'])} [accent]agent-built[/]")
                else:
                    extra_cells.append(fmt_quantity(r["value"], r.get("uncertainty")))
            t.add_row(
                name,
                str(c.get("composition") or ""),
                fmt_num(c.get("e_hull"), 3),
                mark(c.get("thermo_stable")) if c.get("e_hull") is not None else mark(None),
                fmt_num(c.get("min_freq")),
                str(c.get("verdict") or DASH),
                mark(c.get("dyn_stable")) if c.get("verdict") else mark(None),
                fmt_quantity(c.get("value"), c.get("uncertainty")) if c.get("value") is not None else DASH,
                *extra_cells,
                str(c.get("space_group") or DASH),
            )
        return t

    def rounds_table(self) -> Table:
        """One row per review round: best, rubric, reviewer confidence and outcome (UI-5, UI-15).

        Returns
        -------
        rich.table.Table
            The table.
        """
        t = Table(box=None, header_style="muted", pad_edge=False, padding=(0, 2))
        for col in ("round", "best", "rubric", "reviewer", "outcome"):
            t.add_column(col)
        for p in self.rounds:
            scores, llm = p.get("scores") or [], p.get("llm_scores") or []
            outcome = f"stop · {p.get('rule')}" if p.get("stop") else (p.get("gate_note") or "continue")
            t.add_row(
                str(p.get("round")),
                f"[id]{p.get('best') or DASH}[/]",
                fmt_num(scores[-1], 0) if scores else DASH,
                fmt_num(llm[-1], 0) if llm and llm[-1] is not None else DASH,
                _escape(str(outcome)),
            )
        return t

    def summary(
        self, state: RunState | None = None, brief_path: str | None = None, compute_path: str | None = None
    ) -> RenderableType:
        """End-of-run summary box.

        Parameters
        ----------
        state : RunState, optional
            Final state; fields from the events are used when omitted.
        brief_path : str, optional
            Where the lab brief was written.
        compute_path : str, optional
            Where the computational brief was written.

        Returns
        -------
        RenderableType
            The panel.
        """
        status = (state.status if state else None) or (self.end or {}).get("status") or "unknown"
        stop = (state.stop_reason if state else None) or (self.end or {}).get("stop_reason") or self.stop_rule
        best = (state.best if state else None) or (self.end or {}).get("best")
        colour = {"finished": "ok", "failed": "bad", "stopped": "warn", "needs_input": "warn"}.get(
            status, "muted"
        )
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="muted", no_wrap=True)
        grid.add_column()
        title = (state.spec.title if state and state.spec else None) or (self.spec or {}).get("title")
        if title:
            grid.add_row("title", f"[bold]{_escape(str(title))}[/]")
        grid.add_row("status", f"[{colour}]{status}[/]" + (f" · {_escape(str(stop))}" if stop else ""))
        if best:
            grid.add_row("best", f"[accent]{best}[/] {self.candidate_facts(best)}")
        if self.submitted:
            grid.add_row("gate", f"{mark(self.submitted.get('gate_passed'))} code gate")
        if self.validation:
            v = str(self.validation.get("verdict"))
            vc = {"confirmed": "ok", "weakened": "warn", "overturned": "bad"}.get(v, "muted")
            grid.add_row("re-check", f"[{vc}]{v}[/]")
        if self.first_time and self.last_time:
            took = elapsed((self.last_time - self.first_time).total_seconds())
            grid.add_row("time", f"{took} · {self.tool_calls} tool calls · {len(self.rounds)} rounds")
            if self.campaign_span[0] is not None:
                grid.add_row("budgets", self.budget_text())
        tokens = (state.usage if state and state.usage else None) or self.usage
        if tokens and any(tokens.values()):
            grid.add_row("tokens", token_text(tokens))
        lit = state.literature if state is not None else None
        if state is not None and lit is not None and lit.papers:
            n_fail = len(lit.failed_approaches)
            grid.add_row(
                "literature",
                f"{len(lit.papers)} papers · {len(lit.insights)} evidence notes · {n_fail} failed approach"
                f"{'' if n_fail == 1 else 'es'} [faint]· hephaestus literature {self.run_id or state.run_id}[/]",
            )
        if brief_path:
            grid.add_row("lab brief", brief_path)
        if compute_path:
            grid.add_row("compute brief", compute_path)
        rid = self.run_id or (state.run_id if state else "")
        hints = [f"hephaestus replay {rid}", f"hephaestus brief {rid}"]
        if compute_path:
            hints.append(f"hephaestus brief {rid} --compute")
        if status not in ("finished", "needs_input"):
            hints.insert(0, f"hephaestus resume {rid}")
        grid.add_row("run", f"[id]{rid}[/]")
        grid.add_row("next", "[faint]" + "  ·  ".join(hints) + "[/]")
        return panel(grid, "run summary")


# ---------------------------------------------------------------------- call lines


def token_text(usage: dict[str, Any]) -> str:
    """Token use in words: input, output and cache, with the tool builder's share (DL-9).

    Parameters
    ----------
    usage : dict
        ``RunState.usage``: keys such as ``input_tokens``, ``cache_read_input_tokens`` and
        ``capability_input_tokens``.

    Returns
    -------
    str
        E.g. ``"12,400 in · 3,100 out · cache 8,000 read / 1,200 written · tool builder 2,000"``.
    """

    def get(key: str) -> int:
        """Return one count.

        Parameters
        ----------
        key : str
            Usage key.

        Returns
        -------
        int
            The count, 0 when absent.
        """
        return int(usage.get(key) or 0)

    parts = [f"{get('input_tokens'):,} in", f"{get('output_tokens'):,} out"]
    read, wrote = get("cache_read_input_tokens"), get("cache_creation_input_tokens")
    if read or wrote:
        parts.append(f"cache {read:,} read / {wrote:,} written")
    builder = sum(int(v or 0) for k, v in usage.items() if k.startswith("capability_"))
    if builder:
        parts.append(f"tool builder {builder:,}")
    return " · ".join(parts)


def _escape(text: str) -> str:
    """Escape rich markup in free text.

    Parameters
    ----------
    text : str
        Text that may contain square brackets.

    Returns
    -------
    str
        Safe markup.
    """
    from rich.markup import escape

    return escape(strip_codes(text))


def _verb(call: _Call) -> str:
    """Status-line verb for a running call.

    Parameters
    ----------
    call : _Call
        The call.

    Returns
    -------
    str
        E.g. ``"Relaxing c1-1"``.
    """
    if call.tool.startswith("custom_"):
        return f"Running the agent-built {call.tool[7:]} on {call.candidate_id or ''}".strip()
    template = VERBS.get(call.tool, f"Running {call.tool}")
    prop = str(call.input.get("property", "a property")).replace("_", " ")
    return template.format(cid=call.candidate_id or call.input.get("candidate_id", ""), prop=prop).strip()


def _arg(key: str, value: Any) -> str | None:
    """Format one argument of a call line.

    Parameters
    ----------
    key : str
        Argument name.
    value : Any
        Value.

    Returns
    -------
    str or None
        Text, or ``None`` to leave the argument out.
    """
    if value is None or value == [] or value == "":
        return None
    if isinstance(value, bool):
        return key if value else None
    if isinstance(value, int | float):
        return f"{key} {value}" if key in ("fmax", "limit", "duration_ps") else str(value)
    text = str(value)
    if len(text) > 60:
        text = text[:57] + "…"
    return f'"{text}"' if key == "query" else text


def call_line(tool: str, inp: dict[str, Any], took: float | None = None) -> Table:
    """``▸ tool  arg · arg`` for a finished call, with its duration at the right.

    Parameters
    ----------
    tool : str
        Tool name.
    inp : dict
        Tool input.
    took : float, optional
        Duration in seconds, from the events' timestamps.

    Returns
    -------
    rich.table.Table
        The line.
    """
    keys = CALL_ARGS.get(tool, list(inp)[:3])
    args = f" {GLYPH['dot']} ".join(a for a in (_arg(k, inp.get(k)) for k in keys) if a)
    text = Text.assemble((tool, "tool"), ("  ", ""), (args, "id"))
    took_text = ""
    if took is not None and took >= 0.05:
        took_text = elapsed(took) if took >= 1 else f"{took:.1f}s"
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=1, no_wrap=True)
    grid.add_column(ratio=1)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(Text(GLYPH["tool"], style="accent"), text, Text(took_text, style="faint"))
    return grid


def final_step_label(p: dict[str, Any]) -> str:
    """Describe a final re-check step: ``"Full phonons for c2-1 · mace-mp-0-medium"``.

    Parameters
    ----------
    p : dict
        ``final_step`` payload.

    Returns
    -------
    str
        Step name, candidate (``ref`` shown as the reference) and potential.
    """
    name = FINAL_STEP_NAMES.get(str(p.get("tool")), str(p.get("tool")))
    cid = p.get("candidate_id")
    who = "the reference" if cid == "ref" else (cid or "")
    pot = f" {GLYPH['dot']} {p['potential']}" if p.get("potential") else ""
    return f"{name} for {who}{pot}" if who else f"{name}{pot}"


def final_step_line(p: dict[str, Any], took: float | None) -> Table:
    """``▸ Full phonons  c2-1 · mace-mp-0-medium · full   38s`` for a finished re-check step.

    Parameters
    ----------
    p : dict
        ``final_step`` payload.
    took : float or None
        Duration in seconds.

    Returns
    -------
    rich.table.Table
        The line.
    """
    name = FINAL_STEP_NAMES.get(str(p.get("tool")), str(p.get("tool")))
    cid = p.get("candidate_id")
    bits = ["reference" if cid == "ref" else cid, p.get("potential"), p.get("mode")]
    args = f" {GLYPH['dot']} ".join(str(b) for b in bits if b)
    took_text = "" if took is None else (elapsed(float(took)) if float(took) >= 1 else f"{float(took):.1f}s")
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=1, no_wrap=True)
    grid.add_column(ratio=1)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(
        Text(GLYPH["tool"], style="ember"),
        Text.assemble((name, "accent"), ("  ", ""), (args, "id")),
        Text(took_text, style="faint"),
    )
    return grid


def round_rule(title: str) -> Rule:
    """Open a section of the log with a heavy rule: ``━━ ROUND 2 ━━━━━━…``.

    Parameters
    ----------
    title : str
        Section name; shown in capitals.

    Returns
    -------
    rich.rule.Rule
        The rule.
    """
    label = Text.assemble((GLYPH["rule"] * 2 + " ", "rule"), (title.upper(), "accent.bold"))
    return Rule(label, characters=GLYPH["rule"], style="rule", align="left")


def result_line(markup: str) -> Table:
    """``  ╰─ result`` under a call line.

    Parameters
    ----------
    markup : str
        Rich markup.

    Returns
    -------
    rich.table.Table
        The line.
    """
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=4, no_wrap=True)
    grid.add_column(ratio=1)
    grid.add_row(Text(f"  {GLYPH['result']}", style="gutter"), Text.from_markup(markup))
    return grid


# ---------------------------------------------------------------------- per-tool summaries


def _papers(inp: dict[str, Any], r: dict[str, Any], p: dict[str, Any], verbose: bool = False) -> str:
    """Literature search: count and the top paper.

    Parameters
    ----------
    inp : dict
        Tool input.
    r : dict
        Result.
    p : dict
        Event payload.
    verbose : bool
        List more papers.

    Returns
    -------
    str
        Markup.
    """
    papers = r.get("papers") or p.get("papers") or []
    n = r.get("n", len(papers))
    if not papers:
        return f"[muted]{n} papers[/]"
    shown = papers[: 3 if verbose else 1]
    parts = []
    for paper in shown:
        title = str(paper.get("title") or "")
        title = title if len(title) <= 70 else title[:67] + "…"
        meta = ", ".join(x for x in (str(paper.get("year") or ""), _cites(paper.get("citations"))) if x)
        parts.append(f"{_escape(title)} [muted]({meta})[/]" if meta else _escape(title))
    return f"{n} papers · " + " · ".join(parts)


def _cites(n: Any) -> str:
    """Citation count as text.

    Parameters
    ----------
    n : Any
        Count or ``None``.

    Returns
    -------
    str
        ``"120 cites"`` or empty.
    """
    return f"{n} cites" if isinstance(n, int) else ""


def _get_paper(inp: dict[str, Any], r: dict[str, Any], p: dict[str, Any], verbose: bool = False) -> str:
    """``get_paper``: title and year.

    Parameters
    ----------
    inp, r, p : dict
        Input, result and payload.
    verbose : bool
        Unused.

    Returns
    -------
    str
        Markup.
    """
    meta = ", ".join(x for x in (str(r.get("year") or ""), r.get("doi") or "") if x)
    return f"{_escape(str(r.get('title') or ''))} [muted]({_escape(meta)})[/]"


def _trend(inp: dict[str, Any], r: dict[str, Any], p: dict[str, Any], verbose: bool = False) -> str:
    """``publication_trend``: years, a sparkline and the total.

    Parameters
    ----------
    inp, r, p : dict
        Input, result and payload.
    verbose : bool
        Unused.

    Returns
    -------
    str
        Markup.
    """
    per = r.get("papers_per_year") or {}
    if not per:
        return "[muted]no papers[/]"
    years = sorted(per, key=lambda y: int(y))
    counts = [float(per[y]) for y in years]
    return f"{years[0]}\u2013{years[-1]} [accent]{sparkline(counts, 0, max(counts) or 1)}[/] · {int(sum(counts))} papers"


def _evidence(inp: dict[str, Any], r: dict[str, Any], p: dict[str, Any], verbose: bool = False) -> str:
    """``note_evidence``: recorded, and whether the value was dropped (LM-14).

    Parameters
    ----------
    inp, r, p : dict
        Input, result and payload.
    verbose : bool
        Unused.

    Returns
    -------
    str
        Markup.
    """
    n = r.get("n_insights", "?")
    text = f"recorded {inp.get('finding_type', '')} evidence · {n} note{'' if n == 1 else 's'}"
    if r.get("value_dropped_not_in_abstract"):
        text += " · [warn]value dropped: not in the abstract[/]"
    return text


def _mp(inp: dict[str, Any], r: dict[str, Any], p: dict[str, Any], verbose: bool = False) -> str:
    """``lookup_materials_project``: polymorph count and the default entry.

    Parameters
    ----------
    inp, r, p : dict
        Input, result and payload.
    verbose : bool
        Unused.

    Returns
    -------
    str
        Markup.
    """
    n = r.get("n_entries", 0)
    if not n:
        return f"[muted]{_escape(str(r.get('note') or 'not in Materials Project'))}[/]"
    default = r.get("default_polymorph")
    entry: dict[str, Any] = next((x for x in r.get("entries") or [] if x.get("mp_id") == default), {})
    bits = [
        f"{n} polymorph{'s' if n != 1 else ''}",
        f"default [id]{default}[/] {entry.get('space_group', '')}",
    ]
    if entry.get("e_hull_eV_per_atom") is not None:
        bits.append(f"E_hull {fmt_num(entry['e_hull_eV_per_atom'], 3)} eV/atom")
    if entry.get("band_gap_pbe_eV") is not None:
        bits.append(f"gap {fmt_num(entry['band_gap_pbe_eV'])} eV (PBE)")
    if entry.get("n_icsd"):
        bits.append(f"{entry['n_icsd']} ICSD reports")
    return " · ".join(bits)


def _propose(inp: dict[str, Any], r: dict[str, Any], p: dict[str, Any], verbose: bool = False) -> str:
    """``propose_candidate``: the new ID, or why it was refused.

    Parameters
    ----------
    inp, r, p : dict
        Input, result and payload.
    verbose : bool
        Unused.

    Returns
    -------
    str
        Markup.
    """
    if not r.get("accepted"):
        reasons = "; ".join(str(x) for x in r.get("reasons") or []) or "refused"
        return f"{mark(False)} refused: {_escape(reasons)}"
    bits = [f"{mark(True)} accepted as [id]{r.get('candidate_id')}[/]"]
    if r.get("tolerance_factor") is not None:
        bits.append(f"t {fmt_num(r['tolerance_factor'])}")
    if r.get("warnings"):
        bits.append(f"[warn]{len(r['warnings'])} warning{'s' if len(r['warnings']) != 1 else ''}[/]")
    return " · ".join(bits)


def _build(inp: dict[str, Any], r: dict[str, Any], p: dict[str, Any], verbose: bool = False) -> str:
    """``build_structure``: size, source and supercell.

    Parameters
    ----------
    inp, r, p : dict
        Input, result and payload.
    verbose : bool
        Unused.

    Returns
    -------
    str
        Markup.
    """
    sc = r.get("supercell") or []
    bits = [f"{r.get('n_atoms', '?')} atoms", str(r.get("source", ""))]
    if sc and sc != [1, 1, 1]:
        bits.append("\u00d7".join(str(x) for x in sc))
    if r.get("space_group"):
        bits.append(str(r["space_group"]))
    poly = r.get("polymorph") or {}
    if poly.get("matches_structure_type") is False:
        bits.append("[warn]MP phase differs from the proposed prototype[/]")
    return " · ".join(bits)


def _relax(inp: dict[str, Any], r: dict[str, Any], p: dict[str, Any], verbose: bool = False) -> str:
    """``relax``: convergence, energy, symmetry and lattice.

    Parameters
    ----------
    inp, r, p : dict
        Input, result and payload.
    verbose : bool
        Unused.

    Returns
    -------
    str
        Markup.
    """
    conv = "converged" if r.get("converged") else "[warn]not converged[/]"
    bits = [conv, f"{fmt_num(r.get('energy_eV_per_atom'), 3)} eV/atom", str(r.get("space_group") or "")]
    abc = r.get("lattice_abc_A")
    if abc:
        bits.append("a b c " + " ".join(fmt_num(x) for x in abc) + " Å")
    return " · ".join(b for b in bits if b)


def _hull(inp: dict[str, Any], r: dict[str, Any], p: dict[str, Any], verbose: bool = False) -> str:
    """``energy_above_hull``: E_hull and the code verdict.

    Parameters
    ----------
    inp, r, p : dict
        Input, result and payload.
    verbose : bool
        Unused.

    Returns
    -------
    str
        Markup.
    """
    flag = r.get("thermo_stable", p.get("thermo_stable"))
    text = f"E_hull {fmt_num(r.get('e_hull_eV_per_atom'), 3)} eV/atom · {mark(flag)} thermodynamically stable"
    if r.get("threshold") is not None:
        text += f" [muted](≤ {r['threshold']})[/]"
    if r.get("decomposes_to"):
        text += " · decomposes to " + ", ".join(r["decomposes_to"])
    return text


def _phonons(inp: dict[str, Any], r: dict[str, Any], p: dict[str, Any], verbose: bool = False) -> str:
    """``phonons``: lowest frequency, mode-mapping verdict and the code verdict.

    Parameters
    ----------
    inp, r, p : dict
        Input, result and payload.
    verbose : bool
        Unused.

    Returns
    -------
    str
        Markup.
    """
    where = f" at {r['where']}" if r.get("where") else ""
    flag = r.get("dyn_stable", p.get("dyn_stable"))
    bits = [f"min {fmt_num(r.get('min_frequency_THz'))} THz{where}", str(r.get("verdict") or "")]
    bits.append(f"{mark(flag)} dynamically stable")
    rounds = r.get("mode_mapping_rounds") or []
    if rounds:
        bits.append(f"{len(rounds)} mode-mapping round{'s' if len(rounds) != 1 else ''}")
    if r.get("note"):
        bits.append(f"[warn]{_escape(str(r['note']))}[/]")
    return " · ".join(b for b in bits if b)


def _md(inp: dict[str, Any], r: dict[str, Any], p: dict[str, Any], verbose: bool = False) -> str:
    """``run_md``: the code verdict and the largest drift.

    Parameters
    ----------
    inp, r, p : dict
        Input, result and payload.
    verbose : bool
        Unused.

    Returns
    -------
    str
        Markup.
    """
    return (
        f"{mark(r.get('stable'))} stable in MD · max displacement {fmt_num(r.get('max_displacement_A'))} Å "
        f"at {fmt_num(r.get('temperature_K'), 0)} K"
    )


def _predict(inp: dict[str, Any], r: dict[str, Any], p: dict[str, Any], verbose: bool = False) -> str:
    """``predict_property``: value ± uncertainty, model and target (SYS-4).

    Parameters
    ----------
    inp, r, p : dict
        Input, result and payload.
    verbose : bool
        Unused.

    Returns
    -------
    str
        Markup.
    """
    prop = str(r.get("property") or inp.get("property") or "").replace("_", " ")
    text = f"{prop} [bold]{fmt_quantity(r.get('value'), r.get('uncertainty'), r.get('units', ''))}[/]"
    if r.get("model"):
        text += f" [muted]({_escape(str(r['model']))})[/]"
    target = r.get("target") or {}
    if target.get("value") is not None:
        sym = DIRECTION.get(str(target.get("direction")), "vs")
        ref = f" ({target['reference']})" if target.get("reference") else ""
        text += f" · target {sym} {fmt_num(target['value'])}{ref}"
    if r.get("in_domain") is False:
        text += " · [warn]outside the model's domain[/]"
    return text


def _status(inp: dict[str, Any], r: dict[str, Any], p: dict[str, Any], verbose: bool = False) -> str:
    """``candidate_status``: counts.

    Parameters
    ----------
    inp, r, p : dict
        Input, result and payload.
    verbose : bool
        Unused.

    Returns
    -------
    str
        Markup.
    """
    budget = r.get("budget") or {}
    return f"round {r.get('round', '?')} · {len(r.get('candidates') or [])} candidates · {budget.get('tool_calls_used', '?')} tool calls used"


def _review(inp: dict[str, Any], r: dict[str, Any], p: dict[str, Any], verbose: bool = False) -> str:
    """``request_review``: the headline numbers; the panel carries the rest.

    Parameters
    ----------
    inp, r, p : dict
        Input, result and payload.
    verbose : bool
        Unused.

    Returns
    -------
    str
        Markup.
    """
    bits = [f"rubric {r.get('score', DASH)}", f"reviewer {r.get('llm_confidence', DASH)}"]
    if r.get("recommended_next"):
        bits.append(f"next {r['recommended_next']}")
    if r.get("stop_rule"):
        bits.append(f"[accent]stop: {r['stop_rule']}[/]")
    return " · ".join(str(b) for b in bits)


def _submit(inp: dict[str, Any], r: dict[str, Any], p: dict[str, Any], verbose: bool = False) -> str:
    """``submit_final``: accepted or refused by the code gate.

    Parameters
    ----------
    inp, r, p : dict
        Input, result and payload.
    verbose : bool
        Unused.

    Returns
    -------
    str
        Markup.
    """
    if not r.get("accepted"):
        reasons = "; ".join(str(x) for x in r.get("reasons") or []) or str(r.get("note") or "refused")
        return f"{mark(False)} refused by the gate: {_escape(reasons)}"
    return f"{mark(True)} accepted · gate {mark(r.get('gate_passed'))}"


def _generic(inp: dict[str, Any], r: dict[str, Any], p: dict[str, Any], verbose: bool = False) -> str:
    """Fallback for tools without a dedicated summary: a few ``key value`` pairs.

    Parameters
    ----------
    inp, r, p : dict
        Input, result and payload.
    verbose : bool
        Show more pairs.

    Returns
    -------
    str
        Markup.
    """
    items = [(k, v) for k, v in r.items() if isinstance(v, str | int | float | bool) and v is not None]
    shown = items[: 8 if verbose else 4]
    text = " · ".join(f"{k} {fmt_num(v) if isinstance(v, float) else v}" for k, v in shown)
    return _escape(text) or "[muted]done[/]"


def _agent_built(inp: dict[str, Any], r: dict[str, Any], p: dict[str, Any], verbose: bool = False) -> str:
    """``custom_<name>``: the value from an agent-built tool, labelled as such, with no error bar (AT-17).

    Parameters
    ----------
    inp, r, p : dict
        Input, result and payload.
    verbose : bool
        Unused.

    Returns
    -------
    str
        Markup.
    """
    prop = str(r.get("property") or "").replace("_", " ")
    text = f"{prop} [bold]{fmt_quantity(r.get('value'), None, r.get('units', ''))}[/]"
    text += " [accent](agent-built tool, no error bar)[/]"
    target = r.get("target") or {}
    if target.get("value") is not None:
        sym = DIRECTION.get(str(target.get("direction")), "vs")
        ref = f" ({target['reference']})" if target.get("reference") else ""
        text += f" · target {sym} {fmt_num(target['value'])}{ref}"
    return text


def _read_source(inp: dict[str, Any], r: dict[str, Any], p: dict[str, Any], verbose: bool = False) -> str:
    """``read_tool_source``: which source the tool builder read.

    Parameters
    ----------
    inp, r, p : dict
        Input, result and payload.
    verbose : bool
        Unused.

    Returns
    -------
    str
        Markup.
    """
    fn = inp.get("function")
    return f"source of {_escape(str(fn))}" if fn else "our simulation and structure tools"


def _builder_wait(inp: dict[str, Any], r: dict[str, Any], p: dict[str, Any], verbose: bool = False) -> str:
    """``propose_tool``, ``write_tool``, ``request_package``, ``ask_user``, ``give_up``: what was sent.

    Parameters
    ----------
    inp, r, p : dict
        Input, result and payload.
    verbose : bool
        Unused.

    Returns
    -------
    str
        Markup.
    """
    tool = str(p.get("tool"))
    waiting = r.get("status") == "waiting_for_user"
    if tool == "propose_tool":
        what = f"plan for [tool]{_escape(str(inp.get('name') or ''))}[/]"
    elif tool == "write_tool":
        if r.get("passed") is False or (r.get("checks") and not waiting):
            failed = [k for k, v in (r.get("checks") or {}).items() if not (v or {}).get("passed")]
            return f"[bad]checks failed:[/] {', '.join(failed) or 'see the report'}"
        what = f"code for proposal {_escape(str(inp.get('proposal_id') or ''))}"
    elif tool == "request_package":
        if r.get("allowed"):
            return f"{_escape(str(inp.get('module') or ''))} is already allowed"
        what = f"a request to use {_escape(str(inp.get('module') or ''))}"
    elif tool == "ask_user":
        what = f"a question: {_escape(str(inp.get('question') or ''))}"
    else:
        what = f"a proposal to stop: {_escape(str(inp.get('reason') or ''))}"
    return f"sent {what}" + (" [warn]· waiting for you[/]" if waiting else "")


Summariser = Callable[..., str]
SUMMARISERS: dict[str, Summariser] = {
    "search_literature": _papers,
    "related_papers": _papers,
    "get_paper": _get_paper,
    "publication_trend": _trend,
    "note_evidence": _evidence,
    "lookup_materials_project": _mp,
    "propose_candidate": _propose,
    "build_structure": _build,
    "relax": _relax,
    "energy_above_hull": _hull,
    "phonons": _phonons,
    "run_md": _md,
    "predict_property": _predict,
    "candidate_status": _status,
    "request_review": _review,
    "submit_final": _submit,
    "read_tool_source": _read_source,
    "propose_tool": _builder_wait,
    "write_tool": _builder_wait,
    "request_package": _builder_wait,
    "ask_user": _builder_wait,
    "give_up": _builder_wait,
}


def summarise(
    tool: str, inp: dict[str, Any], result: dict[str, Any], payload: dict[str, Any], verbose: bool = False
) -> str:
    """One-line result for a tool call, as rich markup.

    Parameters
    ----------
    tool : str
        Tool name.
    inp : dict
        Tool input.
    result : dict
        Tool result (shortened by the harness).
    payload : dict
        Whole event payload (flat candidate fields, papers).
    verbose : bool
        Show more detail.

    Returns
    -------
    str
        Markup; a failing summariser falls back to the generic one.
    """
    fn = SUMMARISERS.get(tool) or (_agent_built if tool.startswith("custom_") else _generic)
    try:
        return fn(inp, result, payload, verbose)
    except Exception:
        return _generic(inp, result, payload, verbose)


# ---------------------------------------------------------------------- panels


def _target_line(t: dict[str, Any]) -> str:
    """One target in the spec: ``"band gap < SrTiO3 · reference 3.12 ± 0.65 eV"`` (GP-1, GP-11).

    Parameters
    ----------
    t : dict
        A ``Target`` (or the primary target's fields under the same names).

    Returns
    -------
    str
        Markup.
    """
    prop = str(t.get("property", "")).replace("_", " ")
    sym = DIRECTION.get(str(t.get("direction")), str(t.get("direction", "")))
    text = f"{prop} {sym} "
    if t.get("value") is not None:
        text += f"{fmt_num(t['value'])} {t.get('units', '')}".rstrip()
    elif t.get("reference_material"):
        text += f"{t['reference_material']} [muted](computed with the same model)[/]"
    ref = t.get("reference_value") or {}
    if ref:
        text += f" [muted]· reference {fmt_quantity(ref.get('value'), ref.get('uncertainty'), ref.get('units', ''))}[/]"
    if t.get("tolerance") is not None:
        text += f" ± {fmt_num(t['tolerance'])}"
    if t.get("tool"):
        text += f" [accent]· agent-built: {_escape(str(t['tool']))}[/]"
    if t.get("evaluable") is False:
        text += " [warn]· not evaluated (no tool)[/]"
    return text


def _limit(value: Any, what: str) -> str:
    """Describe a budget, where 0 means no limit (OR-9, OR-10).

    Parameters
    ----------
    value : Any
        The budget, or ``None`` when unknown.
    what : str
        Unit, e.g. ``"rounds"``.

    Returns
    -------
    str
        E.g. ``"3 rounds max"``, ``"no limit on rounds"`` or ``"? rounds max"``.
    """
    if value is None:
        return f"? {what} max"
    return f"no limit on {what}" if not value else f"{value} {what} max"


def spec_panel(spec: dict[str, Any], gap: str | None = None) -> Panel:
    """Show the parsed search specification before a run starts (UI-2, GP-1).

    Parameters
    ----------
    spec : dict
        ``SearchSpec`` as a dict.
    gap : str, optional
        A target property no tool computes yet (``ParseResult.gap``); the agent will build one (AT-1).

    Returns
    -------
    rich.panel.Panel
        The panel.
    """
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="muted", no_wrap=True)
    grid.add_column()
    primary = {
        "property": spec.get("target_property"),
        "value": spec.get("target_value"),
        "units": spec.get("target_units", ""),
        "direction": spec.get("direction"),
        "tolerance": spec.get("target_tolerance"),
        "reference_material": spec.get("reference_material"),
        "reference_value": spec.get("reference_value"),
        "tool": spec.get("target_tool"),
        "evaluable": spec.get("target_evaluable", True),
    }
    extras = spec.get("extra_targets") or []
    if spec.get("title"):
        grid.add_row("title", f"[bold]{_escape(str(spec['title']))}[/]")
    grid.add_row("family", _escape(str(spec.get("material_family", ""))))
    if extras:
        grid.add_row("targets", "[muted]all must be met[/]")
        for t in [primary, *extras]:
            grid.add_row("", f"{GLYPH['tool']} {_target_line(t)}")
    else:
        grid.add_row("target", _target_line(primary))
    if gap:
        grid.add_row(
            "new tool",
            f"[warn]No tool computes {_escape(gap.replace('_', ' '))} yet.[/] The agent will read how papers "
            "compute it and propose a tool; you review the plan and then the code before the search starts.",
        )
    th = spec.get("stability_thresholds") or {}
    hull = (
        f"E_hull ≤ {fmt_num(th.get('e_hull_max_eV_per_atom', 0.05))} eV/atom"
        if th.get("thermodynamic", True)
        else "hull check off"
    )
    phon = (
        f"phonons ≥ {fmt_num(th.get('phonon_min_THz', -0.1), 1)} THz"
        if th.get("dynamic", True)
        else "phonons off"
    )
    grid.add_row("stability", f"{hull} · {phon} [muted](decided by code)[/]")
    if spec.get("elements"):
        grid.add_row("elements", ", ".join(spec["elements"]))
    if spec.get("excluded_elements"):
        grid.add_row("excluded", ", ".join(spec["excluded_elements"]))
    cond = spec.get("conditions") or {}
    if cond:
        grid.add_row(
            "conditions", f"{fmt_num(cond.get('temperature_K'), 0)} K · {cond.get('pressure_GPa')} GPa"
        )
    for label, key in (("hard", "hard_constraints"), ("soft", "soft_preferences")):
        if spec.get(key):
            grid.add_row(label, _escape("; ".join(spec[key])))
    for i, a in enumerate(spec.get("assumptions") or []):
        grid.add_row("assumptions" if i == 0 else "", f"[muted]{_escape(a)}[/]")
    return panel(grid, "goal parsed")


def questions_panel(questions: list[str], errors: list[Any]) -> Panel:
    """Clarifying questions and validation errors from the goal parser (GP-5).

    Parameters
    ----------
    questions : list of str
        Questions to answer.
    errors : list
        Error records or messages.

    Returns
    -------
    rich.panel.Panel
        The panel.
    """
    lines = [f"[warn]?[/] {_escape(q)}" for q in questions]
    for err in errors:
        msg = err.get("message") if isinstance(err, dict) else str(err)
        lines.append(f"[bad]{GLYPH['bad']}[/] {_escape(str(msg))}")
    return panel("\n".join(lines) or "The goal could not be parsed.", "needs input", "warn")


def review_panel(
    review: dict[str, Any], config: dict[str, Any] | None = None, spec: dict[str, Any] | None = None
) -> Panel:
    """Show the reviewer's critique for one round, with the code rubric and checks (Section 7, UI-5).

    Parameters
    ----------
    review : dict
        ``request_review`` result.
    config : dict, optional
        Run config, for the confidence threshold.
    spec : dict, optional
        Search spec; with an agent-built target the "within error" check does not apply (AT-17).

    Returns
    -------
    rich.panel.Panel
        The panel.
    """
    threshold = (config or {}).get("confidence_threshold", 72)
    score = review.get("score")
    conf = review.get("llm_confidence")
    head = Table.grid(padding=(0, 2))
    head.add_column(style="muted", no_wrap=True)
    head.add_column()
    head.add_row("best of round", f"[id]{review.get('best_of_round') or DASH}[/]")
    if score is not None:
        head.add_row(
            "code rubric",
            f"[accent]{bar(float(score), 100, 20)}[/] {score}/100 [muted](stop at {threshold})[/]",
        )
    if conf is not None:
        reason = f" [muted]{_escape(str(review.get('llm_confidence_reason') or ''))}[/]"
        head.add_row("reviewer", f"{bar(float(conf), 100, 20)} {conf}/100{reason}")
    for part, pts in (review.get("score_breakdown") or {}).items():
        top = RUBRIC_MAX.get(part, max(int(pts), 1))
        head.add_row(f"  {part.replace('_', ' ')}", f"[faint]{bar(float(pts), top, 10)}[/] {pts}/{top}")
    checks = review.get("checks") or {}
    if checks:
        labels = {
            "target_met_central": "target met",
            "target_met_conservative": "within error",
            "thermo_stable": "on the hull",
            "dyn_stable": "dynamically stable",
            "all_passed": "all checks",
        }
        skip = {"targets"} | ({"target_met_conservative"} if (spec or {}).get("target_tool") else set())
        cells = []
        for k, v in checks.items():
            if k in skip or isinstance(v, list | dict):
                continue
            cells.append(
                f"{mark(v)} {labels.get(k, k)}" if v is not None else f"[muted]off[/] {labels.get(k, k)}"
            )
        head.add_row("checks", "  ".join(cells))
        for r in checks.get("targets") or []:
            head.add_row("", target_fact(r))
    rows: list[RenderableType] = [head]
    findings = review.get("findings") or []
    if findings:
        ft = Table.grid(padding=(0, 2))
        ft.add_column(no_wrap=True)
        ft.add_column(style="muted", no_wrap=True)
        ft.add_column()
        for f in findings:
            sev = str(f.get("severity", "info"))
            text = _escape(str(f.get("text", "")))
            if f.get("suggestion"):
                text += f" [muted]{GLYPH['arrow']} {_escape(str(f['suggestion']))}[/]"
            if f.get("paper_ids"):
                text += f" [faint]{', '.join(f['paper_ids'])}[/]"
            ft.add_row(f"[severity.{sev}]{sev}[/]", str(f.get("category", "")), text)
        rows += [Text(), ft]
    if review.get("recommended_next"):
        why = (
            f" [muted]{GLYPH['dot']} {_escape(str(review.get('why') or ''))}[/]" if review.get("why") else ""
        )
        rows += [Text(), Text.from_markup(f"[muted]next[/]  [bold]{review['recommended_next']}[/]{why}")]
    if review.get("gate_note"):
        rows.append(Text.from_markup(f"[warn]{_escape(str(review['gate_note']))}[/]"))
    if review.get("stop_rule"):
        rows.append(Text.from_markup(f"[accent]stop rule fired: {review['stop_rule']}[/]"))
    return panel(Group(*rows), f"review · round {review.get('round', '?')}")


def target_fact(r: dict[str, Any]) -> str:
    """One target's value and verdict: ``"band gap 2.20 ± 0.65 eV ✓"`` or ``"… (agent-built) ✓"`` (GP-1, AT-17).

    Parameters
    ----------
    r : dict
        Target row: ``{property, value, uncertainty, units, met_central, met_conservative, evaluable, agent_built}``.

    Returns
    -------
    str
        Markup; the tick or cross is the harness's target check.
    """
    prop = str(r.get("property") or "").replace("_", " ")
    if r.get("evaluable") is False:
        return f"{prop} [warn]not evaluated[/]"
    if r.get("value") is None:
        return f"{prop} {DASH}"
    if r.get("agent_built"):
        return f"{prop} {fmt_quantity(r['value'], None, r.get('units', ''))} [accent](agent-built)[/] {mark(r.get('met_central'))}"
    within = " [muted]within error[/]" if r.get("met_conservative") else ""
    return f"{prop} {fmt_quantity(r['value'], r.get('uncertainty'), r.get('units', ''))} {mark(r.get('met_central'))}{within}"


# Readable names for the final re-check's rows (as the web UI's CHECK_NAMES).
CHECK_NAMES = {
    "e_hull_eV_per_atom": "E_hull (eV/atom)",
    "e_hull": "E_hull (eV/atom)",
    "thermo_stable": "on the hull",
    "dyn_stable": "dynamically stable",
    "min_frequency_THz": "lowest phonon (THz)",
    "min_freq": "lowest phonon (THz)",
    "md_stable": "stable in MD",
    "target_met": "target met",
    "target_met_conservative": "target met within error",
    "band_gap": "band gap (eV)",
    "bulk_modulus": "bulk modulus (GPa)",
    "disorder_spread_eV_per_atom": "disorder spread (eV/atom)",
}


def validation_panel(p: dict[str, Any]) -> Panel:
    """Screening versus higher-fidelity values and the verdict (Section 9).

    Parameters
    ----------
    p : dict
        ``final_calculation`` finished payload: ``{verdict, checks, notes}``.

    Returns
    -------
    rich.panel.Panel
        The panel.
    """
    verdict = str(p.get("verdict", "?"))
    colour = {"confirmed": "ok", "weakened": "warn", "overturned": "bad"}.get(verdict, "muted")
    t = Table(box=None, header_style="muted", pad_edge=False, padding=(0, 2))
    for col in ("check", "screening", "final", "Δ"):
        t.add_column(col)

    def show(v: Any) -> str:
        """Format one cell.

        Parameters
        ----------
        v : Any
            Number, flag or ``None``.

        Returns
        -------
        str
            Markup.
        """
        return mark(v) if isinstance(v, bool) else fmt_num(v, 3)

    for c in p.get("checks") or []:
        name = str(c.get("name", ""))
        t.add_row(
            CHECK_NAMES.get(name, name.replace("_", " ")),
            show(c.get("screening_value")),
            show(c.get("final_value")),
            show(c.get("delta")),
        )
    rows: list[RenderableType] = [t]
    for note in p.get("notes") or []:
        rows.append(Text(f"{GLYPH['dot']} {strip_codes(str(note))}", style="muted"))
    return panel(Group(*rows), f"final re-check · {verdict}", colour)


def brief_panel(markdown: str, title: str | None = None) -> Panel:
    """Render the experimental brief (or the computational one) as Markdown (UI-8).

    Parameters
    ----------
    markdown : str
        Brief text.
    title : str, optional
        Panel title; ``"experimental brief"`` by default.

    Returns
    -------
    rich.panel.Panel
        The panel.
    """
    return panel(Markdown(markdown), title or "experimental brief", padding=(1, 2))


def literature_view(
    lit: dict[str, Any],
    cited: set[str] | None = None,
    *,
    text_filter: str | None = None,
    limit: int | None = 15,
    full: bool = False,
) -> RenderableType:
    """Show the literature a run gathered: papers, evidence, failed approaches, trend (Section 2, UI-4).

    Parameters
    ----------
    lit : dict
        ``LiteratureContext`` as a dict.
    cited : set of str, optional
        Paper IDs cited in the brief.
    text_filter : str, optional
        Keep only papers whose title, abstract or authors contain this text (any case).
    limit : int or None
        Most papers to list; ``None`` lists all.
    full : bool
        Show whole abstracts instead of about three lines.

    Returns
    -------
    RenderableType
        Headline counts, the papers, then the evidence and the synthesised lists.
    """
    cited = cited or set()
    papers = sorted(
        lit.get("papers") or [],
        key=lambda x: (-(x.get("relevance") or 0.0), -(x.get("citation_count") or 0)),
    )
    if text_filter:
        needle = text_filter.lower()
        papers = [
            x
            for x in papers
            if needle
            in " ".join([x.get("title") or "", x.get("abstract") or "", *(x.get("authors") or [])]).lower()
        ]
    insights = lit.get("insights") or []
    failed = lit.get("failed_approaches") or []
    n_cited = sum(1 for x in lit.get("papers") or [] if x.get("paper_id") in cited)
    head = Text.assemble(
        (f"{len(lit.get('papers') or [])} papers", "bold"),
        (f"  {GLYPH['sep']}  ", "gutter"),
        (f"{len(insights)} evidence note{'' if len(insights) == 1 else 's'}", "bold"),
        (f"  {GLYPH['sep']}  ", "gutter"),
        (f"{len(failed)} failed approach{'' if len(failed) == 1 else 'es'}", "bold"),
        (f"  {GLYPH['sep']}  ", "gutter"),
        (f"{n_cited} cited in the brief", "bold"),
    )
    rows: list[RenderableType] = [head, Text()]
    shown = papers if limit is None else papers[:limit]
    for i, x in enumerate(shown, 1):
        title = _escape(str(x.get("title") or x.get("paper_id")))
        link = x.get("url") or (f"https://doi.org/{x['doi']}" if x.get("doi") else None)
        title_markup = f"[link={link}]{title}[/link]" if link else title
        meta = [
            str(x["year"]) if x.get("year") else "",
            _cites(x.get("citation_count")),
            str(x.get("found_via") or ""),
        ]
        star = f"  [accent]{GLYPH['mark']} cited in brief[/]" if x.get("paper_id") in cited else ""
        rows.append(Text.from_markup(f"[faint]{i:>2}.[/] [bold]{title_markup}[/]{star}"))
        authors = x.get("authors") or []
        who = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        rel = float(x.get("relevance") or 0.0)
        line2 = f" {GLYPH['dot']} ".join(b for b in [who, *meta, str(x.get("paper_id") or "")] if b)
        rows.append(
            Text.from_markup(
                f"    [muted]{_escape(line2)}[/]  [faint]relevance[/] [accent]{bar(rel, 1.0, 8)}[/]"
            )
        )
        abstract = " ".join(str(x.get("abstract") or "").split())
        if abstract:
            if not full and len(abstract) > 320:
                abstract = abstract[:317].rsplit(" ", 1)[0] + "…"
            rows.append(Padding(Text(abstract, style="faint"), (0, 0, 0, 4)))
        rows.append(Text())
    if len(papers) > len(shown):
        rows.append(
            Text(f"    … {len(papers) - len(shown)} more papers (use --limit or --all)", style="muted")
        )
        rows.append(Text())
    if insights:
        t = Table(box=None, header_style="muted", pad_edge=False, padding=(0, 2))
        for col in ("type", "composition", "value", "finding", "paper"):
            t.add_column(col, no_wrap=col in ("type", "composition", "paper"))
        for ins in insights:
            v = ins.get("value") or {}
            value = fmt_quantity(v.get("value"), v.get("uncertainty"), v.get("units", "")) if v else DASH
            kind = str(ins.get("finding_type") or "")
            style = "bad" if kind == "negative" else "accent"
            t.add_row(
                f"[{style}]{kind}[/]",
                str(ins.get("composition") or DASH),
                value,
                _escape(str(ins.get("summary") or ""))
                + (f" [faint]({_escape(str(ins['conditions']))})[/]" if ins.get("conditions") else ""),
                f"[id]{ins.get('paper_id', '')}[/]",
            )
        rows += [round_rule("evidence notes"), t, Text()]
    for title, key in (
        ("failed approaches", "failed_approaches"),
        ("mechanisms", "mechanisms"),
        ("synthesis routes", "synthesis_routes"),
        ("open questions", "open_questions"),
    ):
        items = lit.get(key) or []
        if items:
            rows.append(round_rule(title))
            rows += [Text.from_markup(f"  [accent]{GLYPH['tool']}[/] {_escape(str(item))}") for item in items]
            rows.append(Text())
    trend = lit.get("field_trend") or {}
    if trend:
        years = sorted(trend, key=lambda y: int(y))
        counts = [float(trend[y]) for y in years]
        top = max(counts) or 1.0
        rows.append(round_rule("papers per year"))
        for y, c in zip(years, counts, strict=True):
            rows.append(Text.from_markup(f"  [muted]{y}[/] [accent]{bar(c, top, 30)}[/] {int(c)}"))
        rows.append(Text())
    if lit.get("coverage_note"):
        rows.append(Text(str(lit["coverage_note"]), style="faint"))
    return Group(*rows)
