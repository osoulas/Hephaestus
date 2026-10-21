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
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from hephaestus.core.types import ProgressEvent, RunState
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
    submitted: dict[str, Any] | None = None
    stop_rule: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] | None = None
    brief: dict[str, Any] | None = None
    end: dict[str, Any] | None = None
    first_time: datetime | None = None
    last_time: datetime | None = None
    _local_start: float | None = None
    in_campaign: bool = False  # draw the timeline gutter from the campaign's start to the run's end
    _pending: list[RenderableType] = field(default_factory=list)  # round line held until its review prints

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
        handler = getattr(self, f"_on_{event.section}", None)
        if handler is None:
            return [self._line(GLYPH["dot"], f"[muted]{event.section} {event.status}[/]", "muted")]
        out: list[RenderableType] = handler(event)
        if self.in_campaign:
            out = [x if isinstance(x, Rule) else Gutter(x) for x in out]
        if event.section == "run" and event.status != "started":
            self.in_campaign = False
        return out

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
                        f"{cfg.get('max_iterations', '?')} rounds max",
                        f"{cfg.get('max_tool_calls', '?')} tool calls max",
                        "novel only" if cfg.get("require_novel") else None,
                    )
                    if x
                )
                .strip()
            )
            head = Text.assemble((f" {verb.upper()} ", "tab.accent"), " ", (e.run_id, "id"))
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
                review_panel(review, self.config),
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
            who = ", ".join(p.get("candidates") or [])
            return [
                round_rule("final re-check"),
                self._line(GLYPH["mark"], f"Re-checking [id]{who}[/] at higher fidelity", "ember"),
            ]
        if e.status == "failed":
            return [result_line(f"[bad]Final re-check failed:[/] {_escape(str(p.get('error', '')))}")]
        self.validation = p
        return [validation_panel(p)]

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
        return [self._line(GLYPH["mark"], f"Brief written · cites {cited} papers{pdf}", "ember")]

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
        if self.stopping:
            verb = "Stopping: submitting the best candidate"
        elif self.calls:
            verb = _verb(next(reversed(self.calls.values())))
        else:
            verb = self.phase
        max_rounds = self.config.get("max_iterations")
        max_calls = self.config.get("max_tool_calls")
        bits = [elapsed(now - start)]
        if max_rounds:
            bits.append(f"round {min(self.round, int(max_rounds))}/{max_rounds}")
        if max_calls:
            bits.append(f"{self.tool_calls}/{max_calls} tools")
        bits.append("^C again to detach" if self.stopping else "^C stop")
        line = ember_frame(now)
        line.append(f" {verb}", style="accent")
        for bit in bits:
            line.append(f"  {GLYPH['sep']} ", style="gutter")
            line.append(bit, style="muted")
        rows: list[RenderableType] = [line]
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
        t.add_column("sg", no_wrap=True)
        best = (self.submitted or {}).get("candidate_id")
        for cid, c in self.candidates.items():
            name = f"[accent]{cid}[/]" if cid == best else f"[id]{cid}[/]"
            t.add_row(
                name,
                str(c.get("composition") or ""),
                fmt_num(c.get("e_hull"), 3),
                mark(c.get("thermo_stable")) if c.get("e_hull") is not None else mark(None),
                fmt_num(c.get("min_freq")),
                str(c.get("verdict") or DASH),
                mark(c.get("dyn_stable")) if c.get("verdict") else mark(None),
                fmt_quantity(c.get("value"), c.get("uncertainty")) if c.get("value") is not None else DASH,
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

    def summary(self, state: RunState | None = None, brief_path: str | None = None) -> RenderableType:
        """End-of-run summary box.

        Parameters
        ----------
        state : RunState, optional
            Final state; fields from the events are used when omitted.
        brief_path : str, optional
            Where the brief was written.

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
        grid.add_row("status", f"[{colour}]{status}[/]" + (f" · {stop}" if stop else ""))
        if best:
            c = self.candidates.get(best, {})
            facts = [str(c.get("composition") or "")]
            if c.get("value") is not None:
                prop = str((self.spec or {}).get("target_property", "value")).replace("_", " ")
                units = (self.spec or {}).get("target_units", "")
                facts.append(f"{prop} {fmt_quantity(c.get('value'), c.get('uncertainty'), units)}")
            if c.get("e_hull") is not None:
                facts.append(f"E_hull {fmt_num(c.get('e_hull'), 3)} eV/atom {mark(c.get('thermo_stable'))}")
            if c.get("verdict"):
                facts.append(f"phonons {c.get('verdict')} {mark(c.get('dyn_stable'))}")
            grid.add_row("best", f"[accent]{best}[/] " + " · ".join(f for f in facts if f))
        if self.submitted:
            grid.add_row("gate", f"{mark(self.submitted.get('gate_passed'))} code gate")
        if self.validation:
            v = str(self.validation.get("verdict"))
            vc = {"confirmed": "ok", "weakened": "warn", "overturned": "bad"}.get(v, "muted")
            grid.add_row("re-check", f"[{vc}]{v}[/]")
        if self.first_time and self.last_time:
            took = elapsed((self.last_time - self.first_time).total_seconds())
            grid.add_row("time", f"{took} · {self.tool_calls} tool calls · {len(self.rounds)} rounds")
        tokens = (state.usage if state and state.usage else None) or self.usage
        if tokens and any(tokens.values()):
            grid.add_row("tokens", " · ".join(f"{k.replace('_', ' ')} {v:,}" for k, v in tokens.items() if v))
        if brief_path:
            grid.add_row("brief", brief_path)
        rid = self.run_id or (state.run_id if state else "")
        hints = [f"hephaestus replay {rid}", f"hephaestus brief {rid}"]
        if status not in ("finished", "needs_input"):
            hints.insert(0, f"hephaestus resume {rid}")
        grid.add_row("run", f"[id]{rid}[/]")
        grid.add_row("next", "[faint]" + "  ·  ".join(hints) + "[/]")
        return panel(grid, "run summary")


# ---------------------------------------------------------------------- call lines


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

    return escape(text)


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
    fn = SUMMARISERS.get(tool, _generic)
    try:
        return fn(inp, result, payload, verbose)
    except Exception:
        return _generic(inp, result, payload, verbose)


# ---------------------------------------------------------------------- panels


def spec_panel(spec: dict[str, Any]) -> Panel:
    """Show the parsed search specification before a run starts (UI-2).

    Parameters
    ----------
    spec : dict
        ``SearchSpec`` as a dict.

    Returns
    -------
    rich.panel.Panel
        The panel.
    """
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="muted", no_wrap=True)
    grid.add_column()
    prop = str(spec.get("target_property", "")).replace("_", " ")
    sym = DIRECTION.get(str(spec.get("direction")), str(spec.get("direction", "")))
    target = f"{prop} {sym} "
    ref_val = spec.get("reference_value") or {}
    if spec.get("target_value") is not None:
        target += f"{fmt_num(spec['target_value'])} {spec.get('target_units', '')}"
    elif spec.get("reference_material"):
        target += f"{spec['reference_material']} [muted](computed with the same model)[/]"
    if ref_val:
        target += f" [muted]· reference {fmt_quantity(ref_val.get('value'), ref_val.get('uncertainty'), ref_val.get('units', ''))}[/]"
    if spec.get("target_tolerance") is not None:
        target += f" ± {fmt_num(spec['target_tolerance'])}"
    grid.add_row("family", _escape(str(spec.get("material_family", ""))))
    grid.add_row("target", target)
    th = spec.get("stability_thresholds") or {}
    grid.add_row(
        "stability",
        f"E_hull ≤ {fmt_num(th.get('e_hull_max_eV_per_atom', 0.05))} eV/atom · "
        f"phonons ≥ {fmt_num(th.get('phonon_min_THz', -0.1), 1)} THz "
        "[muted](decided by code)[/]",
    )
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


def review_panel(review: dict[str, Any], config: dict[str, Any] | None = None) -> Panel:
    """Show the reviewer's critique for one round, with the code rubric and checks (Section 7, UI-5).

    Parameters
    ----------
    review : dict
        ``request_review`` result.
    config : dict, optional
        Run config, for the confidence threshold.

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
            "target_met_central": "target (central)",
            "target_met_conservative": "target (value \u2212 uncertainty)",
            "thermo_stable": "thermo",
            "dyn_stable": "dynamic",
            "all_passed": "all checks",
        }
        head.add_row("checks", "  ".join(f"{mark(v)} {labels.get(k, k)}" for k, v in checks.items()))
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
        t.add_row(
            str(c.get("name", "")),
            show(c.get("screening_value")),
            show(c.get("final_value")),
            show(c.get("delta")),
        )
    rows: list[RenderableType] = [t]
    for note in p.get("notes") or []:
        rows.append(Text(f"{GLYPH['dot']} {note}", style="muted"))
    return panel(Group(*rows), f"final re-check · {verdict}", colour)


def brief_panel(markdown: str) -> Panel:
    """Render the experimental brief as Markdown (UI-8).

    Parameters
    ----------
    markdown : str
        Brief text.

    Returns
    -------
    rich.panel.Panel
        The panel.
    """
    return panel(Markdown(markdown), "experimental brief", padding=(1, 2))
