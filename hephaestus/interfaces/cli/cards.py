"""Agent-built tools in the terminal: plan, code and request cards, the code viewer and the library (Section 14).

These are pure renderers shared by the event log (``render.py``), the decision prompts
(``decide.py``) and the ``tools`` commands. They show what the engine decided and asks; every check
result, trial value and allowed action comes from the engine's payloads (AT-13, AT-27, AT-28).
"""

from __future__ import annotations

from typing import Any

from rich.console import Group, RenderableType
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from hephaestus.interfaces.cli.theme import DASH, fmt_num, mark, panel, strip_codes

# What each decision is called once made (as the web UI's ACTION_TEXT).
ACTION_TEXT = {
    "accept": "Accepted",
    "revise": "Changes requested",
    "cancel_tool": "Tool cancelled",
    "cancel_run": "Run cancelled",
    "allow": "Allowed",
    "deny": "Not allowed",
    "answer": "Answered",
}
CHECK_NAMES = {"static": "Static check", "tests": "Tests", "trial": "Trial run"}


def _t(text: Any) -> str:
    """Escape engine text for rich markup and drop requirement IDs.

    Parameters
    ----------
    text : Any
        Text from a payload.

    Returns
    -------
    str
        Safe markup.
    """
    return escape(strip_codes(str(text or "")))


def _prop(text: Any) -> str:
    """Make a property key readable: ``thermal_expansion`` -> ``thermal expansion``.

    Parameters
    ----------
    text : Any
        Property key.

    Returns
    -------
    str
        Readable name.
    """
    return str(text or "").replace("_", " ")


def _status_pill(decision: dict[str, Any] | None, withdrawn: str | None, waiting: bool) -> str:
    """Markup for a card's state: decided, withdrawn, waiting or not decided.

    Parameters
    ----------
    decision : dict or None
        The decision, once made.
    withdrawn : str or None
        Why the agent withdrew the code.
    waiting : bool
        The run is waiting for this card now.

    Returns
    -------
    str
        Markup.
    """
    if withdrawn is not None:
        return "[muted]withdrawn by the agent[/]"
    if decision:
        action = str(decision.get("action"))
        tone = "bad" if action in ("cancel_tool", "cancel_run", "deny") else "ok"
        return f"[{tone}]{ACTION_TEXT.get(action, action)}[/]"
    return "[warn]needs your decision[/]" if waiting else "[muted]not decided[/]"


def plan_card(
    p: dict[str, Any],
    *,
    decision: dict[str, Any] | None = None,
    waiting: bool = False,
) -> Panel:
    """Show a tool plan: what the tool computes and how, with its citations (AT-6, AT-8).

    Parameters
    ----------
    p : dict
        ``ToolProposal`` at the plan stage.
    decision : dict, optional
        The user's decision, once made.
    waiting : bool
        The run is waiting for this plan.

    Returns
    -------
    rich.panel.Panel
        The card.
    """
    who = "Your library file" if p.get("author") == "user" else "The agent"
    units = f" ({_t(p.get('units'))})" if p.get("units") else ""
    head = Text.from_markup(
        f"[tool]{_t(p.get('name'))}[/] computes {_prop(p.get('property'))}{units}  "
        f"{_status_pill(decision, None, waiting)}\n"
        f"[muted]{who} proposes this tool. Nothing is written until you accept the plan.[/]"
    )
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="muted", no_wrap=True)
    grid.add_column()
    if p.get("description"):
        grid.add_row("what", _t(p["description"]))
    if p.get("method"):
        grid.add_row("method", _t(p["method"]))
    grid.add_row("citations", _t(", ".join(p.get("citations") or [])) or "none")
    blocks = p.get("building_blocks") or []
    grid.add_row("building blocks", _t(", ".join(blocks)) if blocks else "none: its own calculation")
    if p.get("expected_runtime_s") is not None:
        grid.add_row("expected time", f"about {fmt_num(p['expected_runtime_s'], 0)} s per candidate")
    props = list(((p.get("inputs") or {}).get("properties") or {}).keys())
    grid.add_row("inputs", _t(", ".join(props)) if props else "only the candidate")
    if p.get("if_cancelled"):
        style = "warn" if p.get("supplies_target") else "muted"
        grid.add_row("if you cancel it", f"[{style}]{_t(p['if_cancelled'])}[/]")
    rows: list[RenderableType] = [head, Text(), grid]
    if decision and decision.get("message"):
        rows += [Text(), Text.from_markup(f"[accent]You asked:[/] {_t(decision['message'])}")]
    title = f"tool plan · revision {p.get('revision', 1)}"
    return panel(Group(*rows), title, "warn" if waiting and not decision else "accent")


def code_card(
    p: dict[str, Any],
    *,
    decision: dict[str, Any] | None = None,
    withdrawn: str | None = None,
    waiting: bool = False,
    show_source: bool = False,
) -> Panel:
    """Show code under review: check results, the trial against the literature range, building blocks (AT-13).

    Parameters
    ----------
    p : dict
        ``ToolProposal`` at the code stage.
    decision : dict, optional
        The user's decision, once made.
    withdrawn : str, optional
        Why the agent withdrew this code.
    waiting : bool
        The run is waiting for this review.
    show_source : bool
        Include the highlighted source and tests (the prompt pages them instead).

    Returns
    -------
    rich.panel.Panel
        The card.
    """
    who = "Your library file" if p.get("author") == "user" else "The agent"
    units = f" ({_t(p.get('units'))})" if p.get("units") else ""
    rows: list[RenderableType] = []
    if withdrawn is not None:
        rows += [
            Text.from_markup(
                f"[warn]The agent withdrew this code:[/] {_t(withdrawn)} A fixed version follows below."
            ),
            Text(),
        ]
    rows.append(
        Text.from_markup(
            f"[tool]{_t(p.get('name'))}[/] computes {_prop(p.get('property'))}{units}  "
            f"{_status_pill(decision, withdrawn, waiting)}\n"
            f"[muted]{who} wrote this code. It only runs in the campaign once you approve it.[/]"
        )
    )
    checks = p.get("checks") or {}
    if checks:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column()
        for key in ("static", "tests", "trial"):
            c = checks.get(key)
            if c is None:
                continue
            grid.add_row(
                mark(bool(c.get("passed"))), CHECK_NAMES.get(key, key), f"[muted]{_t(c.get('report'))}[/]"
            )
        rows += [Text(), grid]
    trial = p.get("trial") or {}
    if trial.get("value") is not None:
        u = _t(trial.get("units") or p.get("units") or "")
        line = f"Trial run on {_t(trial.get('material'))}: [bold]{float(trial['value']):.4g} {u}[/]"
        rng = trial.get("range")
        if rng:
            line += f" · expected {fmt_num(rng[0], 3)} to {fmt_num(rng[1], 3)} {u} from the literature"
        rows += [Text(), Text.from_markup(line)]
    if p.get("toolkit_calls"):
        rows.append(Text.from_markup(f"[muted]building blocks used:[/] {_t(', '.join(p['toolkit_calls']))}"))
    src, tests = str(p.get("source") or ""), str(p.get("tests") or "")
    if src:
        rows.append(
            Text.from_markup(
                f"[muted]tool {len(src.splitlines())} lines · tests {len(tests.splitlines())} lines[/]"
            )
        )
    if show_source and src:
        rows += [Text(), code_view(src, tests)]
    if decision and decision.get("message"):
        rows += [Text(), Text.from_markup(f"[accent]You asked:[/] {_t(decision['message'])}")]
    title = f"code review · revision {p.get('revision', 1)}"
    return panel(Group(*rows), title, "warn" if waiting and not decision else "accent")


def request_card(
    r: dict[str, Any],
    *,
    decision: dict[str, Any] | None = None,
    waiting: bool = False,
) -> Panel:
    """Show a package request, a question or a proposal to give up.

    Parameters
    ----------
    r : dict
        ``UserRequest``: ``kind`` is ``package``, ``question`` or ``give_up``.
    decision : dict, optional
        The user's answer, once given.
    waiting : bool
        The run is waiting for this answer.

    Returns
    -------
    rich.panel.Panel
        The card.
    """
    kind = str(r.get("kind"))
    pill = _status_pill(decision, None, waiting)
    rows: list[RenderableType] = []
    if kind == "package":
        pkg = _t(r.get("package"))
        rows.append(Text.from_markup(f"Use [tool]{pkg}[/] in the tool?  {pill}"))
        rows.append(
            Text.from_markup(
                "[muted]It is outside the sandbox's allowlist, so the agent needs your permission.[/]"
            )
        )
        rows.append(Text.from_markup(f"[accent]Why:[/] {_t(r.get('question'))}"))
        if r.get("installed"):
            rows.append(
                Text.from_markup(
                    f"[muted]{pkg} is already installed; allowing it lets the tool import it.[/]"
                )
            )
        else:
            pip = _t(r.get("pip_name") or r.get("package"))
            rows.append(
                Text.from_markup(
                    f"[muted]Not installed. Allowing it runs [/]pip install {pip}[muted] in this Python.[/]"
                )
            )
        title = "package request"
    elif kind == "give_up":
        rows.append(Text.from_markup(f"Stop without a tool?  {pill}"))
        rows.append(
            Text.from_markup(
                "[muted]If it stops, the run cannot evaluate this property. You can tell it what to build instead.[/]"
            )
        )
        rows.append(Text.from_markup(f"[accent]Why:[/] {_t(r.get('question'))}"))
        title = "the agent proposes to stop"
    else:
        rows.append(Text.from_markup(f"{_t(r.get('question'))}  {pill}"))
        for i, opt in enumerate(r.get("options") or [], 1):
            rows.append(Text.from_markup(f"  [accent]{i}.[/] {_t(opt)}"))
        title = "question from the tool builder"
    if decision and decision.get("message"):
        said = "You said" if kind == "give_up" else "You answered"
        rows += [Text(), Text.from_markup(f"[accent]{said}:[/] {_t(decision['message'])}")]
    return panel(Group(*rows), title, "warn" if waiting and not decision else "accent")


def code_view(source: str, tests: str = "") -> RenderableType:
    """Highlight a tool's source and tests with line numbers, for paging (AT-13, AT-28).

    Parameters
    ----------
    source : str
        Tool source.
    tests : str
        Test source.

    Returns
    -------
    RenderableType
        The tool, then its tests.
    """
    parts: list[RenderableType] = [
        Text("tool", style="accent.bold"),
        Syntax(source, "python", line_numbers=True, word_wrap=True, theme="ansi_dark"),
    ]
    if tests:
        parts += [
            Text(),
            Text("tests", style="accent.bold"),
            Syntax(tests, "python", line_numbers=True, word_wrap=True, theme="ansi_dark"),
        ]
    return Group(*parts)


def decision_text(decision: dict[str, Any], name: str | None) -> str:
    """Describe a decision in the log: ``"You accepted the plan for toy_expansion"``.

    Parameters
    ----------
    decision : dict
        ``ToolDecision``.
    name : str or None
        Tool name, package name, or ``"question"``.

    Returns
    -------
    str
        Markup.
    """
    action, stage = str(decision.get("action")), str(decision.get("stage"))
    what = {"plan": "the plan", "code": "the code"}.get(stage, "")
    who = f"[tool]{_t(name)}[/]" if name and name != "question" else ""
    msg = f": {_t(decision['message'])}" if decision.get("message") else ""
    if stage in ("plan", "code"):
        verb = {
            "accept": f"You accepted {what} for {who}",
            "revise": f"You asked for changes to {what} for {who}{msg}",
            "cancel_tool": f"You cancelled the tool {who}",
            "cancel_run": "You cancelled the run",
        }.get(action, f"{ACTION_TEXT.get(action, action)} ({what} for {who})")
        return verb
    if stage == "package":
        return {
            "allow": f"You allowed {who}",
            "deny": f"You did not allow {who}",
            "cancel_run": "You cancelled the run",
        }.get(action, ACTION_TEXT.get(action, action))
    if stage == "give_up":
        return {
            "allow": "You let the agent stop without a tool",
            "answer": f"You asked the agent to build something else{msg}",
            "cancel_run": "You cancelled the run",
        }.get(action, ACTION_TEXT.get(action, action))
    return {"answer": f"You answered{msg}", "cancel_run": "You cancelled the run"}.get(
        action, ACTION_TEXT.get(action, action)
    )


def cancelled_text(p: dict[str, Any]) -> str:
    """Describe the end of a tool without a tool (the web UI's four texts).

    Parameters
    ----------
    p : dict
        ``capability`` ``cancelled`` payload: ``{scope, reason?, supplies_target}``.

    Returns
    -------
    str
        Markup.
    """
    if p.get("scope") == "run":
        return "You cancelled the run."
    reason = str(p.get("reason") or "")
    if reason == "budget":
        return "The tool builder ran out of time. The run screens stability only."
    if reason in ("cancelled by the user", "failed", "cancelled_run", ""):
        return "The run continues without the tool and screens stability only."
    return f"No tool: {_t(reason)}. The run screens stability only."


def library_table(lib: dict[str, Any]) -> RenderableType:
    """List the tool library: approved tools and files waiting for review (AT-20, AT-21).

    Parameters
    ----------
    lib : dict
        ``tool_library()`` result: ``{dir, approved, unapproved}``.

    Returns
    -------
    RenderableType
        The folder, then a table per section.
    """
    rows: list[RenderableType] = [Text.from_markup(f"[muted]library[/] {_t(lib.get('dir'))}"), Text()]
    waiting = lib.get("unapproved") or []
    if waiting:
        t = Table(box=None, header_style="muted", pad_edge=False, padding=(0, 2), title_justify="left")
        for col in ("waiting for review", "property", "status", "static check"):
            t.add_column(col)
        for u in waiting:
            h = u.get("header") or {}
            status = "[warn]edited: review again[/]" if u.get("changed") else "[warn]needs review[/]"
            check = u.get("check") or {}
            note = (
                _t(u.get("error"))
                if u.get("error")
                else f"{mark(check.get('passed'))} {_t(check.get('report'))}"
            )
            t.add_row(f"[tool]{_t(u.get('name'))}[/]", _prop(h.get("property")) or DASH, status, note)
        rows += [t, Text()]
    approved = lib.get("approved") or []
    if approved:
        t = Table(box=None, header_style="muted", pad_edge=False, padding=(0, 2))
        for col in ("approved", "property", "units", "written by", "run"):
            t.add_column(col)
        for a in approved:
            who = "you" if a.get("author") == "user" else "the agent"
            t.add_row(
                f"[tool]{_t(a.get('name'))}[/]",
                _prop(a.get("property")),
                _t(a.get("units")) or "no units",
                who,
                f"[id]{_t(a.get('run_id') or DASH)}[/]",
            )
        rows.append(t)
    if not waiting and not approved:
        rows.append(
            Text("No approved tools yet. They appear here when you approve a tool in a run.", style="muted")
        )
    return Group(*rows)
