"""Answer the tool builder from the terminal: plans, code reviews, package requests, questions, give-ups (AT-28).

When the run waits for the user, the stream pauses its live line and calls ``answer_card``. The
card itself is already in the log (the renderer printed it, or replayed it from history on resume),
so this shows the code (paged and highlighted on a terminal), offers exactly the actions the engine
allows, asks for a message or a confirmation where one is needed, and sends the answer with
``Experiment.decide`` (AT-29). Nothing here decides anything itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.console import Console, RenderableType
from rich.text import Text

from hephaestus.interfaces.cli.cards import code_view
from hephaestus.interfaces.cli.render import result_line
from hephaestus.interfaces.python import ExperimentError

Choice = tuple[str, str]  # (key the user types, what it does)
Choose = Callable[[str, list[Choice]], str]


def waiting_decisions(exp: Any) -> list[dict[str, Any]]:
    """List what the run waits for, as the engine sees it: ``{proposal_id, stage, allowed}`` (AT-24).

    Parameters
    ----------
    exp : Experiment
        The run; a handle without ``waiting`` (a replay stand-in) waits for nothing.

    Returns
    -------
    list of dict
        Open waits; empty when nothing waits or the run is not in this process.
    """
    waiting = getattr(exp, "waiting", None)
    return list(waiting()) if callable(waiting) else []


def make_choose(console: Console, ask: Callable[[str], str]) -> Choose:
    """Build a chooser that lists the actions and reads a key, a number or the start of a label.

    Parameters
    ----------
    console : rich.console.Console
        Output console.
    ask : callable
        ``ask(prompt) -> line``.

    Returns
    -------
    callable
        ``choose(question, choices) -> key``; asks again until the answer matches.
    """

    def choose(question: str, choices: list[Choice]) -> str:
        """Ask until one of the choices is picked.

        Parameters
        ----------
        question : str
            What is being decided.
        choices : list of tuple
            ``(key, label)`` pairs.

        Returns
        -------
        str
            The chosen key.
        """
        line = Text.assemble((f"{question}  ", "bold"))
        for key, label in choices:
            line.append(f"[{key}]", style="accent.bold")
            line.append(f" {label}   ", style="")
        console.print(line)
        while True:
            reply = ask("").strip().lower()
            for key, label in choices:
                if reply == key.lower() or (len(reply) > 1 and label.lower().startswith(reply)):
                    return key
            console.print(Text(f"  Type one of: {', '.join(k for k, _ in choices)}", style="muted"))

    return choose


def page(console: Console, renderable: RenderableType) -> None:
    """Show long content in the pager on a terminal (``less -R``), or print it otherwise (AT-28).

    Parameters
    ----------
    console : rich.console.Console
        Output console.
    renderable : RenderableType
        Content, e.g. highlighted code.
    """
    if console.is_terminal:
        with console.pager(styles=True):
            console.print(renderable)
    else:
        console.print(renderable)


def answer_card(
    console: Console,
    exp: Any,
    card: dict[str, Any],
    allowed: list[str],
    *,
    ask: Callable[[str], str],
    confirm: Callable[[str, bool], bool],
    choose: Choose | None = None,
) -> str | None:
    """Show a waiting card, ask for the decision and send it (AT-7, AT-13, AT-28).

    Parameters
    ----------
    console : rich.console.Console
        Output console.
    exp : Experiment
        The live run; ``exp.decide(proposal_id, action, message)`` sends the answer.
    card : dict
        The renderer's card: ``{id, stage, payload, revision}``.
    allowed : list of str
        Actions the engine offers now (from ``waiting_decisions``).
    ask : callable
        ``ask(prompt) -> line`` for messages.
    confirm : callable
        ``confirm(question, default) -> bool`` for cancellations.
    choose : callable, optional
        Chooser; built from ``ask`` when omitted.

    Returns
    -------
    str or None
        The action sent, or ``None`` if the engine refused it (the run keeps waiting).
    """
    choose = choose or make_choose(console, ask)
    stage, p = card["stage"], card["payload"]
    if stage == "code" and p.get("source"):
        page(console, code_view(str(p.get("source")), str(p.get("tests") or "")))
    while True:
        action, message = _pick(console, stage, p, allowed, ask, confirm, choose)
        if action is None:
            continue  # viewed the code, or backed out of a cancellation
        try:
            exp.decide(card["id"], action, message)
        except ExperimentError as exc:
            console.print(result_line(f"[bad]{exc}[/]"))
            return None
        return action


def _pick(
    console: Console,
    stage: str,
    p: dict[str, Any],
    allowed: list[str],
    ask: Callable[[str], str],
    confirm: Callable[[str, bool], bool],
    choose: Choose,
) -> tuple[str | None, str | None]:
    """Ask once for an action on a card, with any message it needs.

    Parameters
    ----------
    console : rich.console.Console
        Output console.
    stage : str
        ``plan``, ``code``, ``package``, ``question`` or ``give_up``.
    p : dict
        The proposal or request.
    allowed : list of str
        Actions the engine offers.
    ask, confirm, choose : callable
        Input helpers.

    Returns
    -------
    tuple
        ``(action, message)``; ``(None, None)`` to ask again.
    """
    cancel_run: list[Choice] = [("q", "cancel the run")] if "cancel_run" in allowed else []
    if stage in ("plan", "code"):
        choices: list[Choice] = [("a", "accept the plan" if stage == "plan" else "approve the code")]
        if "revise" in allowed:
            choices.append(("c", "request changes"))
        if stage == "code" and p.get("source"):
            choices += [("v", "view the code"), ("t", "view the tests")]
        if "cancel_tool" in allowed:
            choices.append(("x", "cancel the tool"))
        choices += cancel_run
        if "revise" not in allowed:
            console.print(
                Text("  No more change requests for this tool: accept it or cancel.", style="muted")
            )
        key = choose("Your decision:", choices)
        if key == "a":
            return "accept", None
        if key == "c":
            what = "plan" if stage == "plan" else "code"
            msg = ask(f"What should change in the {what}? ").strip()
            if not msg:
                console.print(Text("  Say what should change.", style="warn"))
                return None, None
            return "revise", msg
        if key == "v":
            page(console, code_view(str(p.get("source") or "")))
            return None, None
        if key == "t":
            page(console, code_view(str(p.get("tests") or "# no tests")))
            return None, None
        if key == "x":
            note = f" {p['if_cancelled']}" if p.get("supplies_target") and p.get("if_cancelled") else ""
            if confirm(f"Cancel {p.get('name')}? The run continues without it.{note}", False):
                return "cancel_tool", None
            return None, None
        return _confirm_cancel_run(confirm)
    if stage == "package":
        allow = "allow" if p.get("installed") else "allow and install"
        key = choose("Your decision:", [("a", allow), ("d", "don't allow"), *cancel_run])
        if key == "a":
            return "allow", None
        if key == "d":
            return "deny", None
        return _confirm_cancel_run(confirm)
    if stage == "question":
        options = [str(o) for o in p.get("options") or []]
        choices = [(str(i), o) for i, o in enumerate(options, 1)] + [("w", "write an answer"), *cancel_run]
        key = choose("Your answer:", choices)
        if key.isdigit():
            return "answer", options[int(key) - 1]
        if key == "w":
            msg = ask("Your answer: ").strip()
            if not msg:
                console.print(Text("  Type an answer.", style="warn"))
                return None, None
            return "answer", msg
        return _confirm_cancel_run(confirm)
    # give_up
    key = choose("Your decision:", [("s", "let it stop"), ("b", "build something else"), *cancel_run])
    if key == "s":
        return "allow", None
    if key == "b":
        msg = ask(
            "Build this instead (e.g. a lattice-limited mobility proxy at a fixed carrier density): "
        ).strip()
        if not msg:
            console.print(Text("  Say what to build instead.", style="warn"))
            return None, None
        return "answer", msg
    return _confirm_cancel_run(confirm)


def _confirm_cancel_run(confirm: Callable[[str, bool], bool]) -> tuple[str | None, str | None]:
    """Confirm cancelling the whole run (AT-9).

    Parameters
    ----------
    confirm : callable
        ``confirm(question, default) -> bool``.

    Returns
    -------
    tuple
        ``("cancel_run", None)`` when confirmed, else ``(None, None)``.
    """
    if confirm("Cancel the whole run? Nothing else runs.", False):
        return "cancel_run", None
    return None, None
