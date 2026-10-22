"""Run a goal, resume or replay a run, and stream its events to the terminal (OR-7, OR-8, UI-9, UI-16).

``stream`` prints each event above a live status line. A pump thread reads the events, so the main
thread is free to ask the user things while the run waits (agent-built tools, AT-28).

Ctrl+C opens a menu when someone can answer: stop the search (the best candidate still gets the
re-check and the brief), stop everything (resumable), detach, or keep going; during a replay it
changes the speed, pauses or ends it. Without prompts the first Ctrl+C stops the search and the
second detaches; the run folder keeps its snapshot, so ``hephaestus resume`` can carry on.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.text import Text

from hephaestus.core.config import RunConfig
from hephaestus.core.types import RunState
from hephaestus.harness import runner
from hephaestus.interfaces.cli.render import (
    Renderer,
    brief_panel,
    questions_panel,
    result_line,
    spec_panel,
)
from hephaestus.interfaces.cli.theme import GLYPH, ember_frame
from hephaestus.interfaces.python import (
    Experiment,
    ExperimentError,
    backend_ready,
    run_blockers,
    simulation_status,
)

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_NEEDS_INPUT, EXIT_DETACHED = 0, 1, 2, 3, 130
MAX_QUESTION_ROUNDS = 3

DepsSource = runner.Deps | Callable[[], runner.Deps] | None


class Detached(Exception):
    """The user pressed Ctrl+C twice: stop watching and leave the run to be resumed.

    Parameters
    ----------
    run_id : str
        The run left behind.
    """

    def __init__(self, run_id: str) -> None:
        """Keep the run ID.

        Parameters
        ----------
        run_id : str
            Run ID.
        """
        super().__init__(run_id)
        self.run_id = run_id


@dataclass
class Prompts:
    """How the flow asks the user things; the REPL and the ``run`` command supply their own.

    Attributes
    ----------
    ask : callable
        ``ask(question) -> answer``.
    confirm : callable
        ``confirm(question, default) -> bool``.
    interactive : bool
        False means never ask: open questions end the flow with exit code 3.
    choose : callable, optional
        ``choose(question, [(key, label), ...]) -> key``; built from ``ask`` when omitted.
    ask_to_start : bool
        Confirm before starting the campaign (the session does; ``hephaestus run`` does not).
    """

    ask: Callable[[str], str]
    confirm: Callable[[str, bool], bool]
    interactive: bool = True
    choose: Callable[[str, list[tuple[str, str]]], str] | None = None
    ask_to_start: bool = True


def make_deps(source: DepsSource) -> runner.Deps | None:
    """Return fresh dependencies for one ``Experiment``.

    Parameters
    ----------
    source : runner.Deps, callable or None
        Injected dependencies (tests, offline demos), or a factory for them.

    Returns
    -------
    runner.Deps or None
        Dependencies, or ``None`` for the real ones.
    """
    return source() if callable(source) else source


@contextlib.contextmanager
def spinner(console: Console, text: str) -> Iterator[None]:
    """Show an animated status line while a blocking step runs (terminal only).

    Parameters
    ----------
    console : rich.console.Console
        Output console.
    text : str
        What is happening.

    Yields
    ------
    None
        While the step runs.
    """
    if not console.is_terminal:
        yield
        return
    import time

    def frame() -> Text:
        """Render one frame.

        Returns
        -------
        rich.text.Text
            Spinner and text.
        """
        line = ember_frame(time.monotonic())
        line.append(f" {text}…", style="accent")
        return line

    with Live(get_renderable=frame, console=console, refresh_per_second=10, transient=True):
        yield


IDLE_S = 0.3  # quiet time before asking about a waiting card: the run thread blocks while it waits


def stream(
    exp: Experiment,
    console: Console,
    renderer: Renderer,
    *,
    json_lines: bool = False,
    stoppable: bool = True,
    prompts: Prompts | None = None,
) -> RunState | None:
    """Print a run's events (history first, then live) until the run ends, asking the user when it waits.

    Parameters
    ----------
    exp : Experiment
        A started run, or a replay handle.
    console : rich.console.Console
        Output console.
    renderer : Renderer
        Formatter, carrying the display state.
    json_lines : bool
        Print each event as one JSON line instead of formatted output.
    stoppable : bool
        Whether the first Ctrl+C asks the campaign to stop (False for replays: it detaches).
    prompts : Prompts, optional
        How to ask the user; with it, waiting cards are answered here (AT-28) and Ctrl+C opens a
        menu. Without it nothing is asked.

    Returns
    -------
    RunState or None
        The final state.

    Raises
    ------
    Detached
        When the user detaches (or ends a replay).
    """
    from hephaestus.interfaces.cli.decide import answer_card, make_choose, waiting_decisions

    items: queue.Queue[tuple[int, Any]] = queue.Queue()
    done = object()
    generation = [0]

    def pump(events: Iterator[Any]) -> None:
        """Feed events into the queue from a thread, tagged with this pump's generation.

        Parameters
        ----------
        events : iterator
            ``exp.events(...)``.
        """
        gen = generation[0] = generation[0] + 1

        def work() -> None:
            """Read until the stream ends; an error (or a simulated Ctrl+C) is passed on."""
            try:
                for event in events:
                    items.put((gen, event))
            except BaseException as exc:
                items.put((gen, exc))
            finally:
                items.put((gen, done))

        threading.Thread(target=work, daemon=True, name=f"events-{exp.run_id}").start()

    def emit(event: Any) -> None:
        """Print one event.

        Parameters
        ----------
        event : ProgressEvent
            Event.
        """
        if json_lines:
            plain(console, event.model_dump_json())
            return
        for item in renderer.feed(event):
            console.print(item)

    live_ok = console.is_terminal and not json_lines
    live = Live(get_renderable=renderer.status, console=console, refresh_per_second=8, transient=True)
    asking = prompts is not None and prompts.interactive and not json_lines
    choose = (prompts.choose or make_choose(console, prompts.ask)) if asking and prompts else None
    renderer.menu = choose is not None
    checked: set[int] = set()  # cards found not waiting (history), by identity
    interrupts = 0
    paused_replay = False

    @contextlib.contextmanager
    def paused() -> Iterator[None]:
        """Hold the live status line while the user is asked something.

        Yields
        ------
        None
            While asking.
        """
        if live_ok:
            live.stop()
        try:
            yield
        finally:
            if live_ok:
                live.start()

    def maybe_decide() -> None:
        """Answer the card the run waits for, if the engine confirms it waits for it."""
        card = renderer.open_card()
        if card is None or id(card) in checked or renderer.replaying or not asking or prompts is None:
            return
        wait = next(
            (
                w
                for w in waiting_decisions(exp)
                if w["proposal_id"] == card["id"] and w["stage"] == card["stage"]
            ),
            None,
        )
        if wait is None:
            checked.add(id(card))  # decided earlier (history) or not in this process
            return
        with paused():
            answer_card(
                console,
                exp,
                card,
                list(wait.get("allowed") or card["allowed"]),
                ask=prompts.ask,
                confirm=prompts.confirm,
                choose=choose,
            )
        renderer.answered.add((card["id"], card["stage"], card["revision"]))

    def interrupt(from_stream: bool) -> None:
        """Handle Ctrl+C: a menu when someone can answer, else stop then detach.

        Parameters
        ----------
        from_stream : bool
            The interrupt came out of the event iterator (its pump ended), so a new pump is needed.

        Raises
        ------
        Detached
            When the user detaches or ends a replay.
        """
        nonlocal interrupts, paused_replay
        interrupts += 1
        if choose is None:  # nobody to ask: the plain rule
            if interrupts >= 2 or not stoppable:
                raise Detached(exp.run_id)
            renderer.stopping = True
            exp.stop("campaign")
            if not json_lines:
                console.print(
                    result_line(
                        "[warn]Stopping:[/] the campaign submits its best candidate and still writes the "
                        "brief. Press ctrl+c again to detach."
                    )
                )
            if from_stream:
                pump(exp.events(include_history=False))
            return
        with paused():
            try:
                if renderer.replaying:
                    speed = float((getattr(exp, "replay_control", None) or {}).get("speed") or 1.0)
                    key = choose(
                        f"Replay at {speed:g}\u00d7:",
                        [
                            ("i", "instant"),
                            ("f", "faster"),
                            ("s", "slower"),
                            ("p", "pause"),
                            ("e", "end the replay"),
                            ("k", "keep watching"),
                        ],
                    )
                    if key == "e":
                        raise Detached(exp.run_id)
                    if key == "p":
                        exp.stop()
                        paused_replay = True
                    elif key in ("i", "f", "s"):
                        new = 0.0 if key == "i" else (speed * 2 if key == "f" else max(speed / 2, 0.25))
                        exp.set_replay_speed(new)
                        console.print(
                            result_line(
                                f"replay speed {GLYPH['arrow']} {'instant' if new == 0 else f'{new:g}x'}"
                            )
                        )
                else:
                    key = choose(
                        "Stop this run?",
                        [
                            ("s", "stop the search (re-check and brief still run)"),
                            ("e", "stop everything (resume later)"),
                            ("d", "detach"),
                            ("k", "keep running"),
                        ],
                    )
                    if key == "d":
                        raise Detached(exp.run_id)
                    if key in ("s", "e"):
                        renderer.stopping, renderer.stop_all = True, key == "e"
                        exp.stop("campaign" if key == "s" else "all")
                        what = (
                            "the campaign submits its best candidate; the re-check and the brief still run"
                            if key == "s"
                            else "the run ends after the current calculation; resume it later"
                        )
                        console.print(result_line(f"[warn]Stopping:[/] {what}."))
            except KeyboardInterrupt:
                raise Detached(exp.run_id) from None
        if from_stream:
            pump(exp.events(include_history=False))

    pump(exp.events())
    last = time.monotonic()
    with live if live_ok else contextlib.nullcontext():
        while True:
            try:
                try:
                    gen, item = items.get(timeout=0.1)
                except queue.Empty:
                    if time.monotonic() - last >= IDLE_S:
                        maybe_decide()
                    continue
                if gen != generation[0]:
                    continue  # an older pump's leftovers
                if item is done:
                    if paused_replay and choose is not None:
                        paused_replay = False
                        with paused():
                            key = choose("Replay paused.", [("r", "resume"), ("e", "end the replay")])
                        if key == "r":
                            try:
                                exp.resume_replay()
                            except ExperimentError:
                                break  # it had already shown every event
                            pump(exp.events(include_history=False))
                            continue
                        raise Detached(exp.run_id)
                    break
                if isinstance(item, BaseException):
                    if isinstance(item, KeyboardInterrupt):
                        interrupt(from_stream=True)
                        continue
                    raise item
                last = time.monotonic()
                emit(item)
            except KeyboardInterrupt:
                interrupt(from_stream=False)
    return exp.wait()


def finish(
    console: Console, exp: Experiment, renderer: Renderer, state: RunState | None, *, json_lines: bool
) -> int:
    """Print the brief and the summary, and choose the exit code.

    Parameters
    ----------
    console : rich.console.Console
        Output console.
    exp : Experiment
        The run (or replay handle).
    renderer : Renderer
        Formatter with the run's display state.
    state : RunState or None
        Final state.
    json_lines : bool
        Machine-readable mode: print nothing more.

    Returns
    -------
    int
        0 when the run finished, else 1.
    """
    status = state.status if state else None
    if not json_lines:
        brief = exp.brief if exp.exists else None
        if brief:
            console.print(brief_panel(brief))
        path = str(exp.files.brief_md) if brief else None
        compute = exp.files.compute_brief_md if exp.exists else None
        console.print(
            renderer.summary(
                state, brief_path=path, compute_path=str(compute) if compute and compute.exists() else None
            )
        )
    return EXIT_OK if status == "finished" else EXIT_FAILED


def detached_hint(console: Console, run_id: str) -> int:
    """Tell the user how to pick a detached run up again.

    Parameters
    ----------
    console : rich.console.Console
        Output console.
    run_id : str
        Run left behind.

    Returns
    -------
    int
        Exit code 130.
    """
    console.print(
        result_line(
            f"Detached from [id]{run_id}[/]. Its progress is saved: [accent]hephaestus resume {run_id}[/]"
        )
    )
    return EXIT_DETACHED


def preflight(console: Console, config: RunConfig, deps: runner.Deps | None) -> bool:
    """Check the LLM backend before anything calls it: a signed-out Claude Code hangs (OR-11).

    Also warns when Valency needs a sign-in (UI-11). Skipped when a model is injected.

    Parameters
    ----------
    console : rich.console.Console
        Output console.
    config : RunConfig
        Run options.
    deps : runner.Deps or None
        Injected dependencies.

    Returns
    -------
    bool
        False when the run cannot start.
    """
    if deps is not None and deps.llm is not None:
        return True
    ok, error = backend_ready(config.llm_backend)
    if not ok:
        hint = "hephaestus login" if config.llm_backend != "api" else "set ANTHROPIC_API_KEY"
        console.print(
            result_line(f"[bad]Model access '{config.llm_backend}' is not ready:[/] {error or ''} · {hint}")
        )
        return False
    if deps is None or deps.handlers is None:
        try:
            from hephaestus.core import valency_auth

            v = valency_auth.status()
            if v.get("needs_signin") or not v.get("signed_in"):
                console.print(
                    result_line(
                        "[warn]Valency is not signed in[/], so literature search will be unavailable. "
                        "Sign in with [accent]hephaestus signin[/] (or /signin)."
                    )
                )
        except Exception:  # a status check must never block a run
            pass
    return True


def describe_blocker(reason: str) -> str:
    """Reword a run blocker for the terminal, as safe markup (``run_blockers`` words them for the web UI).

    Parameters
    ----------
    reason : str
        One sentence from ``run_blockers``.

    Returns
    -------
    str
        The sentence with the command-line fix, markup escaped.
    """
    from rich.markup import escape

    text = reason.replace("the server's Python", "this Python environment")
    text = text.replace(
        "add one under Connections, or set MP_API_KEY.",
        "run hephaestus login --mp-api-key (or /login mp-api-key), or set MP_API_KEY.",
    )
    return escape(text)


def runnable(console: Console, deps: runner.Deps | None) -> bool:
    """Refuse a run that cannot pass its first step: MACE missing, or no Materials Project key (OR-1).

    ``Experiment.start`` does not check this (only the HTTP API does), so the CLI checks before
    starting or resuming. Skipped when stub tools are injected, as the HTTP API does.

    Parameters
    ----------
    console : rich.console.Console
        Output console.
    deps : runner.Deps or None
        Injected dependencies.

    Returns
    -------
    bool
        True when the run can start.
    """
    if deps is not None and deps.handlers is not None:
        return True
    found = run_blockers(simulation_status())
    if not found:
        return True
    console.print(result_line("[bad]The run cannot start yet:[/]"))
    for reason in found:
        console.print(result_line(describe_blocker(reason)))
    return False


def run_goal(
    console: Console,
    goal: str,
    config: RunConfig,
    prompts: Prompts,
    *,
    deps: DepsSource = None,
    root: Path | None = None,
    verbose: bool = False,
    json_lines: bool = False,
    parse_only: bool = False,
) -> int:
    """Take one goal through the whole flow: parse, ask, confirm, run, stream, brief (SYS-1, UI-1, UI-2).

    Parameters
    ----------
    console : rich.console.Console
        Output console.
    goal : str
        Research goal in plain English.
    config : RunConfig
        Run options.
    prompts : Prompts
        How to ask questions and confirm.
    deps : runner.Deps, callable or None
        Injected dependencies or a factory for them.
    root : pathlib.Path, optional
        Runs folder.
    verbose : bool
        Show thinking and results in full.
    json_lines : bool
        Machine-readable output.
    parse_only : bool
        Stop after the spec is ready (``hephaestus parse``).

    Returns
    -------
    int
        Exit code.
    """
    d = make_deps(deps)
    if config.custom_tools == "ask" and not parse_only and (json_lines or not prompts.interactive):
        console.print(
            result_line(
                "[bad]Agent-built tools need someone to review them[/] (AT-3): with --yes, --json or no "
                "terminal nobody can answer. Use --custom-tools off, or run it interactively."
            )
        )
        return EXIT_FAILED
    if not preflight(console, config, d):
        return EXIT_FAILED
    exp = Experiment(goal, config, root=root, deps=d)
    answers: dict[str, str] = {}
    with spinner(console, "Parsing the goal"):
        parsed = exp.parse()
    for _ in range(MAX_QUESTION_ROUNDS):
        if not parsed.questions:
            break
        if not json_lines:
            if parsed.spec is not None:
                console.print(spec_panel(parsed.spec.model_dump(mode="json"), gap=parsed.gap))
            console.print(questions_panel(parsed.questions, []))
        if not prompts.interactive:
            if json_lines:
                plain(console, _json(parsed.to_dict()))
            return EXIT_NEEDS_INPUT
        for q in parsed.questions:
            answers[q] = prompts.ask(q)
        with spinner(console, "Updating the specification"):
            parsed = exp.parse(answers)
    if parse_only and json_lines:
        plain(console, _json(parsed.to_dict()))
        return EXIT_OK if parsed.ready else (EXIT_NEEDS_INPUT if parsed.questions else EXIT_FAILED)
    if parsed.errors or parsed.spec is None:
        console.print(questions_panel(parsed.questions, parsed.errors))
        return EXIT_NEEDS_INPUT if parsed.questions else EXIT_FAILED
    spec = parsed.spec.model_dump(mode="json")
    if not json_lines:
        console.print(spec_panel(spec, gap=parsed.gap))
        parser = parsed.parser
        console.print(Text(f"  parsed by {parser.get('backend')} · {parser.get('model')}", style="faint"))
    if parse_only:
        return EXIT_OK
    if not runnable(console, d):
        return EXIT_FAILED
    if prompts.interactive and prompts.ask_to_start and not prompts.confirm("Start the campaign?", True):
        console.print(result_line("[muted]Not started. Change options with /config, or type a new goal.[/]"))
        return EXIT_OK
    renderer = Renderer(verbose=verbose, live=True)
    renderer.spec = spec
    try:
        exp.start()
        state = stream(exp, console, renderer, json_lines=json_lines, prompts=prompts)
    except Detached as d_exc:
        return detached_hint(console, d_exc.run_id)
    except ExperimentError as exc:
        console.print(result_line(f"[bad]{exc}[/]"))
        return EXIT_FAILED
    return finish(console, exp, renderer, state, json_lines=json_lines)


def resume_run(
    console: Console,
    run_id: str,
    *,
    deps: DepsSource = None,
    root: Path | None = None,
    verbose: bool = False,
    json_lines: bool = False,
    prompts: Prompts | None = None,
) -> int:
    """Resume an interrupted run, showing its earlier events first (OR-6, UI-16).

    A run stopped while it waited for a tool decision shows the card again and asks (AT-24).

    Parameters
    ----------
    console : rich.console.Console
        Output console.
    run_id : str
        Run to resume.
    deps : runner.Deps, callable or None
        Injected dependencies.
    root : pathlib.Path, optional
        Runs folder.
    verbose : bool
        Full output.
    json_lines : bool
        Machine-readable output.
    prompts : Prompts, optional
        How to ask: tool decisions and the Ctrl+C menu.

    Returns
    -------
    int
        Exit code.
    """
    d = make_deps(deps)
    try:
        exp = Experiment.load(run_id, root, d)
        if not preflight(console, exp.config, d) or not runnable(console, d):
            return EXIT_FAILED
        exp.start(resume=True)
    except ExperimentError as exc:
        console.print(result_line(f"[bad]{exc}[/]"))
        return EXIT_FAILED
    renderer = Renderer(verbose=verbose, live=True)
    try:
        state = stream(exp, console, renderer, json_lines=json_lines, prompts=prompts)
    except Detached as d_exc:
        return detached_hint(console, d_exc.run_id)
    return finish(console, exp, renderer, state, json_lines=json_lines)


def locate_run(ref: str, root: Path | None) -> tuple[str, Path | None]:
    """Resolve a run reference: a run ID, or a path to a (copied) run folder (DL-8).

    Parameters
    ----------
    ref : str
        Run ID or folder path.
    root : pathlib.Path, optional
        Runs folder for a plain ID.

    Returns
    -------
    run_id : str
        Folder name.
    root : pathlib.Path or None
        Folder holding it.
    """
    path = Path(ref)
    if path.is_dir() and (path / "state.json").exists():
        return path.name, path.parent
    return ref, root


def replay_run(
    console: Console,
    ref: str,
    speed: float = 4.0,
    *,
    root: Path | None = None,
    verbose: bool = False,
    json_lines: bool = False,
    prompts: Prompts | None = None,
) -> int:
    """Replay a stored run's events at demo pace, with no network (UI-9, DL-8).

    Ctrl+C offers instant, faster, slower, pause and end when ``prompts`` can ask. A replay
    never asks for a tool decision: the recorded ones show as they were made.

    Parameters
    ----------
    console : rich.console.Console
        Output console.
    ref : str
        Run ID, or a path to a run folder.
    speed : float
        Playback speed; 0 plays with no pauses.
    root : pathlib.Path, optional
        Runs folder.
    verbose : bool
        Full output.
    json_lines : bool
        Machine-readable output.
    prompts : Prompts, optional
        How to ask (the Ctrl+C menu).

    Returns
    -------
    int
        Exit code.
    """
    run_id, base = locate_run(ref, root)
    try:
        handle = Experiment.load(run_id, base).replay(speed)
    except ExperimentError as exc:
        console.print(result_line(f"[bad]{exc}[/]"))
        return EXIT_FAILED
    if not json_lines:
        console.print(
            Text.assemble((" REPLAY ", "tab.accent"), " ", (run_id, "id"), (f" at {speed:g}\u00d7", "muted"))
        )
    renderer = Renderer(verbose=verbose, replaying=True)
    try:
        state = stream(handle, console, renderer, json_lines=json_lines, stoppable=False, prompts=prompts)
    except Detached:
        handle.stop()
        console.print(result_line("[muted]Replay stopped.[/]"))
        return EXIT_DETACHED
    renderer.run_id = run_id
    return finish(console, handle, renderer, state, json_lines=json_lines)


def history_renderer(run_id: str, root: Path | None = None) -> tuple[Experiment, Renderer]:
    """Load a run and feed its stored events through a renderer silently (for ``show``).

    Parameters
    ----------
    run_id : str
        Run ID or folder path.
    root : pathlib.Path, optional
        Runs folder.

    Returns
    -------
    exp : Experiment
        The stored run.
    renderer : Renderer
        Display state rebuilt from the events.

    Raises
    ------
    ExperimentError
        If the run does not exist.
    """
    rid, base = locate_run(run_id, root)
    exp = Experiment.load(rid, base)
    renderer = Renderer()
    for event in exp.events(follow=False):
        renderer.feed(event)
    if renderer.spec is None and exp.state.spec is not None:
        renderer.spec = exp.state.spec.model_dump(mode="json")
    return exp, renderer


def plain(console: Console, text: str) -> None:
    """Print one machine-readable line as is: no markup, highlighting or wrapping.

    Parameters
    ----------
    console : rich.console.Console
        Output console.
    text : str
        The line.
    """
    console.print(text, markup=False, highlight=False, emoji=False, soft_wrap=True)


def _json(obj: Any) -> str:
    """Serialise to one JSON line.

    Parameters
    ----------
    obj : Any
        JSON-serialisable value.

    Returns
    -------
    str
        JSON text.
    """
    import json

    return json.dumps(obj, ensure_ascii=False, default=str)
