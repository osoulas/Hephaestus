"""Run a goal, resume or replay a run, and stream its events to the terminal (OR-7, OR-8, UI-9, UI-16).

``stream`` prints each event above a live status line. The first Ctrl+C asks the campaign to stop:
it submits its best candidate and still writes the brief, so streaming continues to the end. A
second Ctrl+C detaches; the run folder keeps its snapshot, so ``hephaestus resume`` can carry on.
"""

from __future__ import annotations

import contextlib
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
from hephaestus.interfaces.cli.theme import ember_frame
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
    """

    ask: Callable[[str], str]
    confirm: Callable[[str, bool], bool]
    interactive: bool = True


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


def stream(
    exp: Experiment,
    console: Console,
    renderer: Renderer,
    *,
    json_lines: bool = False,
    stoppable: bool = True,
) -> RunState | None:
    """Print a run's events (history first, then live) until the run ends.

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

    Returns
    -------
    RunState or None
        The final state.

    Raises
    ------
    Detached
        On a second Ctrl+C (the first, for replays).
    """

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
    events = exp.events()
    interrupts = 0
    live = Live(get_renderable=renderer.status, console=console, refresh_per_second=8, transient=True)
    with live if live_ok else contextlib.nullcontext():
        while True:
            try:
                for event in events:
                    emit(event)
                break
            except KeyboardInterrupt:
                interrupts += 1
                if interrupts >= 2 or not stoppable:
                    raise Detached(exp.run_id) from None
                renderer.stopping = True
                exp.stop()
                if not json_lines:
                    console.print(
                        result_line(
                            "[warn]Stopping:[/] the campaign submits its best candidate and still writes the "
                            "brief. Press ctrl+c again to detach."
                        )
                    )
                events = exp.events(include_history=False)
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
        console.print(renderer.summary(state, brief_path=path))
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
                console.print(spec_panel(parsed.spec.model_dump(mode="json")))
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
        console.print(spec_panel(spec))
        parser = parsed.parser
        console.print(Text(f"  parsed by {parser.get('backend')} · {parser.get('model')}", style="faint"))
    if parse_only:
        return EXIT_OK
    if not runnable(console, d):
        return EXIT_FAILED
    if prompts.interactive and not prompts.confirm("Start the campaign?", True):
        console.print(result_line("[muted]Not started. Change options with /config, or type a new goal.[/]"))
        return EXIT_OK
    renderer = Renderer(verbose=verbose, live=True)
    renderer.spec = spec
    try:
        exp.start()
        state = stream(exp, console, renderer, json_lines=json_lines)
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
) -> int:
    """Resume an interrupted run, showing its earlier events first (OR-6, UI-16).

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
        state = stream(exp, console, renderer, json_lines=json_lines)
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
) -> int:
    """Replay a stored run's events at demo pace, with no network (UI-9, DL-8).

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
        state = stream(handle, console, renderer, json_lines=json_lines, stoppable=False)
    except Detached:
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
