"""Command-line entry point: the interactive session and the subcommands.

Subcommands: ``run``, ``parse``, ``resume``, ``replay``, ``runs``, ``show``, ``brief``, ``backends``,
``login``, ``signin`` and ``serve``.

The CLI is a renderer on the Python API (``interfaces/python.py``): it starts, stops, resumes and
replays runs and prints their events; every verdict, gate and stop comes from the harness.

Exit codes: 0 done, 1 failed or refused, 2 usage error, 3 the goal needs answers, 130 detached.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console

from hephaestus.core.config import MODELS, RunConfig
from hephaestus.interfaces.cli import actions
from hephaestus.interfaces.cli.render import result_line
from hephaestus.interfaces.cli.stream import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    DepsSource,
    Prompts,
    replay_run,
    resume_run,
    run_goal,
)
from hephaestus.interfaces.cli.theme import make_console

SUBCOMMANDS = {
    "run": "Run a campaign for a goal and stream it.",
    "parse": "Parse a goal into a search specification.",
    "resume": "Resume an interrupted run.",
    "replay": "Replay a stored run, offline.",
    "runs": "List stored runs.",
    "show": "Show a run's spec, rounds, candidates and summary.",
    "brief": "Print a run's experimental brief.",
    "backends": "Show LLM backend readiness.",
    "login": "Sign in to Claude (subscription via Claude Code, or an Anthropic API key); save a Materials Project key.",
    "signin": "Sign in to Valency.",
    "serve": "Serve the web UI and HTTP API.",
}

# RunConfig fields set by the run options, with the attribute argparse stores them under.
OPTION_FIELDS = {
    "backend": "llm_backend",
    "model": "llm_model",
    "novel": "require_novel",
    "fidelity": "band_gap_fidelity",
    "max_rounds": "max_iterations",
    "max_tool_calls": "max_tool_calls",
    "offline": "offline",
    "no_final": "run_final",
}


def add_run_options(parser: argparse.ArgumentParser, suppress: bool) -> None:
    """Add the options that shape a run.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend.
    suppress : bool
        Leave unset options out of the namespace, so a subcommand never overwrites an option
        given before it on the command line.
    """
    d: Any = argparse.SUPPRESS if suppress else None
    flag: Any = argparse.SUPPRESS if suppress else False
    g = parser.add_argument_group("run options")
    g.add_argument("--backend", choices=["auto", "api", "claude-code"], default=d, help="model access")
    g.add_argument("--model", choices=MODELS, default=d, help="Claude model")
    g.add_argument("--novel", action="store_true", default=flag, help="require novel materials")
    g.add_argument("--fidelity", choices=["hse", "pbe"], default=d, help="MEGNet band-gap fidelity")
    g.add_argument("--max-rounds", type=int, default=d, metavar="N", help="maximum review rounds")
    g.add_argument("--max-tool-calls", type=int, default=d, metavar="N", help="tool-call budget")
    g.add_argument(
        "--offline", action="store_true", default=flag, help="serve external calls from the cache only"
    )
    g.add_argument("--no-final", action="store_true", default=flag, help="skip the higher-fidelity re-check")
    g.add_argument(
        "--verbose", "-v", action="store_true", default=flag, help="show thinking and results in full"
    )
    g.add_argument("--root", type=Path, default=d, metavar="DIR", help="runs folder (default ./runs)")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Top-level parser (run options apply to the interactive session) with one subparser per
        entry in ``SUBCOMMANDS``.
    """
    parser = argparse.ArgumentParser(
        prog="hephaestus",
        description="Self-designing experiment agent for materials discovery. "
        "With no command, starts an interactive session.",
    )
    add_run_options(parser, suppress=False)
    sub = parser.add_subparsers(dest="command", metavar="command")
    p = sub.add_parser("run", help=SUBCOMMANDS["run"], description=SUBCOMMANDS["run"])
    p.add_argument("goal", help="research goal in plain English")
    p.add_argument(
        "--yes", "-y", action="store_true", help="don't ask: start at once; open questions exit with code 3"
    )
    p.add_argument("--json", action="store_true", help="print events as JSON lines")
    add_run_options(p, suppress=True)
    p = sub.add_parser("parse", help=SUBCOMMANDS["parse"], description=SUBCOMMANDS["parse"])
    p.add_argument("goal", help="research goal in plain English")
    p.add_argument("--json", action="store_true", help="print the parser's answer as JSON")
    add_run_options(p, suppress=True)
    for name in ("resume", "replay"):
        p = sub.add_parser(name, help=SUBCOMMANDS[name], description=SUBCOMMANDS[name])
        p.add_argument("run_id", help="run ID" + (", or a path to a run folder" if name == "replay" else ""))
        if name == "replay":
            p.add_argument("--speed", type=float, default=4.0, help="playback speed; 0 plays with no pauses")
        p.add_argument("--json", action="store_true", help="print events as JSON lines")
        p.add_argument("--verbose", "-v", action="store_true", default=argparse.SUPPRESS)
        p.add_argument("--root", type=Path, default=argparse.SUPPRESS, metavar="DIR")
    for name in ("runs", "show", "brief"):
        p = sub.add_parser(name, help=SUBCOMMANDS[name], description=SUBCOMMANDS[name])
        if name != "runs":
            p.add_argument("run_id", help="run ID, or a path to a run folder")
        if name == "brief":
            p.add_argument("--pdf", action="store_true", help="print the PDF's path instead")
        p.add_argument("--root", type=Path, default=argparse.SUPPRESS, metavar="DIR")
    for name in ("backends", "signin"):
        sub.add_parser(name, help=SUBCOMMANDS[name], description=SUBCOMMANDS[name])
    p = sub.add_parser("login", help=SUBCOMMANDS["login"], description=SUBCOMMANDS["login"])
    keys = p.add_mutually_exclusive_group()
    keys.add_argument(
        "--api-key",
        action="store_true",
        help="save an Anthropic API key instead of signing in to Claude Code",
    )
    keys.add_argument("--forget-api-key", action="store_true", help="delete the saved Anthropic API key")
    keys.add_argument(
        "--mp-api-key", action="store_true", help="save a Materials Project API key (needed by every run)"
    )
    keys.add_argument(
        "--forget-mp-api-key", action="store_true", help="delete the saved Materials Project key"
    )
    p = sub.add_parser("serve", help=SUBCOMMANDS["serve"], description=SUBCOMMANDS["serve"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return parser


def config_from(args: argparse.Namespace) -> RunConfig:
    """Build the run options from the parsed arguments; unset options keep their defaults.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    RunConfig
        Options for the run.
    """
    updates: dict[str, Any] = {}
    for attr, fld in OPTION_FIELDS.items():
        value = getattr(args, attr, None)
        if attr == "no_final":
            if value:
                updates[fld] = False
        elif attr in ("novel", "offline"):
            if value:
                updates[fld] = True
        elif value is not None:
            updates[fld] = value
    return RunConfig(**updates)


def _ask(console: Console) -> Prompts:
    """Plain prompts for the ``run`` command (no input box).

    Parameters
    ----------
    console : rich.console.Console
        Output console.

    Returns
    -------
    Prompts
        Question and confirmation readers.
    """

    def ask(question: str) -> str:
        """Ask one question on the terminal.

        Parameters
        ----------
        question : str
            The question.

        Returns
        -------
        str
            The answer.
        """
        console.print(f"[warn]?[/] [bold]{question}[/]")
        return input("  ").strip()

    def confirm(question: str, default: bool = True) -> bool:
        """Ask yes or no.

        Parameters
        ----------
        question : str
            The question.
        default : bool
            Answer for an empty reply.

        Returns
        -------
        bool
            The answer.
        """
        reply = input(f"{question} {'[Y/n]' if default else '[y/N]'} ").strip().lower()
        return default if not reply else reply in ("y", "yes")

    return Prompts(ask=ask, confirm=confirm)


def main(
    argv: Sequence[str] | None = None,
    *,
    console: Console | None = None,
    deps: DepsSource = None,
    root: Path | None = None,
    read_line: Any = None,
) -> int:
    """Run the command-line interface.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments without the program name; defaults to ``sys.argv[1:]``.
    console : rich.console.Console, optional
        Output console (tests pass a recording one).
    deps : runner.Deps, callable or None
        Injected dependencies, or a factory for them (tests, offline demos).
    root : pathlib.Path, optional
        Runs folder; ``--root`` overrides it.
    read_line : callable, optional
        Line reader for the interactive session and its questions (tests).

    Returns
    -------
    int
        Process exit code.
    """
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:  # argparse already printed the message
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
    out = console or make_console()
    actions.load_saved_keys()  # keys saved by `login` or the web UI, unless the environment sets them
    folder = getattr(args, "root", None) or root
    verbose = bool(getattr(args, "verbose", False))
    try:
        return _dispatch(args, out, deps, folder, verbose, read_line)
    except KeyboardInterrupt:
        out.print(result_line("[muted]Interrupted.[/]"))
        return 130
    except Exception as exc:  # a CLI never shows a traceback for a known failure
        if isinstance(exc, ValueError | RuntimeError | OSError):
            out.print(result_line(f"[bad]{type(exc).__name__}:[/] {exc}"))
            return EXIT_FAILED
        raise


def _dispatch(
    args: argparse.Namespace,
    out: Console,
    deps: DepsSource,
    folder: Path | None,
    verbose: bool,
    read_line: Any,
) -> int:
    """Route to the subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    out : rich.console.Console
        Output console.
    deps : runner.Deps, callable or None
        Injected dependencies.
    folder : pathlib.Path or None
        Runs folder.
    verbose : bool
        Full output.
    read_line : callable or None
        Line reader.

    Returns
    -------
    int
        Exit code.
    """
    cmd = args.command
    if cmd is None:
        from hephaestus.interfaces.cli.repl import Session

        return Session(
            out, config_from(args), read_line=read_line, deps=deps, root=folder, verbose=verbose
        ).run()
    if cmd in ("run", "parse"):
        json_lines = bool(getattr(args, "json", False))
        prompts = (
            _ask(out) if read_line is None else Prompts(ask=lambda q: read_line("  "), confirm=lambda q, d: d)
        )
        can_ask = read_line is not None or sys.stdin.isatty()
        # Questions are asked when someone can answer; a subcommand never asks to confirm the start.
        prompts.interactive = can_ask and not getattr(args, "yes", False) and not json_lines
        prompts.confirm = lambda question, default: True
        return run_goal(
            out,
            args.goal,
            config_from(args),
            prompts,
            deps=deps,
            root=folder,
            verbose=verbose,
            json_lines=json_lines,
            parse_only=cmd == "parse",
        )
    if cmd == "resume":
        return resume_run(out, args.run_id, deps=deps, root=folder, verbose=verbose, json_lines=args.json)
    if cmd == "replay":
        return replay_run(out, args.run_id, args.speed, root=folder, verbose=verbose, json_lines=args.json)
    if cmd == "runs":
        return actions.runs_view(out, folder)
    if cmd == "show":
        return actions.show_view(out, args.run_id, folder)
    if cmd == "brief":
        return actions.brief_view(out, args.run_id, folder, pdf=args.pdf)
    if cmd == "backends":
        return actions.backends_view(out)
    if cmd == "login":
        mode = next(
            (m for m in actions.LOGIN_MODES[1:] if getattr(args, m.replace("-", "_"), False)), "claude"
        )
        return actions.login(out, mode)
    if cmd == "signin":
        return actions.signin(out)
    if cmd == "serve":
        from hephaestus.interfaces.http import serve

        out.print(result_line(f"Serving on [accent]http://{args.host}:{args.port}[/] · API docs at /docs"))
        serve(args.host, args.port)
        return EXIT_OK
    return EXIT_USAGE
