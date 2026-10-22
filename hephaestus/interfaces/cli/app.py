"""Command-line entry point: the interactive session and the subcommands.

Subcommands: ``run``, ``parse``, ``resume``, ``replay``, ``runs``, ``show``, ``rename``, ``brief``,
``literature``, ``tools``, ``backends``, ``login``, ``signin`` and ``serve``.

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
from rich.markup import escape

from hephaestus.core.config import BAND_GAP_MODELS, MODELS, RunConfig
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
    "rename": "Set a run's title.",
    "brief": "Print a run's experimental brief, or its computational brief.",
    "literature": "Show the papers, evidence and failed approaches a run found.",
    "tools": "List, review, show or remove agent-built tools in the library.",
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
    "band_gap_model": "band_gap_model",
    "max_rounds": "max_iterations",
    "max_tool_calls": "max_tool_calls",
    "offline": "offline",
    "no_final": "run_final",
    "custom_tools": "custom_tools",
    "no_hull": "run_hull",
    "no_phonons": "run_phonons",
    "e_hull_max": "e_hull_max_eV_per_atom",
    "tool_library": "tool_library_dir",
}
FIDELITIES = [str(m["fidelity"]) for m in BAND_GAP_MODELS.values()]  # --fidelity hse means megnet-hse


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
    g.add_argument(
        "--profile",
        choices=list(actions.PROFILES),
        default=d,
        help="preset budgets: quick, standard or thorough (other options override it)",
    )
    g.add_argument("--band-gap-model", choices=list(BAND_GAP_MODELS), default=d, help="band gap model")
    g.add_argument(
        "--fidelity", choices=FIDELITIES, default=d, help="MEGNet fidelity (same as --band-gap-model)"
    )
    g.add_argument(
        "--custom-tools",
        choices=["ask", "off"],
        default=d,
        help="let the agent propose tools for you to review when none computes a target "
        "(default: ask in the interactive session, off for run)",
    )
    g.add_argument("--no-hull", action="store_true", default=flag, help="skip the thermodynamic (hull) check")
    g.add_argument("--no-phonons", action="store_true", default=flag, help="skip the dynamic (phonon) check")
    g.add_argument(
        "--e-hull-max",
        type=float,
        default=d,
        metavar="EV",
        help="hull threshold in eV/atom (replaces the goal's)",
    )
    g.add_argument(
        "--no-elastic", action="store_true", default=flag, help="turn off the elastic-tensor properties"
    )
    g.add_argument("--tool-library", default=d, metavar="DIR", help="agent-built tool library folder")
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
    for name in ("runs", "show", "brief", "rename"):
        p = sub.add_parser(name, help=SUBCOMMANDS[name], description=SUBCOMMANDS[name])
        if name != "runs":
            p.add_argument("run_id", help="run ID, or a path to a run folder")
        if name == "brief":
            p.add_argument(
                "--compute",
                action="store_true",
                help="the computational brief: methods, settings, every number",
            )
            p.add_argument("--pdf", action="store_true", help="print the PDF's path instead")
        if name == "rename":
            p.add_argument("title", nargs="+", help="new title")
        p.add_argument("--root", type=Path, default=argparse.SUPPRESS, metavar="DIR")
    p = sub.add_parser("tools", help=SUBCOMMANDS["tools"], description=SUBCOMMANDS["tools"])
    p.add_argument("action", nargs="?", default="list", choices=actions.TOOL_ACTIONS, help="default: list")
    p.add_argument("name", nargs="?", default=None, help="tool name")
    p.add_argument("--dir", default=None, metavar="DIR", help="library folder (default ./hephaestus_tools)")
    p = sub.add_parser("literature", help=SUBCOMMANDS["literature"], description=SUBCOMMANDS["literature"])
    p.add_argument("run_id", help="run ID, or a path to a run folder")
    p.add_argument("--filter", default=None, metavar="TEXT", help="only papers mentioning TEXT")
    p.add_argument("--limit", type=int, default=15, metavar="N", help="most papers to list (default 15)")
    p.add_argument("--all", action="store_true", help="every paper, with whole abstracts")
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


def config_from(args: argparse.Namespace, *, custom_tools: str | None = None) -> RunConfig:
    """Build the run options from the parsed arguments; unset options keep their defaults.

    A ``--profile`` sets its budgets first; any other option given overrides them.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    custom_tools : str, optional
        Default for ``custom_tools`` when ``--custom-tools`` is not given (``"ask"`` for the
        interactive session, AT-3).

    Returns
    -------
    RunConfig
        Options for the run.
    """
    updates: dict[str, Any] = dict(actions.PROFILES.get(getattr(args, "profile", None) or "", {}))
    if custom_tools is not None:
        updates["custom_tools"] = custom_tools
    fidelity = getattr(args, "fidelity", None)
    if fidelity is not None:
        updates["band_gap_model"] = f"megnet-{fidelity}"
    for attr, fld in OPTION_FIELDS.items():
        value = getattr(args, attr, None)
        if attr in ("no_final", "no_hull", "no_phonons"):
            if value:
                updates[fld] = False
        elif attr in ("novel", "offline"):
            if value:
                updates[fld] = True
        elif value is not None:
            updates[fld] = value
    if getattr(args, "no_elastic", False):
        updates["disabled_capabilities"] = ["elastic"]
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
        if question:  # a choice prints its own options and reads an empty-prompt line
            console.print(f"[warn]?[/] [bold]{escape(question)}[/]")
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


def _scripted(read_line: Any) -> Prompts:
    """Prompts on an injected line reader (tests, embedding), answering like ``_ask``.

    Parameters
    ----------
    read_line : callable
        ``read_line(prompt) -> str``.

    Returns
    -------
    Prompts
        Question and confirmation readers.
    """

    def confirm(question: str, default: bool = True) -> bool:
        """Read yes or no; an empty line takes the default.

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
        reply = str(read_line(f"{question} ")).strip().lower()
        return default if not reply else reply in ("y", "yes")

    return Prompts(ask=lambda q: str(read_line("  ")), confirm=confirm)


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
            out,
            config_from(args, custom_tools="ask"),
            read_line=read_line,
            deps=deps,
            root=folder,
            verbose=verbose,
        ).run()
    if cmd in ("run", "parse"):
        json_lines = bool(getattr(args, "json", False))
        prompts = _ask(out) if read_line is None else _scripted(read_line)
        can_ask = read_line is not None or sys.stdin.isatty()
        # Questions are asked when someone can answer; a subcommand never asks to confirm the start.
        prompts.interactive = can_ask and not getattr(args, "yes", False) and not json_lines
        prompts.ask_to_start = False
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
    if cmd in ("resume", "replay"):
        prompts = _ask(out) if read_line is None else _scripted(read_line)
        prompts.interactive = (read_line is not None or sys.stdin.isatty()) and not args.json
        if cmd == "resume":
            return resume_run(
                out,
                args.run_id,
                deps=deps,
                root=folder,
                verbose=verbose,
                json_lines=args.json,
                prompts=prompts,
            )
        return replay_run(
            out, args.run_id, args.speed, root=folder, verbose=verbose, json_lines=args.json, prompts=prompts
        )
    if cmd == "runs":
        return actions.runs_view(out, folder)
    if cmd == "show":
        return actions.show_view(out, args.run_id, folder)
    if cmd == "brief":
        return actions.brief_view(out, args.run_id, folder, pdf=args.pdf, compute=args.compute)
    if cmd == "rename":
        return actions.rename_cmd(out, args.run_id, " ".join(args.title), folder)
    if cmd == "tools":
        can_ask = read_line is not None or sys.stdin.isatty()
        p = _ask(out) if read_line is None else _scripted(read_line)
        return actions.tools_cmd(
            out,
            args.action,
            args.name,
            args.dir,
            ask=p.ask if can_ask else None,
            confirm=p.confirm if can_ask else None,
        )
    if cmd == "literature":
        return actions.literature_cmd(
            out,
            args.run_id,
            folder,
            text_filter=args.filter,
            limit=None if args.all else args.limit,
            full=args.all,
        )
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
