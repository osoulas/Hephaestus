"""Command-line subcommands: ``run``, ``parse``, ``replay``, ``resume``, ``login``, ``signin``, ``backends``, ``serve``.

``login``, ``backends`` and ``serve`` work; the others report that they are not implemented yet.
"""

import argparse
import getpass
import subprocess
import sys
from collections.abc import Sequence

SUBCOMMANDS = {
    "run": "Run a campaign for a goal.",
    "parse": "Parse a goal into a search specification.",
    "replay": "Replay a stored run.",
    "resume": "Resume an interrupted run.",
    "login": "Sign in to Claude (subscription via Claude Code, or an Anthropic API key).",
    "signin": "Sign in to Valency.",
    "backends": "Show LLM backend readiness.",
    "serve": "Serve the web UI and HTTP API.",
}


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with one subparser per entry in ``SUBCOMMANDS``.
    """
    parser = argparse.ArgumentParser(
        prog="hephaestus", description="Self-designing experiment agent for materials discovery."
    )
    sub = parser.add_subparsers(dest="command")
    subs = {name: sub.add_parser(name, help=help_text) for name, help_text in SUBCOMMANDS.items()}
    subs["login"].add_argument(
        "--api-key",
        action="store_true",
        help="Save an Anthropic API key instead of signing in to Claude Code.",
    )
    subs["login"].add_argument(
        "--forget-api-key", action="store_true", help="Delete the saved Anthropic API key."
    )
    subs["serve"].add_argument("--host", default="127.0.0.1", help="Interface to bind (default 127.0.0.1).")
    subs["serve"].add_argument("--port", type=int, default=8000, help="Port (default 8000).")
    return parser


def login(args: argparse.Namespace) -> int:
    """Sign in to Claude: ``claude auth login --claudeai`` in this terminal, or save an API key (OR-11).

    Parameters
    ----------
    args : argparse.Namespace
        ``api_key`` and ``forget_api_key`` flags.

    Returns
    -------
    int
        0 when signed in or saved, 1 otherwise.
    """
    from hephaestus.llm import auth

    if args.forget_api_key:
        auth.forget_api_key()
        print("Saved API key removed.")
        return 0
    if args.api_key:
        key = getpass.getpass("Anthropic API key (input hidden): ")
        try:
            st = auth.save_api_key(key, check=auth.check_api_key)
        except ValueError as exc:
            print(f"Not saved: {exc}", file=sys.stderr)
            return 1
        print(f"Saved {st['hint']} to {auth.api_key_path()} (owner-only).")
        return 0
    cli = auth.claude_cli_path()
    if cli is None:
        print("Claude Code CLI not found. Install the claude-agent-sdk package.", file=sys.stderr)
        return 1
    code = subprocess.call([cli, "auth", "login", "--claudeai"])
    st = auth.claude_code_status()
    if st["signed_in"]:
        print("Signed in to Claude Code.")
        return 0
    print(f"Not signed in: {st['error'] or f'exit code {code}'}", file=sys.stderr)
    return 1


def backends() -> int:
    """Print whether each LLM backend is ready (OR-11).

    Returns
    -------
    int
        0 when the ``auto`` choice is ready, 1 otherwise.
    """
    from hephaestus.interfaces.python import backend_status
    from hephaestus.llm.auth import load_api_key

    load_api_key()
    st = backend_status()
    for name in ("api", "claude-code"):
        s = st[name]
        note = "ready" if s["ready"] else f"not ready  ({s['error']})"
        print(f"{name:12} {note}")
    auto = st["auto"]
    print(f"{'auto':12} uses {auto['resolves_to']}: {'ready' if auto['ready'] else 'not ready'}")
    return 0 if auto["ready"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code: 0 on success, 2 when the command is not implemented yet.
    """
    args = build_parser().parse_args(argv)
    if args.command == "login":
        return login(args)
    if args.command == "backends":
        return backends()
    if args.command == "serve":
        from hephaestus.interfaces.http import serve

        print(f"Hephaestus at http://{args.host}:{args.port}  (API docs at /docs)")
        serve(args.host, args.port)
        return 0
    command = args.command or "interactive session"
    print(f"hephaestus: {command} is not implemented yet.", file=sys.stderr)
    return 2
