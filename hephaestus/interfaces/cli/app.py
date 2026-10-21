"""Command-line subcommands: ``run``, ``parse``, ``replay``, ``resume``, ``login``, ``signin``, ``backends``, ``serve``.

Placeholder until round 2; every subcommand reports that it is not implemented yet.
"""

import argparse
import sys
from collections.abc import Sequence

SUBCOMMANDS = {
    "run": "Run a campaign for a goal.",
    "parse": "Parse a goal into a search specification.",
    "replay": "Replay a stored run.",
    "resume": "Resume an interrupted run.",
    "login": "Sign in to Claude Code.",
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
    for name, help_text in SUBCOMMANDS.items():
        sub.add_parser(name, help=help_text)
    return parser


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
    command = args.command or "interactive session"
    print(f"hephaestus: {command} is not implemented yet.", file=sys.stderr)
    return 2
