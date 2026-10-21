"""Commands shared by the subcommands and the REPL's slash commands: runs, show, brief, backends, sign-ins.

Each prints to a console and returns an exit code. They only read stored runs and call the Python
API or the sign-in helpers; nothing here computes a result.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from hephaestus.interfaces.cli.render import brief_panel, result_line, spec_panel
from hephaestus.interfaces.cli.stream import EXIT_FAILED, EXIT_OK, history_renderer
from hephaestus.interfaces.cli.theme import DASH, GLYPH, mark
from hephaestus.interfaces.python import ExperimentError, backend_status, device_status, list_runs, models


def runs_view(console: Console, root: Path | None = None, limit: int = 20) -> int:
    """List stored runs, newest first (UI-9, UI-16).

    Parameters
    ----------
    console : rich.console.Console
        Output console.
    root : pathlib.Path, optional
        Runs folder.
    limit : int
        Most rows to show.

    Returns
    -------
    int
        Exit code.
    """
    runs = list_runs(root)
    if not runs:
        console.print(result_line("[muted]No runs yet. Type a goal to start one.[/]"))
        return EXIT_OK
    t = Table(box=None, header_style="muted", pad_edge=False, padding=(0, 2))
    for col in ("run", "status", "best", "stop", "created", "goal"):
        t.add_column(col, no_wrap=col != "goal", overflow="ellipsis")
    colours = {"finished": "ok", "failed": "bad", "running": "accent", "needs_input": "warn"}
    for r in runs[:limit]:
        status = "running" if r.get("running") else str(r.get("status"))
        resumable = status not in ("finished", "running", "created")
        label = f"[{colours.get(status, 'muted')}]{status}[/]" + (
            " [faint](resumable)[/]" if resumable else ""
        )
        goal = str(r.get("raw_goal") or "")
        t.add_row(
            f"[id]{r['run_id']}[/]",
            label,
            str(r.get("best") or DASH),
            str(r.get("stop_reason") or DASH),
            str(r.get("created_at", ""))[:16].replace("T", " "),
            goal if len(goal) <= 60 else goal[:57] + "…",
        )
    console.print(t)
    return EXIT_OK


def show_view(console: Console, run_id: str, root: Path | None = None) -> int:
    """Show a stored run's spec, rounds, candidates and summary, rebuilt from its events.

    Parameters
    ----------
    console : rich.console.Console
        Output console.
    run_id : str
        Run ID or folder path.
    root : pathlib.Path, optional
        Runs folder.

    Returns
    -------
    int
        Exit code.
    """
    try:
        exp, r = history_renderer(run_id, root)
    except ExperimentError as exc:
        console.print(result_line(f"[bad]{exc}[/]"))
        return EXIT_FAILED
    if r.spec:
        console.print(spec_panel(r.spec))
    if r.rounds:
        console.print(Text("Rounds", style="accent"))
        console.print(r.rounds_table())
    if r.candidates:
        console.print(Text("Candidates", style="accent"))
        console.print(r.candidates_table())
    console.print(r.summary(exp.state, brief_path=str(exp.files.brief_md) if exp.brief else None))
    return EXIT_OK


def brief_view(console: Console, run_id: str, root: Path | None = None, pdf: bool = False) -> int:
    """Print a run's brief, or the path of its PDF (UI-8).

    Parameters
    ----------
    console : rich.console.Console
        Output console.
    run_id : str
        Run ID.
    root : pathlib.Path, optional
        Runs folder.
    pdf : bool
        Print the PDF path instead.

    Returns
    -------
    int
        Exit code.
    """
    from hephaestus.interfaces.cli.stream import locate_run
    from hephaestus.interfaces.python import Experiment

    rid, base = locate_run(run_id, root)
    try:
        exp = Experiment.load(rid, base)
    except ExperimentError as exc:
        console.print(result_line(f"[bad]{exc}[/]"))
        return EXIT_FAILED
    if pdf:
        path = exp.brief_pdf
        if path is None:
            console.print(result_line("[warn]No PDF for this run.[/]"))
            return EXIT_FAILED
        console.print(str(path))
        return EXIT_OK
    brief = exp.brief
    if brief is None:
        console.print(result_line(f"[warn]Run {rid} has no brief yet.[/]"))
        return EXIT_FAILED
    console.print(brief_panel(brief))
    return EXIT_OK


def backends_view(console: Console) -> int:
    """Readiness of each model-access backend, the models and the device (OR-11, UI-12).

    Parameters
    ----------
    console : rich.console.Console
        Output console.

    Returns
    -------
    int
        Exit code: 0 when the ``auto`` choice is ready.
    """
    st = backend_status()
    t = Table(box=None, header_style="muted", pad_edge=False, padding=(0, 2))
    for col in ("model access", "ready", "detail"):
        t.add_column(col)
    t.add_row("api", mark(st["api"]["ready"]), str(st["api"].get("error") or "ANTHROPIC_API_KEY found"))
    cc = st["claude-code"]
    detail = cc.get("error") or f"signed in ({cc.get('auth_method') or 'claude.ai'})"
    t.add_row("claude-code", mark(cc["ready"]), str(detail))
    auto = st["auto"]
    t.add_row("auto", mark(auto["ready"]), f"uses {auto['resolves_to']}")
    console.print(t)
    m = models()
    dev = device_status()
    console.print(
        Text(
            f"  models {GLYPH['dot']} " + ", ".join(m["models"]) + f" (default {m['default']})", style="muted"
        )
    )
    console.print(
        Text(
            f"  device {GLYPH['dot']} {dev['device']}" + ("" if dev["torch"] else " (torch not installed)"),
            style="muted",
        )
    )
    return EXIT_OK if auto["ready"] else EXIT_FAILED


def login(console: Console) -> int:
    """Sign in to Claude Code in the terminal (``claude auth login --claudeai``), then re-check.

    Parameters
    ----------
    console : rich.console.Console
        Output console.

    Returns
    -------
    int
        Exit code: 0 when signed in afterwards.
    """
    from hephaestus.llm.auth import claude_cli_path, claude_code_status

    cli = claude_cli_path()
    if cli is None:
        console.print(
            result_line(
                "[bad]Claude Code CLI not found.[/] Install claude-agent-sdk, or put `claude` on PATH."
            )
        )
        return EXIT_FAILED
    console.print(result_line("Opening the Claude sign-in. Follow the prompts below."))
    try:
        subprocess.run([cli, "auth", "login", "--claudeai"], check=False)
    except (OSError, KeyboardInterrupt) as exc:
        console.print(result_line(f"[warn]Sign-in interrupted:[/] {exc}"))
    st = claude_code_status()
    if st["signed_in"]:
        console.print(
            result_line(f"{mark(True)} Signed in to Claude Code ({st.get('auth_method') or 'claude.ai'}).")
        )
        return EXIT_OK
    console.print(result_line(f"{mark(False)} Still signed out: {st.get('error') or ''}"))
    return EXIT_FAILED


def signin(console: Console) -> int:
    """Sign in to Valency in the browser; reuses a stored token when it still works (LM-2, UI-11).

    Parameters
    ----------
    console : rich.console.Console
        Output console.

    Returns
    -------
    int
        Exit code: 0 when signed in afterwards.
    """
    from hephaestus.core import valency_auth

    def show_url(url: str) -> None:
        """Print the sign-in URL in case the browser does not open.

        Parameters
        ----------
        url : str
            Authorisation URL.
        """
        console.print(result_line(f"Opening your browser. If it does not open, visit:\n[accent]{url}[/]"))

    try:
        st = valency_auth.signin(on_url=show_url)
    except Exception as exc:
        console.print(result_line(f"[bad]Valency sign-in failed:[/] {exc}"))
        return EXIT_FAILED
    if st.get("signed_in"):
        console.print(result_line(f"{mark(True)} Signed in to Valency."))
        return EXIT_OK
    console.print(result_line(f"{mark(False)} Not signed in to Valency."))
    return EXIT_FAILED
