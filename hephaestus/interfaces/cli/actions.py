"""Commands shared by the subcommands and the REPL's slash commands: runs, show, brief, backends, sign-ins.

Each prints to a console and returns an exit code. They only read stored runs and call the Python
API or the sign-in helpers; nothing here computes a result.
"""

from __future__ import annotations

import getpass
import subprocess
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from hephaestus.interfaces.cli.render import brief_panel, literature_view, result_line, spec_panel
from hephaestus.interfaces.cli.stream import EXIT_FAILED, EXIT_OK, describe_blocker, history_renderer
from hephaestus.interfaces.cli.theme import DASH, GLYPH, mark
from hephaestus.interfaces.python import (
    ExperimentError,
    backend_status,
    device_status,
    list_runs,
    models,
    run_blockers,
    simulation_status,
)


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


def literature_cmd(
    console: Console,
    run_id: str,
    root: Path | None = None,
    *,
    text_filter: str | None = None,
    limit: int | None = 15,
    full: bool = False,
) -> int:
    """Show the literature a run gathered (UI-4); works on a running run, as its state is saved after every call.

    Parameters
    ----------
    console : rich.console.Console
        Output console.
    run_id : str
        Run ID or folder path.
    root : pathlib.Path, optional
        Runs folder.
    text_filter : str, optional
        Keep papers whose title, abstract or authors contain this text.
    limit : int or None
        Most papers to list; ``None`` lists all.
    full : bool
        Whole abstracts.

    Returns
    -------
    int
        Exit code.
    """
    from hephaestus.interfaces.cli.stream import locate_run
    from hephaestus.interfaces.python import Experiment

    rid, base = locate_run(run_id, root)
    try:
        state = Experiment.load(rid, base).state
    except ExperimentError as exc:
        console.print(result_line(f"[bad]{exc}[/]"))
        return EXIT_FAILED
    if state.literature is None or not (state.literature.papers or state.literature.insights):
        console.print(result_line(f"[muted]Run {rid} has no literature yet.[/]"))
        return EXIT_OK
    cited = set(state.brief.cited_paper_ids) if state.brief else set()
    view = literature_view(
        state.literature.model_dump(mode="json"), cited, text_filter=text_filter, limit=limit, full=full
    )
    console.print(view)
    return EXIT_OK


def load_saved_keys() -> None:
    """Put keys saved under ``~/.hephaestus/`` into the environment, unless already set (DL-7).

    The Anthropic API key and the Materials Project key are saved by ``hephaestus login`` (or the
    web UI's Connections dialog). The server loads them at start-up, and so does every CLI command.
    """
    try:
        from hephaestus.core import credentials
        from hephaestus.llm.auth import load_api_key

        load_api_key()
        credentials.load_secret("materials_project")
    except Exception:  # an unreadable key file must not stop the CLI; readiness checks report it
        pass


def blockers() -> list[str]:
    """Reasons a run cannot pass its first step, one sentence each (empty when runs can start).

    Returns
    -------
    list of str
        From ``run_blockers(simulation_status())``: MACE missing, or no Materials Project key.
    """
    return list(run_blockers(simulation_status()))


def backends_view(console: Console) -> int:
    """Show readiness: model access, simulation packages, the Materials Project key and blockers (OR-11, UI-12).

    Parameters
    ----------
    console : rich.console.Console
        Output console.

    Returns
    -------
    int
        Exit code: 0 when the ``auto`` choice is ready and nothing blocks a run.
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
    sim = simulation_status()
    s = Table(box=None, header_style="muted", pad_edge=False, padding=(0, 2))
    for col in ("simulation", "ready", "detail"):
        s.add_column(col)
    s.add_row(
        "mace",
        mark(sim.get("mace")),
        "screening and final potentials" if sim.get("mace") else escape('pip install -e ".[sim]"'),
    )
    s.add_row(
        "matgl",
        mark(sim.get("matgl")),
        "MEGNet band gap" if sim.get("matgl") else escape('pip install -e ".[sim]"'),
    )
    s.add_row(
        "torch", mark(sim.get("torch")), device_status()["device"] if sim.get("torch") else "not installed"
    )
    mp = _mp_key_status()
    s.add_row("mp key", mark(sim.get("mp_key")), mp)
    console.print(s)
    m = models()
    console.print(
        Text(
            f"  models {GLYPH['dot']} " + ", ".join(m["models"]) + f" (default {m['default']})", style="muted"
        )
    )
    found = blockers()
    for reason in found:
        console.print(result_line(f"[bad]blocked:[/] {describe_blocker(reason)}"))
    return EXIT_OK if auto["ready"] and not found else EXIT_FAILED


def _mp_key_status() -> str:
    """Describe where the Materials Project key comes from, without showing it.

    Returns
    -------
    str
        E.g. ``"saved (…a1B2)"``, ``"from the environment (…a1B2)"`` or a hint to add one.
    """
    from hephaestus.core import credentials

    st = credentials.secret_status("materials_project")
    if not st.get("present"):
        return "missing: hephaestus login --mp-api-key"
    where = "saved" if st.get("source") == "saved" else "from the environment"
    return f"{where} ({st.get('hint')})"


LOGIN_MODES = ("claude", "api-key", "forget-api-key", "mp-api-key", "forget-mp-api-key")


def login(console: Console, mode: str = "claude") -> int:
    """Sign in to Claude, or save or remove an API key (OR-11, DL-7).

    Parameters
    ----------
    console : rich.console.Console
        Output console.
    mode : str
        One of ``LOGIN_MODES``: ``claude`` signs in to Claude Code in this terminal
        (``claude auth login --claudeai``); ``api-key`` and ``mp-api-key`` ask for a key with
        input hidden, check it with one request and save it owner-only; the ``forget-`` modes
        delete the saved key.

    Returns
    -------
    int
        Exit code: 0 when signed in, saved or removed.
    """
    if mode == "api-key":
        return _save_anthropic_key(console)
    if mode == "forget-api-key":
        from hephaestus.llm import auth

        auth.forget_api_key()
        console.print(result_line("Saved Anthropic API key removed."))
        return EXIT_OK
    if mode == "mp-api-key":
        return _save_mp_key(console)
    if mode == "forget-mp-api-key":
        from hephaestus.core import credentials

        credentials.delete_secret("materials_project")
        console.print(result_line("Saved Materials Project key removed."))
        return EXIT_OK
    return _claude_code_login(console)


def _claude_code_login(console: Console) -> int:
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


def _save_anthropic_key(console: Console) -> int:
    """Ask for an Anthropic API key with input hidden, check it and save it owner-only.

    Parameters
    ----------
    console : rich.console.Console
        Output console.

    Returns
    -------
    int
        Exit code: 0 when saved.
    """
    from hephaestus.llm import auth

    key = getpass.getpass("Anthropic API key (input hidden): ")
    try:
        st = auth.save_api_key(key, check=auth.check_api_key)  # looked up now, so tests can replace it
    except ValueError as exc:
        console.print(result_line(f"[bad]Not saved:[/] {exc}"))
        return EXIT_FAILED
    console.print(result_line(f"{mark(True)} Saved {st['hint']} to {auth.api_key_path()} (owner-only)."))
    return EXIT_OK


def _save_mp_key(console: Console) -> int:
    """Ask for a Materials Project API key with input hidden, check it and save it owner-only.

    The format check matches the HTTP API's: letters and digits, at least 16 characters.

    Parameters
    ----------
    console : rich.console.Console
        Output console.

    Returns
    -------
    int
        Exit code: 0 when saved.
    """
    from hephaestus.core import credentials
    from hephaestus.tools import materials_project

    key = getpass.getpass("Materials Project API key (input hidden): ").strip()
    if not (key.isalnum() and len(key) >= 16):
        reason: str | None = "That does not look like a Materials Project API key (32 letters and digits)."
    else:
        reason = materials_project.check_mp_key(key)
    if reason:
        console.print(result_line(f"[bad]Not saved:[/] {reason}"))
        return EXIT_FAILED
    credentials.write_secret("materials_project", key)
    path = credentials.secret_path("materials_project")
    console.print(result_line(f"{mark(True)} Saved …{key[-4:]} to {path} (owner-only)."))
    return EXIT_OK


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
