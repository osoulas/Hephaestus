"""The interactive session: type a goal, answer questions, watch the campaign, read the brief.

Styled after the Claude Code CLI: a welcome panel, a bordered input box with ``/command``
completion and history, a status row with the run options, and the event log above it. Without
a terminal (pipes, tests) the box falls back to plain line input.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.padding import Padding
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from hephaestus.core.config import RunConfig, runs_root, user_dir
from hephaestus.interfaces.cli import actions
from hephaestus.interfaces.cli.decide import make_choose
from hephaestus.interfaces.cli.render import result_line
from hephaestus.interfaces.cli.stream import (
    EXIT_OK,
    DepsSource,
    Prompts,
    describe_blocker,
    make_deps,
    replay_run,
    resume_run,
    run_goal,
    spinner,
)
from hephaestus.interfaces.cli.theme import EMBER, GLYPH, GOLD, SCORCH, STEEL, logo, mark
from hephaestus.interfaces.python import backend_ready, device_status, list_runs, models, tool_library

ReadLine = Callable[[str], str]
PROMPT = f"{GLYPH['tool']} "  # the main goal-or-command prompt
EXAMPLE_GOAL = (
    "a lead-free oxide perovskite with a smaller band gap than SrTiO3 that is thermodynamically "
    "stable (≤ 0.05 eV/atom above hull) and dynamically stable"
)
# Every RunConfig field, grouped as in the web composer (OR-4); /config lists and completes them.
CONFIG_GROUPS: dict[str, tuple[str, ...]] = {
    "search budget": (
        "max_iterations",
        "max_tool_calls",
        "time_budget_s",
        "confidence_threshold",
        "max_parallel_tools",
        "require_novel",
    ),
    "models": (
        "llm_backend",
        "llm_model",
        "band_gap_model",
        "potential",
        "final_potential",
        "cross_check_potential",
    ),
    "stability checks": (
        "run_hull",
        "e_hull_max_eV_per_atom",
        "run_phonons",
        "phonon_mode",
        "phonon_gate",
        "distortion_threshold_meV",
    ),
    "agent-built tools": (
        "custom_tools",
        "max_custom_tools",
        "max_tool_revisions",
        "capability_time_budget_s",
        "custom_tool_timeout_s",
        "tool_library_dir",
        "disabled_capabilities",
    ),
    "final re-check": ("run_final", "final_time_budget_s", "final_dtype"),
    "compute": ("device", "dtype", "max_atoms", "seed", "offline"),
}
CONFIG_KEYS = tuple(k for keys in CONFIG_GROUPS.values() for k in keys)
NO_LIMIT_KEYS = {
    "max_iterations",
    "max_tool_calls",
    "time_budget_s",
    "final_time_budget_s",
    "capability_time_budget_s",
}


@dataclass
class Command:
    """One slash command.

    Attributes
    ----------
    name : str
        Command word, without the slash.
    usage : str
        Argument summary for ``/help``.
    help : str
        What it does.
    handler : callable
        ``handler(session, args) -> bool``; False ends the session.
    choices : callable, optional
        ``choices(session) -> list of str`` for completing the first argument.
    """

    name: str
    usage: str
    help: str
    handler: Callable[[Session, list[str]], bool]
    choices: Callable[[Session], list[str]] | None = None


class Session:
    """One interactive session; ``run`` loops until ``/exit`` or Ctrl+D.

    Parameters
    ----------
    console : rich.console.Console
        Output console.
    config : RunConfig, optional
        Starting options.
    read_line : callable, optional
        ``read_line(prompt) -> str``; defaults to the bordered input box in a terminal, plain
        ``input`` otherwise. Tests pass a scripted reader.
    deps : runner.Deps, callable or None
        Injected dependencies (scripted model, stub tools) or a factory for them.
    root : pathlib.Path, optional
        Runs folder.
    verbose : bool
        Show thinking and results in full.
    """

    def __init__(
        self,
        console: Console,
        config: RunConfig | None = None,
        *,
        read_line: ReadLine | None = None,
        deps: DepsSource = None,
        root: Path | None = None,
        verbose: bool = False,
    ) -> None:
        """Set up the session; nothing is checked or printed until ``run``.

        Parameters
        ----------
        console : rich.console.Console
            Output console.
        config : RunConfig, optional
            Starting options.
        read_line : callable, optional
            Line reader.
        deps : runner.Deps, callable or None
            Injected dependencies.
        root : pathlib.Path, optional
            Runs folder.
        verbose : bool
            Full output.
        """
        self.console = console
        self.config = config or RunConfig()
        self.deps = deps
        self.root = root
        self.verbose = verbose
        self.read_line = read_line or make_reader(self)
        self.scripted = bool((d := make_deps(deps)) is not None and d.llm is not None)

    # ------------------------------------------------------------------ main loop

    def run(self) -> int:
        """Show the banner and handle lines until the user quits.

        Returns
        -------
        int
            Exit code.
        """
        self.banner()
        interrupted = False
        while True:
            try:
                line = self.read_line(PROMPT).strip()
            except KeyboardInterrupt:
                if interrupted:
                    return EXIT_OK
                interrupted = True
                self.console.print(Text("  Press ^C again to exit, or type /exit.", style="muted"))
                continue
            except EOFError:
                return EXIT_OK
            interrupted = False
            if not line:
                continue
            if line.startswith("/"):
                if not self.command(line):
                    return EXIT_OK
                continue
            run_goal(
                self.console,
                line,
                self.config,
                self.prompts(),
                deps=self.deps,
                root=self.root,
                verbose=self.verbose,
            )

    def prompts(self) -> Prompts:
        """Return the session's question, confirmation and choice readers.

        Returns
        -------
        Prompts
            Interactive prompts on the input box.
        """
        return Prompts(ask=self.ask, confirm=self.confirm, choose=make_choose(self.console, self.read_line))

    def ask(self, question: str) -> str:
        """Ask one clarifying question.

        Parameters
        ----------
        question : str
            The parser's question.

        Returns
        -------
        str
            The answer.
        """
        self.console.print(Text.assemble(("? ", "warn"), (question, "bold")))
        return self.read_line("  ").strip()

    def confirm(self, question: str, default: bool = True) -> bool:
        """Ask a yes/no question.

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
        hint = "[Y/n]" if default else "[y/N]"
        reply = self.read_line(f"{question} {hint} ").strip().lower()
        return default if not reply else reply in ("y", "yes")

    def command(self, line: str) -> bool:
        """Run one slash command.

        Parameters
        ----------
        line : str
            Input starting with ``/``.

        Returns
        -------
        bool
            False to end the session.
        """
        name, *args = line[1:].split()
        cmd = COMMANDS.get(name.lower()) or COMMANDS.get(ALIASES.get(name.lower(), ""))
        if cmd is None:
            self.console.print(result_line(f"[warn]Unknown command /{name}.[/] Type /help for the list."))
            return True
        return cmd.handler(self, args)

    # ------------------------------------------------------------------ display

    def status_text(self) -> str:
        """Return the options row under the input box.

        Returns
        -------
        str
            E.g. ``"claude-code · claude-opus-5-5 · novel off · hse · 3 rounds"``.
        """
        c = self.config
        bits = [
            c.llm_backend,
            c.llm_model,
            f"novel {'on' if c.require_novel else 'off'}",
            f"gap {c.band_gap_model}",
            f"{c.max_iterations} rounds" if c.max_iterations else "no round limit",
        ]
        if c.custom_tools == "ask":
            bits.append("new tools ask")
        if c.offline:
            bits.append("offline")
        if self.verbose:
            bits.append("verbose")
        return f" {GLYPH['dot']} ".join(bits)

    def banner(self) -> None:
        """Print the logo, then the readiness of model access, Valency and the device."""
        c = self.config
        if self.scripted:
            access = f"{c.llm_backend} [muted](scripted model)[/]"
            valency = "[muted]not checked (scripted)[/]"
        else:
            with spinner(self.console, "Checking model access"):
                ok, error = backend_ready(c.llm_backend)
            access = f"{c.llm_backend} {mark(ok)}" + ("" if ok else f" [muted]{error or ''} · /login[/]")
            valency = _valency_line()
        dev = device_status()
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="muted", no_wrap=True)
        grid.add_column()
        grid.add_row("model access", access)
        grid.add_row("model", c.llm_model)
        grid.add_row(
            "options",
            f"novel {'on' if c.require_novel else 'off'} · band gap {c.band_gap_model} · "
            f"{c.max_iterations or 'no limit on'} rounds · {c.max_tool_calls or 'no limit on'} tool calls · "
            f"agent-built tools {'ask' if c.custom_tools == 'ask' else 'off'}",
        )
        grid.add_row("literature", valency)
        grid.add_row(
            "device",
            dev["device"] + ("" if dev["torch"] else " [muted](no torch: simulation tools unavailable)[/]"),
        )
        if not self.scripted:
            found = actions.blockers()
            grid.add_row(
                "simulation",
                f"{mark(True)} runs can start"
                if not found
                else f"{mark(False)} "
                + " ".join(describe_blocker(b) for b in found)
                + " [muted]· /backends[/]",
            )
        grid.add_row("runs", str(self.root or runs_root()))
        width = self.console.width
        self.console.print()
        self.console.print(logo(width))
        self.console.print(Text("  self-designing experiments for materials discovery", style="accent"))
        self.console.print(
            Text("  forged in simulation · checked by code · grounded in the literature", style="muted")
        )
        self.console.print()
        rule = Rule(style="gutter", characters="─")
        self.console.print(rule)
        self.console.print(Padding(grid, (0, 2)))
        self.console.print(rule)
        self.console.print(Text("  Type a research goal, for example:", style="muted"))
        self.console.print(Text(f"  {EXAMPLE_GOAL}", style="faint"))
        self.console.print(Text("  /help commands · ^C stops a running campaign · ^D quits", style="muted"))
        self.console.print()


def _valency_line() -> str:
    """Valency sign-in state for the banner (UI-11).

    Returns
    -------
    str
        Markup.
    """
    try:
        from hephaestus.core import valency_auth

        st = valency_auth.status()
    except Exception as exc:
        return f"[warn]Valency status unknown[/] [muted]({exc})[/]"
    if st.get("needs_signin"):
        return f"{mark(False)} Valency sign-in expired [muted]· /signin[/]"
    if not st.get("signed_in"):
        return f"{mark(False)} Valency not signed in [muted]· /signin[/]"
    return f"{mark(True)} Valency signed in"


# ---------------------------------------------------------------------- slash commands


def _help(s: Session, args: list[str]) -> bool:
    """``/help``: list the commands.

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        Ignored.

    Returns
    -------
    bool
        True.
    """
    t = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2))
    t.add_column(style="accent", no_wrap=True)
    t.add_column(style="muted", no_wrap=True)
    t.add_column()
    for cmd in COMMANDS.values():
        t.add_row(f"/{cmd.name}", cmd.usage, cmd.help)
    s.console.print(t)
    s.console.print(Text("  Anything that is not a command is taken as a research goal.", style="muted"))
    return True


def _model(s: Session, args: list[str]) -> bool:
    """``/model [name]``: show or set the Claude model.

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        Optional model ID.

    Returns
    -------
    bool
        True.
    """
    available = models()["models"]
    if not args:
        for m in available:
            here = f" [accent]{GLYPH['arrow']} current[/]" if m == s.config.llm_model else ""
            s.console.print(f"  {m}{here}")
        return True
    if args[0] not in available:
        s.console.print(result_line(f"[warn]Unknown model {args[0]}.[/] Choose from: {', '.join(available)}"))
        return True
    return _set(s, "llm_model", args[0])


def _backend(s: Session, args: list[str]) -> bool:
    """``/backend [auto|api|claude-code]``: show readiness, or choose model access (OR-11, UI-12).

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        Optional backend.

    Returns
    -------
    bool
        True.
    """
    if not args:
        actions.backends_view(s.console)
        return True
    if args[0] not in BACKENDS:
        s.console.print(result_line(f"[warn]Choose one of {', '.join(BACKENDS)}.[/]"))
        return True
    _set(s, "llm_backend", args[0])
    if not s.scripted:
        with spinner(s.console, "Checking model access"):
            ok, error = backend_ready(args[0])
        if not ok:
            s.console.print(result_line(f"[warn]Not ready:[/] {error or ''} · /login"))
    return True


def _novel(s: Session, args: list[str]) -> bool:
    """``/novel [on|off]``: require novel materials (GP-13, UI-13).

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        ``on`` or ``off``; toggles when empty.

    Returns
    -------
    bool
        True.
    """
    value = (not s.config.require_novel) if not args else args[0].lower() in ("on", "yes", "true", "1")
    return _set(s, "require_novel", value)


def _verbose(s: Session, args: list[str]) -> bool:
    """``/verbose [on|off]``: show thinking and results in full.

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        ``on`` or ``off``; toggles when empty.

    Returns
    -------
    bool
        True.
    """
    s.verbose = (not s.verbose) if not args else args[0].lower() in ("on", "yes", "true", "1")
    s.console.print(result_line(f"verbose {'on' if s.verbose else 'off'}"))
    return True


def _config(s: Session, args: list[str]) -> bool:
    """``/config [key value]``: show the run options, or set one (OR-4).

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        ``key value``, or nothing to show all.

    Returns
    -------
    bool
        True.
    """
    if not args:
        t = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2))
        t.add_column(style="muted", no_wrap=True)
        t.add_column()
        data = s.config.model_dump()
        for i, (group, keys) in enumerate(CONFIG_GROUPS.items()):
            if i:
                t.add_row("", "")
            t.add_row(Text(group, style="accent"), "")
            for key in keys:
                t.add_row(f"  {key}", _config_value(key, data.get(key)))
        s.console.print(t)
        s.console.print(
            Text("  /config <key> <value> sets one · /profile quick|standard|thorough", style="faint")
        )
        return True
    if len(args) < 2:
        s.console.print(result_line("[warn]Usage:[/] /config <key> <value>"))
        return True
    return _set(s, args[0], " ".join(args[1:]))


def _config_value(key: str, value: Any) -> str:
    """Format one option for ``/config``: budgets of 0 read "no limit", unset values "none".

    Parameters
    ----------
    key : str
        ``RunConfig`` field.
    value : Any
        Its value.

    Returns
    -------
    str
        Display text.
    """
    if key in NO_LIMIT_KEYS and not value:
        return "0 (no limit)"
    if value is None:
        return "none"
    if isinstance(value, list):
        return ", ".join(map(str, value)) or "none"
    return str(value)


def _profile(s: Session, args: list[str]) -> bool:
    """``/profile quick|standard|thorough``: apply a preset of budgets, as the web composer's.

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        Profile name.

    Returns
    -------
    bool
        True.
    """
    if not args or args[0] not in actions.PROFILES:
        s.console.print(result_line(f"[warn]Usage:[/] /profile {'|'.join(actions.PROFILES)}"))
        return True
    values = actions.PROFILES[args[0]]
    if args[0] == "standard":
        defaults = RunConfig()
        keys = {k for p in actions.PROFILES.values() for k in p}
        values = {k: getattr(defaults, k) for k in keys}
    s.config = RunConfig.model_validate({**s.config.model_dump(), **values})
    shown = " · ".join(f"{k} {_config_value(k, getattr(s.config, k))}" for k in values)
    s.console.print(result_line(f"profile {GLYPH['arrow']} [accent]{args[0]}[/] [muted]{shown}[/]"))
    return True


def _set(s: Session, key: str, value: Any) -> bool:
    """Validate and apply one option change through ``RunConfig`` itself.

    Parameters
    ----------
    s : Session
        Session.
    key : str
        ``RunConfig`` field.
    value : Any
        New value; strings are converted by the model's validation.

    Returns
    -------
    bool
        True.
    """
    if key not in RunConfig.model_fields:
        s.console.print(result_line(f"[warn]Unknown option {key}.[/] See /config."))
        return True
    if isinstance(value, str) and value.lower() in ("none", "null"):
        value = None
    if key == "disabled_capabilities" and isinstance(value, str):
        value = [v.strip() for v in value.split(",") if v.strip()]
    elif key == "disabled_capabilities" and value is None:
        value = []
    if isinstance(value, str) and value.lower() in ("on", "off") and key != "custom_tools":
        value = value.lower() == "on"
    try:
        s.config = RunConfig.model_validate({**s.config.model_dump(), key: value})
    except Exception as exc:
        first = str(exc).splitlines()
        s.console.print(result_line(f"[warn]Invalid value for {key}:[/] {first[-1] if first else exc}"))
        return True
    shown = _config_value(key, getattr(s.config, key))
    s.console.print(result_line(f"{key} {GLYPH['arrow']} [accent]{escape(shown)}[/]"))
    return True


def _runs(s: Session, args: list[str]) -> bool:
    """List stored runs (``/runs``).

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        Ignored.

    Returns
    -------
    bool
        True.
    """
    actions.runs_view(s.console, s.root)
    return True


def _need_id(s: Session, args: list[str], usage: str) -> str | None:
    """Return the run ID argument, or the newest run's when omitted (with a message when there is none).

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        Command arguments.
    usage : str
        Usage text for the error.

    Returns
    -------
    str or None
        Run ID.
    """
    if args:
        return args[0]
    runs = list_runs(s.root)
    if runs:
        return str(runs[0]["run_id"])
    s.console.print(result_line(f"[warn]No runs yet.[/] Usage: {usage}"))
    return None


def _show(s: Session, args: list[str]) -> bool:
    """``/show [id]``: a run's spec, rounds, candidates and summary.

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        Run ID; the newest run by default.

    Returns
    -------
    bool
        True.
    """
    rid = _need_id(s, args, "/show <run id>")
    if rid:
        actions.show_view(s.console, rid, s.root)
    return True


def _brief(s: Session, args: list[str]) -> bool:
    """``/brief [id] [compute] [pdf]``: print a run's lab brief, or its computational brief (UI-8).

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        Run ID (the newest run by default), then ``compute`` and/or ``pdf``.

    Returns
    -------
    bool
        True.
    """
    words = {"compute", "pdf"}
    rid = _need_id(s, [a for a in args if a not in words][:1], "/brief <run id> [compute] [pdf]")
    if rid:
        actions.brief_view(s.console, rid, s.root, pdf="pdf" in args, compute="compute" in args)
    return True


def _rename(s: Session, args: list[str]) -> bool:
    """``/rename [id] <title>``: set a run's title (the newest run's when the first word is not a run).

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        Optional run ID, then the title words.

    Returns
    -------
    bool
        True.
    """
    ids = set(_run_ids(s))
    rid, words = (args[0], args[1:]) if args and args[0] in ids else (None, args)
    if not words:
        s.console.print(result_line("[warn]Usage:[/] /rename [run id] <title>"))
        return True
    rid = rid or _need_id(s, [], "/rename [run id] <title>")
    if rid:
        actions.rename_cmd(s.console, rid, " ".join(words), s.root)
    return True


def _tools(s: Session, args: list[str]) -> bool:
    """``/tools [list|review|show|remove] [name]``: the agent-built tool library (AT-20, AT-21, AT-28).

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        Action (``list`` by default) and tool name.

    Returns
    -------
    bool
        True.
    """
    action = args[0] if args else "list"
    actions.tools_cmd(
        s.console,
        action,
        args[1] if len(args) > 1 else None,
        s.config.tool_library_dir,
        ask=s.read_line,
        confirm=s.confirm,
    )
    return True


def _tool_names(s: Session) -> list[str]:
    """Return the library actions and tool names for completion.

    Parameters
    ----------
    s : Session
        Session.

    Returns
    -------
    list of str
        ``list``, ``review``, ``show``, ``remove``, then tool names.
    """
    try:
        lib = tool_library(s.config.tool_library_dir)
    except Exception:
        return list(actions.TOOL_ACTIONS)
    return list(actions.TOOL_ACTIONS) + [str(t["name"]) for t in lib["approved"] + lib["unapproved"]]


def _literature(s: Session, args: list[str]) -> bool:
    """``/literature [id] [filter]``: the papers, evidence and failed approaches of a run (UI-4).

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        Run ID (the newest run by default), then optional filter words.

    Returns
    -------
    bool
        True.
    """
    rid = _need_id(s, args[:1], "/literature <run id> [filter]")
    if rid:
        text_filter = " ".join(args[1:]) or None
        actions.literature_cmd(s.console, rid, s.root, text_filter=text_filter, full=s.verbose)
    return True


def _resume(s: Session, args: list[str]) -> bool:
    """``/resume <id>``: resume an interrupted run (UI-16).

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        Run ID.

    Returns
    -------
    bool
        True.
    """
    if not args:
        s.console.print(result_line("[warn]Usage:[/] /resume <run id> · /runs lists them"))
        return True
    resume_run(s.console, args[0], deps=s.deps, root=s.root, verbose=s.verbose, prompts=s.prompts())
    return True


def _replay(s: Session, args: list[str]) -> bool:
    """``/replay <id> [speed]``: replay a stored run (UI-9).

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        Run ID and optional speed.

    Returns
    -------
    bool
        True.
    """
    if not args:
        s.console.print(result_line("[warn]Usage:[/] /replay <run id> [speed]"))
        return True
    try:
        speed = float(args[1]) if len(args) > 1 else 4.0
    except ValueError:
        s.console.print(result_line("[warn]Speed must be a number.[/]"))
        return True
    replay_run(s.console, args[0], speed, root=s.root, verbose=s.verbose, prompts=s.prompts())
    return True


def _login(s: Session, args: list[str]) -> bool:
    """``/login [mode]``: sign in to Claude Code, or save or remove an API key (OR-11, DL-7).

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        Optional mode from ``actions.LOGIN_MODES``; Claude Code sign-in when empty.

    Returns
    -------
    bool
        True.
    """
    mode = args[0] if args else "claude"
    if mode not in actions.LOGIN_MODES:
        s.console.print(result_line(f"[warn]Choose one of {', '.join(actions.LOGIN_MODES)}.[/]"))
        return True
    actions.login(s.console, mode)
    return True


def _signin(s: Session, args: list[str]) -> bool:
    """``/signin``: Valency sign-in.

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        Ignored.

    Returns
    -------
    bool
        True.
    """
    actions.signin(s.console)
    return True


def _clear(s: Session, args: list[str]) -> bool:
    """``/clear``: clear the screen.

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        Ignored.

    Returns
    -------
    bool
        True.
    """
    s.console.clear()
    return True


def _exit(s: Session, args: list[str]) -> bool:
    """``/exit``: end the session.

    Parameters
    ----------
    s : Session
        Session.
    args : list of str
        Ignored.

    Returns
    -------
    bool
        False.
    """
    return False


def _run_ids(s: Session) -> list[str]:
    """Return stored run IDs for completion.

    Parameters
    ----------
    s : Session
        Session.

    Returns
    -------
    list of str
        Run IDs, newest first.
    """
    try:
        return [str(r["run_id"]) for r in list_runs(s.root)][:30]
    except Exception:
        return []


BACKENDS = ["auto", "api", "claude-code"]
_ALL = [
    Command("help", "", "List commands", _help),
    Command("model", "[name]", "Show or choose the Claude model", _model, lambda s: list(models()["models"])),
    Command(
        "backend", "[auto|api|claude-code]", "Model access, with readiness", _backend, lambda s: BACKENDS
    ),
    Command("novel", "[on|off]", "Require novel materials", _novel, lambda s: ["on", "off"]),
    Command("config", "[key value]", "Show or set run options", _config, lambda s: list(CONFIG_KEYS)),
    Command(
        "profile",
        "quick|standard|thorough",
        "Preset budgets for the next run",
        _profile,
        lambda s: list(actions.PROFILES),
    ),
    Command("verbose", "[on|off]", "Show thinking and results in full", _verbose, lambda s: ["on", "off"]),
    Command("runs", "", "List stored runs", _runs),
    Command("show", "[id]", "A run's spec, rounds, candidates and summary", _show, _run_ids),
    Command(
        "brief", "[id] [compute] [pdf]", "A run's lab brief, or its computational brief", _brief, _run_ids
    ),
    Command("rename", "[id] <title>", "Set a run's title", _rename, _run_ids),
    Command(
        "tools",
        "[list|review|show|remove] [name]",
        "Agent-built tools in the library",
        _tools,
        _tool_names,
    ),
    Command(
        "literature",
        "[id] [filter]",
        "Papers, evidence and failed approaches of a run",
        _literature,
        _run_ids,
    ),
    Command("resume", "<id>", "Resume an interrupted run", _resume, _run_ids),
    Command("replay", "<id> [speed]", "Replay a stored run, offline", _replay, _run_ids),
    Command("backends", "", "Readiness of each model-access option", lambda s, a: _backend(s, [])),
    Command(
        "login",
        "[mode]",
        "Claude sign-in, or save/remove an API or MP key",
        _login,
        lambda s: list(actions.LOGIN_MODES),
    ),
    Command("signin", "", "Sign in to Valency (literature)", _signin),
    Command("clear", "", "Clear the screen", _clear),
    Command("exit", "", "Quit", _exit),
]
COMMANDS: dict[str, Command] = {c.name: c for c in _ALL}
ALIASES = {"quit": "exit", "q": "exit", "h": "help", "?": "help"}


# ---------------------------------------------------------------------- input box


def completions(session: Session, text: str) -> list[tuple[str, str, str]]:
    """Completions for the text before the cursor (slash commands and their first argument).

    Parameters
    ----------
    session : Session
        Session, for argument choices.
    text : str
        Text before the cursor.

    Returns
    -------
    list of tuple
        ``(completion, display, meta)``; the completion replaces the current word.
    """
    if not text.startswith("/"):
        return []
    parts = text[1:].split(" ")
    if len(parts) == 1:
        word = parts[0].lower()
        return [(f"/{c.name}", f"/{c.name}", c.help) for c in COMMANDS.values() if c.name.startswith(word)]
    cmd = COMMANDS.get(parts[0].lower())
    if cmd is None or cmd.choices is None or len(parts) > 2:
        return []
    word = parts[1]
    return [(x, x, "") for x in cmd.choices(session) if x.startswith(word)]


def make_reader(session: Session) -> ReadLine:
    """Return the line reader: the bordered prompt_toolkit box in a terminal, else ``input``.

    Parameters
    ----------
    session : Session
        Session, for completion and the status row.

    Returns
    -------
    callable
        ``read(prompt) -> str``; raises ``KeyboardInterrupt`` or ``EOFError`` like ``input``.
    """
    import sys

    if not (sys.stdin.isatty() and session.console.is_terminal):
        return input
    try:
        return BoxReader(session)
    except ImportError:  # prompt_toolkit missing
        return input


INPUT_MAX_LINES = 6  # the input box grows with a long goal up to this many rows, then scrolls


def input_area(prompt: str, history: Any = None, completer: Any = None) -> Any:
    """Build the input box's text area: one logical line that wraps and grows to show all of it.

    prompt_toolkit fixes a single-line ``TextArea`` at one row, so a goal wider than the terminal
    showed only its last wrapped row. The height here follows the wrapped text instead.

    Parameters
    ----------
    prompt : str
        Prompt shown before the text, e.g. ``"▸ "``.
    history : prompt_toolkit History, optional
        Input history (the main prompt only).
    completer : prompt_toolkit Completer, optional
        Slash-command completer (the main prompt only).

    Returns
    -------
    prompt_toolkit.widgets.TextArea
        The text area, at least one row and at most ``INPUT_MAX_LINES``.
    """
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.widgets import TextArea

    area = TextArea(
        prompt=[("class:prompt", prompt)],
        multiline=False,
        wrap_lines=True,
        history=history,
        completer=completer,
        complete_while_typing=True,
        dont_extend_height=True,
    )
    area.window.height = Dimension(min=1, max=INPUT_MAX_LINES)
    return area


class BoxReader:
    """Claude Code-style input: a bordered box, a completion menu, hints and the status row.

    Parameters
    ----------
    session : Session
        Session, for completion and the status row.
    input_ : prompt_toolkit Input, optional
        Input source (tests pass a pipe).
    output : prompt_toolkit Output, optional
        Output target (tests pass a dummy).
    """

    def __init__(self, session: Session, input_: Any = None, output: Any = None) -> None:
        """Prepare history and style.

        Parameters
        ----------
        session : Session
            Session.
        input_ : prompt_toolkit Input, optional
            Input source.
        output : prompt_toolkit Output, optional
            Output target.
        """
        from prompt_toolkit.history import FileHistory, InMemoryHistory
        from prompt_toolkit.styles import Style

        self.session = session
        self.input, self.output = input_, output
        try:
            path = user_dir() / "cli_history"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.history: Any = FileHistory(str(path))
        except OSError:
            self.history = InMemoryHistory()
        self.style = Style.from_dict(
            {
                "rule": "#7a3b22",
                "label": f"{GOLD} bold",
                "prompt": f"{GOLD} bold",
                "hint": "#8a8a8a",
                "status": STEEL,
                "completion-menu.completion": "bg:#1f1a17 #d0d0d0",
                "completion-menu.completion.current": f"bg:{EMBER} #000000",
                "completion-menu.meta.completion": f"bg:#1f1a17 {SCORCH}",
                "completion-menu.meta.completion.current": f"bg:{EMBER} #000000",
            }
        )

    def __call__(self, prompt: str) -> str:
        """Read one line between the input rules.

        Parameters
        ----------
        prompt : str
            Prompt text before the cursor; ``PROMPT`` for the main goal-or-command line.

        Returns
        -------
        str
            The entered text.

        Raises
        ------
        KeyboardInterrupt
            On Ctrl+C in an empty box.
        EOFError
            On Ctrl+D in an empty box.
        """
        from prompt_toolkit.application import Application
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.filters import has_completions
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, VSplit, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.layout.menus import CompletionsMenu

        session = self.session
        is_main = prompt == PROMPT
        label = "goal or /command" if is_main else ("answer" if not prompt.strip() else "confirm")

        class Slash(Completer):
            """Complete slash commands and their first argument."""

            def get_completions(self, document: Any, complete_event: Any) -> Any:
                """Yield completions for the text before the cursor.

                Parameters
                ----------
                document : prompt_toolkit.document.Document
                    Current text.
                complete_event : prompt_toolkit.completion.CompleteEvent
                    Unused.

                Yields
                ------
                prompt_toolkit.completion.Completion
                    Candidates.
                """
                before = document.text_before_cursor
                word = before.split(" ")[-1]
                for text, display, meta in completions(session, before):
                    yield Completion(text, start_position=-len(word), display=display, display_meta=meta)

        area = input_area(prompt, history=self.history if is_main else None, completer=Slash() if is_main else None)
        kb = KeyBindings()

        def accept(buffer: Any) -> bool:
            """Finish with the typed text.

            Parameters
            ----------
            buffer : prompt_toolkit.buffer.Buffer
                Input buffer.

            Returns
            -------
            bool
                True keeps the text, so history is stored.
            """
            app.exit(result=buffer.text)
            return True

        area.accept_handler = accept

        @kb.add("c-c")
        def _interrupt(event: Any) -> None:
            """Clear the box, or interrupt when it is already empty.

            Parameters
            ----------
            event : prompt_toolkit.key_binding.KeyPressEvent
                Key event.
            """
            if area.text:
                area.text = ""
            else:
                event.app.exit(exception=KeyboardInterrupt())

        @kb.add("c-d")
        def _eof(event: Any) -> None:
            """Quit when the box is empty.

            Parameters
            ----------
            event : prompt_toolkit.key_binding.KeyPressEvent
                Key event.
            """
            if not area.text:
                event.app.exit(exception=EOFError())

        hints = "  / commands · ↑↓ history · tab completes · ^D quits" if is_main else "  enter to answer"
        top = VSplit(
            [
                Window(
                    FormattedTextControl(
                        [("class:rule", "── "), ("class:label", label), ("class:rule", " ")]
                    ),
                    dont_extend_width=True,
                    height=1,
                ),
                Window(char="─", style="class:rule", height=1),
            ]
        )
        bottom = VSplit(
            [
                Window(char="─", style="class:rule", height=1),
                Window(
                    FormattedTextControl(
                        lambda: [("class:status", f" {session.status_text()} "), ("class:rule", "──")]
                    ),
                    dont_extend_width=True,
                    height=1,
                ),
            ]
        )
        body = HSplit(
            [
                top,
                area,
                bottom,
                ConditionalContainer(CompletionsMenu(max_height=8, scroll_offset=1), filter=has_completions),
                Window(FormattedTextControl([("class:hint", hints)]), height=1),
            ]
        )
        app: Application[str] = Application(
            layout=Layout(body, focused_element=area),
            key_bindings=kb,
            style=self.style,
            full_screen=False,
            erase_when_done=True,
            input=self.input,
            output=self.output,
        )
        text = app.run()
        session.console.print(
            Text.assemble((prompt if prompt.strip() else PROMPT, "accent"), (text, "")), soft_wrap=True
        )
        return text
