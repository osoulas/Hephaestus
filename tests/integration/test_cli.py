"""CLI on scripted runs: subcommands, the interactive session, replay, resume and Ctrl+C (UI-1 to UI-16)."""

import io
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import scripts

from hephaestus.core.types import ProgressEvent, ValidationCheck, ValidationReport
from hephaestus.harness import runner
from hephaestus.interfaces.cli import actions, repl
from hephaestus.interfaces.cli.app import main
from hephaestus.interfaces.cli.render import Renderer
from hephaestus.interfaces.cli.stream import Detached, Prompts, stream
from hephaestus.interfaces.cli.theme import make_console
from hephaestus.llm.scripted import ScriptedLLM
from hephaestus.tools.stubs import check_novelty, stub_handlers

DEMO = Path(__file__).parents[1] / "fixtures" / "events" / "demo"


def final_check(ws: object, sub: Any) -> list[ValidationReport]:
    """Confirm the winner (stand-in for the final re-check).

    Parameters
    ----------
    ws : Workspace
        Unused.
    sub : Submission
        The submission.

    Returns
    -------
    list of ValidationReport
        One confirmed report.
    """
    check = ValidationCheck(name="dyn_stable", screening_value=True, final_value=True)
    return [ValidationReport(candidate_id=sub.candidate_id, verdict="confirmed", checks=[check])]


def factory(parsed: list[dict[str, Any]] | None = None) -> Any:
    """Return a factory of offline dependencies: scripted model, stub tools, stand-in final check.

    Parameters
    ----------
    parsed : list of dict, optional
        Goal-parser answers in order; the ready spec by default.

    Returns
    -------
    callable
        Fresh ``runner.Deps`` per call (each run consumes its script).
    """

    def make() -> runner.Deps:
        """Build one set of dependencies.

        Returns
        -------
        runner.Deps
            Dependencies.
        """
        structured = {"goal_parser": list(parsed or [scripts.PARSED]), "reviewer": list(scripts.REVIEWS)}
        llm = ScriptedLLM(turns=[*scripts.ROUND_1, *scripts.ROUND_2], structured=structured)
        return runner.Deps(
            llm=llm, handlers=stub_handlers(), novelty_fn=check_novelty, final_check=final_check
        )

    return make


def cli(argv: list[str], tmp_path: Path, **kw: Any) -> tuple[int, str]:
    """Run the CLI with a recording console and the runs folder in ``tmp_path``.

    Parameters
    ----------
    argv : list of str
        Arguments.
    tmp_path : pathlib.Path
        Runs folder.
    **kw
        Extra ``main`` keyword arguments (``deps``, ``read_line``).

    Returns
    -------
    tuple
        Exit code and the printed text.
    """
    console = make_console(file=io.StringIO(), width=140, record=True)
    code = main(argv, console=console, root=tmp_path, **kw)
    return code, console.export_text()


def lines(*answers: str) -> Any:
    """Return a scripted line reader.

    Parameters
    ----------
    *answers : str
        Lines to return in order; ``EOFError`` afterwards.

    Returns
    -------
    callable
        ``read(prompt) -> str``.
    """
    queue = list(answers)

    def read(prompt: str) -> str:
        """Return the next scripted line.

        Parameters
        ----------
        prompt : str
            Ignored.

        Returns
        -------
        str
            Next line.

        Raises
        ------
        EOFError
            When the script is used up.
        """
        if not queue:
            raise EOFError
        return queue.pop(0)

    return read


def test_parse_json_prints_the_spec(tmp_path: Path) -> None:
    """``parse --json`` prints the parser's answer as JSON."""
    code, out = cli(["parse", scripts.GOAL, "--json"], tmp_path, deps=factory())
    assert code == 0
    body = json.loads(out.strip().splitlines()[-1])
    assert body["spec"]["target_property"] == "band_gap" and body["questions"] == []


def test_sys1_run_streams_to_brief_and_summary(tmp_path: Path) -> None:
    """``run --yes`` goes from goal to brief unattended and prints the summary."""
    code, out = cli(["run", scripts.GOAL, "--yes"], tmp_path, deps=factory())
    assert code == 0
    assert "GOAL PARSED" in out and "▸ relax  c1-1" in out and "REVIEW · ROUND 1" in out
    assert "EXPERIMENTAL BRIEF" in out and "RUN SUMMARY" in out and "finished · target_met" in out


def test_gp5_open_questions_exit_3_without_a_reader(tmp_path: Path) -> None:
    """With ``--yes``, a goal that needs answers stops with exit code 3."""
    asked = scripts.PARSED | {"questions": ["Which reference material?"]}
    code, out = cli(["run", scripts.GOAL, "--yes"], tmp_path, deps=factory([asked]))
    assert code == 3 and "NEEDS INPUT" in out and "Which reference material?" in out


def test_gp5_questions_are_asked_and_answered(tmp_path: Path) -> None:
    """Without ``--yes``, each question is asked and the answers feed the run."""
    asked = scripts.PARSED | {"questions": ["Which reference material?"]}
    code, out = cli(
        ["run", scripts.GOAL], tmp_path, deps=factory([asked, scripts.PARSED]), read_line=lines("SrTiO3")
    )
    assert code == 0 and "RUN SUMMARY" in out


def test_run_json_lines_are_events(tmp_path: Path) -> None:
    """``--json`` prints one ProgressEvent per line."""
    code, out = cli(["run", scripts.GOAL, "--yes", "--json"], tmp_path, deps=factory())
    assert code == 0
    events = [ProgressEvent.model_validate_json(ln) for ln in out.splitlines() if ln.startswith("{")]
    assert events[0].section == "run" and events[-1].section == "run" and events[-1].status == "finished"


def test_ui9_replay_by_id_and_by_path(tmp_path: Path) -> None:
    """A copied run folder replays offline, by ID under ``--root`` or by its path (DL-8)."""
    runs = tmp_path / "runs"
    shutil.copytree(DEMO, runs / "r-demo")
    code, out = cli(["replay", "r-demo", "--speed", "0", "--root", str(runs)], tmp_path)
    assert code == 0 and "REPLAY  r-demo" in out and "FINAL RE-CHECK · CONFIRMED" in out
    code, out = cli(["replay", str(runs / "r-demo"), "--speed", "0"], tmp_path)
    assert code == 0 and "RUN SUMMARY" in out


def test_ui16_resume_refuses_a_finished_run(tmp_path: Path) -> None:
    """Resuming a finished run prints the refusal and exits 1."""
    cli(["run", scripts.GOAL, "--yes"], tmp_path, deps=factory())
    rid = actions.list_runs(tmp_path)[0]["run_id"]
    code, out = cli(["resume", rid], tmp_path, deps=factory())
    assert code == 1 and "already finished" in out


def test_runs_show_and_brief(tmp_path: Path) -> None:
    """Stored runs are listed, shown and their briefs printed."""
    cli(["run", scripts.GOAL, "--yes"], tmp_path, deps=factory())
    rid = actions.list_runs(tmp_path)[0]["run_id"]
    code, out = cli(["runs"], tmp_path)
    assert code == 0 and rid in out and "finished" in out
    code, out = cli(["show", rid], tmp_path)
    assert code == 0 and "Candidates" in out and "SrTi0.75Fe0.25O3" in out and "Rounds" in out
    code, out = cli(["brief", rid], tmp_path)
    assert code == 0 and "Recommended composition" in out
    code, out = cli(["brief", "nope"], tmp_path)
    assert code == 1 and "No run nope" in out


READY_STATUS = {
    "api": {"ready": False, "error": "ANTHROPIC_API_KEY is not set."},
    "claude-code": {"ready": True, "error": None, "auth_method": "claude.ai"},
    "auto": {"resolves_to": "claude-code", "ready": True},
}


def test_backends_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``backends`` shows each option's readiness without calling out."""
    monkeypatch.setenv("HEPHAESTUS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(actions, "backend_status", lambda: READY_STATUS)
    monkeypatch.setattr(
        actions, "simulation_status", lambda: {"mace": True, "matgl": True, "torch": True, "mp_key": True}
    )
    code, out = cli(["backends"], tmp_path)
    assert code == 0, out
    assert "claude-code" in out and "ANTHROPIC_API_KEY is not set." in out and "uses claude-code" in out
    assert "mace" in out and "blocked" not in out


def test_or1_backends_lists_blockers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing MACE or a missing MP key is reported, with the fix, and fails the check."""
    monkeypatch.setenv("HEPHAESTUS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MP_API_KEY", raising=False)
    monkeypatch.setattr(actions, "backend_status", lambda: READY_STATUS)
    monkeypatch.setattr(
        actions, "simulation_status", lambda: {"mace": False, "matgl": True, "torch": True, "mp_key": False}
    )
    code, out = cli(["backends"], tmp_path)
    assert code == 1
    assert "blocked: MACE is not installed" in out and "Materials Project" in out
    assert "hephaestus login --mp-api-key" in out


def test_dl7_login_saves_and_forgets_an_mp_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``login --mp-api-key`` checks and saves the key owner-only; a malformed key is refused."""
    from hephaestus.tools import materials_project

    home = tmp_path / "home"
    monkeypatch.setenv("HEPHAESTUS_HOME", str(home))
    monkeypatch.delenv("MP_API_KEY", raising=False)
    monkeypatch.setattr(materials_project, "check_mp_key", lambda key: None)
    monkeypatch.setattr("getpass.getpass", lambda prompt: "short")
    code, out = cli(["login", "--mp-api-key"], tmp_path)
    assert code == 1 and "does not look like" in out and not (home / "mp_api_key").exists()
    monkeypatch.setattr("getpass.getpass", lambda prompt: "a1B2" * 8)
    code, out = cli(["login", "--mp-api-key"], tmp_path)
    assert code == 0 and "…a1B2" in out
    assert (home / "mp_api_key").read_text(encoding="utf-8") == "a1B2" * 8
    assert (home / "mp_api_key").stat().st_mode & 0o777 == 0o600
    code, _ = cli(["login", "--forget-mp-api-key"], tmp_path)
    assert code == 0 and not (home / "mp_api_key").exists()


def test_usage_error_exit_2(tmp_path: Path) -> None:
    """Bad arguments are a usage error."""
    code, _ = cli(["run"], tmp_path)
    assert code == 2


def test_ui1_interactive_session_runs_a_goal(tmp_path: Path) -> None:
    """The session takes options, a goal and a confirmation, then runs to the summary."""
    script = lines(
        "/help",
        "/novel on",
        "/novel off",
        "/config max_iterations 3",
        "/bogus",
        scripts.GOAL,
        "y",
        "/runs",
        "/exit",
    )
    code, out = cli([], tmp_path, deps=factory(), read_line=script)
    assert code == 0
    assert "self-designing experiments for materials discovery" in out and "/resume" in out
    assert "require_novel → True" in out and "require_novel → False" in out
    assert "Unknown command /bogus" in out
    assert "RUN SUMMARY" in out and "finished" in out


def test_session_declined_start_runs_nothing(tmp_path: Path) -> None:
    """Answering no to the confirmation starts no run."""
    code, out = cli([], tmp_path, deps=factory(), read_line=lines(scripts.GOAL, "n"))
    assert code == 0 and "Not started" in out and actions.list_runs(tmp_path) == []


def test_slash_completion() -> None:
    """Commands and their first argument complete."""
    session = repl.Session(make_console(file=io.StringIO()), read_line=lines())
    assert [c[0] for c in repl.completions(session, "/mo")] == ["/model"]
    assert [c[0] for c in repl.completions(session, "/backend c")] == ["claude-code"]
    assert repl.completions(session, "a goal") == []


def test_box_reader_reads_a_line() -> None:
    """The bordered input box returns the typed text, and Ctrl+D on an empty box quits."""
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    console = make_console(file=io.StringIO(), record=True)
    session = repl.Session(console, read_line=lines())
    with create_pipe_input() as pipe:
        reader = repl.BoxReader(session, input_=pipe, output=DummyOutput())
        pipe.send_text("a smaller band gap\r")
        assert reader(repl.PROMPT) == "a smaller band gap"
        pipe.send_text("\x04")
        with pytest.raises(EOFError):
            reader(repl.PROMPT)
    assert "▸ a smaller band gap" in console.export_text()


class FakeRun:
    """A run whose event stream is interrupted by Ctrl+C, to test stopping."""

    run_id = "r-fake"

    def __init__(self, interrupts: int) -> None:
        """Set how many times the stream raises ``KeyboardInterrupt``.

        Parameters
        ----------
        interrupts : int
            Interrupts to raise, one per ``events`` call.
        """
        self.interrupts = interrupts
        self.stopped = 0
        self.modes: list[str] = []
        self.calls: list[bool] = []

    def events(self, include_history: bool = True) -> Iterator[ProgressEvent]:
        """Yield one event, then raise or end.

        Parameters
        ----------
        include_history : bool
            Recorded for the test.

        Yields
        ------
        ProgressEvent
            A ``brief_generator`` event.

        Raises
        ------
        KeyboardInterrupt
            While interrupts remain.
        """
        self.calls.append(include_history)
        yield ProgressEvent(
            run_id=self.run_id, section="brief_generator", status="finished", payload={"cited": []}
        )
        if self.interrupts:
            self.interrupts -= 1
            raise KeyboardInterrupt

    def stop(self, mode: str = "campaign") -> bool:
        """Record a stop request.

        Parameters
        ----------
        mode : str
            ``campaign`` or ``all``.

        Returns
        -------
        bool
            True.
        """
        self.stopped += 1
        self.modes.append(mode)
        return True

    def wait(self) -> None:
        """Return no state."""
        return None


def test_or8_first_ctrl_c_stops_and_keeps_streaming() -> None:
    """The first Ctrl+C asks the campaign to stop, then streaming continues without history."""
    run = FakeRun(interrupts=1)
    console = make_console(file=io.StringIO(), width=140, record=True)
    stream(run, console, Renderer())  # type: ignore[arg-type]
    assert run.stopped == 1 and run.calls == [True, False]
    out = console.export_text()
    assert "Stopping" in out and out.count("Brief written") == 2


def test_second_ctrl_c_detaches() -> None:
    """A second Ctrl+C detaches, leaving the run to be resumed."""
    run = FakeRun(interrupts=2)
    with pytest.raises(Detached) as exc:
        stream(run, make_console(file=io.StringIO()), Renderer())  # type: ignore[arg-type]
    assert exc.value.run_id == "r-fake" and run.stopped == 1


def _prompts(*answers: str) -> Prompts:
    """Build prompts that type the given answers in order.

    Parameters
    ----------
    *answers : str
        Lines to type.

    Returns
    -------
    Prompts
        Interactive prompts; confirmations answer yes.
    """
    lines = iter(answers)
    return Prompts(ask=lambda _q: next(lines), confirm=lambda _q, _d: True)


@pytest.mark.parametrize(("key", "mode"), [("s", "campaign"), ("e", "all")])
def test_or8_ctrl_c_menu_stops_the_search_or_everything(key: str, mode: str) -> None:
    """With someone to ask, Ctrl+C offers stop modes; the chosen one reaches ``Experiment.stop``."""
    run = FakeRun(interrupts=1)
    console = make_console(file=io.StringIO(), width=140, record=True)
    stream(run, console, Renderer(), prompts=_prompts(key))  # type: ignore[arg-type]
    assert run.modes == [mode] and run.calls == [True, False]
    out = console.export_text()
    assert "Stop this run?" in out and "Stopping" in out


def test_or8_ctrl_c_menu_detach_and_keep() -> None:
    """Detach raises ``Detached``; keep running does not stop the run."""
    run = FakeRun(interrupts=1)
    with pytest.raises(Detached):
        stream(run, make_console(file=io.StringIO()), Renderer(), prompts=_prompts("d"))  # type: ignore[arg-type]
    assert run.stopped == 0
    run = FakeRun(interrupts=1)
    stream(run, make_console(file=io.StringIO()), Renderer(), prompts=_prompts("keep"))  # type: ignore[arg-type]
    assert run.stopped == 0 and run.calls == [True, False]


class FakeReplay(FakeRun):
    """A replay handle whose stream is interrupted, to test the replay menu."""

    def __init__(self, interrupts: int) -> None:
        """Start at 1x.

        Parameters
        ----------
        interrupts : int
            Interrupts to raise.
        """
        super().__init__(interrupts)
        self.replay_control: dict[str, Any] = {"speed": 1.0}
        self.speeds: list[float] = []

    def set_replay_speed(self, speed: float) -> None:
        """Record a speed change.

        Parameters
        ----------
        speed : float
            New speed.
        """
        self.speeds.append(speed)
        self.replay_control["speed"] = speed


@pytest.mark.parametrize(("key", "speed"), [("f", 2.0), ("s", 0.5), ("i", 0.0)])
def test_ui9_replay_menu_changes_the_speed(key: str, speed: float) -> None:
    """Ctrl+C during a replay offers speeds; the replay keeps streaming at the new one."""
    run = FakeReplay(interrupts=1)
    console = make_console(file=io.StringIO(), width=140, record=True)
    stream(run, console, Renderer(replaying=True), stoppable=False, prompts=_prompts(key))  # type: ignore[arg-type]
    assert run.speeds == [speed] and run.stopped == 0 and run.calls == [True, False]
    assert "Replay at 1" in console.export_text()


def test_ui9_replay_menu_ends_the_replay() -> None:
    """Ending the replay from the menu detaches."""
    run = FakeReplay(interrupts=1)
    with pytest.raises(Detached):
        stream(
            run,
            make_console(file=io.StringIO()),
            Renderer(replaying=True),
            stoppable=False,
            prompts=_prompts("e"),
        )  # type: ignore[arg-type]


def test_or1_run_refused_when_it_cannot_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With real tools but no MACE or MP key, the goal is parsed and the run refused, leaving no folder."""
    from hephaestus.interfaces.cli import stream as stream_mod

    monkeypatch.setattr(
        stream_mod,
        "simulation_status",
        lambda: {"mace": False, "matgl": False, "torch": False, "mp_key": False},
    )

    def real_tools() -> runner.Deps:
        """Scripted model, but the real tool handlers (so the run checks apply).

        Returns
        -------
        runner.Deps
            Dependencies.
        """
        d = factory()()
        d.handlers = None
        return d

    code, out = cli(["run", scripts.GOAL, "--yes"], tmp_path, deps=real_tools)
    assert code == 1
    assert "GOAL PARSED" in out and "The run cannot start yet" in out
    assert 'pip install -e ".[sim]"' in out and "hephaestus login --mp-api-key" in out
    assert actions.list_runs(tmp_path) == [] and not list(tmp_path.glob("r-*"))


def test_ui4_literature_view(tmp_path: Path) -> None:
    """``literature`` lists a run's papers by relevance, its evidence and the coverage note; filters work."""
    runs = tmp_path / "runs"
    shutil.copytree(DEMO, runs / "r-demo")
    code, out = cli(["literature", "r-demo", "--root", str(runs)], tmp_path)
    assert code == 0
    assert "3 papers" in out and "2 cited in the brief" in out
    assert out.index("Fe-doped SrTiO3") < out.index("Solubility limit of Sn")
    assert (
        "EVIDENCE NOTES" in out and "1200.00 C" in out and "SYNTHESIS ROUTES" in out and "Valency Bond" in out
    )
    code, out = cli(["literature", "r-demo", "--root", str(runs), "--filter", "solubility"], tmp_path)
    assert code == 0 and "Solubility limit of Sn" in out and "Fe-doped SrTiO3" not in out
    code, out = cli(["literature", "nope", "--root", str(runs)], tmp_path)
    assert code == 1 and "No run nope" in out
    code, out = cli(["show", "r-demo", "--root", str(runs)], tmp_path)
    assert "hephaestus literature r-demo" in out
