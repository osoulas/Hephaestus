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
from hephaestus.interfaces.cli.stream import Detached, stream
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


def test_backends_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``backends`` shows each option's readiness without calling out."""
    monkeypatch.setattr(
        actions,
        "backend_status",
        lambda: {
            "api": {"ready": False, "error": "ANTHROPIC_API_KEY is not set."},
            "claude-code": {"ready": True, "error": None, "auth_method": "claude.ai"},
            "auto": {"resolves_to": "claude-code", "ready": True},
        },
    )
    code, out = cli(["backends"], tmp_path)
    assert (
        code == 0
        and "claude-code" in out
        and "ANTHROPIC_API_KEY is not set." in out
        and "uses claude-code" in out
    )


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

    def stop(self) -> bool:
        """Record a stop request.

        Returns
        -------
        bool
            True.
        """
        self.stopped += 1
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
