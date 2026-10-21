"""Outer runner: parse, campaign, final check, brief; kill and resume (OR-1, OR-6, round-1 done-when)."""

import json
from pathlib import Path
from typing import Any

import pytest
import scripts

from hephaestus.core.events import EventBus
from hephaestus.core.types import ValidationCheck, ValidationReport
from hephaestus.harness import runner
from hephaestus.llm.scripted import ScriptedLLM
from hephaestus.tools.stubs import check_novelty, stub_handlers


class Killed(BaseException):
    """Simulates the process dying mid-run."""


def final_check(ws: Any, sub: Any) -> list[ValidationReport]:
    """Confirm the winner (stand-in for B's final re-check).

    Parameters
    ----------
    ws : Workspace
        Workspace.
    sub : Submission
        Submission.

    Returns
    -------
    list of ValidationReport
        One confirmed report.
    """
    return [
        ValidationReport(
            candidate_id=sub.candidate_id,
            verdict="confirmed",
            checks=[ValidationCheck(name="dyn_stable", screening_value=True, final_value=True)],
        )
    ]


def deps(tmp_path: Path, turns: list, structured: dict) -> runner.Deps:
    """Return offline dependencies.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Runs folder.
    turns : list
        Orchestrator turns.
    structured : dict
        Structured answers per section.

    Returns
    -------
    runner.Deps
        Dependencies.
    """
    llm = ScriptedLLM(turns=list(turns), structured={k: list(v) for k, v in structured.items()})
    return runner.Deps(
        llm=llm,
        handlers=stub_handlers(),
        novelty_fn=check_novelty,
        final_check=final_check,
        bus=EventBus(tmp_path),
        root=tmp_path,
    )


ANSWERS = {"goal_parser": [scripts.PARSED], "reviewer": scripts.REVIEWS}


def test_sys1_goal_to_brief_unattended(tmp_path: Path) -> None:
    """A goal runs end to end: reference computed, target met, final check, brief on disk."""
    state = runner.run(
        scripts.GOAL, deps=deps(tmp_path, scripts.ROUND_1 + scripts.ROUND_2, ANSWERS), run_id="r1"
    )
    assert state.status == "finished" and state.stop_reason == "target_met"
    assert state.spec and state.spec.excluded_elements == ["Pb"]  # GP-4
    assert state.spec.target_value == pytest.approx(3.12)  # GP-11: model value for SrTiO3
    assert state.best == "c2-1" and state.validation[0].verdict == "confirmed"
    assert len(state.iterations) == 2 and state.iterations[0].critique is not None  # OR-5
    run = tmp_path / "r1"
    for name in (
        "state.json",
        "workspace.json",
        "spec.json",
        "events.jsonl",
        "llm_log.jsonl",
        "brief.md",
        "iter_1/candidates.json",
        "iter_1/critique.json",
        "iter_2/c2-1/sim.json",
        "final/validation.json",
    ):
        assert (run / name).exists(), name
    md = (run / "brief.md").read_text(encoding="utf-8")
    assert "Target not met" not in md and "## Scope and assumptions" in md
    sections = [line for line in md.splitlines() if line.startswith("## ")]
    assert (
        sections[0] == "## Recommended composition and structure"
        and sections[-1] == "## Scope and assumptions"
    )
    order = [
        e["section"] for e in map(json.loads, (run / "events.jsonl").read_text(encoding="utf-8").splitlines())
    ]
    firsts = [
        s
        for i, s in enumerate(order)
        if s in {"goal_parser", "orchestrator", "final_calculation", "brief_generator"} and s not in order[:i]
    ]
    assert firsts == ["goal_parser", "orchestrator", "final_calculation", "brief_generator"]  # OR-1


def test_gp5_questions_stop_before_campaign(tmp_path: Path) -> None:
    """A goal with an unanswered question ends as needs_input with no tool calls."""
    parsed = scripts.PARSED | {"reference_material": None, "questions": ["What band gap do you need?"]}
    state = runner.run(
        "a perovskite with a small band gap", deps=deps(tmp_path, [], {"goal_parser": [parsed]})
    )
    assert state.status == "needs_input" and state.iterations == []


def test_or6_killed_run_resumes_without_repeating_work(tmp_path: Path) -> None:
    """Kill after round 1's review; resume with a fresh conversation; no calculation is repeated."""

    def die(messages: list) -> list:
        """Kill the process.

        Parameters
        ----------
        messages : list
            Ignored.

        Returns
        -------
        list
            Never returns.
        """
        raise Killed

    with pytest.raises(Killed):
        runner.run(scripts.GOAL, deps=deps(tmp_path, [*scripts.ROUND_1, die], ANSWERS), run_id="r2")
    assert json.loads((tmp_path / "r2" / "state.json").read_text(encoding="utf-8"))["status"] == "running"
    d = deps(tmp_path, scripts.ROUND_2, {"reviewer": scripts.REVIEWS[1:]})
    state = runner.resume("r2", deps=d)
    assert state.status == "finished" and state.best == "c2-1" and state.stop_reason == "target_met"
    relaxes = [
        e
        for e in EventBus(tmp_path).history("r2")
        if e.payload.get("tool") == "relax" and e.status == "finished"
    ]
    assert sorted(e.payload["candidate_id"] for e in relaxes) == ["c1-1", "c1-2", "c2-1"]
    with pytest.raises(runner.RunRefused):
        runner.resume("r2", deps=d)


def test_or6_resume_from_state_json_only(tmp_path: Path) -> None:
    """With workspace.json lost, the run resumes from state.json."""

    def die(messages: list) -> list:
        """Kill the process.

        Parameters
        ----------
        messages : list
            Ignored.

        Returns
        -------
        list
            Never returns.
        """
        raise Killed

    with pytest.raises(Killed):
        runner.run(scripts.GOAL, deps=deps(tmp_path, [*scripts.ROUND_1, die], ANSWERS), run_id="r3")
    (tmp_path / "r3" / "workspace.json").unlink()
    state = runner.resume("r3", deps=deps(tmp_path, scripts.ROUND_2, {"reviewer": scripts.REVIEWS[1:]}))
    assert state.status == "finished" and state.best == "c2-1"


def test_sys6_missing_final_check_degrades(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a final check the run still produces a brief that says so."""
    d = deps(tmp_path, scripts.ROUND_1 + scripts.ROUND_2, ANSWERS)
    d.final_check = None
    monkeypatch.setattr(runner, "default_final_check", lambda: None)
    state = runner.run(scripts.GOAL, deps=d, run_id="r4")
    assert state.status == "finished" and state.brief is not None
    assert "Final re-check not run." in state.brief.markdown
    assert any(e.step == "final_calculation" for e in state.errors)


def test_fc1_default_final_check_runs_with_the_run_handlers(tmp_path: Path) -> None:
    """With no injected final check, the real one runs through the run's own (stub) handlers."""
    d = deps(tmp_path, scripts.ROUND_1 + scripts.ROUND_2, ANSWERS)
    d.final_check = None
    state = runner.run(scripts.GOAL, deps=d, run_id="r-fc")
    assert state.status == "finished"
    assert state.validation and state.validation[0].candidate_id == state.submission.candidate_id
    assert state.validation[0].verdict == "confirmed"
    assert (tmp_path / "r-fc" / "final" / "validation.json").exists()
    assert "Final settings: mace-mp-0-medium" in state.brief.markdown
