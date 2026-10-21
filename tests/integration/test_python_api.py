"""Python API: ``Experiment`` run, parse, background start, events, resume, replay and tool steps."""

import json
from pathlib import Path

import pytest
import scripts

from hephaestus.core.types import ValidationCheck, ValidationReport
from hephaestus.harness import runner
from hephaestus.interfaces.python import Experiment, ExperimentError, list_runs
from hephaestus.llm.scripted import ScriptedLLM
from hephaestus.tools.stubs import check_novelty, stub_handlers


class Killed(BaseException):
    """Simulates the process dying mid-run."""


def final_check(ws: object, sub: object) -> list[ValidationReport]:
    """Confirm the winner (stand-in for the final re-check).

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
            candidate_id=sub.candidate_id,  # type: ignore[attr-defined]
            verdict="confirmed",
            checks=[ValidationCheck(name="dyn_stable", screening_value=True, final_value=True)],
        )
    ]


def deps(turns: list | None = None, structured: dict | None = None) -> runner.Deps:
    """Return offline dependencies; the runs folder comes from the ``Experiment``.

    Parameters
    ----------
    turns : list, optional
        Orchestrator turns; both scripted rounds by default.
    structured : dict, optional
        Structured answers per section; parser and both reviews by default.

    Returns
    -------
    runner.Deps
        Dependencies.
    """
    turns = scripts.ROUND_1 + scripts.ROUND_2 if turns is None else turns
    structured = (
        {"goal_parser": [scripts.PARSED], "reviewer": scripts.REVIEWS} if structured is None else structured
    )
    llm = ScriptedLLM(turns=list(turns), structured={k: list(v) for k, v in structured.items()})
    return runner.Deps(llm=llm, handlers=stub_handlers(), novelty_fn=check_novelty, final_check=final_check)


def test_or1_experiment_runs_goal_to_brief(tmp_path: Path) -> None:
    """``Experiment(goal).run()`` finishes with a brief, and ``load`` reads it back."""
    exp = Experiment(scripts.GOAL, root=tmp_path, deps=deps(), run_id="p1")
    state = exp.run()
    assert state.status == "finished" and state.best == "c2-1"
    loaded = Experiment.load("p1", tmp_path)
    assert loaded.state.status == "finished" and loaded.spec is not None
    assert loaded.brief and "## Recommended composition and structure" in loaded.brief
    assert loaded.structure("c2-1") is None  # stub tools store no atoms
    assert [r["run_id"] for r in list_runs(tmp_path)] == ["p1"]
    with pytest.raises(ExperimentError):
        loaded.resume()


def test_gp5_parse_returns_questions_then_answers_make_it_ready(tmp_path: Path) -> None:
    """A question blocks the run; answering it gives a ready spec that the run then uses."""
    asked = scripts.PARSED | {"questions": ["Which reference material?"]}
    d = deps(structured={"goal_parser": [asked, scripts.PARSED], "reviewer": scripts.REVIEWS})
    exp = Experiment(scripts.GOAL, root=tmp_path, deps=d)
    first = exp.parse()
    assert not first.ready and first.questions == ["Which reference material?"]
    assert first.to_dict()["parser"] == {"backend": "scripted", "model": "scripted"}
    second = exp.parse({"Which reference material?": "SrTiO3"})
    assert second.ready and exp.spec is not None
    assert exp.run().status == "finished"  # no third parser answer needed


def test_or7_start_streams_history_then_live_until_the_end(tmp_path: Path) -> None:
    """A background run streams every event once, ending after ``run finished``."""
    exp = Experiment(scripts.GOAL, root=tmp_path, deps=deps()).start()
    events = list(exp.events())
    final = exp.wait(5)
    assert final is not None and final.status == "finished"
    assert [e.seq for e in events] == list(range(len(events)))
    assert (events[-1].section, events[-1].status) == ("run", "finished")
    assert not exp.running
    assert len(list(exp.events())) == len(events)  # history only, then ends


def test_or6_resume_after_kill(tmp_path: Path) -> None:
    """A killed run resumes through ``Experiment.resume`` and does not repeat calculations."""

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
        Experiment(scripts.GOAL, root=tmp_path, deps=deps([*scripts.ROUND_1, die]), run_id="p2").run()
    exp = Experiment.load("p2", tmp_path, deps=deps(scripts.ROUND_2, {"reviewer": scripts.REVIEWS[1:]}))
    assert exp.state.status == "running" and not exp.running
    state = exp.resume()
    assert state.status == "finished" and state.best == "c2-1"


def test_ui9_replay_re_emits_stored_events(tmp_path: Path) -> None:
    """A replay publishes the stored events under a new ID and reads the stored run's state."""
    exp = Experiment(scripts.GOAL, root=tmp_path, deps=deps(), run_id="p3")
    exp.run()
    stored = list(exp.events())
    handle = exp.replay(speed=0)
    replayed = list(handle.events())
    assert handle.run_id.startswith("replay-") and len(replayed) == len(stored)
    assert {e.run_id for e in replayed} == {handle.run_id}
    assert handle.state.run_id == "p3" and handle.brief == exp.brief
    assert not (tmp_path / handle.run_id).exists()


def test_or10_tool_steps_go_through_the_campaign(tmp_path: Path) -> None:
    """Step-by-step calls emit events, count against the budget and are snapshotted."""
    exp = Experiment(scripts.GOAL, root=tmp_path, deps=deps(turns=[]), run_id="p4")
    early = exp.tool("propose_candidate", **scripts.propose(scripts.FE1)["input"])
    assert early["accepted"] is False  # CG-4: evidence must be read first, enforced by code
    research = [s for s in scripts.ROUND_1[0] if "tool" in s]
    for step in research:
        exp.tool(step["tool"], **step["input"])
    proposed = exp.tool("propose_candidate", **scripts.propose(scripts.FE1)["input"])
    cid = proposed["candidate_id"]
    for step in scripts.evaluate(cid)[:2]:
        assert "error" not in exp.tool(step["tool"], **step["input"])
    assert "error" in exp.tool("relax", candidate_id="nope", potential="mace-mp-0-small", fmax=0.02)
    ws = Experiment.load("p4", tmp_path).workspace()
    assert ws is not None and cid in ws.candidates and ws.tool_calls == len(research) + 5
    assert ws.spec.target_value is not None  # reference computed first (GP-11)
    tools = [e.payload.get("tool") for e in exp.events() if e.section == "tool" and e.status != "started"]
    assert tools[-4:] == ["propose_candidate", "build_structure", "relax", "relax"]


def test_or1_tool_steps_need_a_ready_spec(tmp_path: Path) -> None:
    """With an open parser question, ``tool`` refuses rather than guessing."""
    asked = scripts.PARSED | {"questions": ["Which reference material?"]}
    exp = Experiment(scripts.GOAL, root=tmp_path, deps=deps(structured={"goal_parser": [asked]}))
    with pytest.raises(ExperimentError, match="Which reference material"):
        exp.tool("candidate_status")


def test_ui7_structure_falls_back_to_the_stored_state(tmp_path: Path) -> None:
    """A run stored without ``workspace.json`` still serves the relaxed structure as CIF."""
    from pymatgen.core import Lattice, Structure

    exp = Experiment(scripts.GOAL, root=tmp_path, deps=deps(), run_id="p5")
    exp.run()
    (tmp_path / "p5" / "workspace.json").unlink()
    sto = Structure(
        Lattice.cubic(3.9),
        ["Sr", "Ti", "O", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )
    state = json.loads((tmp_path / "p5" / "state.json").read_text(encoding="utf-8"))
    state["iterations"][1]["sims"][0]["relaxed"]["pmg_dict"] = sto.as_dict()
    (tmp_path / "p5" / "state.json").write_text(json.dumps(state), encoding="utf-8")
    cif = Experiment.load("p5", tmp_path).structure("c2-1")
    assert cif is not None and "_atom_site_fract_x" in cif and "Sr" in cif


def test_or4_options_override_config() -> None:
    """Keyword options override ``RunConfig`` fields and are validated."""
    exp = Experiment("goal", llm_backend="api", require_novel=True)
    assert exp.config.llm_backend == "api" and exp.config.require_novel
    with pytest.raises(ValueError):
        Experiment("goal", llm_backend="carrier-pigeon")
