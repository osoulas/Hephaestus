"""More than one property target: gates, parser, references, agent-built tools and the brief."""

from pathlib import Path
from typing import Any

import pytest
import scripts
from factories import candidate_state, demo_spec, workspace
from test_custom_tools import campaign_turns, cfg, deps
from toolkit_fake import GAP, TEST_SOURCE, TOOL_SOURCE, fake_factory

from hephaestus.core.config import RunConfig
from hephaestus.core.types import PropertyPrediction, Quantity, Target
from hephaestus.harness import gates as G
from hephaestus.harness import runner
from hephaestus.harness.goal import parse_goal, validate_spec
from hephaestus.llm.scripted import ScriptedLLM
from hephaestus.tools import custom


@pytest.fixture(autouse=True)
def _cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the cache inside the test's folder.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test folder.
    monkeypatch : pytest.MonkeyPatch
        Sets ``HEPHAESTUS_CACHE_DIR``.
    """
    monkeypatch.setenv("HEPHAESTUS_CACHE_DIR", str(tmp_path / "cache"))


def two_targets() -> Any:
    """Return the demo spec (gap below 3.12 eV) plus a bulk modulus above 180 GPa.

    Returns
    -------
    SearchSpec
        Spec.
    """
    return demo_spec().model_copy(
        update={
            "extra_targets": [Target(property="bulk_modulus", value=180.0, units="GPa", direction="maximise")]
        }
    )


def with_bulk(cs: Any, k: float | None) -> Any:
    """Add a bulk modulus prediction to a candidate.

    Parameters
    ----------
    cs : CandidateState
        Candidate.
    k : float or None
        Bulk modulus in GPa; None for no prediction.

    Returns
    -------
    CandidateState
        The candidate.
    """
    if k is not None:
        q = Quantity(value=k, uncertainty=15.0, units="GPa", source="m")
        cs.predictions["bulk_modulus"] = PropertyPrediction(
            candidate_id=cs.candidate.candidate_id, property="bulk_modulus", value=q, model="m"
        )
    return cs


def test_every_target_must_be_met() -> None:
    """Gap met but bulk modulus not: the target check fails and names both rows."""
    spec = two_targets()
    c = G.check_target(with_bulk(candidate_state(gap=2.0), 150.0), spec)
    assert not c["target_met_central"] and not c["all_passed"]
    assert [(r["property"], r["met_central"]) for r in c["targets"]] == [
        ("band_gap", True),
        ("bulk_modulus", False),
    ]
    assert G.check_target(with_bulk(candidate_state(gap=2.0), 220.0), spec)["target_met_conservative"]
    assert "bulk_modulus" not in G.check_target(candidate_state(gap=2.0), demo_spec())  # one target: no rows


def test_missing_steps_and_gate_name_each_target() -> None:
    """The gate asks for the missing prediction, then names the unmet target."""
    ws = workspace()
    ws.spec = two_targets()
    ws.candidates["c1-1"] = candidate_state(gap=2.0)
    assert "predict_property(bulk_modulus)" in G.missing_steps(ws.candidates["c1-1"], ws.spec)
    with_bulk(ws.candidates["c1-1"], 150.0)
    passed, reasons = G.submission_gate(ws, "c1-1")
    assert not passed and reasons == ["bulk_modulus target not met"]
    with_bulk(ws.candidates["c1-1"], 220.0)
    assert G.submission_gate(ws, "c1-1") == (True, [])


def test_rubric_averages_the_targets() -> None:
    """Property points are averaged: one target met conservatively, one not, gives 15 of 30."""
    _, breakdown = G.score_rubric(with_bulk(candidate_state(gap=2.0), 150.0), two_targets(), 0, True)
    assert breakdown["property"] == 15


def test_best_pick_uses_all_targets() -> None:
    """With equal checks, the candidate nearer every target (relative distance) wins."""
    ws = workspace()
    ws.spec = two_targets()
    ws.candidates["c1-1"] = with_bulk(candidate_state("c1-1", gap=1.0), 150.0)  # far below on gap, misses K
    ws.candidates["c1-2"] = with_bulk(candidate_state("c1-2", gap=2.9), 230.0)  # meets both
    assert G.select_best([G.facts(ws, c) for c in ws.candidates], ws.spec) == "c1-2"


def test_parser_reads_extra_targets_and_validates_them() -> None:
    """A second target from the parser is kept, checked against the registry and needs a value or reference."""
    raw = scripts.PARSED | {
        "extra_targets": [
            {
                "target_property": "bulk_modulus",
                "property_name": "bulk modulus",
                "target_value": None,
                "target_units": "GPa",
                "direction": "maximise",
                "target_tolerance": None,
                "reference_material": "SrTiO3",
            }
        ]
    }
    spec, _, errors = parse_goal(ScriptedLLM(structured={"goal_parser": [raw]}), scripts.GOAL, RunConfig())
    assert (
        spec is not None
        and errors == []
        and [t.property for t in spec.targets()] == ["band_gap", "bulk_modulus"]
    )
    bad = spec.model_copy(
        update={"extra_targets": [Target(property="bulk_modulus", units="eV", direction="maximise")]}
    )
    problems = validate_spec(bad)
    assert any("Units 'eV' do not match 'GPa'" in p for p in problems)
    assert any("for bulk_modulus: ask the user" in p for p in problems)


GOAL = "a lead-free perovskite with a smaller band gap and lower thermal expansion than SrTiO3, stable"
PARSED_TWO = scripts.PARSED | {
    "extra_targets": [
        {
            "target_property": "unsupported",
            "property_name": "thermal expansion",
            "target_value": None,
            "target_units": "1e-6/K",
            "direction": "minimise",
            "target_tolerance": None,
            "reference_material": "SrTiO3",
        }
    ]
}


def test_two_targets_end_to_end_with_an_agent_built_second_target(tmp_path: Path) -> None:
    """Band gap (built in) and thermal expansion (library tool): both references, both checked, both in the brief."""
    custom.save_tool(tmp_path / "lib", "toy_expansion", TOOL_SOURCE, TEST_SOURCE)
    turns = campaign_turns()
    turns[2] = [
        *turns[2],
        {"tool": "predict_property", "input": {"candidate_id": "c1-1", "property": "band_gap"}},
    ]
    d = deps(tmp_path, turns, None, reviews=[scripts.review(scripts.FE2, 90)])
    d.llm.answers["goal_parser"] = [PARSED_TWO]
    state = runner.run(GOAL, cfg=cfg(tmp_path), deps=d, run_id="r1")
    assert state.status == "finished", state.errors
    gap, tex = state.spec.targets()
    assert (
        gap.value == pytest.approx(3.12)
        and tex.value == pytest.approx(10.0)
        and tex.tool == "custom_toy_expansion"
    )
    assert state.submission and state.submission.gate_passed  # FE1: gap 2.6 < 3.12 and 8 < 10
    md = (tmp_path / "r1" / "brief.md").read_text(encoding="utf-8")
    assert "Thermal expansion" in md and "(agent-built tool)" in md
    preds = {p.property for p in state.iterations[0].predictions}
    assert preds == {"band_gap", GAP}


def test_a_second_target_without_a_tool_is_not_evaluated_and_cannot_be_met(tmp_path: Path) -> None:
    """Cancelling the tool for the second target leaves the first one checked and the gate failing on the second."""
    from test_custom_tools import decider, plan

    turns = [[{"tool": "propose_tool", "input": plan()}], *campaign_turns()]
    turns[3] = [
        *turns[3],
        {"tool": "predict_property", "input": {"candidate_id": "c1-1", "property": "band_gap"}},
    ]
    turns[-1] = [scripts.submit("c1-1", accept_not_met=True)]
    turns[3] = [s for s in turns[3] if not s.get("tool", "").startswith("custom_")]
    d = deps(tmp_path, turns, decider("cancel_tool")[0], reviews=[scripts.review(scripts.FE2, 90)])
    d.llm.answers["goal_parser"] = [PARSED_TWO]
    d.toolkit_factory = fake_factory
    state = runner.run(GOAL, cfg=cfg(tmp_path), deps=d, run_id="r1")
    assert state.status == "finished", state.errors
    gap, tex = state.spec.targets()
    assert gap.evaluable and gap.value == pytest.approx(3.12) and not tex.evaluable
    assert state.stop_reason != "target_met" and state.submission and not state.submission.gate_passed
    assert (tmp_path / "r1" / "capability_thermal_expansion" / "state.json").exists()
