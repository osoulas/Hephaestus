"""Code-only verdicts: target check, rubric, best pick, stop rules, submission gate."""

import pytest
from factories import candidate_state, demo_spec, workspace

from hephaestus.core.config import RunConfig
from hephaestus.core.types import Critique, Finding, Insight
from hephaestus.harness import gates as G


def critique(score: int = 80, conf: int | None = 80, critical: bool = False, stop: bool = False) -> Critique:
    """Return a critique with the given score, confidence and findings.

    Parameters
    ----------
    score : int
        Code rubric score.
    conf : int or None
        Reviewer confidence.
    critical : bool
        Add a critical finding.
    stop : bool
        Reviewer recommends stopping, with a reason.

    Returns
    -------
    Critique
        The critique.
    """
    findings = (
        [Finding(severity="critical", category="reliability", text="gap is model artefact")]
        if critical
        else []
    )
    return Critique(
        iteration=1,
        candidate_id="c1-1",
        score=score,
        llm_confidence=conf,
        findings=findings,
        stop_recommended=stop,
        stop_reason="done" if stop else None,
    )


PASS = {"all_passed": True, "dyn_stable": True}
FAIL_DYN = {"all_passed": False, "dyn_stable": False}


def test_sc1_sc2_target_check_central_and_conservative() -> None:
    """2.0 ± 0.65 < 3.12 on both readings; 2.8 ± 0.65 only on the central value."""
    spec = demo_spec()
    assert G.check_target(candidate_state(gap=2.0), spec)["target_met_conservative"]
    c = G.check_target(candidate_state(gap=2.8), spec)
    assert c["target_met_central"] and not c["target_met_conservative"] and c["all_passed"]


def test_sc1_missing_results_never_pass() -> None:
    """No prediction, no hull or no phonons means the check fails."""
    spec = demo_spec()
    for kwargs in ({"gap": None}, {"e_hull": None}, {"dyn": None}):
        assert not G.check_target(candidate_state(**kwargs), spec)["all_passed"]  # type: ignore[arg-type]


def test_se13_thermo_threshold_and_stale_hull() -> None:
    """E_hull 0.06 fails at 0.05; a stale hull fails regardless."""
    spec = demo_spec()
    assert not G.check_target(candidate_state(e_hull=0.06), spec)["thermo_stable"]
    cs = candidate_state(e_hull=0.0)
    cs.hull_stale = True
    assert not G.check_target(cs, spec)["thermo_stable"]


def test_target_directions() -> None:
    """Maximise and hit behave as documented."""
    spec = demo_spec().model_copy(update={"direction": "maximise", "target_value": 100.0})
    assert G.target_met(101, spec) and not G.target_met(100, spec)
    spec = spec.model_copy(update={"direction": "hit", "target_tolerance": 0.5})
    assert G.target_met(100.4, spec) and not G.target_met(100.6, spec)


def test_sc8_rubric_full_marks_and_sum() -> None:
    """A perfect candidate scores 100 and the score equals the breakdown's sum."""
    score, breakdown = G.score_rubric(candidate_state(gap=2.0, e_hull=0.01), demo_spec(), 0, True)
    assert score == 100 == sum(breakdown.values())


def test_sc8_rubric_partial() -> None:
    """Central-only target, E_hull between half and full threshold, failed novelty search."""
    score, b = G.score_rubric(candidate_state(gap=2.8, e_hull=0.04), demo_spec(), None, False)
    assert b == {"property": 15, "thermo": 10, "dynamic": 20, "reliability": 15, "novelty_synthesis": 0}
    assert score == 60


def test_select_best_prefers_checks_then_distance() -> None:
    """More checks passed beats closer to target; ties go to the smaller gap."""
    ws = workspace()
    ws.candidates = {
        "c1-1": candidate_state("c1-1", gap=1.0, dyn=False),
        "c1-2": candidate_state("c1-2", gap=2.5),
        "c1-3": candidate_state("c1-3", gap=2.0),
    }
    assert G.select_best([G.facts(ws, c) for c in ws.candidates], ws.spec) == "c1-3"


def test_synthesis_route_matches_parent_composition() -> None:
    """A route for SrTiO3 counts for Fe-doped SrTiO3, not for BaSnO3."""
    ins = [
        Insight(
            composition="SrTiO3", property="synthesis", finding_type="synthesis", summary="s", paper_id="p"
        )
    ]
    assert G.has_synthesis_route("SrTi0.875Fe0.125O3", ins)
    assert not G.has_synthesis_route("BaSnO3", ins)


@pytest.mark.parametrize(
    ("crit", "expected_note"),
    [
        (critique(conf=60), "reviewer confidence 60"),
        (critique(conf=None), "reviewer confidence None"),
        (critique(score=70), "code rubric 70"),
        (critique(critical=True), "critical finding"),
    ],
)
def test_or2_or12_target_met_blocked(crit: Critique, expected_note: str) -> None:
    """Passing code checks do not stop the run without all four parts (OR-2, OR-12).

    Parameters
    ----------
    crit : Critique
        Critique that should block.
    expected_note : str
        Text expected in the gate note.
    """
    d = G.should_stop([crit], [PASS], RunConfig(max_iterations=5), 0)
    assert d.rule is None
    assert d.gate_note is not None and expected_note in d.gate_note


def test_or2_target_met_fires() -> None:
    """All four parts hold: the run stops with "target_met"."""
    assert G.should_stop([critique()], [PASS], RunConfig(max_iterations=5), 0).rule == "target_met"


def test_or2_rule_order() -> None:
    """User stop beats max rounds beats target met."""
    cfg = RunConfig(max_iterations=1)
    assert G.should_stop([critique()], [PASS], cfg, 0, user_stop=True).rule == "user_stop"
    assert G.should_stop([critique()], [PASS], cfg, 0).rule == "max_rounds"


def test_or2_remaining_rules() -> None:
    """Reviewer stop, plateau, persistent instability and wall time each fire."""
    cfg = RunConfig(max_iterations=10, time_budget_s=100)
    assert G.should_stop([critique(score=10, stop=True)], [FAIL_DYN], cfg, 0).rule == "reviewer_stop"
    three = [critique(score=s) for s in (50, 52, 53)]
    assert G.should_stop(three, [PASS, PASS, FAIL_DYN], cfg, 0).rule == "plateau"
    two = [critique(score=s) for s in (20, 50)]
    assert G.should_stop(two, [FAIL_DYN, FAIL_DYN], cfg, 0).rule == "persistent_instability"
    assert G.should_stop([critique(score=10)], [FAIL_DYN], cfg, 100).rule == "wall_time"
    assert G.should_stop(two, [PASS | {"all_passed": False}, FAIL_DYN], cfg, 0).rule is None


def test_submit_gate_refuses_bad_submission() -> None:
    """Missing calculations and failed checks are listed as reasons."""
    ws = workspace()
    ws.candidates["c1-1"] = candidate_state(dyn=None, e_hull=0.2)
    passed, reasons = G.submission_gate(ws, "c1-1")
    assert not passed
    assert "missing: phonons" in reasons and "not thermodynamically stable" in reasons


def test_gp13_submit_gate_novelty() -> None:
    """With novelty required, prior reports or a failed search block submission."""
    ws = workspace(require_novel=True)
    ws.candidates["c1-1"] = candidate_state()
    assert G.submission_gate(ws, "c1-1", novelty_hits=0) == (True, [])
    assert not G.submission_gate(ws, "c1-1", novelty_hits=3)[0]
    assert not G.submission_gate(ws, "c1-1", novelty_hits=None)[0]
