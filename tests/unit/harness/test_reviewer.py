"""Reviewer: validation, one retry, and the minimal critique when the LLM fails (Section 7)."""

from typing import Any

from factories import candidate_state, workspace

from hephaestus.core.types import Paper
from hephaestus.harness import gates as G
from hephaestus.harness.reviewer import phonon_interpretation, review_round, same_composition
from hephaestus.llm.scripted import ScriptedLLM


def answer(next_comp: str = "SrTi0.75Fe0.25O3", conf: int = 70, **over: Any) -> dict[str, Any]:
    """Return a valid reviewer answer.

    Parameters
    ----------
    next_comp : str
        Recommended composition.
    conf : int
        Reviewer confidence.
    **over
        Fields to override.

    Returns
    -------
    dict
        Answer matching ``REVIEW_SCHEMA``.
    """
    base = {
        "findings": [
            {"severity": "major", "category": "reliability", "text": "MEGNet error 0.65 eV.", "suggestion": "",
             "paper_ids": []},
            {"severity": "minor", "category": "synthesis", "text": "Solid-state route known.", "suggestion": "",
             "paper_ids": ["p1"]},
        ],
        "next_composition": next_comp,
        "next_rationale": "More Fe.",
        "stop_recommended": False,
        "stop_reason": None,
        "llm_confidence": conf,
        "llm_confidence_reason": "Within error bar.",
    }  # fmt: skip
    return base | over


def run(llm: ScriptedLLM | None):  # type: ignore[no-untyped-def]
    """Review a one-candidate round with the given model.

    Parameters
    ----------
    llm : ScriptedLLM or None
        Reviewer.

    Returns
    -------
    tuple
        ``(critique, errors)``.
    """
    ws = workspace()
    ws.papers["p1"] = Paper(paper_id="p1", title="t")
    ws.candidates["c1-1"] = cs = candidate_state()
    f = G.facts(ws, "c1-1")
    score, breakdown = G.score_rubric(cs, ws.spec, 0, False)
    return review_round(
        llm, ws, "summary", [f], "c1-1", G.check_target(cs, ws.spec), score, breakdown, None, []
    )


def test_sc12_valid_review_keeps_code_fields_and_confidence() -> None:
    """Code fields come from code; the reviewer's confidence and reason are recorded."""
    crit, errors = run(ScriptedLLM(structured={"reviewer": [answer()]}))
    assert errors == []
    assert crit.score == 93 and crit.target_met_conservative
    assert crit.llm_confidence == 70 and crit.llm_confidence_reason


def test_sc7_repeat_recommendation_is_asked_again_once() -> None:
    """Recommending a tried composition triggers one retry that names the problem."""
    llm = ScriptedLLM(structured={"reviewer": [answer(next_comp="SrTi0.875Fe0.125O3"), answer()]})
    crit, errors = run(llm)
    assert crit.next_composition == "SrTi0.75Fe0.25O3" and errors == []
    assert "already tried" in llm.structured_calls[1][1]


def test_sc4_sc5_sc9_problems_recorded_after_second_failure() -> None:
    """Missing reliability/sceptic findings and unknown IDs survive as an error after the retry."""
    bad = answer(findings=[{"severity": "info", "category": "literature", "text": "x", "suggestion": "",
                            "paper_ids": ["nope"]}])  # fmt: skip
    _, errors = run(ScriptedLLM(structured={"reviewer": [bad, bad]}))
    assert errors[-1].code == "invalid_critique"
    for tag in ("SC-4", "SC-5", "SC-9"):
        assert tag in errors[-1].message


def test_sc10_llm_failure_gives_minimal_critique() -> None:
    """With the reviewer down, the critique has code checks and no confidence."""
    crit, errors = run(ScriptedLLM(structured={"reviewer": [RuntimeError("down"), RuntimeError("down")]}))
    assert crit.score == 93 and crit.llm_confidence is None
    assert "Review unavailable" in crit.findings[0].text
    assert errors and errors[0].code == "llm_failed"


def test_lm15_same_composition() -> None:
    """Notation differences do not hide a repeat."""
    assert same_composition("SrTiO3", "Sr1Ti1O3")
    assert not same_composition("SrTiO3", "BaTiO3")


def test_sc3_phonon_interpretation() -> None:
    """Each phonon verdict gets a physical reading."""
    cs = candidate_state(dyn=False)
    assert "tilting" in phonon_interpretation(cs)
    cs.phonon_details = {
        "verdict": "stable_after_distortion",
        "mode_mapping_rounds": [{"space_group": "I4/mcm"}],
    }
    assert "I4/mcm" in phonon_interpretation(cs)
