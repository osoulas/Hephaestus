"""Independent reviewer: critique and LLM confidence (Section 7).

Code fixes the facts (best of round, checks, rubric); one structured LLM call adds
findings, a next composition, a stop recommendation and the reviewer's own confidence. The
answer is validated (SC-4 to SC-7, SC-9) and asked for again once; if the LLM fails, a
minimal critique with the code facts is recorded (SC-10).
"""

from __future__ import annotations

import json
from typing import Any

from hephaestus.core.types import Critique, ErrorRecord, Finding
from hephaestus.harness.prompts import REVIEW_SCHEMA, REVIEWER_SYSTEM
from hephaestus.llm.base import LLM, LLMError
from hephaestus.tools.registry import CandidateState, Workspace

WHERE = {
    "Γ": "a zone-centre mode: polar (ferroelectric) displacement or a strain instability",
    "R": "out-of-phase octahedral tilting (a⁻a⁻a⁻ type), typical when the tolerance factor is below 1",
    "M": "in-phase octahedral tilting (a⁰a⁰c⁺ type)",
    "X": "an antipolar or cooperative Jahn-Teller-like distortion",
}


def same_composition(a: str, b: str) -> bool:
    """Whether two formulae describe the same composition, independent of notation (LM-15).

    Parameters
    ----------
    a, b : str
        Formulae, e.g. ``"SrTiO3"`` and ``"Sr1Ti1O3"``.

    Returns
    -------
    bool
        True when their normalised atomic fractions agree; falls back to string equality.
    """
    from pymatgen.core import Composition

    try:
        return Composition(a).fractional_composition.almost_equals(Composition(b).fractional_composition)
    except Exception:
        return a.strip() == b.strip()


def phonon_interpretation(cs: CandidateState) -> str:
    """Physical reading of a candidate's phonon result (SC-3).

    Parameters
    ----------
    cs : CandidateState
        Candidate with phonon results.

    Returns
    -------
    str
        One or two sentences.
    """
    if cs.phonon is None:
        return "Phonons not computed."
    d = cs.phonon_details
    verdict = d.get("verdict") or ("stable" if cs.phonon.stable else "unstable")
    where = cs.phonon.instability_at
    reading = WHERE.get(where or "", f"an instability at {where}") if where else ""
    rounds = d.get("mode_mapping_rounds") or []
    if verdict == "stable":
        return f"No imaginary modes (min {cs.phonon.min_frequency_THz:.2f} THz)."
    if verdict == "spurious_soft_mode":
        dE = rounds[-1].get("dE_meV_per_atom") if rounds else None
        return (
            f"Soft mode at {where} ({reading}) is spurious: freezing it in changed the energy by "
            f"{dE} meV/atom, below the threshold."
        )
    if verdict == "stable_after_distortion":
        sg = rounds[-1].get("space_group") if rounds else cs.space_group
        return f"Real instability at {where} ({reading}); the structure relaxed into {sg} and is now stable."
    return f"Unstable: {reading or 'imaginary modes'} (min {cs.phonon.min_frequency_THz:.2f} THz) persists."


def validate_critique(
    c: Critique, previous: list[Critique], tried: list[str], known_ids: set[str]
) -> list[str]:
    """Check a critique against SC-4 to SC-7 and SC-9.

    Parameters
    ----------
    c : Critique
        The new critique.
    previous : list of Critique
        Earlier critiques, for repeated recommendations.
    tried : list of str
        Compositions already proposed in this run.
    known_ids : set of str
        Retrieved paper IDs.

    Returns
    -------
    list of str
        Problems; empty when valid.
    """
    from pymatgen.core import Composition

    problems = []
    try:
        Composition(c.next_composition)
        if not c.next_composition.strip():
            raise ValueError
    except Exception:
        problems.append(f"next_composition {c.next_composition!r} is not an exact formula (SC-6).")
    else:
        if any(same_composition(c.next_composition, t) for t in tried):
            problems.append(f"{c.next_composition} was already tried; recommend a new composition (SC-7).")
        if any(
            p.next_composition and same_composition(c.next_composition, p.next_composition) for p in previous
        ):
            problems.append(f"{c.next_composition} was already recommended (SC-7).")
    categories = {f.category for f in c.findings}
    if "reliability" not in categories:
        problems.append("Add a reliability finding on the potential and property model (SC-4).")
    if not categories & {"novelty", "synthesis"}:
        problems.append("Add a sceptic challenge on novelty or synthesis (SC-5).")
    for f in c.findings:
        if f.category == "literature" and not f.paper_ids:
            problems.append(f"Literature finding without a paper ID: {f.text[:60]!r} (SC-9).")
        unknown = [p for p in f.paper_ids if p not in known_ids]
        if unknown:
            problems.append(f"Unknown paper IDs {unknown}; cite only retrieved papers (SC-9).")
    if c.stop_recommended and not c.stop_reason:
        problems.append("stop_recommended needs a stop_reason.")
    return problems


def review_prompt(
    ws: Workspace,
    summary: str,
    round_facts: list[dict[str, Any]],
    best_id: str | None,
    score: int,
    breakdown: dict[str, int],
    novelty: dict[str, Any] | None,
    previous: list[Critique],
) -> str:
    """Build the reviewer's prompt.

    Parameters
    ----------
    ws : Workspace
        Run workspace (evidence, papers, spec).
    summary : str
        The orchestrator's summary of the round.
    round_facts : list of dict
        Code facts for this round's candidates.
    best_id : str or None
        Code-chosen best of the round.
    score : int
        Its rubric score.
    breakdown : dict
        Rubric breakdown.
    novelty : dict or None
        Novelty search for the best candidate.
    previous : list of Critique
        Earlier reviews.

    Returns
    -------
    str
        The prompt.
    """
    candidates = []
    for f in round_facts:
        cs = ws.candidates[f["candidate_id"]]
        candidates.append(
            {
                **f,
                "rationale": cs.candidate.rationale,
                "evidence_ids": cs.candidate.evidence_ids,
                "risks": cs.candidate.risks,
                "phonon_interpretation": phonon_interpretation(cs),
                "domain_note": cs.domain_note,
            }
        )
    payload = {
        "goal": ws.spec.raw_goal,
        "target": {
            "property": ws.spec.target_property,
            "direction": ws.spec.direction,
            "value": ws.spec.target_value,
            "units": ws.spec.target_units,
            "reference": ws.spec.reference_material,
        },
        "all_targets": [t.model_dump(exclude={"reference_value"}) for t in ws.spec.targets()],
        "conditions": ws.spec.conditions.model_dump(),
        "round": ws.round,
        "orchestrator_summary": summary,
        "candidates": candidates,
        "best_of_round": best_id,
        "rubric": {"score": score, "breakdown": breakdown},
        "novelty_search": novelty,
        "evidence": [i.model_dump() for i in ws.insights[-40:]],
        "papers": [
            {"paper_id": p.paper_id, "title": p.title, "year": p.year} for p in list(ws.papers.values())[-40:]
        ],
        "tried_compositions": [cs.candidate.composition for cs in ws.candidates.values()],
        "agent_built_tool": agent_tool_review(ws),
        "earlier_reviews": [
            {
                "round": c.iteration,
                "best": c.candidate_id,
                "score": c.score,
                "next": c.next_composition,
                "confidence": c.llm_confidence,
            }
            for c in previous
        ],
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def review_round(
    llm: LLM | None,
    ws: Workspace,
    summary: str,
    round_facts: list[dict[str, Any]],
    best_id: str | None,
    checks: dict[str, bool],
    score: int,
    breakdown: dict[str, int],
    novelty: dict[str, Any] | None,
    previous: list[Critique],
) -> tuple[Critique, list[ErrorRecord]]:
    """Review one round: code facts plus one validated LLM critique (Section 7).

    Parameters
    ----------
    llm : LLM or None
        Reviewer model; ``None`` records a minimal critique (SC-10).
    ws : Workspace
        Run workspace.
    summary : str
        Orchestrator's summary.
    round_facts : list of dict
        Code facts for this round's candidates.
    best_id : str or None
        Code-chosen best.
    checks : dict
        ``check_target`` for the best candidate.
    score : int
        Rubric score.
    breakdown : dict
        Rubric breakdown.
    novelty : dict or None
        Novelty search result.
    previous : list of Critique
        Earlier critiques.

    Returns
    -------
    critique : Critique
        The critique; code fields always come from code.
    errors : list of ErrorRecord
        Reviewer failures or validation problems that remained.
    """
    base = {
        "iteration": ws.round,
        "candidate_id": best_id,
        "target_met_central": checks.get("target_met_central", False),
        "target_met_conservative": checks.get("target_met_conservative", False),
        "thermo_stable": checks.get("thermo_stable", False),
        "dyn_stable": checks.get("dyn_stable", False),
        "score": score,
        "score_breakdown": breakdown,
    }
    errors: list[ErrorRecord] = []
    if llm is None:
        return _minimal(base, "no reviewer model"), errors
    prompt = review_prompt(ws, summary, round_facts, best_id, score, breakdown, novelty, previous)
    tried = [cs.candidate.composition for cs in ws.candidates.values()]
    known = set(ws.papers)
    critique: Critique | None = None
    problems: list[str] = []
    for attempt in range(2):
        ask = (
            prompt
            if attempt == 0
            else prompt + "\n\nYour previous answer had problems; fix them:\n- " + "\n- ".join(problems)
        )
        try:
            raw = llm.structured("reviewer", REVIEWER_SYSTEM, ask, REVIEW_SCHEMA)
            critique = Critique(
                **base,
                findings=[Finding(**f) for f in raw.get("findings", [])],
                next_composition=str(raw.get("next_composition", "")),
                next_rationale=str(raw.get("next_rationale", "")),
                stop_recommended=bool(raw.get("stop_recommended", False)),
                stop_reason=raw.get("stop_reason"),
                llm_confidence=_clamp(raw.get("llm_confidence")),
                llm_confidence_reason=raw.get("llm_confidence_reason"),
            )
        except (LLMError, ValueError, TypeError) as exc:
            errors.append(ErrorRecord(step="reviewer", code="llm_failed", message=str(exc)[:500]))
            if critique is None and attempt == 1:
                return _minimal(base, str(exc)), errors
            continue
        problems = validate_critique(critique, previous, tried, known)
        if not problems:
            return critique, errors
    if critique is None:
        return _minimal(base, "reviewer failed twice"), errors
    errors.append(ErrorRecord(step="reviewer", code="invalid_critique", message="; ".join(problems)[:500]))
    return critique, errors


def _clamp(value: Any) -> int | None:
    """Clamp a confidence to 0..100, or ``None`` when missing.

    Parameters
    ----------
    value : Any
        Raw confidence.

    Returns
    -------
    int or None
        Clamped integer.
    """
    if value is None:
        return None
    return max(0, min(100, int(value)))


def _minimal(base: dict[str, Any], reason: str) -> Critique:
    """Build the critique recorded when the reviewer is unavailable (SC-10).

    Parameters
    ----------
    base : dict
        Code fields.
    reason : str
        Why the review is missing.

    Returns
    -------
    Critique
        Code checks and score only; no reviewer confidence, so "target met" cannot fire.
    """
    note = Finding(severity="info", category="other", text=f"Review unavailable: {reason[:200]}")
    return Critique(**base, findings=[note], llm_confidence=None, llm_confidence_reason=None)


def agent_tool_review(ws: Workspace) -> dict[str, Any] | None:
    """Return the agent-built tool behind the target, with its source, for the reviewer (AT-17).

    Parameters
    ----------
    ws : Workspace
        Run workspace.

    Returns
    -------
    dict or None
        ``{name, property, units, note, source}``; ``None`` when the target uses a built-in model.
    """
    from hephaestus.tools.custom import load_sources

    name = (ws.spec.target_tool or "").removeprefix("custom_")
    tool = ws.custom_tools.get(name)
    if tool is None:
        return None
    try:
        source = load_sources(tool)[0]
    except Exception as exc:
        source = f"(source unavailable: {exc})"
    return {
        "name": tool.name,
        "property": tool.property,
        "units": tool.units,
        "note": "Written by the agent and approved by the user; its values carry no uncertainty. Judge the method.",
        "source": source[:12000],
    }
