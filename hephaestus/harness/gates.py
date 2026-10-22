"""Target check, rubric, stop rules and submission gate; code only (SC-1, SC-8, OR-2).

Nothing here calls an LLM. The LLM reads these results and cannot override them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from hephaestus.core.config import RunConfig, budget_limit
from hephaestus.core.types import Critique, Insight, Quantity, SearchSpec
from hephaestus.tools.registry import CandidateState, Workspace

STOP_RULES = [
    "user_stop",
    "max_rounds",
    "target_met",
    "reviewer_stop",
    "plateau",
    "persistent_instability",
    "wall_time",
]
PLATEAU_POINTS = 5
PLATEAU_ROUNDS = 3
RUBRIC_MAX = {"property": 30, "thermo": 20, "dynamic": 20, "reliability": 15, "novelty_synthesis": 15}


def target_met(value: float, spec: SearchSpec) -> bool:
    """Whether a value meets the property target on its own (no uncertainty).

    Parameters
    ----------
    value : float
        Predicted value, in ``spec.target_units``.
    spec : SearchSpec
        Target value and direction. ``hit`` uses ``target_tolerance`` (default 0).

    Returns
    -------
    bool
        Strictly below (minimise), strictly above (maximise), or within tolerance (hit).
    """
    if spec.target_value is None:
        return False
    t = spec.target_value
    if spec.direction == "minimise":
        return value < t
    if spec.direction == "maximise":
        return value > t
    return abs(value - t) <= (spec.target_tolerance or 0.0)


def pessimistic(q: Quantity, spec: SearchSpec) -> float:
    """Shift a value by its uncertainty in the unfavourable direction (SC-2).

    Parameters
    ----------
    q : Quantity
        Prediction with uncertainty; ``None`` uncertainty counts as zero.
    spec : SearchSpec
        Direction of the target.

    Returns
    -------
    float
        ``value + u`` when minimising, ``value - u`` when maximising; for ``hit``, the value
        moved ``u`` further from the target.
    """
    u = q.uncertainty or 0.0
    if spec.direction == "minimise":
        return q.value + u
    if spec.direction == "maximise":
        return q.value - u
    t = spec.target_value or 0.0
    return q.value + u if q.value >= t else q.value - u


def distance_to_target(value: float | None, spec: SearchSpec) -> float:
    """Signed distance to the target; lower is better, negative means past it.

    Parameters
    ----------
    value : float or None
        Predicted value; ``None`` is infinitely far.
    spec : SearchSpec
        Target value and direction.

    Returns
    -------
    float
        ``value - target`` (minimise), ``target - value`` (maximise), ``|value - target|`` (hit).
    """
    if value is None or spec.target_value is None:
        return math.inf
    t = spec.target_value
    if spec.direction == "minimise":
        return value - t
    if spec.direction == "maximise":
        return t - value
    return abs(value - t)


def check_target(cs: CandidateState, spec: SearchSpec) -> dict[str, bool | None]:
    """Code target check: property on two readings plus both stability verdicts (SC-1, SC-2, SE-13).

    Parameters
    ----------
    cs : CandidateState
        Candidate state.
    spec : SearchSpec
        Target and stability thresholds.

    Returns
    -------
    dict
        ``target_met_central``, ``target_met_conservative``, ``thermo_stable``, ``dyn_stable``,
        ``all_passed``. Missing results count as not passed; a stability check turned off for
        the run is ``None`` and does not count towards ``all_passed``.
    """
    rows = target_rows(cs, spec)
    central = all(r["met_central"] for r in rows)
    conservative = all(r["met_conservative"] for r in rows)
    st = spec.stability_thresholds
    thermo = cs.e_hull is not None and not cs.hull_stale and cs.e_hull <= st.e_hull_max_eV_per_atom
    dyn = cs.phonon is not None and cs.phonon.stable
    out: dict[str, Any] = {
        "target_met_central": central,
        "target_met_conservative": conservative,
        "thermo_stable": thermo if st.thermodynamic else None,
        "dyn_stable": dyn if st.dynamic else None,
        "all_passed": central and (thermo or not st.thermodynamic) and (dyn or not st.dynamic),
    }
    if len(rows) > 1:
        out["targets"] = rows
    return out


def target_rows(cs: CandidateState, spec: SearchSpec) -> list[dict[str, Any]]:
    """Check every property target on its own; all must be met (central and conservative readings).

    Parameters
    ----------
    cs : CandidateState
        Candidate state.
    spec : SearchSpec
        Targets.

    Returns
    -------
    list of dict
        Per target: ``property``, ``value``, ``uncertainty``, ``units``, ``met_central``,
        ``met_conservative``, ``evaluable`` and ``agent_built``. A target with no prediction, or
        that cannot be evaluated in this run, is not met.
    """
    rows = []
    for i, t in enumerate(spec.targets()):
        view = spec.target_view(i)
        pred = cs.predictions.get(t.property) if t.evaluable else None
        rows.append(
            {
                "property": t.property,
                "value": pred.value.value if pred else None,
                "uncertainty": pred.value.uncertainty if pred else None,
                "units": pred.value.units if pred else t.units,
                "target": t.value,
                "direction": t.direction,
                "met_central": pred is not None and target_met(pred.value.value, view),
                # An agent-built tool has no uncertainty, so both readings agree (AT-17)
                "met_conservative": pred is not None and target_met(pessimistic(pred.value, view), view),
                "evaluable": t.evaluable,
                "agent_built": bool(pred and pred.model.startswith("agent-built:")),
            }
        )
    return rows


def missing_steps(cs: CandidateState, spec: SearchSpec) -> list[str]:
    """List the calculations the submission gate still needs for a candidate.

    Parameters
    ----------
    cs : CandidateState
        Candidate state.
    spec : SearchSpec
        Gives the target property.

    Returns
    -------
    list of str
        Tool names still to run, in order.
    """
    out = []
    if cs.record is None:
        out.append("build_structure")
    if cs.energy_eV_per_atom is None:
        out.append("relax")
    st = spec.stability_thresholds
    if st.thermodynamic and (cs.e_hull is None or cs.hull_stale):
        out.append("energy_above_hull")
    if st.dynamic and cs.phonon is None:
        out.append("phonons")
    for t in spec.targets():
        if t.evaluable and t.property not in cs.predictions:
            out.append(t.tool or f"predict_property({t.property})")
    return out


def facts(ws: Workspace, candidate_id: str) -> dict[str, Any]:
    """Code-computed facts for one candidate, as used by the gate, the reviewer and the UI.

    Parameters
    ----------
    ws : Workspace
        The run's workspace.
    candidate_id : str
        Candidate ID.

    Returns
    -------
    dict
        Composition, round, key results, the target check and what is missing.
    """
    cs = ws.get(candidate_id)
    pred = cs.predictions.get(ws.spec.target_property)
    checks = check_target(cs, ws.spec)
    return {
        "candidate_id": candidate_id,
        "composition": cs.candidate.composition,
        "round": cs.round,
        "source": cs.record.source if cs.record else None,
        "converged": cs.converged,
        "space_group": cs.space_group,
        "e_hull": cs.e_hull,
        "hull_stale": cs.hull_stale,
        "phonon_verdict": cs.phonon_details.get("verdict"),
        "min_freq": cs.phonon.min_frequency_THz if cs.phonon else None,
        "value": pred.value.value if pred else None,
        "uncertainty": pred.value.uncertainty if pred else None,
        "units": pred.value.units if pred else ws.spec.target_units,
        "in_domain": pred.in_domain if pred else None,
        "agent_built": bool(pred and pred.model.startswith("agent-built:")),
        **checks,
        "distance": _finite(distance_all(cs, ws.spec)),  # None when a target has no value (JSON has no inf)
        "missing": missing_steps(cs, ws.spec),
    }


def _rank_key(f: dict[str, Any], spec: SearchSpec) -> tuple[Any, ...]:
    """Sort key: most checks passed first, then closest to target, then ID.

    Parameters
    ----------
    f : dict
        Output of ``facts``.
    spec : SearchSpec
        Target.

    Returns
    -------
    tuple
        Ascending sort key.
    """
    passed = sum(bool(f[k]) for k in ("target_met_central", "thermo_stable", "dyn_stable"))
    distance = f.get("distance", distance_to_target(f["value"], spec))
    distance = math.inf if distance is None else distance
    return (-passed, distance, f["candidate_id"])


def _finite(x: float) -> float | None:
    """Return ``x``, or ``None`` when it is not finite.

    Parameters
    ----------
    x : float
        Number.

    Returns
    -------
    float or None
        ``x`` or ``None``.
    """
    return x if math.isfinite(x) else None


def distance_all(cs: CandidateState, spec: SearchSpec) -> float:
    """Distance to every target at once; lower is better (ranking only).

    One target: the plain distance, in its units. Several: the sum of distances relative to each
    target value, so properties in different units can be added.

    Parameters
    ----------
    cs : CandidateState
        Candidate state.
    spec : SearchSpec
        Targets.

    Returns
    -------
    float
        ``inf`` when any target has no prediction.
    """
    targets = spec.targets()
    total = 0.0
    for i, t in enumerate(targets):
        pred = cs.predictions.get(t.property) if t.evaluable else None
        d = distance_to_target(pred.value.value if pred else None, spec.target_view(i))
        if len(targets) > 1 and math.isfinite(d):
            d /= max(abs(t.value or 0.0), 1e-9)
        total += d
    return total


def select_best(all_facts: list[dict[str, Any]], spec: SearchSpec) -> str | None:
    """Pick the best candidate: most checks passed, then closest to target (OR-3, Section 7).

    Parameters
    ----------
    all_facts : list of dict
        Facts of the candidates to choose from.
    spec : SearchSpec
        Target.

    Returns
    -------
    str or None
        Candidate ID, or ``None`` for an empty list.
    """
    if not all_facts:
        return None
    return min(all_facts, key=lambda f: _rank_key(f, spec))["candidate_id"]


def has_synthesis_route(composition: str, insights: list[Insight]) -> bool:
    """Whether the literature gives a synthesis route for this composition or its parent.

    A synthesis insight counts when its composition's elements are a subset of the candidate's,
    so a route for SrTiO3 counts for SrTi0.875Fe0.125O3.

    Parameters
    ----------
    composition : str
        Candidate formula.
    insights : list of Insight
        Recorded evidence.

    Returns
    -------
    bool
        True if any matching synthesis insight exists.
    """
    elements = _elements(composition)
    for ins in insights:
        if ins.finding_type == "synthesis" and ins.composition:
            other = _elements(ins.composition)
            if other and other <= elements:
                return True
    return False


def _elements(formula: str) -> set[str]:
    """Element symbols in a formula, or an empty set when it does not parse.

    Parameters
    ----------
    formula : str
        Formula string.

    Returns
    -------
    set of str
        Element symbols.
    """
    from pymatgen.core import Composition

    try:
        return {el.symbol for el in Composition(formula).elements}
    except Exception:
        return set()


def score_rubric(
    cs: CandidateState,
    spec: SearchSpec,
    novelty_hits: int | None,
    synthesis_route: bool,
) -> tuple[int, dict[str, int]]:
    """Confidence rubric, in code (SC-8).

    Property 30 (30 conservative, 15 central only); thermodynamic 20 (20 at half the E_hull
    threshold, 10 at the threshold); dynamic 20; reliability 15 (5 each: converged, potential
    domain, property-model domain); novelty and synthesis 15 (8 no prior report, 7 route).
    A stability check turned off for the run is left out and the score rescaled to 0 to 100.

    Parameters
    ----------
    cs : CandidateState
        The scored candidate.
    spec : SearchSpec
        Target and thresholds.
    novelty_hits : int or None
        Prior reports of the exact composition; ``None`` when the search failed (no points).
    synthesis_route : bool
        Whether the literature gives a plausible route.

    Returns
    -------
    score : int
        Sum of the breakdown, 0 to 100.
    breakdown : dict
        Points per component.
    """
    checks = check_target(cs, spec)
    thr = spec.stability_thresholds.e_hull_max_eV_per_atom
    # 30 per target met conservatively, 15 at the central value; averaged over the targets
    rows = target_rows(cs, spec)
    prop = round(
        sum(30 if r["met_conservative"] else 15 if r["met_central"] else 0 for r in rows) / len(rows)
    )
    thermo = 0
    if checks["thermo_stable"] and cs.e_hull is not None:
        thermo = 20 if cs.e_hull <= thr / 2 else 10
    preds = [cs.predictions.get(t.property) for t in spec.targets()]
    models_in_domain = all(p is not None and p.in_domain for p in preds)
    reliability = 5 * sum([cs.converged, cs.potential_in_domain is True, models_in_domain])
    breakdown = {
        "property": prop,
        "thermo": thermo,
        "dynamic": 20 if checks["dyn_stable"] else 0,
        "reliability": reliability,
        "novelty_synthesis": (8 if novelty_hits == 0 else 0) + (7 if synthesis_route else 0),
    }
    # A stability check turned off for the run is left out, and the rest scaled back to 100
    st = spec.stability_thresholds
    off = [k for k, on in (("thermo", st.thermodynamic), ("dynamic", st.dynamic)) if not on]
    for k in off:
        del breakdown[k]
    available = 100 - sum(RUBRIC_MAX[k] for k in off)
    return round(100 * sum(breakdown.values()) / available), breakdown


@dataclass
class StopDecision:
    """Result of the stop rules after a review (OR-2)."""

    rule: str | None
    gate_note: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def stop(self) -> bool:
        """Whether a stop rule fired.

        Returns
        -------
        bool
            True when ``rule`` is set.
        """
        return self.rule is not None


def target_met_gate(critique: Critique, all_passed: bool, threshold: int) -> tuple[bool, str | None]:
    """Apply the four-part "target met" rule and say why it was blocked (OR-2 rule 3, OR-12).

    Parameters
    ----------
    critique : Critique
        This round's critique (code score and reviewer confidence).
    all_passed : bool
        Code checks for the best candidate.
    threshold : int
        Confidence threshold for both the rubric and the reviewer.

    Returns
    -------
    met : bool
        True when all four parts hold.
    gate_note : str or None
        Why the stop was blocked although the code checks passed.
    """
    if not all_passed:
        return False, None
    reasons = []
    if critique.score < threshold:
        reasons.append(f"code rubric {critique.score} < {threshold}")
    if critique.llm_confidence is None or critique.llm_confidence < threshold:
        reasons.append(f"reviewer confidence {critique.llm_confidence} < {threshold}")
    critical = [f.text for f in critique.findings if f.severity == "critical"]
    if critical:
        reasons.append("critical finding: " + "; ".join(critical))
    if reasons:
        return False, "Code checks pass but the stop is blocked: " + ", ".join(reasons) + "."
    return True, None


def should_stop(
    critiques: list[Critique],
    best_checks: list[dict[str, bool | None]],
    cfg: RunConfig,
    elapsed_s: float,
    user_stop: bool = False,
) -> StopDecision:
    """Apply the stop rules in order after a review (OR-2, OR-9).

    Parameters
    ----------
    critiques : list of Critique
        Every review so far, oldest first; the last is this round's.
    best_checks : list of dict
        ``check_target`` output for each round's best candidate, aligned with ``critiques``.
    cfg : RunConfig
        Thresholds and budgets.
    elapsed_s : float
        Campaign wall time so far, in seconds.
    user_stop : bool
        The user asked to stop.

    Returns
    -------
    StopDecision
        The first rule that fired, or none; with a gate note when "target met" was blocked.
    """
    latest = critiques[-1]
    met, gate_note = target_met_gate(latest, best_checks[-1]["all_passed"], cfg.confidence_threshold)
    scores = [c.score for c in critiques[-PLATEAU_ROUNDS:]]
    unstable = [c["dyn_stable"] is False for c in best_checks[-2:]]
    fired = [
        ("user_stop", user_stop),
        ("max_rounds", len(critiques) >= budget_limit(cfg.max_iterations)),
        ("target_met", met),
        ("reviewer_stop", latest.stop_recommended and bool(latest.stop_reason)),
        ("plateau", len(scores) == PLATEAU_ROUNDS and max(scores) - min(scores) < PLATEAU_POINTS),
        ("persistent_instability", len(unstable) == 2 and all(unstable)),
        ("wall_time", elapsed_s >= budget_limit(cfg.time_budget_s)),
    ]
    rule = next((name for name, hit in fired if hit), None)
    return StopDecision(rule=rule, gate_note=gate_note)


def submission_gate(
    ws: Workspace, candidate_id: str, novelty_hits: int | None = None
) -> tuple[bool, list[str]]:
    """Apply the submission gate: required calculations, checks and novelty (``submit_final``).

    Parameters
    ----------
    ws : Workspace
        The run's workspace.
    candidate_id : str
        Submitted candidate.
    novelty_hits : int or None
        Prior reports of the composition; only used when ``cfg.require_novel``. ``None``
        (search failed) fails the novelty part.

    Returns
    -------
    passed : bool
        Whether every part of the gate holds.
    reasons : list of str
        Why it failed; empty when it passed.
    """
    cs = ws.get(candidate_id)
    reasons = [f"missing: {step}" for step in missing_steps(cs, ws.spec)]
    checks = check_target(cs, ws.spec)
    for key, label in [
        ("target_met_central", "property target not met"),
        ("thermo_stable", "not thermodynamically stable"),
        ("dyn_stable", "not dynamically stable"),
    ]:
        if key == "target_met_central" and len(ws.spec.targets()) > 1:
            continue  # named per target below
        if key == "target_met_central" and not ws.spec.target_evaluable:
            continue
        if checks[key] is False:  # None: the check is turned off for the run
            reasons.append(label)
    rows = target_rows(cs, ws.spec)
    if len(rows) > 1:
        for r in rows:
            if r["evaluable"] and not r["met_central"] and r["value"] is not None:
                reasons.append(f"{r['property']} target not met")
    for t in ws.spec.targets():
        if not t.evaluable:
            reasons.append(f"{t.property} was not evaluated: the user cancelled the tool for it")
    if ws.cfg.require_novel and novelty_hits != 0:
        reasons.append(
            "novelty required but the composition has prior reports"
            if novelty_hits
            else "novelty required but the novelty search failed"
        )
    return not reasons, reasons
