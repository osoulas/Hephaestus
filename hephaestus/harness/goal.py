"""Goal parser: free text to ``SearchSpec`` (Section 1).

One structured LLM call drafts the spec and up to three clarifying questions; code
then expands implied constraints and validates the result.
"""

from __future__ import annotations

import json
import re
from typing import Any

from hephaestus.core.config import RunConfig
from hephaestus.core.types import Conditions, ErrorRecord, SearchSpec, StabilityThresholds
from hephaestus.llm.base import LLM, LLMError

TOXIC = {"lead": "Pb", "cadmium": "Cd", "mercury": "Hg", "arsenic": "As", "thallium": "Tl"}
STABILITY_CONSTRAINTS = ("thermodynamically stable", "dynamically stable")
NOVELTY_CONSTRAINT = "novel: not an experimentally known compound and not previously reported"

PARSER_SYSTEM = """\
You turn a materials research goal into a structured search specification. Extract only what \
the goal says or clearly implies; never invent a target value. If the goal compares with a \
known material ("smaller band gap than SrTiO3"), set reference_material and leave target_value \
null: code computes the reference with the same model as the candidates. Map the property to \
one of the supported keys, or to "unsupported" if none fits. Separate hard constraints (must \
hold) from soft preferences. Record every assumption in plain words. Ask at most three \
clarifying questions, only when a required field (target property, value or reference, \
direction) is missing or ambiguous. Give the run a short title (at most eight words, sentence \
case, no full stop) that names the material family and the target, e.g. "Lead-free perovskite \
with a smaller gap than SrTiO3"."""
TITLE_MAX = 70


def property_registry(fidelity: str = "hse") -> dict[str, dict[str, Any]]:
    """Return the property registry (PP-1).

    Parameters
    ----------
    fidelity : {"hse", "pbe"}
        Band gap fidelity.

    Returns
    -------
    dict
        Property key to ``{units, published_error, ...}``.
    """
    from hephaestus.tools.simulation import property_registry as registry

    return registry(fidelity)


def goal_schema(keys: list[str]) -> dict[str, Any]:
    """JSON schema of the parser's answer.

    Parameters
    ----------
    keys : list of str
        Supported property keys.

    Returns
    -------
    dict
        Schema.
    """
    s, sn, nn = {"type": "string"}, {"type": ["string", "null"]}, {"type": ["number", "null"]}
    sa = {"type": "array", "items": {"type": "string"}}
    props = {
        "title": s,
        "material_family": s,
        "elements": sa,
        "excluded_elements": sa,
        "target_property": {"type": "string", "enum": [*keys, "unsupported"]},
        "target_value": nn,
        "target_units": s,
        "direction": {"type": "string", "enum": ["maximise", "minimise", "hit"]},
        "target_tolerance": nn,
        "reference_material": sn,
        "temperature_K": nn,
        "pressure_GPa": nn,
        "e_hull_max_eV_per_atom": nn,
        "hard_constraints": sa,
        "soft_preferences": sa,
        "assumptions": sa,
        "questions": sa,
    }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def parse_goal(
    llm: LLM, raw_goal: str, cfg: RunConfig, answers: dict[str, str] | None = None
) -> tuple[SearchSpec | None, list[str], list[ErrorRecord]]:
    """Draft a specification and clarifying questions from a goal (GP-1 to GP-8, GP-11 to GP-13).

    Parameters
    ----------
    llm : LLM
        Model for the structured call.
    raw_goal : str
        The user's goal.
    cfg : RunConfig
        Novelty and fidelity options.
    answers : dict, optional
        Answers to earlier clarifying questions, merged into the goal.

    Returns
    -------
    spec : SearchSpec or None
        Expanded draft; ``None`` when the LLM failed.
    questions : list of str
        At most three; empty once answered.
    errors : list of ErrorRecord
        Parser failures and validation problems.
    """
    registry = property_registry(cfg.band_gap_fidelity)
    prompt = json.dumps(
        {
            "goal": raw_goal,
            "answers": answers or {},
            "supported_properties": {k: v.get("units") for k, v in registry.items()},
        },
        ensure_ascii=False,
    )
    try:
        raw = llm.structured("goal_parser", PARSER_SYSTEM, prompt, goal_schema(list(registry)))
    except LLMError as exc:
        return None, [], [ErrorRecord(step="goal_parser", code="llm_failed", message=str(exc)[:500])]
    conditions = Conditions(
        temperature_K=raw.get("temperature_K") or 300.0,
        pressure_GPa=raw.get("pressure_GPa") or 0.0001,
    )
    spec = SearchSpec(
        raw_goal=raw_goal,
        title=run_title(raw.get("title"), raw_goal),
        material_family=raw.get("material_family") or "",
        elements=list(raw.get("elements") or []),
        excluded_elements=list(raw.get("excluded_elements") or []),
        target_property=raw.get("target_property") or "unsupported",
        target_value=raw.get("target_value"),
        target_units=raw.get("target_units") or "",
        direction=raw.get("direction") or "minimise",
        target_tolerance=raw.get("target_tolerance"),
        reference_material=raw.get("reference_material"),
        conditions=conditions,
        hard_constraints=list(raw.get("hard_constraints") or []),
        soft_preferences=list(raw.get("soft_preferences") or []),
        stability_thresholds=StabilityThresholds(
            e_hull_max_eV_per_atom=raw.get("e_hull_max_eV_per_atom") or 0.05,
        ),
        assumptions=list(raw.get("assumptions") or []),
    )
    if raw.get("temperature_K") is None or raw.get("pressure_GPa") is None:
        spec.assumptions.append("Operating conditions default to 300 K and ambient pressure (GP-7).")
    spec = expand_constraints(spec, cfg)
    questions = [] if answers else list(raw.get("questions") or [])[:3]
    errors = [
        ErrorRecord(step="goal_parser", code="invalid_spec", message=m) for m in validate_spec(spec, registry)
    ]
    return spec, questions, errors


def run_title(proposed: str | None, raw_goal: str) -> str:
    """Return the run's title: the parser's proposal, tidied, else the start of the goal.

    Parameters
    ----------
    proposed : str or None
        Title from the LLM.
    raw_goal : str
        The goal, as typed.

    Returns
    -------
    str
        A title of at most ``TITLE_MAX`` characters.
    """
    title = " ".join((proposed or "").split()).strip(" .\"'")
    if not title:
        words = raw_goal.split()
        title = " ".join(words[:10])
        title = title[0].upper() + title[1:] if title else "Untitled run"
        if len(words) > 10:
            title += "…"
    if len(title) > TITLE_MAX:
        title = title[: TITLE_MAX - 1].rsplit(" ", 1)[0] + "…"
    return title


def expand_constraints(spec: SearchSpec, cfg: RunConfig) -> SearchSpec:
    """Turn wording into explicit fields (GP-4, GP-12, GP-13).

    Parameters
    ----------
    spec : SearchSpec
        Draft spec.
    cfg : RunConfig
        ``require_novel``.

    Returns
    -------
    SearchSpec
        A copy with excluded elements from "X-free"/"no X"/"without X", the two stability
        constraints and, when required, the novelty constraint.
    """
    spec = spec.model_copy(deep=True)
    text = spec.raw_goal.lower()
    for word, symbol in TOXIC.items():
        if (
            re.search(rf"\b{word}[- ]free\b|\bno {word}\b|\bwithout {word}\b", text)
            and symbol not in spec.excluded_elements
        ):
            spec.excluded_elements.append(symbol)
    spec = apply_stability_options(spec, cfg)
    if cfg.require_novel and NOVELTY_CONSTRAINT not in spec.hard_constraints:
        spec.hard_constraints.append(NOVELTY_CONSTRAINT)
    return spec


def apply_stability_options(spec: SearchSpec, cfg: RunConfig) -> SearchSpec:
    """Apply the run's stability options to a spec (SE-13, GP-12).

    The run's E_hull threshold, when set, replaces the goal's; a check turned off for the run is
    not required by the gate, and its hard constraint is replaced by an assumption.

    Parameters
    ----------
    spec : SearchSpec
        Spec from the parser, or one given by the caller.
    cfg : RunConfig
        ``run_hull``, ``e_hull_max_eV_per_atom`` and ``run_phonons``.

    Returns
    -------
    SearchSpec
        A copy with the thresholds, the stability hard constraints and assumptions updated.
    """
    spec = spec.model_copy(deep=True)
    thr = spec.stability_thresholds
    if cfg.e_hull_max_eV_per_atom is not None:
        thr.e_hull_max_eV_per_atom = cfg.e_hull_max_eV_per_atom
    thr.thermodynamic, thr.dynamic = cfg.run_hull, cfg.run_phonons
    checks = [
        (
            thr.thermodynamic,
            "thermodynamically stable",
            f"E_hull ≤ {thr.e_hull_max_eV_per_atom} eV/atom",
            "Thermodynamic stability (energy above hull) is not checked in this run.",
        ),
        (
            thr.dynamic,
            "dynamically stable",
            "no imaginary mode that lowers the energy when frozen in",
            "Dynamic stability (phonons) is not checked in this run.",
        ),
    ]
    for on, name, rule, skipped in checks:
        hc = spec.hard_constraints
        at = next((i for i, h in enumerate(hc) if h.lower().startswith(name)), None)
        if on:
            # Keep the goal's own wording in place unless the run changed the threshold
            if at is None:
                hc.append(f"{name}: {rule}")
            elif name.startswith("thermo") and cfg.e_hull_max_eV_per_atom is not None:
                hc[at] = f"{name}: {rule}"
            spec.assumptions = [a for a in spec.assumptions if a != skipped]
        else:
            spec.hard_constraints = [h for h in hc if not h.lower().startswith(name)]
            if skipped not in spec.assumptions:
                spec.assumptions.append(skipped)
    return spec


def validate_spec(spec: SearchSpec, registry: dict[str, dict[str, Any]] | None = None) -> list[str]:
    """Check a spec before a run starts (GP-2, GP-5, GP-8).

    Parameters
    ----------
    spec : SearchSpec
        Spec to check.
    registry : dict, optional
        Property registry; defaults to ``property_registry()``.

    Returns
    -------
    list of str
        Problems; empty when the spec is valid.
    """
    from pymatgen.core import Element

    reg = registry or property_registry()
    problems = []
    for sym in [*spec.elements, *spec.excluded_elements]:
        try:
            Element(sym)
        except Exception:
            problems.append(f"Unknown element symbol {sym!r}.")
    if spec.target_property not in reg:
        problems.append(f"Property {spec.target_property!r} is not supported; supported: {', '.join(reg)}.")
    else:
        units = reg[spec.target_property].get("units")
        if units and spec.target_units and spec.target_units.replace("³", "3") != units.replace("³", "3"):
            problems.append(f"Units {spec.target_units!r} do not match {units!r} for {spec.target_property}.")
    if spec.target_value is None and not spec.reference_material:
        problems.append("No target value or reference material: ask the user (GP-5).")
    if spec.direction == "hit" and spec.target_tolerance is None:
        problems.append("A 'hit' target needs a tolerance.")
    if set(spec.elements) & set(spec.excluded_elements):
        problems.append("An element is both required and excluded.")
    return problems
