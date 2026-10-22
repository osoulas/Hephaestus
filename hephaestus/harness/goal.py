"""Goal parser: free text to ``SearchSpec`` (Section 1).

One structured LLM call drafts the spec and up to three clarifying questions; code
then expands implied constraints and validates the result.
"""

from __future__ import annotations

import json
import re
from typing import Any

from hephaestus.core.config import RunConfig
from hephaestus.core.types import Conditions, ErrorRecord, SearchSpec, StabilityThresholds, Target
from hephaestus.llm.base import LLM, LLMError

TOXIC = {"lead": "Pb", "cadmium": "Cd", "mercury": "Hg", "arsenic": "As", "thallium": "Tl"}
STABILITY_CONSTRAINTS = ("thermodynamically stable", "dynamically stable")
NOVELTY_CONSTRAINT = "novel: not an experimentally known compound and not previously reported"

PARSER_SYSTEM = """\
You turn a materials research goal into a structured search specification. Extract only what \
the goal says or clearly implies; never invent a target value. If the goal compares with a \
known material ("smaller band gap than SrTiO3"), set reference_material and leave target_value \
null: code computes the reference with the same model as the candidates. Map the property to \
one of the supported keys, or to "unsupported" if none fits; either way give property_name, \
the property asked for as a short snake_case name (e.g. "thermal_expansion_coefficient"), and \
its units in target_units. When the goal sets more than one property target (e.g. "a smaller \
band gap than SrTiO3 and a higher bulk modulus"), put the main one in the target_* fields and \
each other one in extra_targets, in the same form; every target must be met. Stability and \
the hull are not property targets. Separate hard constraints (must hold) from soft preferences. Record every assumption in plain words. Ask at most three \
clarifying questions, only when a required field (target property, value or reference, \
direction) is missing or ambiguous. Give the run a short title (at most eight words, sentence \
case, no full stop) that names the material family and the target, e.g. "Lead-free perovskite \
with a smaller gap than SrTiO3"."""
TITLE_MAX = 70


def property_registry(fidelity: str = "hse", cfg: RunConfig | None = None) -> dict[str, dict[str, Any]]:
    """Return the property registry (PP-1); with ``cfg``, the run's registry (AT-1, AT-26).

    Parameters
    ----------
    fidelity : {"hse", "pbe"}
        Band gap fidelity; ignored when ``cfg`` is given.
    cfg : RunConfig, optional
        Run options: removes disabled capabilities and adds approved library tools.

    Returns
    -------
    dict
        Property key to ``{units, published_error, ...}``.
    """
    if cfg is not None:
        from hephaestus.tools.custom import run_registry

        return run_registry(cfg)
    from hephaestus.tools.simulation import property_registry as registry

    return registry(fidelity)


PROPERTY_NAME_RE = re.compile(r"[^a-z0-9]+")


def property_key(name: str | None) -> str:
    """Normalise a free-text property name to a snake_case key.

    Parameters
    ----------
    name : str or None
        Property name from the parser, e.g. ``"Thermal expansion"``.

    Returns
    -------
    str
        ``"thermal_expansion"``; ``"unsupported"`` when empty.
    """
    key = PROPERTY_NAME_RE.sub("_", (name or "").strip().lower()).strip("_")[:60]
    return key or "unsupported"


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
        "property_name": s,
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
    target = {
        "target_property": props["target_property"],
        "property_name": s,
        "target_value": nn,
        "target_units": s,
        "direction": props["direction"],
        "target_tolerance": nn,
        "reference_material": sn,
    }
    props["extra_targets"] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": target,
            "required": list(target),
            "additionalProperties": False,
        },
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
    registry = property_registry(cfg=cfg)
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
        target_property=target_key(raw),
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
        extra_targets=[
            Target(
                property=target_key(x),
                value=x.get("target_value"),
                units=x.get("target_units") or "",
                direction=x.get("direction") or "minimise",
                tolerance=x.get("target_tolerance"),
                reference_material=x.get("reference_material"),
            )
            for x in raw.get("extra_targets") or []
            if isinstance(x, dict)
        ],
    )
    if raw.get("temperature_K") is None or raw.get("pressure_GPa") is None:
        spec.assumptions.append("Operating conditions default to 300 K and ambient pressure (GP-7).")
    spec = expand_constraints(spec, cfg)
    questions = [] if answers else list(raw.get("questions") or [])[:3]
    errors = [
        ErrorRecord(step="goal_parser", code="invalid_spec", message=m)
        for m in validate_spec(spec, registry, gaps_ok=cfg.custom_tools == "ask")
    ]
    return spec, questions, errors


def target_key(raw: dict[str, Any]) -> str:
    """Return the target property: a registry key, or the plain name of an unsupported one (GP-2).

    Parameters
    ----------
    raw : dict
        Parser answer with ``target_property`` and ``property_name``.

    Returns
    -------
    str
        A registry key, a snake_case property name (a capability gap), or ``"unsupported"``.
    """
    key = raw.get("target_property") or "unsupported"
    return key if key != "unsupported" else property_key(raw.get("property_name"))


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


def validate_spec(
    spec: SearchSpec, registry: dict[str, dict[str, Any]] | None = None, gaps_ok: bool = False
) -> list[str]:
    """Check a spec before a run starts (GP-2, GP-5, GP-8, AT-3).

    Parameters
    ----------
    spec : SearchSpec
        Spec to check.
    registry : dict, optional
        Property registry; defaults to ``property_registry()``.
    gaps_ok : bool
        Accept a named property that no tool computes: the run opens a capability phase for
        it (``RunConfig.custom_tools == "ask"``). ``"unsupported"`` is always a problem.

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
    seen: set[str] = set()
    for t in spec.targets():
        if t.property in seen:
            problems.append(f"{t.property} is a target twice.")
        seen.add(t.property)
        if t.property not in reg:
            if (not gaps_ok or t.property == "unsupported") and not t.tool:
                problems.append(f"Property {t.property!r} is not supported; supported: {', '.join(reg)}.")
        else:
            units = reg[t.property].get("units")
            if units and t.units and t.units.replace("³", "3") != units.replace("³", "3"):
                problems.append(f"Units {t.units!r} do not match {units!r} for {t.property}.")
        if t.value is None and not t.reference_material:
            name = "" if len(spec.targets()) == 1 else f" for {t.property}"
            problems.append(f"No target value or reference material{name}: ask the user (GP-5).")
        if t.direction == "hit" and t.tolerance is None:
            problems.append("A 'hit' target needs a tolerance.")
    if set(spec.elements) & set(spec.excluded_elements):
        problems.append("An element is both required and excluded.")
    return problems


def validate_for_run(spec: SearchSpec, cfg: RunConfig) -> list[str]:
    """Check a spec against the run's own registry (AT-1, AT-3, AT-26).

    Parameters
    ----------
    spec : SearchSpec
        Spec to check.
    cfg : RunConfig
        Disabled capabilities, the tool library and ``custom_tools``.

    Returns
    -------
    list of str
        Problems; empty when valid. A capability gap is a problem only with ``custom_tools="off"``.
    """
    return validate_spec(spec, property_registry(cfg=cfg), gaps_ok=cfg.custom_tools == "ask")
