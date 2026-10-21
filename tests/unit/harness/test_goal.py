"""Goal parser: expansion and validation in code (Section 1)."""

import scripts

from hephaestus.core.config import RunConfig
from hephaestus.harness.goal import NOVELTY_CONSTRAINT, expand_constraints, parse_goal, validate_spec
from hephaestus.llm.scripted import ScriptedLLM


def parse(raw: dict, goal: str = scripts.GOAL, **cfg: object):  # type: ignore[no-untyped-def]
    """Parse a goal with a scripted answer.

    Parameters
    ----------
    raw : dict
        Scripted parser answer.
    goal : str
        Goal text.
    **cfg
        ``RunConfig`` overrides.

    Returns
    -------
    tuple
        ``(spec, questions, errors)``.
    """
    return parse_goal(ScriptedLLM(structured={"goal_parser": [raw]}), goal, RunConfig(**cfg))  # type: ignore[arg-type]


def test_gp4_gp7_gp12_expansion() -> None:
    """Lead-free excludes Pb; default conditions are recorded; both stability constraints added."""
    spec, questions, errors = parse(scripts.PARSED)
    assert spec is not None and questions == [] and errors == []
    assert spec.excluded_elements == ["Pb"]
    assert any("300 K" in a for a in spec.assumptions)
    assert sum("stable" in h for h in spec.hard_constraints) == 2


def test_gp13_novelty_constraint() -> None:
    """With require_novel, the novelty constraint is added."""
    spec, _, _ = parse(scripts.PARSED, require_novel=True)
    assert spec is not None and NOVELTY_CONSTRAINT in spec.hard_constraints


def test_gp2_unsupported_property_flagged() -> None:
    """Creep rate is flagged, not swapped."""
    spec, _, errors = parse(scripts.PARSED | {"target_property": "unsupported"}, "low creep rate alloy")
    assert spec is not None and any("not supported" in e.message for e in errors)


def test_gp5_gp8_validation() -> None:
    """No target or reference is a question; unknown element symbols are rejected."""
    spec, _, _ = parse(scripts.PARSED)
    assert spec is not None
    bad = spec.model_copy(update={"reference_material": None, "elements": ["Xx"]})
    problems = validate_spec(bad)
    assert any("GP-5" in p for p in problems) and any("Xx" in p for p in problems)


def test_expand_is_idempotent() -> None:
    """Expanding twice adds nothing new."""
    spec, _, _ = parse(scripts.PARSED)
    assert spec is not None
    assert expand_constraints(spec, RunConfig()) == spec
