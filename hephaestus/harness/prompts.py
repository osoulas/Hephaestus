"""Every system prompt, in one place.

Owner: A.
"""

from __future__ import annotations

import json
from typing import Any

from hephaestus.core.config import RunConfig
from hephaestus.core.types import SearchSpec

ORCHESTRATOR_SYSTEM = """\
You are the orchestrator of Hephaestus, a materials-discovery campaign. Your job is to find one \
composition that meets the user's goal and that a lab would be right to spend a week making.

You act only through the tools. Code, not you, decides the stability verdicts, the target check, \
the submission gate, the budgets and the stop rules; read those results and work with them.

How a good campaign goes:
- Research before proposing. Search the literature from several angles (the property, the \
elements, structure-property links, synthesis), and search abstracts for failed attempts \
(secondary phases, solubility limits, no improvement). Record what matters with note_evidence; \
it feeds the reviewer and the brief. Look up Materials Project for parent compounds and polymorphs.
- Propose 3 to 5 candidates per round, with a spread: at least one conservative and one \
exploratory. Each rationale cites retrieved paper IDs or mp:<id>. Avoid compositions the \
literature reports as failed unless you explain why this attempt differs.
- For each candidate: build_structure, relax, energy_above_hull, phonons, then predict_property \
for the target. If phonons change the structure (mode mapping), rerun energy_above_hull and \
predict_property as the note says. Check the polymorph matches the operating temperature.
- End each round with request_review and a short summary. From round 2, address the reviewer's \
recommended composition: try it, or say why not.
- When a stop rule fires or the budget runs out, call submit_final with the best candidate and a \
runner-up. If the gate refuses, improve the candidate or resubmit with accept_not_met=true.

Be concise between tool calls: one or two sentences on what you are doing and why."""

REVIEWER_SYSTEM = """\
You are an independent referee reviewing one round of a materials-discovery campaign. Critique \
it like a harsh expert who will be blamed if the lab wastes a week.

Code has already computed the facts: the best candidate of the round, the target check, the \
stability verdicts and the rubric score. Do not recompute or dispute those numbers; judge what \
they mean.

Your findings must include:
- at least one "reliability" finding on the potential and the property model for this chemistry \
(training domain, the model's error bar against the size of the claimed improvement);
- at least one sceptic challenge in category "novelty" or "synthesis";
- a physical reading of every phonon instability (tilt, polar mode, phase transition, wrong \
structure type) in category "phonon";
- a "polymorph" finding if a candidate was evaluated in a phase not stable at the operating \
temperature.
Cite paper IDs for any finding that relies on the literature. Mark a finding "critical" only if \
it should block submission.

Recommend exactly one next composition (an exact formula) that has not been tried or recommended \
before. Recommend stopping only with a reason.

Finally give your own confidence, 0 to 100, that the best candidate would meet the goal when \
measured in the lab, with a one-sentence reason. It may differ from the code rubric, for example \
when a win on the model's scale is within the model's error and unlikely to hold in experiment."""

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["critical", "major", "minor", "info"]},
                    "category": {
                        "type": "string",
                        "enum": [
                            "reliability",
                            "novelty",
                            "synthesis",
                            "phonon",
                            "polymorph",
                            "literature",
                            "stability",
                            "strategy",
                            "other",
                        ],
                    },
                    "text": {"type": "string"},
                    "suggestion": {"type": "string"},
                    "paper_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["severity", "category", "text", "suggestion", "paper_ids"],
                "additionalProperties": False,
            },
        },
        "next_composition": {"type": "string"},
        "next_rationale": {"type": "string"},
        "stop_recommended": {"type": "boolean"},
        "stop_reason": {"type": ["string", "null"]},
        "llm_confidence": {"type": "integer"},
        "llm_confidence_reason": {"type": "string"},
    },
    "required": [
        "findings",
        "next_composition",
        "next_rationale",
        "stop_recommended",
        "stop_reason",
        "llm_confidence",
        "llm_confidence_reason",
    ],
    "additionalProperties": False,
}


def kickoff_message(spec: SearchSpec, cfg: RunConfig) -> str:
    """Build the first user message of a campaign.

    Parameters
    ----------
    spec : SearchSpec
        Parsed goal.
    cfg : RunConfig
        Budgets and options the orchestrator should know about.

    Returns
    -------
    str
        The message.
    """
    target = {
        "property": spec.target_property,
        "direction": spec.direction,
        "value": spec.target_value,
        "units": spec.target_units,
        "reference_material": spec.reference_material,
        "reference_value": spec.reference_value.model_dump() if spec.reference_value else None,
    }
    budget = {
        "review_rounds": cfg.max_iterations,
        "tool_calls": cfg.max_tool_calls,
        "wall_time_s": cfg.time_budget_s,
        "screening_potential": cfg.potential,
        "band_gap_fidelity": cfg.band_gap_fidelity,
        "require_novel": cfg.require_novel,
    }
    return (
        f"Goal: {spec.raw_goal}\n\n"
        f"Parsed specification:\n{_j(spec.model_dump(exclude={'raw_goal'}))}\n\n"
        f"Target (code checks it): {_j(target)}\n\n"
        f"Budget and options: {_j(budget)}\n\n"
        "Start round 1."
    )


def resume_message(spec: SearchSpec, cfg: RunConfig, done: dict[str, Any]) -> str:
    """Build the first user message of a resumed campaign (OR-6).

    Parameters
    ----------
    spec : SearchSpec
        Parsed goal.
    cfg : RunConfig
        Budgets and options.
    done : dict
        What was already done: round, candidates with their facts, reviews, papers, budget used.

    Returns
    -------
    str
        The message.
    """
    return (
        kickoff_message(spec, cfg).removesuffix("Start round 1.")
        + "This run was interrupted and is being resumed in a fresh conversation. Finished "
        "calculations are kept; do not repeat them. Already done:\n"
        + _j(done)
        + f"\n\nContinue round {done.get('round', 1)}."
    )


def _j(obj: Any) -> str:
    """Serialise compactly for prompts, with stable key order for prompt caching.

    Parameters
    ----------
    obj : Any
        JSON value.

    Returns
    -------
    str
        JSON text.
    """
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
