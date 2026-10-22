"""Every system prompt, in one place."""

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

You may request several tool calls in one turn. Calls on different candidates, and literature \
and database lookups, run in parallel; calls on the same candidate run in the order you give them; \
propose_candidate, candidate_status, request_review and submit_final each run alone. So evaluate \
several candidates in one turn, and keep a lookup whose result you need for a later call in an \
earlier turn.

Be concise between tool calls: one or two sentences on what you are doing and why."""

CAPABILITY_SYSTEM = """\
You are the tool builder of Hephaestus, a materials-discovery agent. The user's goal needs a \
property that none of our tools can compute. Your job is to give the campaign a trustworthy tool \
for it, with the user's approval at every step.

1. Research the method. Search the literature for how papers compute this property with \
interatomic potentials or DFT, and record the method with note_evidence (finding_type "method"). \
Read our own tools with read_tool_source for inspiration and to judge scope: the function list \
first, then the source of anything close to what you need.
2. Propose one tool with propose_tool: what it computes, the method with citations, inputs, \
units, the building blocks it uses (may be none), expected runtime, and what the run loses if \
the user cancels it. Then end your turn: the user's decision arrives as the next message.
3. When the plan is accepted, write the tool and its tests with write_tool. Code checks them, \
runs the tests and a trial run in a sandbox; fix any failure it reports. When the checks pass, \
read the trial value and its details against the literature and the physics before ending your \
turn. If they show a bug, call write_tool again straight away with the fix and a reason: that \
withdraws the code the user is reviewing, so they cannot approve the broken version. If you are \
unsure whether a result is right, say why with ask_user. Otherwise end your turn: the user \
reviews the code.
4. If the user asks for changes, revise and propose (or write) again with the same name.
5. If the method needs a Python package outside the allowlist (for example phonopy), call \
request_package and end your turn; once the user allows it, list it in TOOL["packages"] and import it.
6. The user never reads your plain text as a question. To ask anything (a choice between \
approaches, what to do when a request cannot be met), call ask_user and end your turn. Never end \
a turn expecting an answer you did not ask for with ask_user.
7. If no trustworthy tool can be built within the constraints, call give_up with the reason \
instead of repeating yourself.

The tool file (one tool per file):
- A literal header `TOOL = {"name": ..., "property": ..., "units": ..., "description": ..., \
"method": ..., "citations": [...], "inputs": {JSON schema, defaults for every input}, \
"building_blocks": [...], "trial_range": [min, max], "packages": [...], "author": "agent"}`. \
`trial_range` is the plausible value for the trial material in the tool's units, from the \
literature: code fails the trial when the value falls outside it; `packages` lists only \
imports beyond the allowlist that the user allowed (usually none). `name` is snake_case; `property` is the gap \
property's key exactly.
- `def run(structure, inputs, hp) -> dict` returning `{"value": float, "units": <the header's \
units>, "details": {...}}`. `structure` is a pymatgen `Structure.as_dict()`; use \
`pymatgen.core.Structure.from_dict`. All atomistic-model evaluations go through `hp`: \
`hp.single_point(structure)` -> {energy_eV, forces_eV_per_A, stress_GPa (3x3, negative under \
compression)}; `hp.relax(structure, fix_cell=False, fmax=0.01)` -> {structure, \
energy_eV_per_atom, converged}; `hp.phonons(structure, supercell=None)` -> {frequencies_THz, \
qpoints, min_frequency_THz, supercell}; `hp.md(structure, temperature_K, duration_ps)` -> \
{energies_eV, temperatures_K, final_structure, max_displacement_A, stable}; `hp.log(message)`. \
They run with the candidate's own potential.
- Allowed imports only: math, cmath, statistics, itertools, functools, collections, typing, \
dataclasses, numpy, scipy, pymatgen.core, pymatgen.analysis, pymatgen.symmetry, ase, ase.build, \
ase.units, ase.eos, plus any package the user allowed through request_package. No file, network or process access, no names or attributes starting with \
an underscore, no getattr/eval/exec/open. At most 500 lines.
- The test file has plain `def test_*` functions whose arguments are any of `run` (your run \
function), `hp`, `structure` (a relaxed trial structure of the reference material) and `tool` \
(the header). Test physics you can check: signs, symmetry, a known limit.

Keep the tool small, deterministic and honest: compute exactly what the plan says. Be concise \
between tool calls."""

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
        "computed_by": spec.target_tool or "predict_property",
        "direction": spec.direction,
        "value": spec.target_value,
        "units": spec.target_units,
        "reference_material": spec.reference_material,
        "reference_value": spec.reference_value.model_dump() if spec.reference_value else None,
    }
    budget = {
        "review_rounds": cfg.max_iterations or "no limit",
        "tool_calls": cfg.max_tool_calls or "no limit",
        "wall_time_s": cfg.time_budget_s or "no limit",
        "energy_above_hull": "on" if cfg.run_hull else "off for this run: do not call it",
        "phonons": "on" if cfg.run_phonons else "off for this run: do not call it",
        "screening_potential": cfg.potential,
        "band_gap_fidelity": cfg.band_gap_fidelity,
        "require_novel": cfg.require_novel,
    }
    note = ""
    if spec.target_tool:
        note = (
            f"The target property comes from the agent-built tool {spec.target_tool}, approved by the user: "
            f"call it in place of predict_property for {spec.target_property}. Its results carry no "
            "uncertainty.\n\n"
        )
    elif not spec.target_evaluable:
        note = (
            f"No tool can compute {spec.target_property} in this run (the user cancelled the tool for it): "
            "screen for stability only, and submit the best candidate with accept_not_met=true.\n\n"
        )
    return (
        f"Goal: {spec.raw_goal}\n\n"
        f"Parsed specification:\n{_j(spec.model_dump(exclude={'raw_goal'}))}\n\n"
        f"Target (code checks it): {_j(target)}\n\n"
        f"{note}"
        f"Budget and options: {_j(budget)}\n\n"
        "Start round 1."
    )


def capability_kickoff(
    spec: SearchSpec, cfg: RunConfig, library: list[dict[str, Any]], done: dict[str, Any] | None
) -> str:
    """Build the first user message of the capability phase (Section 14).

    Parameters
    ----------
    spec : SearchSpec
        Parsed goal; ``target_property`` is the gap.
    cfg : RunConfig
        Budgets and limits.
    library : list of dict
        Approved library tools ``{name, property, units, description}``.
    done : dict or None
        For a resumed phase: proposals and decisions so far.

    Returns
    -------
    str
        The message.
    """
    limits = {
        "max_tools": cfg.max_custom_tools,
        "max_revisions_per_tool": cfg.max_tool_revisions,
        "tool_timeout_s": cfg.custom_tool_timeout_s,
        "screening_potential": cfg.potential,
        "trial_material": spec.reference_material or "SrTiO3",
    }
    text = (
        f"Goal: {spec.raw_goal}\n\n"
        f"Missing property (the gap): {spec.target_property!r}, units {spec.target_units!r}, "
        f"direction {spec.direction}, reference {spec.reference_material or 'none'}.\n\n"
        f"Tools already in the library (for other properties): {_j(library)}\n\n"
        f"Limits: {_j(limits)}\n\n"
    )
    if done:
        return text + "This phase was interrupted and is resumed. So far:\n" + _j(done) + "\n\nContinue."
    return text + "Start by researching how this property is computed."


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
