"""Scripted orchestrator and reviewer turns for the demo goal, shared by integration tests."""

from typing import Any

FE1, FE2, BAFE = "SrTi0.875Fe0.125O3", "SrTi0.75Fe0.25O3", "BaTi0.875Fe0.125O3"


def evaluate(cid: str) -> list[dict[str, Any]]:
    """Return the five evaluation calls for one candidate.

    Parameters
    ----------
    cid : str
        Candidate ID.

    Returns
    -------
    list of dict
        Scripted steps.
    """
    return [
        {
            "tool": "build_structure",
            "input": {"candidate_id": cid, "source": "template", "mp_id": None, "seed": 0},
        },
        {"tool": "relax", "input": {"candidate_id": cid, "potential": "mace-mp-0-small", "fmax": 0.02}},
        {"tool": "energy_above_hull", "input": {"candidate_id": cid}},
        {"tool": "phonons", "input": {"candidate_id": cid, "mode": "fast"}},
        {"tool": "predict_property", "input": {"candidate_id": cid, "property": "band_gap"}},
    ]


def propose(comp: str, evidence: str = "stub:fe-sto") -> dict[str, Any]:
    """Return a ``propose_candidate`` step.

    Parameters
    ----------
    comp : str
        Composition.
    evidence : str
        Evidence ID.

    Returns
    -------
    dict
        Scripted step.
    """
    return {
        "tool": "propose_candidate",
        "input": {
            "composition": comp,
            "structure_type": "perovskite_cubic",
            "rationale": f"Fe 3d states narrow the gap ({evidence}).",
            "evidence_ids": [evidence],
            "prior_estimate": 2.5,
            "novelty": "variant",
            "synthesis_feasibility": "easy",
            "risks": [],
        },
    }


def submit(cid: str, runner: str | None = None, accept_not_met: bool = False) -> dict[str, Any]:
    """Return a ``submit_final`` step.

    Parameters
    ----------
    cid : str
        Candidate.
    runner : str, optional
        Runner-up.
    accept_not_met : bool
        Accept a failing gate.

    Returns
    -------
    dict
        Scripted step.
    """
    return {
        "tool": "submit_final",
        "input": {
            "candidate_id": cid,
            "runner_up_id": runner,
            "justification": "best",
            "accept_not_met": accept_not_met,
        },
    }


ROUND_1 = [
    [
        {"thinking": "Research first."},
        {"text": "Searching the literature for gap narrowing in SrTiO3."},
        {
            "tool": "search_literature",
            "input": {"query": "SrTiO3 band gap narrowing doping", "mode": "semantic", "limit": 5},
        },
        {
            "tool": "search_literature",
            "input": {"query": "SrTiO3 secondary phase solubility", "mode": "abstract", "limit": 5},
        },
        {
            "tool": "note_evidence",
            "input": {
                "paper_id": "stub:sto-ss",
                "composition": "SrTiO3",
                "property": "synthesis",
                "value": 1200,
                "units": "C",
                "conditions": "12 h",
                "finding_type": "synthesis",
                "summary": "Solid state from SrCO3 + TiO2 at 1200 C.",
            },
        },
        {"tool": "lookup_materials_project", "input": {"composition": "SrTiO3"}},
    ],
    [propose(FE1), propose(BAFE)],
    evaluate("c1-1") + evaluate("c1-2"),
    [
        {
            "tool": "request_review",
            "input": {"summary": "Fe on the B site lowers the gap; Ba variant is off hull."},
        }
    ],
]
ROUND_2 = [
    [{"text": "Trying the reviewer's recommendation."}, propose(FE2)],
    evaluate("c2-1"),
    [submit("c1-2")],
    [
        {
            "tool": "request_review",
            "input": {"summary": "Higher Fe content lowers the gap further and stays on hull."},
        }
    ],
    [submit("c2-1", "c1-1")],
]


def review(next_comp: str, conf: int) -> dict[str, Any]:
    """Return a valid reviewer answer.

    Parameters
    ----------
    next_comp : str
        Recommended composition.
    conf : int
        Reviewer confidence.

    Returns
    -------
    dict
        Answer.
    """
    return {
        "findings": [
            {
                "severity": "major",
                "category": "reliability",
                "text": "MEGNet ±0.65 eV.",
                "suggestion": "",
                "paper_ids": [],
            },
            {
                "severity": "minor",
                "category": "synthesis",
                "text": "Route known.",
                "suggestion": "",
                "paper_ids": ["stub:sto-ss"],
            },
        ],
        "next_composition": next_comp,
        "next_rationale": "More Fe narrows the gap.",
        "stop_recommended": False,
        "stop_reason": None,
        "llm_confidence": conf,
        "llm_confidence_reason": "reason",
    }


REVIEWS = [review(FE2, 60), review("SrTi0.625Fe0.375O3", 80)]

GOAL = (
    "a lead-free oxide perovskite with a smaller band gap than SrTiO3 that is thermodynamically stable "
    "(≤ 0.05 eV/atom above hull) and dynamically stable"
)
PARSED = {
    "material_family": "oxide perovskite",
    "elements": [],
    "excluded_elements": [],
    "target_property": "band_gap",
    "target_value": None,
    "target_units": "eV",
    "direction": "minimise",
    "target_tolerance": None,
    "reference_material": "SrTiO3",
    "temperature_K": None,
    "pressure_GPa": None,
    "e_hull_max_eV_per_atom": 0.05,
    "hard_constraints": ["oxide perovskite ABO3"],
    "soft_preferences": [],
    "assumptions": [],
    "questions": [],
}
