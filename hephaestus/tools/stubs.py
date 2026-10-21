"""Deterministic stand-ins for every science and data tool, for tests and offline demos.

They follow the result shapes in ``docs/TOOLS.md`` but compute nothing: outcomes come from a
table keyed by composition. Use ``stub_handlers()`` as the ``handlers`` of a campaign.
"""

from __future__ import annotations

import re
from typing import Any

from hephaestus.core.types import (
    Candidate,
    Insight,
    Paper,
    PhononResult,
    PropertyPrediction,
    Quantity,
    StructureRecord,
)
from hephaestus.tools.registry import CandidateState, Handler, ToolError, Workspace

DEFAULT_OUTCOMES: dict[str, dict[str, Any]] = {
    "SrTiO3": {"gap": 3.12, "e_hull": 0.0, "dyn": True},
    "BaSnO3": {"gap": 2.9, "e_hull": 0.0, "dyn": True},
    "SrTi0.875Fe0.125O3": {"gap": 2.6, "e_hull": 0.03, "dyn": True},
    "SrTi0.75Fe0.25O3": {"gap": 2.2, "e_hull": 0.02, "dyn": True},
    "SrTi0.875Sn0.125O3": {"gap": 3.0, "e_hull": 0.01, "dyn": True},
    "CaTiO3": {"gap": 3.3, "e_hull": 0.0, "dyn": False, "where": "R"},
    "BaTi0.875Fe0.125O3": {"gap": 2.4, "e_hull": 0.09, "dyn": True},
}
DEFAULT = {"gap": 3.5, "e_hull": 0.08, "dyn": False, "where": "M"}

STUB_PAPERS = [
    Paper(
        paper_id="stub:fe-sto",
        title="Fe-doped SrTiO3: visible-light absorption from Fe 3d states",
        year=2019,
        citation_count=120,
        abstract="Fe substitution on the Ti site narrows the optical gap of SrTiO3 from 3.25 eV to 2.6 eV.",
        relevance=0.9,
    ),
    Paper(
        paper_id="stub:sn-sto",
        title="Solubility limit of Sn in SrTiO3 and SnO2 secondary phases",
        year=2016,
        citation_count=35,
        abstract="Above 30 % Sn, SnO2 segregates; solid-state synthesis at 1400 C.",
        relevance=0.7,
    ),
    Paper(
        paper_id="stub:sto-ss",
        title="Solid-state synthesis of SrTiO3 from SrCO3 and TiO2",
        year=2012,
        citation_count=60,
        abstract="SrTiO3 forms phase-pure after calcination at 1200 C for 12 h.",
        relevance=0.6,
    ),
]


def _outcome(ws: Workspace, comp: str) -> dict[str, Any]:
    """Return the scripted outcome for a composition.

    Parameters
    ----------
    ws : Workspace
        Unused; kept for a uniform signature.
    comp : str
        Formula.

    Returns
    -------
    dict
        ``gap``, ``e_hull``, ``dyn`` and optionally ``where``.
    """
    return DEFAULT_OUTCOMES.get(comp, DEFAULT)


def _paper_dict(p: Paper, n: int = 700) -> dict[str, Any]:
    """Compact paper record as tools return it.

    Parameters
    ----------
    p : Paper
        Paper.
    n : int
        Abstract length cap.

    Returns
    -------
    dict
        Tool-shaped paper.
    """
    return {
        "paper_id": p.paper_id,
        "title": p.title,
        "year": p.year,
        "citations": p.citation_count,
        "source": p.source,
        "abstract": p.abstract[:n],
    }


def search_literature(ws: Workspace, query: str, mode: str, limit: int) -> dict[str, Any]:
    """Return the stub papers and record them (``search_literature``).

    Parameters
    ----------
    ws : Workspace
        Run workspace.
    query : str
        Query text.
    mode : str
        Search mode.
    limit : int
        Maximum papers.

    Returns
    -------
    dict
        ``{n, papers}``.
    """
    ws.queries.append(f"{mode}:{query}")
    papers = STUB_PAPERS[: max(1, min(limit, 15))]
    for p in papers:
        ws.papers.setdefault(p.paper_id, p)
    return {"n": len(papers), "papers": [_paper_dict(p) for p in papers]}


def note_evidence(
    ws: Workspace,
    paper_id: str,
    composition: str | None,
    property: str,
    value: float | None,
    units: str | None,
    conditions: str,
    finding_type: str,
    summary: str,
) -> dict[str, Any]:
    """Record evidence, dropping values absent from the abstract (``note_evidence``, LM-14).

    Parameters
    ----------
    ws : Workspace
        Run workspace.
    paper_id, composition, property, conditions, finding_type, summary : str
        Evidence fields.
    value : float or None
        Reported value.
    units : str or None
        Its units.

    Returns
    -------
    dict
        ``{recorded, value_dropped_not_in_abstract, n_insights}``.

    Raises
    ------
    ToolError
        If the paper was not retrieved.
    """
    if paper_id not in ws.papers:
        raise ToolError(f"Paper {paper_id} has not been retrieved in this run.")
    dropped = False
    q = None
    if value is not None:
        if any(
            abs(float(x) - value) < 1e-9 for x in re.findall(r"\d+(?:\.\d+)?", ws.papers[paper_id].abstract)
        ):
            q = Quantity(value=value, uncertainty=None, units=units or "", source=paper_id)
        else:
            dropped = True
    ws.insights.append(
        Insight(
            composition=composition,
            property=property,
            value=q,
            conditions=conditions,
            finding_type=finding_type,  # type: ignore[arg-type]
            summary=summary,
            paper_id=paper_id,
        )
    )
    return {"recorded": True, "value_dropped_not_in_abstract": dropped, "n_insights": len(ws.insights)}


def lookup_materials_project(ws: Workspace, composition: str) -> dict[str, Any]:
    """Return one cubic polymorph for known compositions (``lookup_materials_project``).

    Parameters
    ----------
    ws : Workspace
        Run workspace.
    composition : str
        Formula.

    Returns
    -------
    dict
        Tool-shaped MP result.
    """
    if composition not in ("SrTiO3", "BaSnO3", "CaTiO3"):
        return {
            "formula": composition,
            "n_entries": 0,
            "default_polymorph": None,
            "note": "Not in MP.",
            "entries": [],
        }
    entry = {
        "mp_id": "mp-5229",
        "space_group": "Pm-3m",
        "e_hull_eV_per_atom": 0.0,
        "experimental": True,
        "n_icsd": 50,
        "band_gap_pbe_eV": 1.8,
        "nsites": 5,
    }
    return {
        "formula": composition,
        "n_entries": 1,
        "default_polymorph": "mp-5229",
        "note": "",
        "entries": [entry],
    }


def propose_candidate(
    ws: Workspace,
    composition: str,
    structure_type: str,
    rationale: str,
    evidence_ids: list[str],
    prior_estimate: float | None,
    novelty: str,
    synthesis_feasibility: str,
    risks: list[str],
) -> dict[str, Any]:
    """Register a candidate with simplified code checks (``propose_candidate``).

    Parameters
    ----------
    ws : Workspace
        Run workspace.
    composition, structure_type, rationale, novelty, synthesis_feasibility : str
        Candidate fields.
    evidence_ids, risks : list of str
        Evidence and risks.
    prior_estimate : float or None
        Orchestrator's prior.

    Returns
    -------
    dict
        Acceptance with ``candidate_id``, or ``{accepted: false, reasons}``.
    """
    from pymatgen.core import Composition

    reasons = []
    els = {e.symbol for e in Composition(composition).elements}
    if bad := els & set(ws.spec.excluded_elements):
        reasons.append(f"excluded elements {sorted(bad)} (CG-2)")
    if any(cs.candidate.composition == composition for cs in ws.candidates.values()):
        reasons.append("composition already proposed (CG-6)")
    if not evidence_ids:
        reasons.append("no evidence_ids (CG-4)")
    unknown = [e for e in evidence_ids if not e.startswith("mp:") and e not in ws.papers]
    if unknown:
        reasons.append(f"unknown evidence IDs {unknown} (CG-4)")
    if reasons:
        return {"accepted": False, "reasons": reasons}
    n = sum(1 for cs in ws.candidates.values() if cs.round == ws.round) + 1
    cid = f"c{ws.round}-{n}"
    prior = Quantity(
        value=prior_estimate, uncertainty=None, units=ws.spec.target_units, source="orchestrator prior"
    )
    cand = Candidate(
        candidate_id=cid,
        iteration=ws.round,
        composition=composition,
        structure_type=structure_type,
        rationale=rationale,
        evidence_ids=evidence_ids,
        prior_estimate=prior if prior_estimate is not None else None,
        novelty=novelty,  # type: ignore[arg-type]
        synthesis_feasibility=synthesis_feasibility,  # type: ignore[arg-type]
        risks=risks,
    )
    ws.candidates[cid] = CandidateState(candidate=cand, round=ws.round)
    return {
        "accepted": True,
        "candidate_id": cid,
        "sites": {"A": {}, "B": {}, "X": {"O": 1.0}},
        "tolerance_factor": 1.0,
    }


def build_structure(
    ws: Workspace, candidate_id: str, source: str, mp_id: str | None, seed: int
) -> dict[str, Any]:
    """Attach a placeholder structure (``build_structure``).

    Parameters
    ----------
    ws : Workspace
        Run workspace.
    candidate_id : str
        Candidate.
    source : str
        ``auto``, ``mp`` or ``template``.
    mp_id : str or None
        MP entry.
    seed : int
        Disorder seed.

    Returns
    -------
    dict
        Tool-shaped build result.
    """
    cs = ws.get(candidate_id)
    src = f"mp:{mp_id}" if source == "mp" and mp_id else "template:perovskite_cubic"
    cs.record = StructureRecord(
        candidate_id=candidate_id,
        cif=f"data_{candidate_id}\n",
        pmg_dict={},
        source=src,
        supercell=[2, 2, 2],
        n_atoms=40,
        disorder_seed=seed,
    )
    for field in ("structure", "energy_eV_per_atom", "e_hull", "e_form", "phonon", "md", "space_group"):
        setattr(cs, field, None)
    cs.predictions = {}
    cs.converged = False
    return {
        "candidate_id": candidate_id,
        "source": src,
        "n_atoms": 40,
        "supercell": [2, 2, 2],
        "composition_error": 0.0,
        "disorder_seed": seed,
    }


def relax(ws: Workspace, candidate_id: str, potential: str, fmax: float) -> dict[str, Any]:
    """Pretend to relax (``relax``).

    Parameters
    ----------
    ws : Workspace
        Run workspace.
    candidate_id : str
        Candidate.
    potential : str
        Potential name.
    fmax : float
        Force threshold, eV/Å.

    Returns
    -------
    dict
        Tool-shaped relax result.

    Raises
    ------
    ToolError
        If the structure was not built.
    """
    cs = ws.get(candidate_id)
    if cs.record is None:
        raise ToolError("Call build_structure first.")
    cs.structure, cs.energy_eV_per_atom, cs.converged, cs.potential = {}, -7.5, True, potential
    cs.space_group, cs.potential_in_domain, cs.domain_note = "Pm-3m", True, "All elements in MPtrj."
    cs.e_hull = cs.e_form = cs.phonon = None
    cs.predictions = {}
    ws.compute_s += 1.0
    return {
        "candidate_id": candidate_id,
        "potential": potential,
        "converged": True,
        "energy_eV_per_atom": -7.5,
        "space_group": "Pm-3m",
        "lattice_abc_A": [3.9, 3.9, 3.9],
        "volume_per_atom_A3": 11.9,
        "wall_time_s": 1.0,
        "domain_note": cs.domain_note,
    }


def energy_above_hull(ws: Workspace, candidate_id: str) -> dict[str, Any]:
    """Return the scripted E_hull (``energy_above_hull``).

    Parameters
    ----------
    ws : Workspace
        Run workspace.
    candidate_id : str
        Candidate.

    Returns
    -------
    dict
        Tool-shaped hull result with the code verdict.

    Raises
    ------
    ToolError
        If the candidate was not relaxed.
    """
    cs = ws.get(candidate_id)
    if cs.energy_eV_per_atom is None:
        raise ToolError("Call relax first.")
    cs.e_hull = float(_outcome(ws, cs.candidate.composition)["e_hull"])
    cs.e_form, cs.hull_stale = -3.0, False
    thr = ws.spec.stability_thresholds.e_hull_max_eV_per_atom
    return {
        "candidate_id": candidate_id,
        "e_hull_eV_per_atom": cs.e_hull,
        "formation_energy_eV_per_atom": -3.0,
        "decomposes_to": [],
        "n_competing_phases": 4,
        "thermo_stable": cs.e_hull <= thr,
        "threshold": thr,
        "wall_time_s": 1.0,
    }


def phonons(ws: Workspace, candidate_id: str, mode: str) -> dict[str, Any]:
    """Return the scripted phonon verdict (``phonons``).

    Parameters
    ----------
    ws : Workspace
        Run workspace.
    candidate_id : str
        Candidate.
    mode : str
        ``fast`` or ``full``.

    Returns
    -------
    dict
        Tool-shaped phonon result with the code verdict.

    Raises
    ------
    ToolError
        If the candidate was not relaxed.
    """
    cs = ws.get(candidate_id)
    if cs.energy_eV_per_atom is None:
        raise ToolError("Call relax first.")
    out = _outcome(ws, cs.candidate.composition)
    stable = bool(out["dyn"])
    where = None if stable else out.get("where", "R")
    cs.phonon = PhononResult(
        mode=mode,
        stable=stable,
        min_frequency_THz=0.2 if stable else -1.5,  # type: ignore[arg-type]
        instability_at=where,
        supercell=[2, 2, 2],
    )
    verdict = "stable" if stable else "unstable"
    cs.phonon_details = {"verdict": verdict, "mode_mapping_rounds": []}
    return {
        "candidate_id": candidate_id,
        "mode": mode,
        "min_frequency_THz": cs.phonon.min_frequency_THz,
        "where": where,
        "dyn_stable": stable,
        "verdict": verdict,
        "mode_mapping_rounds": [],
        "supercell": [2, 2, 2],
        "wall_time_s": 1.0,
    }


def predict_property(ws: Workspace, candidate_id: str, property: str) -> dict[str, Any]:
    """Return the scripted band gap (``predict_property``).

    Parameters
    ----------
    ws : Workspace
        Run workspace.
    candidate_id : str
        Candidate.
    property : str
        Property key; only ``band_gap`` is scripted.

    Returns
    -------
    dict
        Tool-shaped prediction.

    Raises
    ------
    ToolError
        If the candidate was not relaxed, or the property is not ``band_gap``.
    """
    cs = ws.get(candidate_id)
    if cs.energy_eV_per_atom is None:
        raise ToolError("Call relax first.")
    if property != "band_gap":
        raise ToolError(f"Stub predicts only band_gap, not {property}.")
    gap = float(_outcome(ws, cs.candidate.composition)["gap"])
    model = "stub-MEGNet (hse)"
    cs.predictions[property] = PropertyPrediction(
        candidate_id=candidate_id,
        property=property,
        value=Quantity(value=gap, uncertainty=0.65, units="eV", source=model),
        model=model,
    )
    return {
        "candidate_id": candidate_id,
        "property": property,
        "value": gap,
        "uncertainty": 0.65,
        "units": "eV",
        "model": model,
        "in_domain": True,
        "scale_note": "HSE fidelity",
        "target": {
            "direction": ws.spec.direction,
            "value": ws.spec.target_value,
            "reference": ws.spec.reference_material,
        },
    }


def check_novelty(ws: Workspace, composition: str, property: str = "band_gap") -> dict[str, Any]:
    """Return no prior reports for doped compositions, many for parents.

    Parameters
    ----------
    ws : Workspace
        Run workspace.
    composition : str
        Formula.
    property : str
        Property key.

    Returns
    -------
    dict
        ``{hits, paper_ids}``.
    """
    parent = "." not in composition
    return {"hits": 12 if parent else 0, "paper_ids": ["stub:sto-ss"] if parent else []}


def stub_handlers() -> dict[str, Handler]:
    """Return handlers for every B and C tool.

    Returns
    -------
    dict
        Tool name to stub handler.
    """
    return {
        "search_literature": search_literature,
        "related_papers": lambda ws, paper_id, relation, limit: search_literature(
            ws, paper_id, relation, limit
        ),
        "get_paper": lambda ws, paper_id: (
            _paper_dict(ws.papers[paper_id], 3000) if paper_id in ws.papers else {"error": "unknown"}
        ),
        "publication_trend": lambda ws, query: {"query": query, "papers_per_year": {2020: 3, 2021: 5}},
        "note_evidence": note_evidence,
        "lookup_materials_project": lookup_materials_project,
        "propose_candidate": propose_candidate,
        "build_structure": build_structure,
        "relax": relax,
        "energy_above_hull": energy_above_hull,
        "phonons": phonons,
        "run_md": lambda ws, candidate_id, duration_ps: {
            "candidate_id": candidate_id,
            "stable": True,
            "max_displacement_A": 0.3,
            "temperature_K": 300,
        },
        "predict_property": predict_property,
    }
