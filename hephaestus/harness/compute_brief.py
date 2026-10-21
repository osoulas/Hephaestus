"""Computational brief: the DFT follow-up for the recommended candidate (Section 10).

The lab brief says how to make and test the candidate. This brief says how to check it with
first-principles calculations before anyone goes to the lab: which settings, which calculations
in which order, what the screening predicted for each, and when to stop. It is built by code from
the run's stored results and the candidate's chemistry (magnetic ions, heavy elements, partial
occupancy, the target property), so every number in it comes from the run or from a fixed table.
"""

from __future__ import annotations

import re
from typing import Any

from pymatgen.core import Composition, Element, Structure

from hephaestus.core.types import Brief, Paper, RunState
from hephaestus.harness.brief import (
    DASH,
    _fmt,
    candidate_label,
    collect_facts,
    id_label,
    paper_url,
    property_name,
    strip_codes,
)
from hephaestus.tools.registry import CandidateState, Workspace
from hephaestus.tools.simulation import MECHANICAL

TITLE = "Computational brief"
SECTIONS = [
    "Starting point",
    "Calculation settings",
    "Workflow",
    "Screening results to check",
    "Starting structure",
    "Methods to compare against",
    "Scope and assumptions",
]
# Materials Project's fitted Hubbard U values for oxides and fluorides, in eV.
HUBBARD_U = {"Co": 3.32, "Cr": 3.7, "Fe": 5.3, "Mn": 3.9, "Mo": 4.38, "Ni": 6.2, "V": 3.25, "W": 6.2}
# Open-shell d or f ions that need spin polarisation.
MAGNETIC = {"V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Ru", "Rh", "Os", "Ir"} | {
    e.symbol for e in Element if e.is_lanthanoid and e.symbol not in {"La", "Lu"}
}
# Heavy elements whose states can sit at the band edges, so spin-orbit coupling matters.
SOC = {"Sn", "Sb", "Te", "I", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi"}
DFT_WORDS = re.compile(
    r"\b(DFT|first[- ]principles|ab initio|density[- ]functional|HSE0?6?|hybrid functional|"
    r"GGA|PBE(?:sol)?|r2SCAN|phonon|band structure)\b",
    re.I,
)
VERDICTS = {
    "stable": "no imaginary modes",
    "spurious_soft_mode": "a soft mode that did not lower the energy when frozen in, so not a real instability",
    "stable_after_distortion": "a real instability: the structure was distorted along the soft mode and re-relaxed",
    "unstable": "unstable",
}


def chemistry(composition: str) -> dict[str, Any]:
    """Work out what the candidate's elements mean for the DFT settings.

    Parameters
    ----------
    composition : str
        Formula, e.g. ``"SrTi0.875Fe0.125O3"``.

    Returns
    -------
    dict
        ``elements``, ``hubbard_u`` (element to U in eV, for oxides and fluorides),
        ``magnetic``, ``soc``, ``f_elements`` and ``partial`` (any non-integer amount).
    """
    comp = Composition(composition)
    els = [e.symbol for e in comp.elements]
    anion_ok = any(a in els for a in ("O", "F"))
    return {
        "elements": els,
        "hubbard_u": {e: HUBBARD_U[e] for e in els if e in HUBBARD_U} if anion_ok else {},
        "magnetic": [e for e in els if e in MAGNETIC],
        "soc": [e for e in els if e in SOC],
        "f_elements": [e for e in els if Element(e).is_lanthanoid and e not in {"La", "Lu"}],
        "partial": any(abs(n - round(n)) > 1e-6 for n in comp.values()),
    }


def _structure(cs: CandidateState | None) -> Structure | None:
    """Return the candidate's current structure, if stored.

    Parameters
    ----------
    cs : CandidateState or None
        Candidate.

    Returns
    -------
    Structure or None
        Relaxed (or mode-mapped) structure.
    """
    if cs is None or not cs.structure:
        return None
    try:
        return Structure.from_dict(cs.structure)
    except Exception:
        return None


def settings_table(chem: dict[str, Any], mechanical: bool) -> str:
    """Recommended DFT settings as a Markdown table.

    Parameters
    ----------
    chem : dict
        Output of ``chemistry``.
    mechanical : bool
        Whether the target is an elastic property (needs a higher cutoff).

    Returns
    -------
    str
        Table with parameter, recommendation and reason.
    """
    rows = [
        (
            "Code and potentials",
            "VASP with the PAW sets Materials Project uses (semicore where it does), or Quantum ESPRESSO with SSSP",
            "Keeps energies comparable with MP's competing phases",
        ),
        (
            "Geometry functional",
            "PBEsol" + ("+U" if chem["hubbard_u"] else "") + " (or r2SCAN)",
            "Better lattice constants than PBE for oxides and other solids",
        ),
        (
            "Hull functional",
            "PBE"
            + ("+U" if chem["hubbard_u"] else "")
            + " with MP settings and MP2020 corrections, or r2SCAN for every phase",
            "Lets you use MP's competing phases directly; r2SCAN avoids mixing +U and plain GGA",
        ),
        (
            "Plane-wave cutoff",
            "520 eV"
            + (", and 1.3 \u00d7 the largest ENMAX for stresses and elastic constants" if mechanical else ""),
            "MP's standard; stresses converge more slowly than energies",
        ),
        (
            "k-points",
            "KSPACING 0.22 Å⁻¹ to relax; 0.15 Å⁻¹ for densities of states",
            "Converge the total energy to 1 meV/atom first",
        ),
        (
            "Convergence",
            "EDIFF 1e-6 eV (1e-8 for phonons); forces below 0.01 eV/Å",
            "Phonons need tight forces",
        ),
        (
            "Smearing",
            "Gaussian, 0.05 eV; tetrahedron method for densities of states",
            "Insulator-safe; no spurious partial occupations",
        ),
    ]
    if chem["magnetic"]:
        rows.append(
            (
                "Spin",
                f"Spin-polarised; start high-spin on {', '.join(chem['magnetic'])}",
                "Open-shell ions; a non-magnetic start can trap a wrong ground state",
            )
        )
    if chem["hubbard_u"]:
        us = ", ".join(f"{e} {u} eV" for e, u in chem["hubbard_u"].items())
        rows.append(("Hubbard U (Dudarev)", us, "Materials Project's fitted values, so the hull matches MP"))
    if chem["f_elements"]:
        rows.append(
            (
                "4f electrons",
                f"Test 4f-in-core PAW sets against +U for {', '.join(chem['f_elements'])}",
                "Open 4f shells make plain GGA unreliable",
            )
        )
    if chem["soc"]:
        rows.append(
            (
                "Spin-orbit coupling",
                f"Include for the band structure ({', '.join(chem['soc'])})",
                "Heavy-element states at the band edges shift the gap",
            )
        )
    lines = ["| Setting | Recommendation | Why |", "| --- | --- | --- |"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(lines)


def workflow_steps(
    ws: Workspace, cs: CandidateState | None, chem: dict[str, Any], facts: dict[str, Any], n_atoms: int | None
) -> list[str]:
    """Ordered calculations, each with what decides it.

    Parameters
    ----------
    ws : Workspace
        Workspace (spec, config).
    cs : CandidateState or None
        The recommended candidate.
    chem : dict
        Output of ``chemistry``.
    facts : dict
        Output of ``collect_facts``.
    n_atoms : int or None
        Atoms in the starting cell.

    Returns
    -------
    list of str
        Markdown steps, "**Name.** what; pass if ..., stop if ...".
    """
    spec = ws.spec
    target = spec.target_property
    thr = spec.stability_thresholds
    cell = f"the {n_atoms}-atom starting cell" if n_atoms else "the starting cell"
    steps = [
        f"**Convergence tests.** On {cell}, raise the cutoff and k-point density until the energy "
        "changes by less than 1 meV/atom and stresses by less than 0.5 kbar.",
        f"**Relax.** Full relaxation (cell shape, volume and ions) of {cell}. Restart once from the "
        "relaxed cell so the basis matches the final volume. Pass if it keeps the space group "
        f"{facts.get('win_space_group') or 'found in screening'}; stop if it collapses to a different structure type.",
    ]
    if chem["magnetic"]:
        steps.append(
            "**Magnetic order.** Compare ferromagnetic and the simple antiferromagnetic orders (A-, C- and "
            "G-type on a \u221a2\u00d7\u221a2\u00d72 or 2\u00d72\u00d72 cell). Carry the lowest-energy order into every later step."
        )
    if chem["partial"]:
        steps.append(
            "**Site ordering.** The screened cell is one ordering of a partly occupied site. Enumerate the "
            "symmetry-distinct orderings in the same supercell (or build a special quasirandom structure) "
            "and carry the lowest-energy one forward; report the spread."
        )
    elif "double" in (facts.get("win_structure_type") or ""):
        steps.append(
            "**B-site order.** Compare the rock-salt ordered cell with one antisite swap; a small energy "
            "cost means antisite disorder in real samples."
        )
    if thr.thermodynamic:
        own = Composition(facts.get("win_composition") or "X").reduced_formula
        phases = [p for p in facts.get("win_decomposes_to") or [] if Composition(p).reduced_formula != own]
        steps.append(
            "**Energy above hull.** Recompute every competing phase of the chemical system at the same "
            f"settings{' (' + ', '.join(phases) + ' set the hull locally)' if phases else ''}. "
            f"Pass if E_hull ≤ {_fmt(thr.e_hull_max_eV_per_atom)} eV/atom; stop if it is well above."
        )
    if thr.dynamic:
        sc = (
            "x".join(str(n) for n in cs.phonon.supercell)
            if cs and cs.phonon and cs.phonon.supercell
            else None
        )
        verdict = (cs.phonon_details or {}).get("verdict") if cs else None
        extra = (
            " Screening found a real instability and distorted the structure: check the distortion survives "
            "and that the undistorted cell is higher in energy."
            if verdict == "stable_after_distortion"
            else " Screening saw a soft mode that raised no energy when frozen in: confirm it is not a real instability."
            if verdict == "spurious_soft_mode"
            else ""
        )
        steps.append(
            "**Phonons.** Finite displacements (phonopy) on a "
            + (
                f"{sc} supercell (as in screening) or larger"
                if sc
                else "supercell at least 10 Å on each side"
            )
            + f", or DFPT. Pass if no mode is below {_fmt(thr.phonon_min_THz)} THz; if one is, freeze it in, "
            "re-relax and repeat." + extra
        )
    if target == "band_gap":
        ref = spec.reference_material
        soc = f", with spin-orbit coupling for {', '.join(chem['soc'])}" if chem["soc"] else ""
        steps.append(
            "**Band structure (hybrid DFT).** HSE06 (25 % exact exchange, screening 0.2 Å⁻¹"
            + (", no +U" if chem["hubbard_u"] else "")
            + f") on the relaxed geometry{soc}: self-consistent run, then the band structure along a "
            "standard high-symmetry path (seekpath) and the projected density of states. Pass if the gap "
            f"meets the target {_target_text(facts)}; stop if it misses by more than the screening error bar."
        )
        if ref:
            steps.append(
                f"**Reference at the same level.** Compute {ref} with identical settings and compare "
                "gaps, not absolute values: the target is relative."
            )
        steps.append(
            "**Optical response.** Frequency-dependent dielectric function (HSE06 or GW on top) for the "
            "absorption edge, and effective masses at the band edges."
        )
    elif target in MECHANICAL:
        steps.append(
            "**Elastic tensor.** Stress-strain with ±0.5 % and ±1 % strains (ions relaxed) at the higher cutoff, "
            "or the finite-difference elastic tensor. Pass if the Born stability criteria hold and the "
            f"{property_name(target).lower()} meets the target {_target_text(facts)}."
        )
        steps.append(
            "**Electronic structure.** PBEsol band structure and density of states, to confirm the "
            "candidate is an insulator or metal as expected (hybrid DFT only if the gap matters)."
        )
    else:
        steps.append(
            f"**{property_name(target)}.** Compute it with the method standard for this property, at the "
            f"converged settings. Pass if it meets the target {_target_text(facts)}."
        )
    t = spec.conditions.temperature_K
    steps.append(
        f"**Finite temperature (optional).** A few picoseconds of ab initio MD at {_fmt(t)} K (NVT); pass if "
        "the structure holds with no diffusion or amorphisation."
    )
    if facts.get("run_id"):
        steps.append(
            f"**Fallback.** If a step fails, repeat the cheapest failing step for the runner-up, "
            f"{facts.get('run_composition')} ({candidate_label(facts.get('run_id'))})."
        )
    return steps


def _target_text(facts: dict[str, Any]) -> str:
    """Return the target as text, e.g. "(< 3.12 eV)".

    Parameters
    ----------
    facts : dict
        Facts.

    Returns
    -------
    str
        Target in brackets, or empty when there is none.
    """
    if facts.get("target_value") is None:
        return ""
    sign = {"minimise": "<", "maximise": ">", "hit": "≈"}.get(facts.get("direction") or "", "")
    return f"({sign} {_fmt(facts['target_value'])} {facts.get('target_units') or ''})".replace(" )", ")")


def expected_table(ws: Workspace, state: RunState, cs: CandidateState | None, facts: dict[str, Any]) -> str:
    """Tabulate what the screening predicted, so the DFT results can be checked against it.

    Parameters
    ----------
    ws : Workspace
        Workspace.
    state : RunState
        Run state (final re-check).
    cs : CandidateState or None
        The recommended candidate.
    facts : dict
        Facts.

    Returns
    -------
    str
        Markdown table.
    """
    if cs is None:
        return "No candidate was evaluated."
    pot = cs.potential or ws.cfg.potential
    rows: list[tuple[str, str, str]] = []
    if cs.lattice:
        lat = cs.lattice
        rows.append(
            (
                "Lattice (a, b, c)",
                f"{_fmt(lat.a)}, {_fmt(lat.b)}, {_fmt(lat.c)} Å ({pot})",
                "DFT within a few per cent; more means the screening structure is off",
            )
        )
    if cs.space_group:
        rows.append(("Space group", f"{cs.space_group} ({pot})", "Should survive the DFT relaxation"))
    if cs.e_hull is not None:
        rows.append(
            (
                "Energy above hull",
                f"{_fmt(cs.e_hull)} eV/atom ({pot}, MP phases re-relaxed with it)",
                f"Limit {_fmt(ws.spec.stability_thresholds.e_hull_max_eV_per_atom)} eV/atom",
            )
        )
    if cs.e_form is not None:
        rows.append(
            ("Formation energy", f"{_fmt(cs.e_form)} eV/atom ({pot})", "Sanity check on the references")
        )
    if cs.phonon:
        verdict = VERDICTS.get((cs.phonon_details or {}).get("verdict") or "", "")
        rows.append(
            (
                "Lowest phonon frequency",
                f"{_fmt(cs.phonon.min_frequency_THz)} THz ({cs.phonon.mode} phonons{', ' + verdict if verdict else ''})",
                "Imaginary modes shown as negative",
            )
        )
    pred = cs.predictions.get(ws.spec.target_property)
    if pred:
        unc = f" ± {_fmt(pred.value.uncertainty)}" if pred.value.uncertainty is not None else ""
        rows.append(
            (
                property_name(ws.spec.target_property),
                f"{_fmt(pred.value.value)}{unc} {pred.value.units} ({pred.model})",
                f"Target {_target_text(facts).strip('()')}" if _target_text(facts) else DASH,
            )
        )
    ref = ws.spec.reference_value
    if ws.spec.reference_material and ref is not None:
        rows.append(
            (
                f"Reference {ws.spec.reference_material}",
                f"{_fmt(ref.value)} {ref.units} on the same model",
                "Compute it at your DFT level too",
            )
        )
    if state.validation:
        v = state.validation[0]
        rows.append(("Final re-check", v.verdict.capitalize(), "Higher-fidelity potential, full phonons"))
    lines = ["| Quantity | Screening value | Use it to check |", "| --- | --- | --- |"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(lines)


def compare_papers(papers: dict[str, Paper], limit: int = 6) -> list[Paper]:
    """Papers the run read that report first-principles work, most relevant first.

    Parameters
    ----------
    papers : dict
        The run's papers.
    limit : int
        Most to return.

    Returns
    -------
    list of Paper
        Papers whose title or abstract mentions DFT methods.
    """
    hits = [p for p in papers.values() if DFT_WORDS.search(f"{p.title} {p.abstract or ''}")]
    hits.sort(key=lambda p: (-p.relevance, -(p.citation_count or 0)))
    return hits[:limit]


def assemble_compute_brief(state: RunState, ws: Workspace) -> Brief:
    """Assemble the computational brief from stored results (EB-2).

    Parameters
    ----------
    state : RunState
        Run state with the submission.
    ws : Workspace
        Workspace.

    Returns
    -------
    Brief
        Markdown brief; ``cited_paper_ids`` lists the methods papers.
    """
    facts = collect_facts(state, ws)
    win = facts.get("win_id")
    cs = ws.candidates.get(win) if win else None
    comp = facts.get("win_composition")
    lines = [f"# {TITLE}: {comp or 'no candidate'}", ""]
    if cs is None or not comp:
        lines += ["No candidate was submitted, so there is nothing to compute.", ""]
        return Brief(run_id=state.run_id, markdown="\n".join(lines))
    chem = chemistry(comp)
    s = _structure(cs)
    n_atoms = len(s) if s is not None else None
    target = ws.spec.target_property
    mechanical = target in MECHANICAL
    steps = workflow_steps(ws, cs, chem, facts, n_atoms)
    first = f"convergence tests and a full relaxation of the {n_atoms or ''}-atom cell".replace(" -atom", "")
    deciding = (
        "HSE06 band gap"
        if target == "band_gap"
        else "DFT elastic tensor"
        if mechanical
        else f"DFT {property_name(target).lower()}"
    )
    special = [
        *(["spin-polarised"] if chem["magnetic"] else []),
        *(["spin-orbit coupling"] if chem["soc"] else []),
        *(["site ordering"] if chem["partial"] else []),
    ]
    lines += [
        f"- **What to compute:** Check {comp} from first principles before it goes to the lab.",
        f"- **Starting structure:** {n_atoms or 'n/a'} atoms, {cs.space_group or 'space group n/a'}, relaxed with {cs.potential or ws.cfg.potential}.",
        "- **Level of theory:** PBEsol"
        + ("+U" if chem["hubbard_u"] else "")
        + " geometry"
        + (", HSE06 gaps" if target == "band_gap" else "")
        + (f"; {', '.join(special)}" if special else "")
        + ".",
        f"- **Deciding calculation:** The {deciding}.",
        f"- **First calculation:** {first[0].upper() + first[1:]}.",
        "",
    ]
    body: dict[str, str] = {}
    pred = cs.predictions.get(target)
    body["Starting point"] = (
        f"**{comp}** ({candidate_label(win)}), {(facts.get('win_structure_type') or 'structure').replace('_', ' ')}, "
        f"space group {cs.space_group or 'n/a'}. Screening used the machine-learned potential "
        f"{cs.potential or ws.cfg.potential}"
        + (f" and {pred.model} for the {property_name(target).lower()}" if pred else "")
        + ". Those models are fast but approximate; the calculations below check the claims that matter "
        "at density-functional level, cheapest and most decisive first."
    )
    body["Calculation settings"] = settings_table(chem, mechanical)
    body["Workflow"] = "\n".join(f"{n}. {step}" for n, step in enumerate(steps, 1))
    body["Screening results to check"] = expected_table(ws, state, cs, facts)
    if s is not None:
        note = (
            " It already includes the distortion found by mode mapping."
            if (cs.phonon_details or {}).get("verdict") == "stable_after_distortion"
            else ""
        )
        body["Starting structure"] = (
            f"Relaxed with {cs.potential or ws.cfg.potential}; start the DFT relaxation from it.{note}\n\n"
            f"```\n{s.to(fmt='poscar').strip()}\n```"
        )
    else:
        body["Starting structure"] = "The structure was not stored for this run."
    papers = compare_papers(ws.papers)
    refs = []
    for n, p in enumerate(papers, 1):
        url = paper_url(p)
        link = f" [{id_label(p.paper_id)}]({url})" if url else f" {id_label(p.paper_id)}"
        refs.append(f"{n}. **{p.title}**" + (f" ({p.year})" if p.year else "") + "." + link)
    body["Methods to compare against"] = (
        "\n".join(refs) if refs else "None of the papers the run read report first-principles calculations."
    )
    body["Scope and assumptions"] = "\n".join(
        [
            "- Settings are starting points; run the convergence tests before trusting any number.",
            "- Hubbard U values are Materials Project's fitted set for oxides and fluorides.",
            "- Screening values come from machine-learned models; DFT values will differ, and a relative "
            "target should be judged against the reference computed the same way.",
            *([f"- {a}" for a in ws.spec.assumptions]),
        ]
    )
    for title in SECTIONS:
        lines += [f"## {title}", "", body[title], ""]
    return Brief(
        run_id=state.run_id,
        markdown=strip_codes("\n".join(lines)),
        cited_paper_ids=[p.paper_id for p in papers],
    )
