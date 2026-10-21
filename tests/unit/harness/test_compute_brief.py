"""Computational brief: DFT settings and workflow from stored results and the candidate's chemistry."""

from factories import candidate_state, workspace
from pymatgen.core import Lattice, Structure

from hephaestus.core.types import RunState, Submission
from hephaestus.harness.compute_brief import SECTIONS, assemble_compute_brief, chemistry


def brief(composition: str = "SrTi0.875Fe0.125O3", with_structure: bool = True, **cfg: object) -> str:
    """Build the computational brief for one submitted candidate.

    Parameters
    ----------
    composition : str
        Candidate formula.
    with_structure : bool
        Store a relaxed structure on the candidate.
    **cfg
        ``RunConfig`` overrides.

    Returns
    -------
    str
        The brief's Markdown.
    """
    ws = workspace(**cfg)
    cs = candidate_state(composition=composition)
    if with_structure:
        s = Structure(
            Lattice.cubic(3.9),
            ["Sr", "Ti", "O", "O", "O"],
            [[0, 0, 0], [0.5] * 3, [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        )
        cs.structure = s.as_dict()
    ws.candidates["c1-1"] = cs
    state = RunState(run_id="r-test", raw_goal=ws.spec.raw_goal, config=ws.cfg, spec=ws.spec)
    state.submission = Submission(candidate_id="c1-1", justification="best", gate_passed=True)
    return assemble_compute_brief(state, ws).markdown


def test_chemistry_rules() -> None:
    """Fe in an oxide gets +U and spin; Bi gets spin-orbit coupling; fractional amounts are partial."""
    fe = chemistry("SrTi0.875Fe0.125O3")
    assert fe["hubbard_u"] == {"Fe": 5.3} and fe["magnetic"] == ["Fe"] and fe["partial"]
    bi = chemistry("BiFeO3")
    assert bi["soc"] == ["Bi"] and not bi["partial"]
    assert chemistry("FeS2")["hubbard_u"] == {}  # U only for oxides and fluorides, as in MP
    assert not chemistry("SrTiO3")["magnetic"]


def test_brief_has_every_section_in_order() -> None:
    """All sections appear once, in order, under a computational-brief title."""
    md = brief()
    assert md.startswith("# Computational brief: SrTi0.875Fe0.125O3")
    heads = [line[3:] for line in md.splitlines() if line.startswith("## ")]
    assert heads == SECTIONS


def test_band_gap_target_asks_for_hybrid_band_structure_and_reference() -> None:
    """A band-gap target gets an HSE06 band structure and the reference at the same level."""
    md = brief()
    assert "HSE06" in md and "Reference at the same level" in md and "SrTiO3" in md
    assert "Hubbard U" in md and "Fe 5.3 eV" in md and "Magnetic order" in md and "Site ordering" in md


def test_structure_is_given_as_poscar() -> None:
    """The stored structure is included as a POSCAR; without one, the brief says so."""
    assert "```\nSr1 Ti1 O3" in brief()
    assert "was not stored" in brief(with_structure=False)


def test_checks_turned_off_are_left_out() -> None:
    """With the hull and phonons off, their steps are not in the workflow."""
    from hephaestus.harness.goal import apply_stability_options

    ws_md = brief()
    assert "**Energy above hull.**" in ws_md and "**Phonons.**" in ws_md
    ws = workspace(run_hull=False, run_phonons=False)
    ws.spec = apply_stability_options(ws.spec, ws.cfg)
    ws.candidates["c1-1"] = candidate_state()
    state = RunState(run_id="r-test", raw_goal=ws.spec.raw_goal, config=ws.cfg, spec=ws.spec)
    state.submission = Submission(candidate_id="c1-1", justification="best", gate_passed=True)
    md = assemble_compute_brief(state, ws).markdown
    assert "**Energy above hull.**" not in md and "**Phonons.**" not in md


def test_no_submission_gives_a_short_brief() -> None:
    """Without a submitted candidate the brief says there is nothing to compute."""
    ws = workspace()
    state = RunState(run_id="r-test", raw_goal=ws.spec.raw_goal, config=ws.cfg, spec=ws.spec)
    assert "nothing to compute" in assemble_compute_brief(state, ws).markdown
