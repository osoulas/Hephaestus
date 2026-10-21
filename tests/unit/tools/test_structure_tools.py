"""Tool handlers propose_candidate and build_structure (CG-2 to CG-12, SG-1, SG-10, SG-12)."""

from pathlib import Path

import pytest
from factories import workspace
from mp_fake import FakeMP, install_fake_mp

from hephaestus.core.types import Candidate, Paper
from hephaestus.tools import structures as S
from hephaestus.tools.registry import CandidateState, ToolError, Workspace, execute


@pytest.fixture
def fake_mp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeMP:
    """Route MP calls to recorded fixtures and the cache to a temporary folder.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Patches the fetch functions and the cache location.
    tmp_path : pathlib.Path
        Cache folder.

    Returns
    -------
    FakeMP
        The fake, with its call log.
    """
    return install_fake_mp(monkeypatch, tmp_path)


def _ws(**cfg: object) -> Workspace:
    """Return a workspace with one retrieved paper.

    Parameters
    ----------
    **cfg
        ``RunConfig`` overrides.

    Returns
    -------
    Workspace
        Fresh workspace.
    """
    ws = workspace(**cfg)
    ws.papers["p1"] = Paper(paper_id="p1", title="t")
    return ws


def _propose(ws: Workspace, composition: str, structure_type: str = "perovskite_cubic", **kw: object) -> dict:
    """Call propose_candidate with defaults.

    Parameters
    ----------
    ws : Workspace
        Workspace.
    composition : str
        Formula.
    structure_type : str
        Prototype key.
    **kw
        Overrides.

    Returns
    -------
    dict
        Tool result.
    """
    args = {
        "composition": composition,
        "structure_type": structure_type,
        "rationale": "r",
        "evidence_ids": ["p1"],
        "prior_estimate": 2.5,
        "novelty": "variant",
        "synthesis_feasibility": "easy",
        "risks": [],
    } | kw
    return S.propose_candidate(ws, **args)  # type: ignore[arg-type]


def test_propose_accepts_and_reports_sites() -> None:
    """A valid ordered candidate gets an ID, its sites and the tolerance factor."""
    ws = _ws()
    out = _propose(ws, "Sr2FeMoO6", "double_perovskite")
    assert out["accepted"] and out["candidate_id"] == "c1-1"
    assert out["sites"] == {"A": {"Sr": 1.0}, "B": {"Fe": 1.0}, "B'": {"Mo": 1.0}, "X": {"O": 1.0}}
    assert 0.9 < out["tolerance_factor"] < 1.1
    prior = ws.candidates["c1-1"].candidate.prior_estimate
    assert prior is not None and prior.value == 2.5


def test_cg2_excluded_element_rejected() -> None:
    """Lead is refused when the spec excludes it."""
    ws = _ws()
    ws.spec.excluded_elements = ["Pb"]
    out = _propose(ws, "PbTiO3")
    assert not out["accepted"] and "CG-2" in out["reasons"][0]


def test_cg3_rejected_unless_declared() -> None:
    """An unbalanced ordered formula is refused, unless an off-stoichiometric risk is declared."""
    ws = _ws()
    out = _propose(ws, "BaBiO3")
    assert not out["accepted"] and any("CG-3" in r for r in out["reasons"])
    out = _propose(ws, "BaBiO3", risks=["off-stoichiometric: Bi3+/Bi5+ charge disproportionation"])
    assert out["accepted"] and out["warnings"]


def test_cg9_template_mismatch_rejected() -> None:
    """A composition that cannot fill the prototype is refused with the reason."""
    out = _propose(_ws(), "Gd2Ti2O7", "perovskite_cubic")
    assert not out["accepted"] and any("CG-9" in r for r in out["reasons"])
    assert _propose(_ws(), "Gd2Ti2O7", "pyrochlore")["accepted"]


def test_cg6_repeat_rejected_even_if_written_differently() -> None:
    """SrTiO3 and Sr2Ti2O6 are the same composition."""
    ws = _ws()
    assert _propose(ws, "SrTiO3")["accepted"]
    out = _propose(ws, "Sr2Ti2O6")
    assert not out["accepted"] and any("CG-6" in r for r in out["reasons"])


def test_cg4_evidence_required() -> None:
    """Evidence must be a retrieved paper or an mp: ID."""
    assert any("CG-4" in r for r in _propose(_ws(), "SrTiO3", evidence_ids=[])["reasons"])
    assert any("CG-4" in r for r in _propose(_ws(), "SrTiO3", evidence_ids=["nope"])["reasons"])
    assert _propose(_ws(), "SrTiO3", evidence_ids=["mp:mp-5229"])["accepted"]


@pytest.mark.parametrize(
    ("formula", "risks", "shared"),
    [
        ("SrTi0.875Fe0.125O3", [], "site B would hold Fe 0.125 + Ti 0.875"),
        ("Sr0.75La0.25Ti0.75Al0.25O3", [], "site A would hold La 0.25 + Sr 0.75"),
        ("SrTiO2.875", ["off-stoichiometric: oxygen vacancies"], "site X would hold O"),
    ],
)
def test_disordered_candidate_rejected(formula: str, risks: list[str], shared: str) -> None:
    """Doped solid solutions and partly vacant sites are parked: refused, naming the shared site."""
    out = _propose(_ws(), formula, risks=risks)
    assert not out["accepted"]
    assert any("not supported yet" in r and shared in r for r in out["reasons"])


def test_ordered_mixed_cation_prototype_is_accepted() -> None:
    """Two B cations in an ordered prototype are fine; the same formula as a cubic perovskite is not."""
    assert _propose(_ws(), "Sr2FeMoO6", "double_perovskite")["accepted"]
    assert not _propose(_ws(), "SrFe0.5Mo0.5O3", "perovskite_cubic")["accepted"]


def test_cg12_novelty_rejects_known_compound(fake_mp: FakeMP) -> None:
    """With require_novel, an ICSD-backed MP compound is refused with its IDs."""
    out = _propose(_ws(require_novel=True), "LaFeO3", "perovskite_orthorhombic")
    assert not out["accepted"] and any("CG-12" in r for r in out["reasons"])


def test_sg1_build_auto_uses_mp_default(fake_mp: FakeMP) -> None:
    """LaFeO3 builds as the observed Pnma polymorph from MP, listing the alternatives."""
    ws = _ws()
    cid = _propose(ws, "LaFeO3", "perovskite_orthorhombic")["candidate_id"]
    out = S.build_structure(ws, cid, "auto", None, 0)
    assert out["source"].startswith("mp:") and out["polymorph"]["space_group"] == "Pnma"
    assert out["polymorph"]["other_polymorphs"]
    assert out["space_group"] == "Pnma" and out["n_atoms"] == 20
    assert ws.candidates[cid].record is not None


def test_build_template_for_ordered_compound(fake_mp: FakeMP) -> None:
    """source=template builds the prototype cell without asking MP."""
    ws = _ws()
    cid = _propose(ws, "Y2Sn2O7", "pyrochlore")["candidate_id"]
    out = S.build_structure(ws, cid, "template", None, 7)
    assert out["source"] == "template:pyrochlore"
    assert out["n_atoms"] == 22 and out["supercell"] == [1, 1, 1] and out["space_group"] == "Fd-3m"
    assert fake_mp.calls == []


def test_build_refuses_disordered_candidate(fake_mp: FakeMP) -> None:
    """A disordered candidate that bypassed propose_candidate is still not built."""
    ws = _ws()
    cand = Candidate(
        candidate_id="c1-1",
        iteration=1,
        composition="SrTi0.875Fe0.125O3",
        structure_type="perovskite_cubic",
        rationale="r",
    )
    ws.candidates["c1-1"] = CandidateState(candidate=cand, round=1)
    with pytest.raises(ToolError, match="disordered"):
        S.build_structure(ws, "c1-1", "auto", None, 0)
    assert ws.candidates["c1-1"].errors[-1].code == "disordered"


def test_build_mp_unavailable_falls_back_to_template(fake_mp: FakeMP) -> None:
    """Offline with nothing cached, auto degrades to the template and says so (SYS-6)."""
    ws = _ws(offline=True)
    cid = _propose(ws, "SrTiO3")["candidate_id"]
    out = S.build_structure(ws, cid, "auto", None, 0)
    assert out["source"] == "template:perovskite_cubic"
    assert "unavailable" in out["notes"][0]


def test_build_resets_later_results(fake_mp: FakeMP) -> None:
    """Rebuilding clears every later result, including ones added to CandidateState later."""
    ws = _ws()
    cid = _propose(ws, "SrTiO3")["candidate_id"]
    cs = ws.candidates[cid]
    cs.energy_eV_per_atom, cs.e_hull, cs.converged, cs.space_group = -7.0, 0.01, True, "Pm-3m"
    cs.elastic, cs.domain_note, cs.wall_time_s = {"K": 170.0}, "old structure", 12.0
    S.build_structure(ws, cid, "template", None, 0)
    after = ws.candidates[cid]
    assert after is cs
    assert (after.energy_eV_per_atom, after.e_hull, after.converged, after.space_group) == (
        None,
        None,
        False,
        None,
    )
    assert (after.elastic, after.domain_note, after.wall_time_s) == (None, "", 0.0)
    assert after.record is not None and after.candidate.composition == "SrTiO3"


def test_sg10_build_failure_is_named_and_recorded(fake_mp: FakeMP) -> None:
    """source=mp without an ID fails with a named code, kept in the candidate's errors."""
    ws = _ws()
    cid = _propose(ws, "SrTiO3")["candidate_id"]
    with pytest.raises(ToolError, match="mp_missing"):
        S.build_structure(ws, cid, "mp", None, 0)
    assert ws.candidates[cid].errors[-1].code == "mp_missing"


def test_handlers_registered_and_dispatched(fake_mp: FakeMP) -> None:
    """The real handlers are registered and run through execute without raising."""
    ws = _ws()
    out, err = execute(
        ws,
        "propose_candidate",
        {
            "composition": "Y2Sn2O7",
            "structure_type": "pyrochlore",
            "rationale": "r",
            "evidence_ids": ["p1"],
            "prior_estimate": None,
            "novelty": "known",
            "synthesis_feasibility": "easy",
            "risks": [],
        },
    )
    assert not err and out["accepted"]
    out, err = execute(
        ws, "build_structure", {"candidate_id": "c1-1", "source": "template", "mp_id": None, "seed": 0}
    )
    assert not err and out["space_group"] == "Fd-3m"


def test_mp_structure_type_accepted_when_in_mp(fake_mp: FakeMP) -> None:
    """structure_type 'mp' skips the prototype fit and reports MP's default polymorph."""
    out = _propose(_ws(), "LaFeO3", "mp")
    assert out["accepted"] and out["sites"] is None and out["tolerance_factor"] is None
    assert out["mp_default_polymorph"]["space_group"] == "Pnma"


@pytest.mark.parametrize(
    ("formula", "reason"),
    [("Gd3Ga5O12", "not in Materials Project"), ("SrTi0.875Fe0.125O3", "exact (integer) composition")],
)
def test_mp_structure_type_needs_an_mp_entry(fake_mp: FakeMP, formula: str, reason: str) -> None:
    """Without an MP entry there is nothing to build, so the candidate is refused up front."""
    out = _propose(_ws(), formula, "mp")
    assert not out["accepted"] and any(reason in r for r in out["reasons"])


def test_mp_structure_type_offline_without_cache_is_refused(fake_mp: FakeMP) -> None:
    """Offline with nothing cached, an 'mp' candidate cannot be checked and is refused."""
    out = _propose(_ws(offline=True), "LaFeO3", "mp")
    assert not out["accepted"] and any("unavailable" in r for r in out["reasons"])


def test_mp_structure_type_builds_from_mp_only(fake_mp: FakeMP) -> None:
    """An 'mp' candidate builds its MP polymorph; source=template is a named failure."""
    ws = _ws()
    cid = _propose(ws, "LaFeO3", "mp")["candidate_id"]
    with pytest.raises(ToolError, match="no_template"):
        S.build_structure(ws, cid, "template", None, 0)
    out = S.build_structure(ws, cid, "auto", None, 0)
    assert out["polymorph"]["space_group"] == "Pnma"
    assert out["polymorph"]["matches_structure_type"] is None


def test_mp_structure_type_offline_build_does_not_fall_back(fake_mp: FakeMP) -> None:
    """With no prototype, an unreachable MP is an error rather than a template build."""
    ws = _ws()
    cid = _propose(ws, "LaFeO3", "mp")["candidate_id"]
    ws.cfg.offline = True  # proposing cached LaFeO3, so the offline build still works
    out = S.build_structure(ws, cid, "auto", None, 0)
    assert out["source"].startswith("mp:")
    ws2 = _ws(offline=True)
    cand = Candidate(
        candidate_id="c1-1", iteration=1, composition="BaSnO3", structure_type="mp", rationale="r"
    )
    ws2.candidates["c1-1"] = CandidateState(candidate=cand, round=1)
    with pytest.raises(ToolError, match="mp_unavailable"):
        S.build_structure(ws2, "c1-1", "auto", None, 0)


def test_polymorph_mismatch_is_flagged(fake_mp: FakeMP) -> None:
    """LaFeO3 proposed as cubic but built as MP's Pnma phase carries a note; a matching type does not."""
    ws = _ws()
    cid = _propose(ws, "LaFeO3", "perovskite_cubic")["candidate_id"]
    out = S.build_structure(ws, cid, "auto", None, 0)
    assert out["polymorph"]["matches_structure_type"] is False
    assert any("does not have the space group of the proposed perovskite_cubic" in n for n in out["notes"])
    ws = _ws()
    cid = _propose(ws, "LaFeO3", "perovskite_orthorhombic")["candidate_id"]
    out = S.build_structure(ws, cid, "auto", None, 0)
    assert out["polymorph"]["matches_structure_type"] is True
    assert not any("does not have the space group" in n for n in out["notes"])


def test_cg6_close_formulas_are_different_compositions() -> None:
    """Fe3O4 and Fe2O3 are distinct candidates; only the same composition counts as a repeat."""
    ws = _ws()
    assert _propose(ws, "Fe3O4", "spinel")["accepted"]
    assert _propose(ws, "Fe2O3", "corundum")["accepted"]


def test_inverse_spinel_warning_at_proposal() -> None:
    """Proposing a known inverse spinel on the spinel template warns."""
    out = _propose(_ws(), "NiFe2O4", "spinel")
    assert out["accepted"] and any("inverse spinel" in w for w in out["warnings"])


def test_sg8_oversized_mp_cell(fake_mp: FakeMP) -> None:
    """An MP cell above max_atoms: auto falls back to the template; mp_id or 'mp' candidates fail."""
    ws = _ws(max_atoms=10)
    cid = _propose(ws, "LaFeO3", "perovskite_cubic")["candidate_id"]
    out = S.build_structure(ws, cid, "auto", None, 0)
    assert out["source"] == "template:perovskite_cubic" and out["n_atoms"] <= 10
    assert any("above max_atoms=10" in n for n in out["notes"])
    pnma = next(e["mp_id"] for e in S.mp.mp_polymorphs("LaFeO3") if e["space_group"] == "Pnma")
    with pytest.raises(ToolError, match="supercell"):
        S.build_structure(ws, cid, "mp", pnma, 0)
    ws_mp = _ws(max_atoms=10)  # a fresh run: LaFeO3 was already proposed in ws (CG-6)
    mp_cid = _propose(ws_mp, "LaFeO3", "mp")["candidate_id"]
    with pytest.raises(ToolError, match="supercell"):
        S.build_structure(ws_mp, mp_cid, "auto", None, 0)
