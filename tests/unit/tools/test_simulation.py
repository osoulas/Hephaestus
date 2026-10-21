"""Simulation engine on toy calculators: relaxation, hull, phonon verdicts, MD (Section 5)."""

from pathlib import Path

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from factories import workspace
from toy import A0, DoubleWell, toy_cell

from hephaestus.core.types import Candidate, StructureRecord
from hephaestus.tools import simulation as S
from hephaestus.tools.registry import CandidateState, Workspace, execute


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the cache at a temporary folder.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.
    monkeypatch : pytest.MonkeyPatch
        Patches the environment.
    """
    monkeypatch.setenv("HEPHAESTUS_CACHE_DIR", str(tmp_path / "cache"))


def test_se1_se3_relax_shrinks_strained_cell_and_reports_convergence() -> None:
    """A stretched Cu cell relaxes back; convergence follows the optimiser."""
    atoms = bulk("Cu", "fcc", a=3.8)
    out, _, converged, steps = S.relax_atoms(atoms, EMT(), fmax=0.01)
    assert converged and steps > 0
    assert out.get_volume() < atoms.get_volume()
    _, _, conv2, _ = S.relax_atoms(atoms, EMT(), fmax=1e-6, max_steps=2)
    assert not conv2


def test_se5_hull_known_answer() -> None:
    """AB at -1.0 eV/atom with elements at 0 is on the hull; at +0.1 above the A + B tie-line it is not."""
    phases = [
        {"composition": "Sr", "energy_eV_per_atom": 0.0},
        {"composition": "O", "energy_eV_per_atom": 0.0},
        {"composition": "SrO", "energy_eV_per_atom": -1.0},
    ]
    e_hull, e_form, products = S.energy_above_hull("SrO", -1.0, phases)
    assert e_hull == pytest.approx(0.0, abs=1e-9) and e_form == pytest.approx(-1.0)
    e_hull, _, products = S.energy_above_hull("Sr2O", -0.5, phases)  # tie-line at 2/3·0 + ... = -0.6667
    assert e_hull == pytest.approx(-0.5 - (-2.0 / 3.0)) and sorted(products) == ["Sr", "SrO"]
    with pytest.raises(ValueError, match="elemental reference"):
        S.energy_above_hull("SrTiO3", -1.0, phases)


def test_subsystems_cover_every_combination() -> None:
    """Sr-Ti-O has 7 subsystems."""
    assert S.subsystems(["Ti", "O", "Sr"]) == ["O", "Sr", "Ti", "O-Sr", "O-Ti", "Sr-Ti", "O-Sr-Ti"]


def test_supercell_and_labels() -> None:
    """Supercells reach 7 Å; q-points are named."""
    assert S.supercell_dims(toy_cell(), 7.0) == [3, 3, 3]
    assert [S.q_label(q) for q in ([0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0.5, -0.5, 0.5])] == [
        "Γ",
        "X",
        "M",
        "R",
    ]


@pytest.mark.parametrize(
    ("A", "verdict", "dyn"),
    [
        ((1.0, 1.0, 1.0), "stable", True),
        ((-0.02, 1.0, 1.0), "spurious_soft_mode", True),  # well depth 0.1 meV < 2 meV
        ((-1.0, 1.0, 1.0), "stable_after_distortion", True),  # 250 meV, unstable along x only
        ((-1.0, -1.0, -1.0), "unstable", False),  # still unstable after 2 distortions
    ],
)
def test_se13_se14_mode_mapping_verdicts(A: tuple[float, float, float], verdict: str, dyn: bool) -> None:
    """Each verdict arises from the physics it names; only "unstable" fails the gate.

    Parameters
    ----------
    A : tuple of float
        Quadratic coefficients per direction.
    verdict : str
        Expected verdict.
    dyn : bool
        Expected ``dyn_stable``.
    """
    calc = DoubleWell(A0, A)
    res = S.phonon_check(toy_cell(), 0.0, calc, threshold_meV=2.0, fmax=1e-3)
    assert res["verdict"] == verdict and res["dyn_stable"] is dyn
    if verdict == "stable_after_distortion":
        assert res["rounds"][0]["real"] and res["rounds"][0]["dE_meV_per_atom"] < -200
        assert res["energy_eV_per_atom"] == pytest.approx(-0.25, abs=1e-3)
    if verdict == "spurious_soft_mode":
        assert not res["rounds"][0]["real"] and res["min_frequency_THz"] < -0.1


def test_se13_strict_gate_fails_any_imaginary_mode() -> None:
    """With phonon_gate="strict", a spurious soft mode fails."""
    res = S.phonon_check(toy_cell(), 0.0, DoubleWell(A0, (-0.02, 1, 1)), gate="strict")
    assert res["verdict"] == "unstable" and not res["dyn_stable"] and res["rounds"] == []


def test_se9_md_on_stable_crystal() -> None:
    """Cu at 300 K for 0.2 ps stays intact."""
    res = S.md_check(bulk("Cu", "fcc", a=3.61, cubic=True), EMT(), 300.0, 0.2)
    assert res["stable"] and res["max_displacement_A"] < 0.5


# ---------------------------------------------------------------- tool handlers


def ws_with_cu(monkeypatch: pytest.MonkeyPatch, calc: object) -> Workspace:
    """Workspace with one built Cu candidate and every potential mapped to ``calc``.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Patches potentials and the phase source.
    calc : Calculator
        Calculator to use.

    Returns
    -------
    Workspace
        The workspace.
    """
    monkeypatch.setattr(S, "FACTORIES", {name: (lambda d, t: calc) for name in S.FACTORIES})
    monkeypatch.setattr(S, "_LOADED", {})
    monkeypatch.setattr(
        S,
        "PHASE_SOURCE",
        lambda els: [
            {"composition": "Cu", "mp_id": "mp-30", "pmg_dict": S.to_pmg(bulk("Cu", "fcc", a=3.6)).as_dict()},
        ],
    )
    ws = workspace()
    cand = Candidate(
        candidate_id="c1-1", iteration=1, composition="Cu", structure_type="perovskite_cubic", rationale="r"
    )
    rec = StructureRecord(
        candidate_id="c1-1",
        cif="",
        pmg_dict=S.to_pmg(bulk("Cu", "fcc", a=3.7)).as_dict(),
        source="mp:mp-30",
        n_atoms=1,
    )
    ws.candidates["c1-1"] = CandidateState(candidate=cand, round=1, record=rec)
    return ws


def test_tools_relax_hull_phonons_md(monkeypatch: pytest.MonkeyPatch) -> None:
    """The handlers chain on a workspace and set the code verdicts."""
    ws = ws_with_cu(monkeypatch, EMT())
    r, err = execute(ws, "energy_above_hull", {"candidate_id": "c1-1"})
    assert err and "relax first" in r["error"]
    r, err = execute(ws, "relax", {"candidate_id": "c1-1", "potential": "mace-mp-0-small", "fmax": 0.01})
    assert not err and r["converged"] and r["space_group"] == "Fm-3m"
    r, err = execute(ws, "energy_above_hull", {"candidate_id": "c1-1"})
    assert not err and r["thermo_stable"] and abs(r["e_hull_eV_per_atom"]) < 1e-3
    r, err = execute(ws, "phonons", {"candidate_id": "c1-1", "mode": "fast"})
    assert not err and r["verdict"] == "stable" and r["dyn_stable"]
    r, err = execute(ws, "run_md", {"candidate_id": "c1-1", "duration_ps": 0.5})
    assert not err and r["stable"], r
    cs = ws.candidates["c1-1"]
    assert cs.potential_in_domain and cs.phonon is not None and ws.compute_s > 0


def test_se14_distortion_marks_hull_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    """When mode mapping changes the structure, the hull is stale and predictions are cleared."""
    calc = DoubleWell(A0, (-1.0, 1.0, 1.0))
    ws = ws_with_cu(monkeypatch, calc)
    ws.candidates["c1-1"].record = StructureRecord(
        candidate_id="c1-1", cif="", pmg_dict=S.to_pmg(toy_cell()).as_dict(), source="mp:x", n_atoms=1
    )
    monkeypatch.setattr(
        S, "PHASE_SOURCE", lambda els: [{"composition": "Cu", "pmg_dict": S.to_pmg(toy_cell()).as_dict()}]
    )
    execute(ws, "relax", {"candidate_id": "c1-1", "potential": "mace-mp-0-small", "fmax": 0.01})
    execute(ws, "energy_above_hull", {"candidate_id": "c1-1"})
    r, err = execute(ws, "phonons", {"candidate_id": "c1-1", "mode": "fast"})
    assert not err and r["verdict"] == "stable_after_distortion" and "rerun energy_above_hull" in r["note"]
    assert ws.candidates["c1-1"].hull_stale


def test_dl3_relaxation_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second identical relaxation does not call the calculator."""
    calls = {"n": 0}

    class Counting(EMT):
        """EMT that counts calculations."""

        def calculate(self, *a: object, **k: object) -> None:
            """Count, then compute.

            Parameters
            ----------
            *a, **k : object
                Passed to EMT.
            """
            calls["n"] += 1
            super().calculate(*a, **k)

    atoms = bulk("Cu", "fcc", a=3.7)
    S.relax_cached(atoms, "mace-mp-0-small", Counting(), 0.01, False, "float32")
    n = calls["n"]
    S.relax_cached(atoms, "mace-mp-0-small", Counting(), 0.01, False, "float32")
    assert n > 0 and calls["n"] == n


def test_se12_domain_check() -> None:
    """Noble gases are outside MPtrj."""
    assert S.domain_check(["Sr", "Ti", "O"], "m")[0]
    assert not S.domain_check(["Xe", "F"], "m")[0]
