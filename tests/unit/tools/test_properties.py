"""Elastic tensor and the properties derived from it (Section 6, PP-1 to PP-6)."""

from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.eos import EquationOfState
from ase.units import GPa
from factories import workspace

from hephaestus.core.types import Candidate, StructureRecord
from hephaestus.tools import simulation as S
from hephaestus.tools.registry import PROPERTY_KEYS, TOOLS, CandidateState, execute


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


def isotropic(K: float, G: float) -> np.ndarray:
    """Elastic tensor of an isotropic solid.

    Parameters
    ----------
    K, G : float
        Bulk and shear modulus, GPa.

    Returns
    -------
    numpy.ndarray
        6x6 tensor, GPa.
    """
    C = np.zeros((6, 6))
    C[:3, :3] = K - 2 * G / 3
    C[np.arange(3), np.arange(3)] = K + 4 * G / 3
    C[np.arange(3, 6), np.arange(3, 6)] = G
    return C


def test_pp1_registry_covers_every_tool_property() -> None:
    """The registry, the property keys and the tool's enum agree; every entry has units and an error note."""
    reg = S.property_registry()
    tool = next(t for t in TOOLS if t["name"] == "predict_property")
    assert list(reg) == PROPERTY_KEYS == tool["input_schema"]["properties"]["property"]["enum"]
    assert all("units" in v and v["error_note"] for v in reg.values())


def test_isotropic_tensor_gives_exact_moduli_and_isotropy() -> None:
    """VRH returns K and G exactly; A^U = 0, Zener = 1, Born stable; known ratios."""
    p = S.elastic_properties(isotropic(100.0, 60.0), 5.0, 0.08)
    v = p["values"]
    assert v["bulk_modulus"] == pytest.approx(100) and v["shear_modulus"] == pytest.approx(60)
    assert v["anisotropy_index"] == pytest.approx(0, abs=1e-9) and v["zener_ratio"] == pytest.approx(1)
    assert v["youngs_modulus"] == pytest.approx(9 * 100 * 60 / 360)
    assert v["poisson_ratio"] == pytest.approx((300 - 120) / (2 * 360))
    assert v["pugh_ratio"] == pytest.approx(100 / 60) and p["born_stable"]
    assert p["uncertainties"]["youngs_modulus"] > 0 and p["uncertainties"]["zener_ratio"] is None  # SYS-4


def test_sound_velocities_and_debye_temperature() -> None:
    """v_t = sqrt(G/rho), v_l = sqrt((K + 4G/3)/rho), and the Debye formula."""
    v = S.from_moduli(100.0, 60.0, 5.0, 0.08)
    assert v["sound_velocity_transverse"] == pytest.approx(np.sqrt(60e9 / 5000) / 1000)
    assert v["sound_velocity_longitudinal"] == pytest.approx(np.sqrt(180e9 / 5000) / 1000)
    assert 300 < v["debye_temperature"] < 800


def test_born_instability_is_flagged() -> None:
    """A negative C44 fails the Born criteria and no derived properties are given."""
    C = isotropic(100.0, 60.0)
    C[3, 3] = -10
    p = S.elastic_properties(C, 5.0, 0.08)
    assert not p["born_stable"] and "youngs_modulus" not in p["values"]


def test_pp3_emt_copper_elastic_bulk_matches_equation_of_state() -> None:
    """For EMT Cu the strain-stress bulk modulus agrees with an energy-volume fit within 2 %, with cubic symmetry."""
    atoms, _, _, _ = S.relax_atoms(bulk("Cu", "fcc", a=3.6, cubic=True), EMT(), fmax=1e-4)
    C = S.elastic_tensor(atoms, EMT())
    assert C[1, 1] == pytest.approx(C[0, 0], rel=1e-3) and C[2, 2] == pytest.approx(C[0, 0], rel=1e-3)
    assert C[3, 3] == pytest.approx(C[4, 4], rel=1e-3) and abs(C[0, 3]) < 1
    vols, energies = [], []
    for x in np.linspace(0.98, 1.02, 7):
        a = atoms.copy()
        a.set_cell(atoms.cell * x, scale_atoms=True)
        a.calc = EMT()
        vols.append(a.get_volume())
        energies.append(a.get_potential_energy())
    _, _, B = EquationOfState(vols, energies).fit()
    K = S.elastic_properties(C, 8.9, 4 / atoms.get_volume())["values"]["bulk_modulus"]
    assert pytest.approx(B / GPa, rel=0.02) == K


def test_predict_property_tool_returns_related_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """One mechanical call stores every mechanical prediction and returns them as related."""
    monkeypatch.setattr(S, "FACTORIES", {name: (lambda d, t: EMT()) for name in S.FACTORIES})
    monkeypatch.setattr(S, "_LOADED", {})
    ws = workspace()
    cand = Candidate(
        candidate_id="c1-1", iteration=1, composition="Cu", structure_type="perovskite_cubic", rationale="r"
    )
    rec = StructureRecord(
        candidate_id="c1-1",
        cif="",
        source="mp:mp-30",
        n_atoms=4,
        pmg_dict=S.to_pmg(bulk("Cu", "fcc", a=3.6, cubic=True)).as_dict(),
    )
    ws.candidates["c1-1"] = CandidateState(candidate=cand, round=1, record=rec)
    r, err = execute(ws, "predict_property", {"candidate_id": "c1-1", "property": "bulk_modulus"})
    assert err and "relax first" in r["error"]
    execute(ws, "relax", {"candidate_id": "c1-1", "potential": "mace-mp-0-small", "fmax": 0.005})
    r, err = execute(ws, "predict_property", {"candidate_id": "c1-1", "property": "bulk_modulus"})
    assert not err and r["units"] == "GPa" and r["uncertainty"] == 15.0 and r["mechanically_stable"]
    assert {"shear_modulus", "youngs_modulus", "debye_temperature"} <= set(r["related"])
    t = ws.compute_s
    r2, _ = execute(ws, "predict_property", {"candidate_id": "c1-1", "property": "debye_temperature"})
    assert ws.compute_s == t and r2["value"] > 0  # no second elastic calculation
    assert set(S.MECHANICAL) <= set(ws.candidates["c1-1"].predictions)
    dumped = ws.model_dump_json()  # snapshot stays plain JSON (OR-6)
    assert type(ws).model_validate_json(dumped).model_dump_json() == dumped
    r, err = execute(ws, "predict_property", {"candidate_id": "c1-1", "property": "density"})
    assert not err and r["value"] == pytest.approx(8.9, rel=0.05)
    r, err = execute(ws, "predict_property", {"candidate_id": "c1-1", "property": "formation_energy"})
    assert err and "energy_above_hull" in r["error"]
    r, err = execute(ws, "predict_property", {"candidate_id": "c1-1", "property": "creep_rate"})
    assert err and "not available" in r["error"]  # PP-6


def test_pp5_widen_factor() -> None:
    """Out-of-domain chemistry doubles the error; mixed anions widen it by half."""
    assert S.widen_factor(["Sr", "Ti", "O"], True) == 1.0
    assert S.widen_factor(["Sr", "Ta", "O", "N"], True) == 1.5
    assert S.widen_factor(["Xe", "F"], False) == 2.0


class FakeMEGNet:
    """Band-gap model stand-in: the gap depends on the fidelity index; counts calls."""

    def __init__(self, gaps: dict[int, float]) -> None:
        """Set the gap per fidelity index.

        Parameters
        ----------
        gaps : dict
            Fidelity index to gap, eV.
        """
        self.gaps = gaps
        self.calls = 0

    def predict_structure(self, structure: object, state_attr: object) -> float:
        """Return the gap for the requested fidelity.

        Parameters
        ----------
        structure : Structure
            Ignored.
        state_attr : torch.Tensor
            One-element fidelity index.

        Returns
        -------
        float
            Gap, eV.
        """
        self.calls += 1
        return self.gaps[int(state_attr[0])]  # type: ignore[index]


@pytest.fixture
def fake_megnet(monkeypatch: pytest.MonkeyPatch) -> FakeMEGNet:
    """Replace MEGNet with a fake: PBE 1.8 eV, HSE 3.5 eV.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Patches the factory.

    Returns
    -------
    FakeMEGNet
        The fake.
    """
    pytest.importorskip("torch")
    fake = FakeMEGNet({0: 1.8, 2: 3.5})
    monkeypatch.setattr(S, "BANDGAP_FACTORY", lambda: fake)
    monkeypatch.setattr(S, "_BANDGAP", {})
    return fake


def test_pp1_band_gap_uses_chosen_fidelity_and_caches(fake_megnet: FakeMEGNet) -> None:
    """HSE and PBE map to their state attributes; a repeat call is served from the cache."""
    s = S.to_pmg(bulk("Cu", "fcc", a=3.6))
    assert S.band_gap(s, "hse") == 3.5 and S.band_gap(s, "pbe") == 1.8
    S.band_gap(s, "hse")
    assert fake_megnet.calls == 2
    with pytest.raises(S.ToolError, match="fidelity"):
        S.band_gap(s, "lda")


def test_band_gap_is_floored_at_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A small negative prediction (a metal) is reported as 0 eV."""
    pytest.importorskip("torch")
    monkeypatch.setattr(S, "BANDGAP_FACTORY", lambda: FakeMEGNet({2: -0.2}))
    monkeypatch.setattr(S, "_BANDGAP", {})
    assert S.band_gap(S.to_pmg(bulk("Cu", "fcc", a=3.6)), "hse") == 0.0


def test_pp4_band_gap_tool_reports_error_bar_and_target(
    monkeypatch: pytest.MonkeyPatch, fake_megnet: FakeMEGNet
) -> None:
    """predict_property(band_gap) gives ±0.65 eV at HSE, the target, and stores the prediction."""
    monkeypatch.setattr(S, "FACTORIES", {name: (lambda d, t: EMT()) for name in S.FACTORIES})
    monkeypatch.setattr(S, "_LOADED", {})
    ws = workspace()
    cand = Candidate(
        candidate_id="c1-1", iteration=1, composition="Cu", structure_type="perovskite_cubic", rationale="r"
    )
    rec = StructureRecord(
        candidate_id="c1-1",
        cif="",
        source="mp:mp-30",
        n_atoms=1,
        pmg_dict=S.to_pmg(bulk("Cu", "fcc", a=3.6)).as_dict(),
    )
    ws.candidates["c1-1"] = CandidateState(candidate=cand, round=1, record=rec)
    execute(ws, "relax", {"candidate_id": "c1-1", "potential": "mace-mp-0-small", "fmax": 0.02})
    r, err = execute(ws, "predict_property", {"candidate_id": "c1-1", "property": "band_gap"})
    assert not err and r["value"] == 3.5 and r["uncertainty"] == 0.65 and r["units"] == "eV"
    assert r["model"].startswith("MEGNet-BandGap-mfi") and r["target"]["direction"] == "minimise"
    assert ws.candidates["c1-1"].predictions["band_gap"].value.uncertainty == 0.65
    ws.cfg.band_gap_fidelity = "pbe"
    r, _ = execute(ws, "predict_property", {"candidate_id": "c1-1", "property": "band_gap"})
    assert r["value"] == 1.8 and r["uncertainty"] == 0.57
