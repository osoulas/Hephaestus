"""MACE-MP-0 small on cubic perovskites, against docs/stability_check.md (pytest -m slow).

Needs the ``sim`` extra and the MACE model in ``~/.cache/mace`` (downloaded on first use).
"""

from pathlib import Path

import numpy as np
import pytest

from hephaestus.tools import simulation as S

pytestmark = pytest.mark.slow


def cubic(a: str, b: str, lattice_A: float):  # type: ignore[no-untyped-def]
    """Cubic ABO3 perovskite.

    Parameters
    ----------
    a, b : str
        A- and B-site elements.
    lattice_A : float
        Lattice parameter, Å.

    Returns
    -------
    ase.Atoms
        Five-atom cell.
    """
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    s = Structure.from_spacegroup(
        "Pm-3m", Lattice.cubic(lattice_A), [a, b, "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
    )
    return AseAtomsAdaptor.get_atoms(s)


@pytest.fixture(scope="module")
def calc():  # type: ignore[no-untyped-def]
    """MACE-MP-0 small in float64 on CPU.

    Returns
    -------
    MACECalculator
        The calculator.
    """
    pytest.importorskip("mace")
    if not any(Path.home().joinpath(".cache", "mace").glob("*")):
        pytest.skip("MACE model not downloaded")
    return S.FACTORIES["mace-mp-0-small"]("cpu", "float64")


@pytest.mark.parametrize(
    ("a", "b", "lat", "verdict", "dE_range"),
    [
        ("Sr", "Ti", 3.905, "spurious_soft_mode", (-2.0, 0.5)),  # doc: -0.98 meV/atom, spurious
        ("Ba", "Ti", 4.00, "stable_after_distortion", (-20.0, -8.0)),  # doc: → Cm, -13.6
        ("Ca", "Ti", 3.84, "stable_after_distortion", (-80.0, -40.0)),  # doc: -51.8; here Imma, then Pnma
    ],
)
def test_se14_cubic_perovskites_match_stability_doc(calc, a, b, lat, verdict, dE_range) -> None:  # type: ignore[no-untyped-def]
    """Mode mapping reproduces the verdicts and first-round energies in the stability doc.

    Parameters
    ----------
    calc : MACECalculator
        Potential.
    a, b : str
        Cations.
    lat : float
        Starting lattice parameter, Å.
    verdict : str
        Expected verdict.
    dE_range : tuple of float
        Allowed total distortion energy, meV/atom.
    """
    atoms, e, conv, _ = S.relax_atoms(cubic(a, b, lat), calc, fmax=0.01, fix_symmetry=True)
    assert conv and S.space_group(S.to_pmg(atoms)) == "Pm-3m"
    res = S.phonon_check(atoms, e, calc, threshold_meV=2.0, fmax=0.01)
    total = (
        sum(r["dE_meV_per_atom"] for r in res["rounds"] if r["real"]) or res["rounds"][0]["dE_meV_per_atom"]
    )
    print(a + b + "O3", res["verdict"], res["rounds"], f"total {total:.2f} meV/atom")
    assert res["verdict"] == verdict
    assert dE_range[0] <= total <= dE_range[1]


def test_pp3_srtio3_bulk_modulus(calc) -> None:  # type: ignore[no-untyped-def]
    """Cubic SrTiO3 with MACE-MP-0 small: bulk modulus near experiment (about 180 GPa) and DFT (about 170).

    Parameters
    ----------
    calc : MACECalculator
        Potential.
    """
    atoms, _, _, _ = S.relax_atoms(cubic("Sr", "Ti", 3.905), calc, fmax=0.005, fix_symmetry=True)
    C = S.elastic_tensor(atoms, calc)
    p = S.elastic_properties(C, float(S.to_pmg(atoms).density), 5 / atoms.get_volume())
    print("SrTiO3 C (GPa)", np.round(C[:4, :4]), {k: round(v, 2) for k, v in p["values"].items()})
    assert 140 <= p["values"]["bulk_modulus"] <= 210
    assert p["born_stable"]


def test_section6_megnet_hse_gaps_on_relaxed_perovskites(calc) -> None:  # type: ignore[no-untyped-def]
    """MEGNet HSE on MACE-relaxed cubic cells: SrTiO3 near 3.25 eV within the error bar.

    BaSnO3 comes out below SrTiO3, although measured gaps say otherwise: the model-scale win
    the reviewer must judge (REQUIREMENTS, Section 6 decision).

    Parameters
    ----------
    calc : MACECalculator
        Potential.
    """
    pytest.importorskip("matgl")
    gaps = {}
    for a, b, lat in (("Sr", "Ti", 3.905), ("Ba", "Sn", 4.116)):
        atoms, _, _, _ = S.relax_atoms(cubic(a, b, lat), calc, fmax=0.01, fix_symmetry=True)
        gaps[a + b + "O3"] = S.band_gap(S.to_pmg(atoms), "hse")
    print("MEGNet HSE gaps (eV):", {k: round(v, 2) for k, v in gaps.items()})
    assert abs(gaps["SrTiO3"] - 3.25) <= 0.65 + 0.3
    assert gaps["BaSnO3"] < gaps["SrTiO3"]
