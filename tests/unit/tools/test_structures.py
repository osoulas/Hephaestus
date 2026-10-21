"""Structure generator and candidate checks (Sections 3 and 4)."""

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from hephaestus.tools import structures as S
from hephaestus.tools.structures import StructureBuildError


def test_sg5_sites_by_radius_not_order() -> None:
    """Sr goes on A however the formula is written; Gd on the pyrochlore A site."""
    assert S.assign_sites("TiSrO3", "perovskite_cubic").sites == {
        "A": {"Sr": 1.0},
        "B": {"Ti": 1.0},
        "X": {"O": 1.0},
    }
    pyro = S.assign_sites("Ti2Gd2O7", "pyrochlore").sites
    assert pyro["A"] == {"Gd": 1.0} and pyro["B"] == {"Ti": 1.0}


def test_sg5_dopants_follow_size() -> None:
    """La substitutes for Sr on A and Al for Ti on B."""
    sites = S.assign_sites("Sr0.75La0.25Ti0.75Al0.25O3", "perovskite_cubic").sites
    assert sites["A"] == {"La": 0.25, "Sr": 0.75}
    assert sites["B"] == {"Al": 0.25, "Ti": 0.75}


def test_sg5_olivine_orders_cations() -> None:
    """Mg fills both olivine M sites; in LiFePO4, Li takes M1 and Fe takes M2 (triphylite)."""
    assert S.assign_sites("Mg2SiO4", "olivine").sites["T"] == {"Si": 1.0}
    lfp = S.assign_sites("LiFePO4", "olivine").sites
    assert (lfp["M1"], lfp["M2"], lfp["T"]) == ({"Li": 1.0}, {"Fe": 1.0}, {"P": 1.0})


@pytest.mark.parametrize(
    ("formula", "key", "expected"),
    [
        ("Fe2VAl", "full_heusler", {"X": {"Fe": 1.0}, "Y": {"V": 1.0}, "Z": {"Al": 1.0}}),
        ("Mg2SiO4", "spinel", {"A": {"Si": 1.0}, "B": {"Mg": 1.0}, "X": {"O": 1.0}}),
        ("Fe3O4", "spinel", {"A": {"Fe": 1.0}, "B": {"Fe": 1.0}, "X": {"O": 1.0}}),
    ],
)
def test_sg5_ordered_layout_preferred(formula: str, key: str, expected: dict) -> None:
    """When whole elements can fill every site, no site is shared (so the compound stays ordered)."""
    sites = S.assign_sites(formula, key).sites
    assert sites == expected
    assert S.disorder_reason(sites) is None


def test_inverse_spinel_is_flagged() -> None:
    """Known inverse spinels carry a warning; normal ones and other templates do not."""
    note = S.inverse_spinel_note("NiFe2O4", "spinel")
    assert note is not None and note.startswith("NiFe2O4 is a known")
    assert S.inverse_spinel_note("MgAl2O4", "spinel") is None
    assert S.inverse_spinel_note("NiFe2O4", "rocksalt") is None


def test_sg5_antiperovskite_orders_anions() -> None:
    """O takes the octahedral anion site and Br the large cuboctahedral one."""
    sites = S.assign_sites("Li3OBr", "antiperovskite").sites
    assert sites["X"] == {"O": 1.0} and sites["Y"] == {"Br": 1.0}


def test_cg3_charge_balance() -> None:
    """Common oxidation states balance Sr0.75La0.25Ti0.75Al0.25O3 but not SrTi0.5Nb0.5O2."""
    assert S.oxidation_states("Sr0.75La0.25Ti0.75Al0.25O3") is not None
    assert S.oxidation_states("SrTi0.5Nb0.5O2") is None


@pytest.mark.parametrize("formula", ["LaMnO3", "LaNiO3", "SrFeO3", "LiNiO2", "Fe3O4", "SrTaO2N"])
def test_cg3_well_established_states_balance(formula: str) -> None:
    """Mn3+, Ni3+, Fe4+, mixed-valence Fe3O4 and oxynitrides are not falsely rejected."""
    assert S.oxidation_states(formula) is not None


def test_sg10_wrong_template_is_named_failure() -> None:
    """SrTiO3 cannot fill a pyrochlore; the failure names the reason."""
    with pytest.raises(StructureBuildError) as exc:
        S.assign_sites("SrTiO3", "pyrochlore")
    assert exc.value.code == "no_template"
    assert "anion/cation ratio" in exc.value.message


def test_sg10_unknown_structure_type() -> None:
    """An unknown prototype key is a named failure, not a guess."""
    with pytest.raises(StructureBuildError) as exc:
        S.assign_sites("SrTiO3", "garnet")
    assert exc.value.code == "unknown_structure_type"


def test_sg4_supercell_srti0875fe0125o3_is_40_atoms() -> None:
    """One Fe in eight B sites needs the 2x2x2 cell, matched exactly."""
    build = S.build_from_template("SrTi0.875Fe0.125O3", "perovskite_cubic", seed=0)
    assert build.supercell == [2, 2, 2]
    assert len(build.structure) == 40
    assert build.composition_error == 0.0
    assert build.structure.composition.reduced_formula == "Sr8Ti7FeO24"


def test_sg4_relative_tolerance_rejects_coarse_cells() -> None:
    """0.1 Fe is met by 1/10 (50 atoms), never by 1/8 or 1/6."""
    build = S.build_from_template("SrTi0.9Fe0.1O3", "perovskite_cubic")
    assert sorted(build.supercell) == [1, 2, 5]


def test_sg8_never_exceeds_max_atoms() -> None:
    """A dopant level needing more than the cap is refused, with advice."""
    with pytest.raises(StructureBuildError) as exc:
        S.build_from_template("SrTi0.97Fe0.03O3", "perovskite_cubic", max_atoms=80)
    assert exc.value.code == "supercell"
    build = S.build_from_template("Gd2Ti1.75Zr0.25O7", "pyrochlore", max_atoms=80)
    assert len(build.structure) <= 80


def test_sg6_same_seed_identical_and_different_seed_differs() -> None:
    """Disorder is reproducible from the seed, and the seed changes the arrangement."""

    def dopant_sites(seed: int) -> frozenset[tuple[str, tuple[float, ...]]]:
        """Return where La and Al sit for one seed.

        Parameters
        ----------
        seed : int
            Disorder seed.

        Returns
        -------
        frozenset
            ``(element, rounded fractional coordinates)`` of the dopants.
        """
        s = S.build_from_template("Sr0.75La0.25Ti0.75Al0.25O3", "perovskite_cubic", seed=seed).structure
        return frozenset(
            (x.specie.symbol, tuple(round(c, 3) for c in x.frac_coords))
            for x in s
            if x.specie.symbol in "LaAl"
        )

    assert dopant_sites(3) == dopant_sites(3)
    assert len({dopant_sites(seed) for seed in range(6)}) > 1


def test_sg9_overlap_rejected() -> None:
    """Two atoms 0.5 Å apart are reported."""
    s = Structure(Lattice.cubic(5.0), ["Ti", "O"], [[0, 0, 0], [0.1, 0, 0]])
    problems = S.check_geometry(s)
    assert problems and "Ti0-O1" in problems[0]


@pytest.mark.parametrize(
    ("formula", "a_measured"),
    [("SrTiO3", 3.905), ("BaTiO3", 4.00), ("SrZrO3", 4.10), ("BaZrO3", 4.193), ("BaSnO3", 4.116)],
)
def test_sg13_cubic_lattice_within_2pct(formula: str, a_measured: float) -> None:
    """Template lattice parameters land within 2 % of measured (pseudo)cubic cells."""
    build = S.build_from_template(formula, "perovskite_cubic")
    assert build.structure.lattice.a == pytest.approx(a_measured, rel=0.02)


def test_off_stoichiometry_becomes_anion_vacancies() -> None:
    """Up to 5 % missing anions are placed as vacancies; more is refused."""
    build = S.build_from_template("SrTi0.75Fe0.25O2.875", "perovskite_cubic")
    assert build.structure.composition.reduced_formula == "Sr8Ti6Fe2O23"
    with pytest.raises(StructureBuildError):
        S.assign_sites("SrTiO2.5", "perovskite_cubic")


def test_descriptors() -> None:
    """Tolerance factor near 1 for SrTiO3; pyrochlore radius ratio in the stable window."""
    t = S.descriptors(S.assign_sites("SrTiO3", "perovskite_cubic"))["tolerance_factor"]
    assert t == pytest.approx(1.0, abs=0.02)
    ratio = S.descriptors(S.assign_sites("Gd2Ti2O7", "pyrochlore"))["radius_ratio_A_B"]
    assert 1.46 <= ratio <= 1.78


@pytest.mark.parametrize(
    ("formula", "key", "sg"),
    [
        ("Gd2Ti1.75Zr0.25O7", "pyrochlore", None),
        ("Y2Sn2O7", "pyrochlore", 227),
        ("Na2S", "antifluorite", 225),
        ("CeO2", "fluorite", 225),
        ("MgAl1.75Ga0.25O4", "spinel", None),
        ("ZnFe2O4", "spinel", 227),
        ("LaFeO3", "perovskite_orthorhombic", 62),
        ("La2NiO4", "ruddlesden_popper", 139),
        ("Sr2FeMoO6", "double_perovskite", 225),
        ("Fe2O3", "corundum", 167),
        ("MnTiO3", "ilmenite", 148),
        ("LiNiO2", "layered_rocksalt", 166),
        ("SnO2", "rutile", 136),
        ("GaN", "wurtzite", 186),
        ("GaAs", "zincblende", 216),
        ("CsBr", "cesium_chloride", 221),
        ("Ni3Al", "l12", 221),
        ("ZrNiSn", "half_heusler", 216),
        ("Co2MnSi", "full_heusler", 225),
    ],
)
def test_sg3_every_family_builds(formula: str, key: str, sg: int | None) -> None:
    """A composition per family builds a valid structure of the right formula and symmetry."""
    build = S.build_from_template(formula, key)
    s = build.structure
    assert s.composition.reduced_formula == S.integer_composition(formula).reduced_formula
    assert S.check_geometry(s) == []
    if sg is not None:
        assert SpacegroupAnalyzer(s, symprec=0.01).get_space_group_number() == sg
