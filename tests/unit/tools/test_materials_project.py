"""Materials Project lookup: polymorph choice, caching, offline replay (SG-1, SG-12, CG-12, DL-3)."""

from pathlib import Path
from typing import Any

import pytest
from factories import workspace
from mp_fake import FakeMP, install_fake_mp

from hephaestus.tools import materials_project as mp
from hephaestus.tools.registry import ToolError


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


def test_sg1_lafeo3_default_is_pnma(fake_mp: FakeMP) -> None:
    """LaFeO3 defaults to the observed Pnma phase, not the 0 K R-3c ground state."""
    entries = mp.mp_polymorphs("LaFeO3")
    assert entries[0]["space_group"] == "R-3c"
    default = mp.choose_polymorph(entries)
    assert default is not None and default["space_group"] == "Pnma"


def test_sg1_srtio3_default_is_cubic(fake_mp: FakeMP) -> None:
    """SrTiO3 defaults to cubic Pm-3m, the most-reported observed phase near the hull."""
    default = mp.choose_polymorph(mp.mp_polymorphs("Sr2Ti2O6"))
    assert default is not None and default["space_group"] == "Pm-3m"


def test_sg1_without_observed_entries_takes_lowest() -> None:
    """With no ICSD-backed entry in the window, the lowest-energy entry wins."""
    entries = [
        {"mp_id": "a", "e_hull_eV_per_atom": 0.02, "experimental": False, "n_icsd": 0},
        {"mp_id": "b", "e_hull_eV_per_atom": 0.0, "experimental": False, "n_icsd": 0},
        {"mp_id": "c", "e_hull_eV_per_atom": 0.2, "experimental": True, "n_icsd": 9},
    ]
    default = mp.choose_polymorph(entries)
    assert default is not None and default["mp_id"] == "b"
    assert mp.choose_polymorph([]) is None


def test_sg12_legacy_mp_id_builds_that_entry(fake_mp: FakeMP) -> None:
    """A legacy ID resolves to the current entry, and other polymorphs are listed."""
    found = mp.lookup_mp_structure("SrTiO3", "mp-5229")
    assert found is not None
    assert found["entry"]["space_group"] == "Pm-3m"
    assert found["structure"].composition.reduced_formula == "SrTiO3"
    assert len(found["alternatives"]) == len(mp.mp_polymorphs("SrTiO3")) - 1


def test_sg12_mp_id_of_other_composition_is_refused(fake_mp: FakeMP) -> None:
    """An mp_id for a different formula is an error, not a silent swap."""
    with pytest.raises(ToolError, match="not LaFeO3"):
        mp.lookup_mp_structure("LaFeO3", "mp-5229")


def test_cg12_known_compound_reports_mp_ids(fake_mp: FakeMP) -> None:
    """A compound with ICSD-backed entries is known, with their IDs."""
    known = mp.is_known_compound("LaFeO3")
    assert known["known"] and known["mp_ids"]
    assert mp.is_known_compound("SrTi0.875Fe0.125O3") == {"known": False, "mp_ids": []}


def test_dl3_second_lookup_makes_no_client_call(fake_mp: FakeMP) -> None:
    """Repeating a lookup is served from the cache."""
    ws = workspace()
    first = mp.lookup_materials_project(ws, "SrTiO3")
    n = len(fake_mp.calls)
    second = mp.lookup_materials_project(ws, "SrTiO3")
    assert first == second
    assert len(fake_mp.calls) == n


def test_sys8_offline_miss_is_tool_error(fake_mp: FakeMP) -> None:
    """Offline, an uncached composition is a tool error; a cached one still works."""
    with pytest.raises(ToolError, match="offline"):
        mp.lookup_materials_project(workspace(offline=True), "BaSnO3")
    mp.lookup_materials_project(workspace(), "BaSnO3")
    assert mp.lookup_materials_project(workspace(offline=True), "BaSnO3")["n_entries"] > 0


def test_lookup_result_shape(fake_mp: FakeMP) -> None:
    """The tool result has the TOOLS.md keys; MP phonons are attached as information."""
    out = mp.lookup_materials_project(workspace(), "SrTiO3")
    assert set(out) == {"formula", "n_entries", "default_polymorph", "note", "entries"}
    row = next(e for e in out["entries"] if e["mp_id"] == out["default_polymorph"])
    for key in (
        "mp_id",
        "space_group",
        "e_hull_eV_per_atom",
        "experimental",
        "n_icsd",
        "band_gap_pbe_eV",
        "nsites",
    ):
        assert key in row
    assert row["phonon"]["available"]
    assert row["phonon"]["min_frequency_THz"] < 0 and row["phonon"]["where"] == "R"
    assert "information only" in out["note"]
    assert "structure" not in row


def test_solid_solution_is_not_in_mp(fake_mp: FakeMP) -> None:
    """Fractional formulas return no entries and a pointer to the end-members, without a request."""
    out = mp.lookup_materials_project(workspace(), "SrTi0.875Fe0.125O3")
    assert out["n_entries"] == 0 and "end-members" in out["note"]
    assert fake_mp.calls == []


def test_phonon_extremes_finds_min_and_label() -> None:
    """The lowest frequency and its labelled q-point are found."""
    freqs = [[0.0, 1.0, -2.0], [3.0, 4.0, 5.0]]
    qpts = [[0, 0, 0], [0.25, 0, 0], [0.5, 0.5, 0.5]]
    labels = {"\\Gamma": [0, 0, 0], "R": [0.5, 0.5, 0.5]}
    assert mp.phonon_extremes(freqs, qpts, labels) == {"min_frequency_THz": -2.0, "where": "R"}
    assert mp.phonon_extremes([[1.0, 0.5]], [[0, 0, 0], [0.1, 0, 0]])["where"] == "(0.100, 0.000, 0.000)"


def test_failed_phonon_fetch_degrades(fake_mp: FakeMP, monkeypatch: pytest.MonkeyPatch) -> None:
    """A phonon download failure leaves the lookup intact with available: false (SYS-6)."""

    def boom(phonon_id: str, method: str) -> dict[str, Any]:
        """Fail like a network error.

        Parameters
        ----------
        phonon_id, method : str
            Ignored.

        Returns
        -------
        dict
            Never returns.

        Raises
        ------
        ConnectionError
            Always.
        """
        raise ConnectionError("down")

    monkeypatch.setattr(mp, "_fetch_phonon", boom)
    out = mp.lookup_materials_project(workspace(), "SrTiO3")
    row = next(e for e in out["entries"] if e["mp_id"] == out["default_polymorph"])
    assert row["phonon"]["available"] is False and "down" in row["phonon"]["error"]


def test_competing_phases_queries_all_subsystems(fake_mp: FakeMP) -> None:
    """Every subsystem of the chemical system is asked for, stable phases only."""
    mp.competing_phases(["O", "Ti", "Sr"])
    ((_, query),) = fake_mp.calls
    assert set(query["chemsys"]) == {"O", "Sr", "Ti", "O-Sr", "O-Ti", "Sr-Ti", "O-Sr-Ti"}
    assert query["is_stable"] is True
