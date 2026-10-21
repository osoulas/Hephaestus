"""Real Materials Project calls; run with ``pytest -m live`` and an MP API key."""

from pathlib import Path

import pytest
from factories import workspace

from hephaestus.tools import materials_project as mp

pytestmark = pytest.mark.live


def test_live_sg1_lafeo3_default_is_pnma(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The live API gives Pnma as LaFeO3's default polymorph, and a repeat is served from cache."""
    monkeypatch.setenv("HEPHAESTUS_CACHE_DIR", str(tmp_path))
    out = mp.lookup_materials_project(workspace(), "LaFeO3")
    default = next(e for e in out["entries"] if e["mp_id"] == out["default_polymorph"])
    assert default["space_group"] == "Pnma"
    assert mp.lookup_materials_project(workspace(offline=True), "LaFeO3") == out


def test_live_srtio3_cubic_phonons_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cubic SrTiO3 carries MP's own phonons, with the known soft mode at R."""
    monkeypatch.setenv("HEPHAESTUS_CACHE_DIR", str(tmp_path))
    out = mp.lookup_materials_project(workspace(), "SrTiO3")
    cubic = next(e for e in out["entries"] if e["mp_id"] == out["default_polymorph"])
    assert cubic["space_group"] == "Pm-3m"
    assert cubic["phonon"]["available"] and cubic["phonon"]["min_frequency_THz"] < 0
