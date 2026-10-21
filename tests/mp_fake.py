"""Recorded Materials Project responses in place of the network, for tool tests."""

import json
from pathlib import Path
from typing import Any

import pytest

from hephaestus.tools import materials_project as mp

MP_FIX = Path(__file__).parent / "fixtures" / "mp"


class FakeMP:
    """Serves recorded MP documents and counts calls; stands in for the network boundary."""

    def __init__(self) -> None:
        """Start with no calls."""
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def summary(self, **query: Any) -> list[dict[str, Any]]:
        """Return recorded summary documents for a query.

        Parameters
        ----------
        **query
            ``formula=``, ``material_ids=`` or ``chemsys=``.

        Returns
        -------
        list of dict
            Raw documents, or empty when nothing was recorded.
        """
        self.calls.append(("summary", query))
        if "formula" in query:
            path = MP_FIX / f"summary_{query['formula']}.json"
        elif "material_ids" in query:
            path = MP_FIX / f"by_id_{query['material_ids'][0]}.json"
        else:
            return []
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def phonon(self, phonon_id: str, method: str) -> dict[str, Any]:
        """Return the recorded cubic SrTiO3 phonon band.

        Parameters
        ----------
        phonon_id : str
            Phonon document ID.
        method : str
            Phonon method.

        Returns
        -------
        dict
            Frequencies, q-points and labels.
        """
        self.calls.append(("phonon", {"id": phonon_id, "method": method}))
        data = json.loads((MP_FIX / "phonon_SrTiO3_Pm-3m.json").read_text(encoding="utf-8"))
        if phonon_id != data["phonon_id"]:
            raise RuntimeError("no such phonon document")
        return data


def install_fake_mp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeMP:
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
    fake = FakeMP()
    monkeypatch.setattr(mp, "_fetch_summary", fake.summary)
    monkeypatch.setattr(mp, "_fetch_phonon", fake.phonon)
    monkeypatch.setenv("HEPHAESTUS_CACHE_DIR", str(tmp_path / "cache"))
    return fake
