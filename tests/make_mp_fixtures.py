"""Record Materials Project fixtures for the offline tests: ``python tests/make_mp_fixtures.py``.

Needs an MP API key (``MP_API_KEY`` or pymatgen's ``.pmgrc.yaml``). Writes raw summary documents
for a few compositions, one lookup by legacy ID, and the lowest phonon band of cubic SrTiO3 to
``tests/fixtures/mp/``. Rerun when ``SUMMARY_FIELDS`` changes.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from hephaestus.tools import materials_project as mp

OUT = Path(__file__).parent / "fixtures" / "mp"
FORMULAS = ["SrTiO3", "LaFeO3", "BaSnO3"]


def jsonable(x: Any) -> Any:
    """Convert MP values (enums, numpy, pymatgen objects) to plain JSON.

    Parameters
    ----------
    x : Any
        Value from an MP document.

    Returns
    -------
    Any
        JSON-serialisable value.
    """
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [jsonable(v) for v in x]
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, np.generic):
        return x.item()
    if hasattr(x, "as_dict"):
        return jsonable(x.as_dict())
    return x


def write(name: str, value: Any) -> None:
    """Write one fixture as UTF-8 JSON.

    Parameters
    ----------
    name : str
        File stem.
    value : Any
        Content.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(jsonable(value), indent=1) + "\n", encoding="utf-8")


def main() -> None:
    """Fetch and write every fixture."""
    for formula in FORMULAS:
        write(f"summary_{formula}", mp._fetch_summary(formula=formula))
    write("by_id_mp-5229", mp._fetch_summary(material_ids=["mp-5229"]))
    cubic = next(
        mp._entry(d) for d in mp._fetch_summary(formula="SrTiO3") if d["symmetry"]["symbol"] == "Pm-3m"
    )
    method = next(m for m in mp.PHONON_METHODS if cubic["phonon_ids"].get(m))
    pid = cubic["phonon_ids"][method][0]
    bs = mp._fetch_phonon(pid, method)
    freqs = np.asarray(bs["frequencies"])
    write(
        "phonon_SrTiO3_Pm-3m",
        {
            "phonon_id": pid,
            "method": method,
            "frequencies": np.round(freqs[:1], 4).tolist(),  # lowest band only
            "qpoints": np.round(np.asarray(bs["qpoints"]), 5).tolist(),
            "labels_dict": bs["labels_dict"],
        },
    )
    print(f"Wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
