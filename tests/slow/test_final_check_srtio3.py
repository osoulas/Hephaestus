"""Final re-check on SrTiO3 with real MACE and MEGNet (pytest -m slow; Section 9).

Uses MACE-MP-0 small as the "final" potential so no extra model download is needed; the final
settings (float64, fmax 0.005 eV/Å, full phonons) still differ from screening. Competing phases
are built locally, so no Materials Project key is needed.
"""

from pathlib import Path
from typing import Any

import pytest
from factories import workspace

from hephaestus.core.types import Candidate, StructureRecord, Submission
from hephaestus.harness.final_check import validate_finalists
from hephaestus.tools import simulation as S
from hephaestus.tools.registry import CandidateState, load_default_handlers

pytestmark = pytest.mark.slow


def _phases(elements: list[str]) -> list[dict[str, Any]]:
    """Stable phases of Sr-Ti-O built from textbook structures.

    Parameters
    ----------
    elements : list of str
        Chemical system (ignored; always Sr-Ti-O).

    Returns
    -------
    list of dict
        ``mp_id``, ``composition`` and ``pmg_dict`` per phase.
    """
    from pymatgen.core import Lattice, Molecule, Structure

    o2 = Molecule(["O", "O"], [[0, 0, 0], [0, 0, 1.21]]).get_boxed_structure(10, 10, 10)
    phases = {
        "Sr": Structure.from_spacegroup("Fm-3m", Lattice.cubic(6.08), ["Sr"], [[0, 0, 0]]),
        "Ti": Structure.from_spacegroup(
            "P6_3/mmc", Lattice.hexagonal(2.95, 4.69), ["Ti"], [[1 / 3, 2 / 3, 0.25]]
        ),
        "O2": o2,
        "SrO": Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(5.16), ["Sr", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        ),
        "TiO2": Structure.from_spacegroup(
            "P4_2/mnm", Lattice.tetragonal(4.59, 2.96), ["Ti", "O"], [[0, 0, 0], [0.305, 0.305, 0]]
        ),
    }
    return [
        {"mp_id": k, "composition": s.composition.formula, "pmg_dict": s.as_dict()} for k, s in phases.items()
    ]


def test_fc1_fc2_fc7_srtio3_final_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SrTiO3 is re-checked at final settings; it stays on the hull and dynamically stable."""
    pytest.importorskip("mace")
    pytest.importorskip("matgl")
    if not any(Path.home().joinpath(".cache", "mace").glob("*")):
        pytest.skip("MACE model not downloaded")
    from pymatgen.core import Lattice, Structure

    monkeypatch.setenv("HEPHAESTUS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(S, "PHASE_SOURCE", _phases)
    ws = workspace(final_potential="mace-mp-0-small", device="cpu")
    ws.spec = ws.spec.model_copy(update={"reference_material": None})
    sto = Structure.from_spacegroup(
        "Pm-3m", Lattice.cubic(3.905), ["Sr", "Ti", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
    )
    cand = Candidate(
        candidate_id="c1-1",
        iteration=1,
        composition="SrTiO3",
        structure_type="perovskite_cubic",
        rationale="t",
    )
    rec = StructureRecord(candidate_id="c1-1", cif="", pmg_dict=sto.as_dict(), source="mp:mp-5229", n_atoms=5)
    ws.candidates["c1-1"] = CandidateState(candidate=cand, round=1, record=rec)
    handlers = load_default_handlers()
    for name, args in [
        ("relax", {"potential": "mace-mp-0-small", "fmax": 0.02}),
        ("energy_above_hull", {}),
        ("phonons", {"mode": "fast"}),
        ("predict_property", {"property": "band_gap"}),
    ]:
        handlers[name](ws, candidate_id="c1-1", **args)

    report = validate_finalists(ws, Submission(candidate_id="c1-1", gate_passed=True), handlers)[0]

    checks = {c.name: c for c in report.checks}
    assert report.notes[0].startswith(
        "Final settings: mace-mp-0-small (float64), fmax 0.005 eV/Å, full phonons"
    )
    assert checks["thermo_stable"].final_value is True
    assert checks["dyn_stable"].final_value is True
    assert abs(checks["band_gap"].delta) < 0.2
    assert report.verdict == "confirmed", report.notes
    assert any("Ordered compound" in n for n in report.notes)
