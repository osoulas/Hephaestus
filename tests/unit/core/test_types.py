"""Shared types: every fixture loads and round-trips through JSON with no loss (SYS-5)."""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from hephaestus.core import types as T
from hephaestus.core.config import RunConfig

FIXTURE_TYPES: dict[str, type[BaseModel]] = {
    "quantity": T.Quantity,
    "search_spec": T.SearchSpec,
    "paper": T.Paper,
    "insight": T.Insight,
    "literature_context": T.LiteratureContext,
    "candidate": T.Candidate,
    "structure_record": T.StructureRecord,
    "phonon_result": T.PhononResult,
    "simulation_result": T.SimulationResult,
    "property_prediction": T.PropertyPrediction,
    "critique": T.Critique,
    "validation_report": T.ValidationReport,
    "brief": T.Brief,
    "progress_event": T.ProgressEvent,
    "submission": T.Submission,
    "run_state": T.RunState,
    "run_config": RunConfig,
}


@pytest.mark.parametrize("name", sorted(FIXTURE_TYPES))
def test_sys5_fixture_round_trips(fixtures_dir: Path, name: str) -> None:
    """Each fixture validates and serialises back to identical JSON.

    Parameters
    ----------
    fixtures_dir : pathlib.Path
        ``tests/fixtures``.
    name : str
        Fixture stem.
    """
    model = FIXTURE_TYPES[name]
    raw = json.loads((fixtures_dir / f"{name}.json").read_text(encoding="utf-8"))
    obj = model.model_validate(raw)
    again = model.model_validate_json(obj.model_dump_json())
    assert again == obj
    assert json.loads(again.model_dump_json()) == json.loads(obj.model_dump_json())


def test_records_reject_unknown_fields() -> None:
    """A misspelt field is an error, not silently dropped."""
    with pytest.raises(ValidationError):
        T.Quantity(value=1.0, uncertainty=None, units="eV", source="x", uncertainity=0.1)  # type: ignore[call-arg]


def test_dl12_unicode_round_trips() -> None:
    """Units such as "Å" and "≤" survive JSON."""
    q = T.Quantity(value=3.9, uncertainty=0.01, units="Å", source="≤ test")
    assert T.Quantity.model_validate_json(q.model_dump_json()) == q


def test_pp1_band_gap_model_replaces_fidelity() -> None:
    """Old configs with band_gap_fidelity load as the matching band gap model; the fidelity is derived."""
    from hephaestus.core.config import BAND_GAP_MODELS, RunConfig

    old = RunConfig.model_validate({"band_gap_fidelity": "pbe"})
    assert old.band_gap_model == "megnet-pbe" and old.band_gap_fidelity == "pbe"
    assert "band_gap_fidelity" not in old.model_dump()
    assert RunConfig().band_gap_model == "megnet-hse" and set(BAND_GAP_MODELS) == {
        "megnet-hse",
        "megnet-pbe",
        "megnet-gllb-sc",
        "megnet-scan",
    }


def test_or9_or10_budget_of_zero_means_no_limit() -> None:
    """A budget of 0 is no limit; negative budgets are rejected."""
    import math

    import pytest
    from pydantic import ValidationError

    from hephaestus.core.config import RunConfig, budget_limit

    assert budget_limit(0) == math.inf and budget_limit(90) == 90
    assert RunConfig(max_tool_calls=0, time_budget_s=0).max_tool_calls == 0
    with pytest.raises(ValidationError):
        RunConfig(max_iterations=-1)
