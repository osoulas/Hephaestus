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
