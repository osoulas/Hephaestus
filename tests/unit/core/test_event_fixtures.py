"""The recorded event fixtures still match the current contracts (UI and CLI build against them)."""

import json
from pathlib import Path

from hephaestus.core.types import ProgressEvent, RunState


def test_or7_recorded_events_validate(fixtures_dir: Path) -> None:
    """Every recorded event is a valid ProgressEvent and the stream covers each section."""
    lines = (fixtures_dir / "events" / "demo" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    events = [ProgressEvent.model_validate(json.loads(line)) for line in lines]
    assert [e.seq for e in events] == list(range(len(events)))
    sections = {e.section for e in events}
    assert {
        "run",
        "goal_parser",
        "tool",
        "agent",
        "orchestrator",
        "final_calculation",
        "brief_generator",
    } <= sections


def test_recorded_state_validates(fixtures_dir: Path) -> None:
    """The recorded final state loads as RunState."""
    state = RunState.model_validate_json(
        (fixtures_dir / "events" / "demo" / "state.json").read_text(encoding="utf-8")
    )
    assert state.status == "finished" and state.brief is not None
