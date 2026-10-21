"""Event bus: fan-out, history on disk, and replay (OR-7, DL-10, UI-9)."""

from pathlib import Path

from hephaestus.core.events import EventBus


def test_or7_subscriber_receives_events_in_order(tmp_path: Path) -> None:
    """Events reach a subscriber in publication order with increasing seq."""
    bus = EventBus(tmp_path)
    sub = bus.subscribe("r1")
    bus.emit("r1", "tool", "started", {"tool": "relax"}, iteration=1)
    bus.emit("r1", "tool", "finished", {"tool": "relax"}, iteration=1)
    bus.close_run("r1")
    got = list(sub)
    assert [e.status for e in got] == ["started", "finished"]
    assert [e.seq for e in got] == [0, 1]


def test_dl10_history_loads_from_disk_in_new_process(tmp_path: Path) -> None:
    """A fresh bus (a restarted server) sees earlier events and continues the sequence."""
    EventBus(tmp_path).emit("r1", "run", "started")
    bus = EventBus(tmp_path)
    assert len(bus.history("r1")) == 1
    event = bus.emit("r1", "run", "finished")
    assert event.seq == 1
    sub = bus.subscribe("r1")
    bus.close_run("r1")
    assert [e.status for e in sub] == ["started", "finished"]


def test_ui9_replay_reemits_without_writing(tmp_path: Path) -> None:
    """Replay publishes a stored run under a new ID and leaves no files for it."""
    bus = EventBus(tmp_path)
    for status in ("started", "finished"):
        bus.emit("r1", "tool", status)  # type: ignore[arg-type]
    seen = []
    remove = bus.add_listener(seen.append)
    out = bus.replay("r1", "replay-1", speed=0)
    remove()
    assert [e.run_id for e in out] == ["replay-1", "replay-1"]
    assert len(seen) == 2
    assert not (tmp_path / "replay-1").exists()
