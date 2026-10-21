"""Run-folder helpers: atomic UTF-8 writes and JSON lines (DL-1, DL-12)."""

from pathlib import Path

import pytest

from hephaestus.core import store
from hephaestus.core.types import Quantity


def test_dl12_atomic_write_round_trips_unicode(tmp_path: Path) -> None:
    """Text with "≤" and "Å" survives a write and read."""
    path = tmp_path / "a" / "b.txt"
    store.atomic_write_text(path, "≤ 0.05 eV/atom, a = 3.9 Å")
    assert path.read_text(encoding="utf-8") == "≤ 0.05 eV/atom, a = 3.9 Å"
    assert not list(path.parent.glob(".*.tmp"))


def test_dl12_atomic_write_retries_refused_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A replace refused twice (Windows: file open elsewhere) succeeds on retry."""
    real = store.os.replace
    calls = {"n": 0}

    def flaky(src: str, dst: str) -> None:
        """Refuse the first two replaces.

        Parameters
        ----------
        src, dst : str
            Paths passed to ``os.replace``.
        """
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("in use")
        real(src, dst)

    monkeypatch.setattr(store.os, "replace", flaky)
    store.atomic_write_text(tmp_path / "x.json", "{}", delay_s=0)
    assert (tmp_path / "x.json").read_text(encoding="utf-8") == "{}"


def test_save_and_load_model(tmp_path: Path) -> None:
    """A shared type saved as JSON loads back equal."""
    q = Quantity(value=1.0, uncertainty=0.1, units="eV", source="t")
    store.save_json(tmp_path / "q.json", q)
    assert store.load_model(tmp_path / "q.json", Quantity) == q


def test_read_jsonl_skips_torn_last_line(tmp_path: Path) -> None:
    """A half-written final line from a crash is ignored."""
    path = tmp_path / "events.jsonl"
    store.append_jsonl(path, {"a": 1})
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"a": ')
    assert store.read_jsonl(path) == [{"a": 1}]


def test_dl1_run_dir_created(tmp_path: Path) -> None:
    """Each run gets its own folder named by run ID."""
    rid = store.new_run_id()
    assert store.run_dir(rid, tmp_path).is_dir()
    assert rid.startswith("r-")


def test_dl3_second_call_served_from_cache(tmp_path: Path) -> None:
    """A repeated call with the same inputs does not recompute."""
    calls = []

    def compute() -> dict[str, float]:
        """Count calls and return a value.

        Returns
        -------
        dict
            A fixed payload.
        """
        calls.append(1)
        return {"a": 3.905}

    first = store.cached("mp", {"formula": "SrTiO3"}, compute, root=tmp_path)
    second = store.cached("mp", {"formula": "SrTiO3"}, compute, root=tmp_path)
    assert first == second == {"a": 3.905}
    assert len(calls) == 1


def test_dl3_key_ignores_dict_order_but_not_version() -> None:
    """Key order is canonical; a version change gives a new key."""
    assert store.cache_key("mp", {"a": 1, "b": 2}) == store.cache_key("mp", {"b": 2, "a": 1})
    assert store.cache_key("mp", {"a": 1}) != store.cache_key("mp", {"a": 1}, version="2")
    assert store.cache_key("mp", {"a": 1}) != store.cache_key("valency", {"a": 1})


def test_sys8_offline_miss_raises(tmp_path: Path) -> None:
    """Offline, a miss raises instead of calling out."""
    with pytest.raises(store.CacheMiss):
        store.cached("mp", {"formula": "X"}, lambda: 1, offline=True, root=tmp_path)
    store.cached("mp", {"formula": "X"}, lambda: 1, root=tmp_path)
    assert store.cached("mp", {"formula": "X"}, lambda: 2, offline=True, root=tmp_path) == 1
