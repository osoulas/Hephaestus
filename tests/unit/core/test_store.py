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


def test_dl3_cached_second_call_does_not_compute(tmp_path: Path) -> None:
    """The second identical call is served from disk; a new version recomputes."""
    calls = []

    def compute() -> dict:
        """Count calls.

        Returns
        -------
        dict
            A result.
        """
        calls.append(1)
        return {"e": -1.5}

    for _ in range(2):
        assert store.cached("relax", {"s": "SrTiO3", "fmax": 0.02}, compute, "v1", tmp_path) == {"e": -1.5}
    assert len(calls) == 1
    store.cached("relax", {"fmax": 0.02, "s": "SrTiO3"}, compute, "v1", tmp_path)  # key order irrelevant
    assert len(calls) == 1
    store.cached("relax", {"s": "SrTiO3", "fmax": 0.02}, compute, "v2", tmp_path)
    assert len(calls) == 2


def test_dl3_concurrent_identical_inputs_compute_once(tmp_path: Path) -> None:
    """Callers with identical inputs at the same moment share one computation."""
    import threading
    import time

    calls = []
    gate = threading.Barrier(4)

    def compute() -> int:
        """Slow computation that counts calls.

        Returns
        -------
        int
            A result.
        """
        calls.append(1)
        time.sleep(0.05)
        return 7

    def work() -> None:
        """Ask for the same key as the other threads."""
        gate.wait()
        assert store.cached("relax", {"s": "same"}, compute, "v", tmp_path) == 7

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1


def test_sys8_offline_miss_raises(tmp_path: Path) -> None:
    """Offline, a miss raises instead of computing; a cached entry is still served."""
    with pytest.raises(store.CacheMiss):
        store.cached("mp", {"formula": "X"}, lambda: 1, root=tmp_path, offline=True)
    store.cached("mp", {"formula": "X"}, lambda: 1, root=tmp_path)
    assert store.cached("mp", {"formula": "X"}, lambda: 2, root=tmp_path, offline=True) == 1
