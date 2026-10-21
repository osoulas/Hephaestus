"""Parallel tool calls: one at a time, exact budget, shared results computed once."""

import threading
import time
from pathlib import Path
from typing import Any

import pytest
import scripts
from ase.build import bulk
from ase.calculators.emt import EMT
from factories import workspace

from hephaestus.core.events import EventBus
from hephaestus.core.types import Candidate, StructureRecord
from hephaestus.harness.agent import Campaign
from hephaestus.llm.scripted import ScriptedLLM
from hephaestus.tools import simulation as S
from hephaestus.tools.registry import HANDLERS, CandidateState, Workspace
from hephaestus.tools.stubs import check_novelty, stub_handlers


def run_concurrently(fn: Any, n: int) -> list[Any]:
    """Call ``fn(i)`` from ``n`` threads released together.

    Parameters
    ----------
    fn : callable
        Work for thread ``i``.
    n : int
        Number of threads.

    Returns
    -------
    list
        Results in thread order.
    """
    gate = threading.Barrier(n)
    out: list[Any] = [None] * n

    def work(i: int) -> None:
        """Wait for the others, then call ``fn``.

        Parameters
        ----------
        i : int
            Thread index.
        """
        gate.wait()
        out[i] = fn(i)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


class Tracker:
    """Slow handlers that record which calls overlap."""

    def __init__(self) -> None:
        """Start with nothing running."""
        self.lock = threading.Lock()
        self.active: dict[str, int] = {}
        self.peak_total = 0
        self.peak_same = 0
        self.order: list[str] = []
        self.overlapped_exclusive = False

    def handler(self, ws: Workspace, candidate_id: str, tag: str = "") -> dict:
        """Sleep while recording concurrency.

        Parameters
        ----------
        ws : Workspace
            Workspace.
        candidate_id : str
            Candidate the call works on.
        tag : str
            Label recorded in ``order``.

        Returns
        -------
        dict
            Empty result.
        """
        with self.lock:
            self.active[candidate_id] = self.active.get(candidate_id, 0) + 1
            self.peak_same = max(self.peak_same, self.active[candidate_id])
            self.peak_total = max(self.peak_total, sum(self.active.values()))
            self.order.append(f"{candidate_id}:{tag}")
        time.sleep(0.03)
        with self.lock:
            self.active[candidate_id] -= 1
        return {}

    def exclusive(self, ws: Workspace) -> dict:
        """Record whether anything else was running.

        Parameters
        ----------
        ws : Workspace
            Workspace.

        Returns
        -------
        dict
            Empty result.
        """
        with self.lock:
            self.overlapped_exclusive |= sum(self.active.values()) > 0
        time.sleep(0.03)
        return {}


def campaign_with(tracker: Tracker, tmp_path: Path) -> tuple[Campaign, EventBus]:
    """Campaign whose tools are the tracker's handlers.

    Parameters
    ----------
    tracker : Tracker
        Records concurrency.
    tmp_path : pathlib.Path
        Runs folder.

    Returns
    -------
    tuple
        ``(campaign, bus)``.
    """
    bus = EventBus(tmp_path)
    camp = Campaign(workspace(), ScriptedLLM(), handlers={"relax": tracker.handler}, bus=bus)
    camp._run_control = lambda name, args: (tracker.exclusive(camp.ws), False)  # type: ignore[method-assign]
    return camp, bus


def test_calls_on_different_candidates_run_in_parallel(tmp_path: Path) -> None:
    """Relaxations of four different candidates overlap; the budget counts every call exactly."""
    tracker = Tracker()
    camp, bus = campaign_with(tracker, tmp_path)
    run_concurrently(lambda i: camp.call_tool("relax", {"candidate_id": f"c1-{i}"}), 4)
    assert tracker.peak_total > 1 and camp.ws.tool_calls == 4
    events = [e for e in bus.history("r-test") if e.section == "tool"]
    for cid in {e.payload["call_id"] for e in events}:  # every call has one started, then one finished
        assert [e.status for e in events if e.payload["call_id"] == cid] == ["started", "finished"]


def test_calls_on_the_same_candidate_never_overlap(tmp_path: Path) -> None:
    """Eight simultaneous calls on one candidate run one at a time."""
    tracker = Tracker()
    camp, _ = campaign_with(tracker, tmp_path)
    run_concurrently(lambda i: camp.call_tool("relax", {"candidate_id": "c1-1"}), 8)
    assert tracker.peak_same == 1 and camp.ws.tool_calls == 8


def test_run_control_runs_alone(tmp_path: Path) -> None:
    """request_review never overlaps with a candidate call."""
    tracker = Tracker()
    camp, _ = campaign_with(tracker, tmp_path)

    def call(i: int) -> object:
        """Mix candidate calls and reviews.

        Parameters
        ----------
        i : int
            Thread index.

        Returns
        -------
        object
            Tool outcome.
        """
        if i % 3 == 0:
            return camp.call_tool("request_review", {"summary": "s"})
        return camp.call_tool("relax", {"candidate_id": f"c1-{i}"})

    run_concurrently(call, 9)
    assert not tracker.overlapped_exclusive


def test_run_batch_keeps_order_per_candidate_and_parallelises_across(tmp_path: Path) -> None:
    """A turn's calls on one candidate run in the order given; different candidates overlap."""
    tracker = Tracker()
    camp, _ = campaign_with(tracker, tmp_path)
    calls = [("relax", {"candidate_id": c, "tag": str(n)}) for n in range(3) for c in ("c1-1", "c1-2")]
    out = camp.run_batch(calls)
    assert len(out) == 6 and tracker.peak_total == 2
    for c in ("c1-1", "c1-2"):
        assert [x for x in tracker.order if x.startswith(c)] == [f"{c}:0", f"{c}:1", f"{c}:2"]


def test_concurrent_elastic_requests_on_one_candidate_compute_the_tensor_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two mechanical properties requested at once share one elastic calculation."""
    monkeypatch.setenv("HEPHAESTUS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(S, "FACTORIES", {name: (lambda d, t: EMT()) for name in S.FACTORIES})
    monkeypatch.setattr(S, "_LOADED", {})
    calls = [0]
    real = S.elastic_tensor

    def counting(*a: Any, **k: Any) -> Any:
        """Count elastic calculations.

        Parameters
        ----------
        *a, **k : Any
            Passed through.

        Returns
        -------
        numpy.ndarray
            The tensor.
        """
        calls[0] += 1
        return real(*a, **k)

    monkeypatch.setattr(S, "elastic_tensor", counting)
    ws = workspace()
    cand = Candidate(
        candidate_id="c1-1", iteration=1, composition="Cu", structure_type="perovskite_cubic", rationale="r"
    )
    rec = StructureRecord(
        candidate_id="c1-1",
        cif="",
        source="mp:mp-30",
        n_atoms=4,
        pmg_dict=S.to_pmg(bulk("Cu", "fcc", a=3.6, cubic=True)).as_dict(),
    )
    ws.candidates["c1-1"] = CandidateState(candidate=cand, round=1, record=rec)
    camp = Campaign(
        ws,
        ScriptedLLM(),
        handlers=dict(HANDLERS),
        bus=EventBus(tmp_path),
    )
    camp.call_tool("relax", {"candidate_id": "c1-1", "potential": "mace-mp-0-small", "fmax": 0.01})
    props = ["bulk_modulus", "shear_modulus", "debye_temperature"]
    results = run_concurrently(
        lambda i: camp.call_tool("predict_property", {"candidate_id": "c1-1", "property": props[i]}), 3
    )
    assert all(not err for _, err in results) and calls[0] == 1


def test_calls_after_submission_are_refused(tmp_path: Path) -> None:
    """A turn that submits and then keeps calling gets errors for the later calls, and the run ends."""
    turns = [
        *scripts.ROUND_1[:3],
        [scripts.submit("c1-1", accept_not_met=True), {"tool": "candidate_status", "input": {}}],
    ]
    camp = Campaign(
        workspace(),
        ScriptedLLM(turns=turns),
        handlers=stub_handlers(),
        bus=EventBus(tmp_path),
        novelty_fn=check_novelty,
    )
    sub = camp.run()
    assert sub is not None and sub.candidate_id == "c1-1"
    last = camp.messages[-1]["content"]
    assert len(last) == 2 and last[1]["is_error"] and "campaign is over" in last[1]["content"]


def test_same_structure_on_two_candidates_is_relaxed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two candidates with identical structures relaxed at the same time share one relaxation."""
    monkeypatch.setenv("HEPHAESTUS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(S, "FACTORIES", {name: (lambda d, t: EMT()) for name in S.FACTORIES})
    monkeypatch.setattr(S, "_LOADED", {})
    calls = [0]
    real = S.relax_atoms

    def counting(*a: Any, **k: Any) -> Any:
        """Count relaxations.

        Parameters
        ----------
        *a, **k : Any
            Passed through.

        Returns
        -------
        tuple
            Relaxation result.
        """
        calls[0] += 1
        return real(*a, **k)

    monkeypatch.setattr(S, "relax_atoms", counting)
    ws = workspace()
    pmg = S.to_pmg(bulk("Cu", "fcc", a=3.7, cubic=True)).as_dict()
    for cid in ("c1-1", "c1-2"):
        cand = Candidate(
            candidate_id=cid, iteration=1, composition="Cu", structure_type="perovskite_cubic", rationale="r"
        )
        rec = StructureRecord(candidate_id=cid, cif="", source="mp:mp-30", n_atoms=4, pmg_dict=pmg)
        ws.candidates[cid] = CandidateState(candidate=cand, round=1, record=rec)
    camp = Campaign(ws, ScriptedLLM(), handlers=dict(HANDLERS), bus=EventBus(tmp_path))
    out = run_concurrently(
        lambda i: camp.call_tool(
            "relax", {"candidate_id": f"c1-{i + 1}", "potential": "mace-mp-0-small", "fmax": 0.01}
        ),
        2,
    )
    assert all(not err for _, err in out) and calls[0] == 1
    assert ws.candidates["c1-1"].energy_eV_per_atom == ws.candidates["c1-2"].energy_eV_per_atom


def test_each_thread_gets_its_own_calculator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Threads share the loaded model but not the calculator object (ASE keeps results on it)."""
    monkeypatch.setattr(S, "FACTORIES", {"mace-mp-0-small": lambda d, t: EMT()})
    monkeypatch.setattr(S, "_LOADED", {})
    calcs = run_concurrently(lambda i: S.load_potential("mace-mp-0-small", "cpu"), 3)
    assert len({id(c) for c in calcs}) == 3 and len(S._LOADED) == 1


def test_se15_gpu_calls_serialise_cpu_calls_do_not() -> None:
    """Only CUDA calculators take the GPU lock."""

    class Fake:
        """Calculator stand-in with a device."""

        def __init__(self, device: str) -> None:
            """Set the device.

            Parameters
            ----------
            device : str
                Device string.
            """
            self.device = device

    assert S.model_lock(Fake("cuda:0")) is S.GPU_LOCK
    assert S.model_lock(Fake("cpu")) is not S.GPU_LOCK and S.model_lock(EMT()) is not S.GPU_LOCK
