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


def test_or7_concurrent_calls_run_one_at_a_time(tmp_path: Path) -> None:
    """Eight simultaneous calls never overlap; the budget counts all eight; events pair up."""
    active, peak = [0], [0]

    def slow(ws: Workspace, candidate_id: str) -> dict:
        """Record how many calls are inside a handler at once.

        Parameters
        ----------
        ws : Workspace
            Workspace.
        candidate_id : str
            Ignored.

        Returns
        -------
        dict
            Empty result.
        """
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        time.sleep(0.01)
        active[0] -= 1
        return {}

    bus = EventBus(tmp_path)
    camp = Campaign(workspace(), ScriptedLLM(), handlers={"slow": slow}, bus=bus)
    run_concurrently(lambda i: camp.call_tool("slow", {"candidate_id": f"c{i}"}), 8)
    assert peak[0] == 1 and camp.ws.tool_calls == 8
    statuses = [e.status for e in bus.history("r-test") if e.section == "tool"]
    assert statuses == ["started", "finished"] * 8  # never interleaved


def test_concurrent_elastic_requests_compute_the_tensor_once(
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
