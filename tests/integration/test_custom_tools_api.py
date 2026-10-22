"""Agent-built tools through the Python API and the HTTP API (AT-21, AT-27, AT-29)."""

import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_custom_tools import GOAL, PHASE_TURNS, campaign_turns, cfg, deps
from toolkit_fake import TEST_SOURCE, TOOL_SOURCE

from hephaestus.interfaces import http
from hephaestus.interfaces.python import Experiment, ExperimentError


@pytest.fixture(autouse=True)
def _cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the cache in the test folder and never check real backends.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test folder.
    monkeypatch : pytest.MonkeyPatch
        Patches the environment.
    """
    monkeypatch.setenv("HEPHAESTUS_CACHE_DIR", str(tmp_path / "cache"))


def wait_for(exp: Experiment, stage: str, timeout: float = 10.0) -> dict[str, Any]:
    """Wait until a proposal of ``exp`` waits at ``stage``.

    Parameters
    ----------
    exp : Experiment
        Run.
    stage : str
        ``plan`` or ``code``.
    timeout : float
        Seconds.

    Returns
    -------
    dict
        The waiting proposal.
    """
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        for p in exp.tool_proposals():
            if p["waiting"] and p["stage"] == stage:
                return p
        time.sleep(0.05)
    raise AssertionError(f"nothing waiting at {stage}")


def test_at29_python_api_decides_step_by_step(tmp_path: Path) -> None:
    """Without a callback, ``tool_proposals`` shows what waits and ``decide`` answers it."""
    d = deps(tmp_path, PHASE_TURNS + campaign_turns(), None)
    exp = Experiment(GOAL, cfg(tmp_path), root=tmp_path, deps=d)
    assert exp.parse().gap == "thermal_expansion"
    exp.start()
    plan = wait_for(exp, "plan")
    assert plan["name"] == "toy_expansion" and "revise" in plan["allowed"]
    with pytest.raises(ExperimentError, match="not waiting"):
        exp.decide("t9", "accept")
    exp.decide("t1", "accept")
    code = wait_for(exp, "code")
    assert code["source"] == TOOL_SOURCE and code["checks"]["tests"]["passed"]
    exp.decide("t1", "accept")
    state = exp.wait(30)
    assert state is not None and state.status == "finished"
    assert state.spec and state.spec.target_tool == "custom_toy_expansion"


def test_at27_http_tool_decisions_and_library(tmp_path: Path) -> None:
    """The UI's endpoints: proposals, decisions (409 when not offered) and the library."""
    lib = tmp_path / "lib"
    app = http.create_app(
        tmp_path,
        deps_factory=lambda: deps(tmp_path, PHASE_TURNS + campaign_turns(), None),
        library_dir=lib,
    )
    client = TestClient(app)
    config = cfg(tmp_path).model_dump(mode="json")
    run_id = client.post("/runs", json={"raw_goal": GOAL, "config": config}).json()["run_id"]
    exp = Experiment.load(run_id, tmp_path)
    wait_for(exp, "plan")
    props = client.get(f"/runs/{run_id}/tool-proposals").json()
    assert props[0]["waiting"] and props[0]["stage"] == "plan"
    bad = client.post(f"/runs/{run_id}/tool-decisions", json={"proposal_id": "t1", "action": "nonsense"})
    assert bad.status_code == 409
    ok = client.post(f"/runs/{run_id}/tool-decisions", json={"proposal_id": "t1", "action": "accept"})
    assert ok.status_code == 200 and ok.json()["decision"]["stage"] == "plan"
    wait_for(exp, "code")
    client.post(f"/runs/{run_id}/tool-decisions", json={"proposal_id": "t1", "action": "accept"})
    for _ in range(200):
        if client.get(f"/runs/{run_id}").json()["status"] == "finished":
            break
        time.sleep(0.05)
    tools = client.get("/tools").json()
    assert [t["name"] for t in tools["approved"]] == ["toy_expansion"] and tools["approved"][0][
        "source"
    ] == TOOL_SOURCE
    assert client.delete("/tools/toy_expansion").json() == {"removed": True}
    assert client.delete("/tools/toy_expansion").status_code == 404


def test_at21_http_approves_a_hand_written_tool_as_shown(tmp_path: Path) -> None:
    """Approval carries the reviewed fingerprint; a stale one is refused."""
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "toy_expansion.py").write_text(TOOL_SOURCE, encoding="utf-8")
    (lib / "test_toy_expansion.py").write_text(TEST_SOURCE, encoding="utf-8")
    client = TestClient(
        http.create_app(tmp_path, deps_factory=lambda: deps(tmp_path, [], None), library_dir=lib)
    )
    f = client.get("/tools").json()["unapproved"][0]
    assert f["check"]["passed"]
    assert client.post("/tools/toy_expansion/approve", json={"sha256": "0" * 64}).status_code == 409
    assert (
        client.post("/tools/toy_expansion/approve", json={"sha256": f["sha256"]}).json()["name"]
        == "toy_expansion"
    )


def test_at24_stop_while_waiting_cancels_the_run(tmp_path: Path) -> None:
    """A user stop while a plan waits ends the run as cancelled, with no campaign."""
    d = deps(tmp_path, PHASE_TURNS, None)
    exp = Experiment(GOAL, cfg(tmp_path), root=tmp_path, deps=d)
    exp.start()
    wait_for(exp, "plan")
    threading.Thread(target=exp.stop).start()
    state = exp.wait(10)
    assert state is not None and state.status == "stopped" and state.stop_reason == "cancelled by user"
