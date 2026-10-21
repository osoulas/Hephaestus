"""HTTP API on scripted runs: parse, start, SSE events, state, brief, structure, resume, replay."""

import json
from pathlib import Path
from typing import Any

import pytest
import scripts
from fastapi.testclient import TestClient
from test_python_api import deps

from hephaestus.interfaces import http


def not_ready(check_claude_code: bool = True) -> dict[str, Any]:
    """Backend status with nothing signed in.

    Parameters
    ----------
    check_claude_code : bool
        Ignored.

    Returns
    -------
    dict
        Every backend not ready.
    """
    off = {"ready": False, "error": "signed out", "auth_method": None}
    return {"api": off, "claude-code": off, "auto": {"resolves_to": "claude-code", "ready": False}}


@pytest.fixture(autouse=True)
def _offline_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never run ``claude auth status`` or import torch in tests.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Patches the status helpers.
    """
    monkeypatch.setattr(http, "backend_status", not_ready)
    monkeypatch.setattr(http, "device_status", lambda: {"torch": False, "cuda": False, "device": "cpu"})


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Return a client on a scripted app writing runs to ``tmp_path``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Runs folder.

    Returns
    -------
    fastapi.testclient.TestClient
        Client.
    """
    return TestClient(http.create_app(tmp_path, deps_factory=deps))


def stream(client: TestClient, run_id: str) -> list[dict[str, Any]]:
    """Read a run's SSE stream to its ``end`` event.

    Parameters
    ----------
    client : TestClient
        Client.
    run_id : str
        Run ID.

    Returns
    -------
    list of dict
        The ``progress`` events' data.
    """
    out, kind = [], None
    with client.stream("GET", f"/runs/{run_id}/events") as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        for line in resp.iter_lines():
            if line.startswith("event: "):
                kind = line[7:]
            elif line.startswith("data: ") and kind == "progress":
                out.append(json.loads(line[6:]))
            elif line.startswith("data: ") and kind == "end":
                break
    return out


def test_ui2_ui3_ui8_parse_run_stream_and_brief(client: TestClient) -> None:
    """Parse, start with the spec, stream to the end, then read state, brief and structure."""
    parsed = client.post("/parse", json={"raw_goal": scripts.GOAL}).json()
    assert parsed["questions"] == [] and parsed["spec"]["reference_material"] == "SrTiO3"
    run_id = client.post("/runs", json={"spec": parsed["spec"]}).json()["run_id"]
    events = stream(client, run_id)
    assert (events[-1]["section"], events[-1]["status"]) == ("run", "finished")
    assert not any(e["section"] == "goal_parser" for e in events)  # spec given: parser skipped
    state = client.get(f"/runs/{run_id}").json()
    assert state["status"] == "finished" and state["best"] == "c2-1" and state["running"] is False
    brief = client.get(f"/runs/{run_id}/brief")
    assert brief.status_code == 200 and "## Scope and assumptions" in brief.text
    assert client.get(f"/runs/{run_id}/brief", params={"format": "pdf"}).status_code == 404
    assert client.get(f"/runs/{run_id}/structure/c2-1").status_code == 404  # stub tools store no atoms
    assert client.get(f"/runs/{run_id}/structure/zz").status_code == 404
    assert [r["run_id"] for r in client.get("/runs").json()] == [run_id]
    assert client.post(f"/runs/{run_id}/resume").status_code == 409
    assert client.post(f"/runs/{run_id}/stop").json() == {"stopping": False}


def test_ui16_history_survives_a_new_server(client: TestClient, tmp_path: Path) -> None:
    """A second app on the same folder (a restart) serves the earlier events from disk."""
    run_id = client.post("/runs", json={"raw_goal": scripts.GOAL}).json()["run_id"]
    first = stream(client, run_id)
    again = stream(TestClient(http.create_app(tmp_path, deps_factory=deps)), run_id)
    assert [e["seq"] for e in again] == [e["seq"] for e in first]


def test_ui9_replay(client: TestClient) -> None:
    """A replay streams the stored events under its own ID."""
    run_id = client.post("/runs", json={"raw_goal": scripts.GOAL}).json()["run_id"]
    stored = stream(client, run_id)
    replay_id = client.post(f"/replay/{run_id}", json={"speed": 0}).json()["replay_id"]
    replayed = stream(client, replay_id)
    assert len(replayed) == len(stored) and {e["run_id"] for e in replayed} == {replay_id}
    assert client.post("/replay/missing").status_code == 404


def test_or11_unready_backend_answers_503(tmp_path: Path) -> None:
    """With backend checks on, a signed-out backend is refused instead of hanging."""
    c = TestClient(http.create_app(tmp_path, deps_factory=deps, check_backends=True))
    r = c.post("/runs", json={"raw_goal": scripts.GOAL, "config": {"llm_backend": "claude-code"}})
    assert r.status_code == 503 and "signed out" in r.json()["detail"]
    assert c.post("/parse", json={"raw_goal": "x", "llm_backend": "api"}).status_code == 503


def test_ui12_status_endpoints(client: TestClient) -> None:
    """Health, models and config defaults answer at once."""
    health = client.get("/health").json()
    assert health["ok"] is True and "valency" in health
    assert "claude-opus-5-5" in client.get("/models").json()["models"]
    assert client.get("/config/defaults").json()["confidence_threshold"] == 72
    assert client.get("/runs/nope").status_code == 404
    assert client.post("/runs", json={}).status_code == 422
    assert client.get("/").status_code == 200


def test_ui1_web_ui_is_served_offline(client: TestClient) -> None:
    """The static UI loads from the package with no external assets (UI-9: works with Wi-Fi off)."""
    index = client.get("/")
    assert index.status_code == 200 and "<title>Hephaestus</title>" in index.text
    app_js = client.get("/js/app.js")
    assert app_js.status_code == 200 and "javascript" in app_js.headers["content-type"]
    for path in ["/css/app.css", "/js/run.js", "/js/viewer.js", "/favicon.svg"]:
        assert client.get(path).status_code == 200, path
    web = Path(http.WEB_DIR)
    for f in web.rglob("*"):
        if f.suffix in {".html", ".js", ".css"}:
            text = f.read_text(encoding="utf-8")
            assert "https://cdn" not in text and "fonts.googleapis" not in text, f


def test_ui12_claude_sign_in_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Start, follow and finish a Claude Code sign-in over HTTP with a fake CLI."""
    from test_auth import fake_login, wait_for

    login = fake_login(tmp_path, "code")
    c = TestClient(http.create_app(tmp_path, deps_factory=deps, claude_login=login))
    st = c.post("/auth/claude/start").json()
    assert st["status"] in ("starting", "waiting")
    wait_for(login, {"waiting"})
    st = c.get("/auth/claude").json()
    assert st["status"] == "waiting" and st["code_prompt"] and st["url"].startswith("https://")
    assert c.post("/auth/claude/code", json={"code": "good"}).status_code == 200
    wait_for(login, {"signed_in"})
    assert c.get("/auth/claude").json()["signed_in"] is True
    assert c.post("/auth/claude/code", json={"code": "again"}).status_code == 409


def test_ui12_api_key_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Save, report and forget an API key; the key itself never comes back."""
    monkeypatch.setenv("HEPHAESTUS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    c = TestClient(http.create_app(tmp_path, deps_factory=deps, key_check=None))
    assert c.get("/auth/anthropic").json()["present"] is False
    assert c.post("/auth/anthropic", json={"api_key": "nope"}).status_code == 422
    key = "sk-ant-api03-secret-ABCD"
    saved = c.post("/auth/anthropic", json={"api_key": key})
    assert saved.status_code == 200 and saved.json()["hint"] == "sk-…ABCD" and key not in saved.text
    assert c.delete("/auth/anthropic").json()["present"] is False


def test_or1_runs_refused_without_mace_or_mp_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With checks on, a run that would fail at its first step is refused with the fix, not started."""
    ready = {"ready": True, "error": None, "auth_method": None}
    monkeypatch.setattr(
        http,
        "backend_status",
        lambda check_claude_code=True: {
            "api": ready,
            "claude-code": ready,
            "auto": {"resolves_to": "api", "ready": True},
        },
    )
    monkeypatch.setattr(
        http, "simulation_status", lambda: {"mace": False, "matgl": True, "torch": True, "mp_key": False}
    )
    c = TestClient(http.create_app(tmp_path, deps_factory=deps, check_backends=True))
    r = c.post("/runs", json={"raw_goal": scripts.GOAL})
    assert (
        r.status_code == 503
        and "MACE is not installed" in r.json()["detail"]
        and "Materials Project" in r.json()["detail"]
    )
    assert not list(tmp_path.glob("r-*"))
    assert c.get("/health").json()["simulation"]["mace"] is False


def test_dl7_materials_project_key_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Save, report and forget an MP key owner-only; it goes into MP_API_KEY and never comes back."""
    monkeypatch.setenv("HEPHAESTUS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MP_API_KEY", raising=False)
    c = TestClient(
        http.create_app(
            tmp_path,
            deps_factory=deps,
            mp_check=lambda k: "Materials Project rejected this key." if k.startswith("bad") else None,
        )
    )
    assert c.get("/auth/materials-project").json()["present"] is False
    assert c.post("/auth/materials-project", json={"api_key": "short"}).status_code == 422
    assert c.post("/auth/materials-project", json={"api_key": "bad" + "x" * 29}).status_code == 422
    key = "a1B2" * 8
    saved = c.post("/auth/materials-project", json={"api_key": key})
    assert saved.json() == {"present": True, "source": "saved", "hint": "…a1B2"} and key not in saved.text
    import os

    assert os.environ["MP_API_KEY"] == key
    assert (tmp_path / "home" / "mp_api_key").read_text(encoding="utf-8") == key
    assert c.delete("/auth/materials-project").json()["present"] is False and "MP_API_KEY" not in os.environ
