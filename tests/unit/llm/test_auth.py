"""Claude Code sign-in from the UI and the saved Anthropic API key (OR-11, UI-12, DL-7)."""

import os
import stat
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from hephaestus.llm import auth
from hephaestus.llm.auth import ClaudeLogin

FAKE_CLI = """
import sys, time
print("Opening browser to sign in…")
print("If the browser didn't open, visit: https://claude.test/oauth/authorize?code=true&state=abc")
sys.stdout.write("Paste code here if prompted > ")
sys.stdout.flush()
mode = sys.argv[1]
if mode == "auto":          # the browser callback finishes the sign-in by itself
    time.sleep(0.2)
    sys.exit(0)
if mode == "hang":
    time.sleep(30)
code = sys.stdin.readline().strip()
if code == "good":
    sys.exit(0)
print("Invalid code")
sys.exit(1)
"""


def fake_login(tmp_path: Path, mode: str, signed_in: bool = True, **kw: Any) -> ClaudeLogin:
    """Return a sign-in runner driving a fake CLI.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Where the fake CLI is written.
    mode : {"auto", "code", "hang"}
        How the fake CLI finishes.
    signed_in : bool
        What the status check reports afterwards.
    **kw
        Extra ``ClaudeLogin`` arguments.

    Returns
    -------
    ClaudeLogin
        The runner.
    """
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLI, encoding="utf-8")
    return ClaudeLogin(
        command=[sys.executable, str(script), mode],
        verify=lambda: {"signed_in": signed_in, "error": None if signed_in else "still signed out"},
        **kw,
    )


def wait_for(login: ClaudeLogin, statuses: set[str], timeout: float = 10.0) -> dict[str, Any]:
    """Poll until the sign-in reaches one of ``statuses``.

    Parameters
    ----------
    login : ClaudeLogin
        Runner.
    statuses : set of str
        Target statuses.
    timeout : float
        Seconds to wait.

    Returns
    -------
    dict
        The state reached.
    """
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        st = login.state()
        if st["status"] in statuses:
            return st
        time.sleep(0.05)
    raise AssertionError(f"stuck at {login.state()}")


def test_ui12_sign_in_finishes_by_itself_after_authorising(tmp_path: Path) -> None:
    """The browser callback ends the CLI; the state turns signed_in and readiness is refreshed."""
    done: list[bool] = []
    login = fake_login(tmp_path, "auto", on_done=lambda: done.append(True))
    login.start()
    st = wait_for(login, {"signed_in", "failed"})
    assert st["status"] == "signed_in" and st["url"].startswith("https://claude.test/oauth")
    assert done == [True]


def test_ui12_code_fallback(tmp_path: Path) -> None:
    """When the page shows a code, submitting it completes the sign-in; a wrong code fails."""
    login = fake_login(tmp_path, "code")
    st = login.start()
    st = wait_for(login, {"waiting"})
    assert st["code_prompt"] and st["url"]
    with pytest.raises(ValueError):
        login.submit_code("   ")
    login.submit_code("good")
    assert wait_for(login, {"signed_in", "failed"})["status"] == "signed_in"

    login.start()
    wait_for(login, {"waiting"})
    login.submit_code("bad")
    st = wait_for(login, {"signed_in", "failed"})
    assert st["status"] == "failed" and "Invalid code" in st["error"]


def test_ui12_cancel_and_timeout(tmp_path: Path) -> None:
    """A sign-in can be cancelled, and one left open past the timeout fails."""
    login = fake_login(tmp_path, "hang")
    login.start()
    wait_for(login, {"waiting"})
    assert login.cancel()["status"] == "cancelled"
    with pytest.raises(ValueError):
        login.submit_code("good")

    slow = fake_login(tmp_path, "hang", timeout_s=0.3)
    slow.start()
    st = wait_for(slow, {"failed"})
    assert "timed out" in st["error"]


def test_or11_exit_zero_but_still_signed_out_is_reported(tmp_path: Path) -> None:
    """A CLI that exits cleanly while the status check still says signed out is a failure."""
    login = fake_login(tmp_path, "auto", signed_in=False)
    login.start()
    st = wait_for(login, {"signed_in", "failed"})
    assert st["status"] == "failed" and st["error"] == "still signed out"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``~/.hephaestus`` at a temporary folder and clear API credentials.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.
    monkeypatch : pytest.MonkeyPatch
        Environment patcher.

    Returns
    -------
    pathlib.Path
        The folder.
    """
    monkeypatch.setenv("HEPHAESTUS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    return tmp_path / "home"


def test_dl7_api_key_is_saved_owner_only_and_never_echoed(home: Path) -> None:
    """A saved key is owner-only on disk, active at once, and status shows only a hint."""
    key = "sk-ant-api03-" + "x" * 20 + "WXYZ"
    st = auth.save_api_key(key, check=None)
    assert st == {"present": True, "source": "saved", "hint": "sk-…WXYZ"}
    assert os.environ["ANTHROPIC_API_KEY"] == key
    path = auth.api_key_path()
    assert path.read_text(encoding="utf-8") == key
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    del os.environ["ANTHROPIC_API_KEY"]
    assert auth.load_api_key() and os.environ["ANTHROPIC_API_KEY"] == key
    assert auth.forget_api_key() == {"present": False, "source": None, "hint": None}
    assert not path.exists() and "ANTHROPIC_API_KEY" not in os.environ


def test_or11_bad_keys_are_refused(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed or rejected keys raise and are not saved; an environment key wins over a saved one."""
    with pytest.raises(ValueError, match="does not look like"):
        auth.save_api_key("hello there", check=None)
    with pytest.raises(ValueError, match="rejected"):
        auth.save_api_key("sk-ant-bad", check=lambda k: "The API rejected this key.")
    assert not auth.api_key_path().exists()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env-1234")
    assert not auth.load_api_key()
    assert auth.api_key_status()["source"] == "environment"
