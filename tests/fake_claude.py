"""A fake ``claude auth login`` CLI for sign-in tests and offline UI demos."""

import sys
import time
from pathlib import Path
from typing import Any

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
