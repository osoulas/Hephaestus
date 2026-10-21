"""Claude Code login and the Valency OAuth token store in ``~/.hephaestus/`` (LM-2, DL-7).

A signed-out Claude Code CLI hangs instead of failing, so ``claude_code_status`` must be
checked before the ``claude-code`` backend starts (OR-11).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def claude_cli_path() -> str | None:
    """Locate the Claude Code CLI: the Agent SDK's bundled copy, else ``claude`` on PATH.

    Returns
    -------
    str or None
        Path to the executable, or ``None`` when not installed.
    """
    try:
        import claude_agent_sdk

        bundled = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
        if bundled.exists():
            return str(bundled)
        exe = bundled.with_suffix(".exe")
        if exe.exists():
            return str(exe)
    except ImportError:
        pass
    return shutil.which("claude")


def claude_code_status(timeout_s: float = 20.0) -> dict[str, Any]:
    """Ask the CLI whether it is signed in (``claude auth status``).

    Parameters
    ----------
    timeout_s : float
        Seconds before giving up.

    Returns
    -------
    dict
        ``{signed_in, auth_method, error}``; never includes tokens or the email address.
    """
    cli = claude_cli_path()
    if cli is None:
        return {"signed_in": False, "auth_method": None, "error": "Claude Code CLI not found."}
    try:
        proc = subprocess.run(
            [cli, "auth", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            encoding="utf-8",
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"signed_in": False, "auth_method": None, "error": f"auth status failed: {exc}"}
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        data = {}
    signed_in = bool(data.get("loggedIn"))
    return {
        "signed_in": signed_in,
        "auth_method": data.get("authMethod"),
        "error": None
        if signed_in
        else (proc.stderr.strip()[:200] or "Not signed in: run `python -m hephaestus login`."),
    }
