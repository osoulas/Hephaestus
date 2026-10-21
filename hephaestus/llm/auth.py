"""Claude Code sign-in and status, and the saved Anthropic API key in ``~/.hephaestus/`` (OR-11, UI-12, DL-7).

A signed-out Claude Code CLI hangs instead of failing, so ``claude_code_status`` must be
checked before the ``claude-code`` backend starts (OR-11).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable
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


# ---------------------------------------------------------------------- Claude Code sign-in

URL_RE = re.compile(r"https://\S+")
CODE_PROMPT = "Paste code"
LOGIN_TIMEOUT_S = 600.0


class ClaudeLogin:
    """Run ``claude auth login --claudeai`` in the background for the UI's "Sign in to Claude" (UI-12).

    The CLI opens the sign-in page itself; its callback finishes the sign-in once the user
    authorises. It also prints a fallback URL whose page shows a code, which ``submit_code``
    passes to the CLI. Nothing the CLI prints is logged or kept beyond the URL and a short error.
    """

    def __init__(
        self,
        command: list[str] | None = None,
        verify: Callable[[], dict[str, Any]] | None = None,
        on_done: Callable[[], None] | None = None,
        timeout_s: float = LOGIN_TIMEOUT_S,
    ) -> None:
        """Create an idle sign-in.

        Parameters
        ----------
        command : list of str, optional
            Command to run; defaults to the bundled CLI's ``auth login --claudeai``.
        verify : callable, optional
            Returns ``claude_code_status()``-like ``{signed_in, error}`` after the CLI exits.
        on_done : callable, optional
            Called once a sign-in succeeds (e.g. to refresh cached readiness).
        timeout_s : float
            Seconds before an unfinished sign-in is cancelled.
        """
        self.command = command
        self.verify = verify or claude_code_status
        self.on_done = on_done
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._state: dict[str, Any] = {"status": "idle", "url": None, "code_prompt": False, "error": None}
        self._output = ""

    def state(self) -> dict[str, Any]:
        """Return the sign-in state.

        Returns
        -------
        dict
            ``{status, url, code_prompt, error}``; ``status`` is ``idle``, ``starting``,
            ``waiting``, ``verifying``, ``signed_in``, ``failed`` or ``cancelled``.
        """
        with self._lock:
            return dict(self._state)

    def start(self) -> dict[str, Any]:
        """Start the sign-in, or return the running one's state.

        Returns
        -------
        dict
            :meth:`state` just after starting.
        """
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return dict(self._state)
            cmd = self.command
            if cmd is None:
                cli = claude_cli_path()
                if cli is None:
                    self._state = {
                        "status": "failed",
                        "url": None,
                        "code_prompt": False,
                        "error": "Claude Code CLI not found.",
                    }
                    return dict(self._state)
                cmd = [cli, "auth", "login", "--claudeai"]
            self._output = ""
            self._state = {"status": "starting", "url": None, "code_prompt": False, "error": None}
            try:
                self._proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                )
            except OSError as exc:
                self._state.update(status="failed", error=f"Could not start the sign-in: {exc}")
                return dict(self._state)
            proc = self._proc
        threading.Thread(target=self._read, args=(proc,), daemon=True, name="claude-login").start()
        timer = threading.Timer(self.timeout_s, self._expire, args=(proc,))
        timer.daemon = True
        timer.start()
        return self.state()

    def submit_code(self, code: str) -> dict[str, Any]:
        """Pass the code shown on the fallback page to the CLI.

        Parameters
        ----------
        code : str
            The code from the sign-in page.

        Returns
        -------
        dict
            :meth:`state`.

        Raises
        ------
        ValueError
            If no sign-in is waiting, or the code is empty.
        """
        code = code.strip()
        if not code:
            raise ValueError("Paste the code from the sign-in page.")
        with self._lock:
            proc = self._proc
            waiting = self._state["status"] in ("starting", "waiting")
            if proc is None or proc.poll() is not None or proc.stdin is None or not waiting:
                raise ValueError("No sign-in is waiting for a code; start one first.")
            proc.stdin.write((code + "\n").encode())
            proc.stdin.flush()
            self._state.update(status="verifying")
        return self.state()

    def cancel(self) -> dict[str, Any]:
        """Stop a running sign-in.

        Returns
        -------
        dict
            :meth:`state`.
        """
        with self._lock:
            proc = self._proc
            if proc is not None and proc.poll() is None:
                proc.terminate()
                self._state.update(status="cancelled", error=None)
        return self.state()

    def _expire(self, proc: subprocess.Popen[bytes]) -> None:
        """Cancel a sign-in left unfinished past the timeout.

        Parameters
        ----------
        proc : subprocess.Popen
            The sign-in process it was started for.
        """
        with self._lock:
            if proc is self._proc and proc.poll() is None:
                proc.terminate()
                self._state.update(status="failed", error="The sign-in timed out. Try again.")

    def _read(self, proc: subprocess.Popen[bytes]) -> None:
        """Follow the CLI's output until it exits, then record the outcome.

        Parameters
        ----------
        proc : subprocess.Popen
            The sign-in process.
        """
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            with self._lock:
                self._output = (self._output + chunk.decode("utf-8", "replace"))[-4000:]
                if self._state["url"] is None and (m := URL_RE.search(self._output)):
                    self._state["url"] = m.group(0)
                if CODE_PROMPT in self._output:
                    self._state["code_prompt"] = True
                if self._state["status"] == "starting" and self._state["url"]:
                    self._state["status"] = "waiting"
        code = proc.wait()
        with self._lock:
            if proc is not self._proc or self._state["status"] in ("cancelled", "failed"):
                return
            self._state["status"] = "verifying"
            tail = [ln.strip() for ln in self._output.splitlines() if ln.strip() and not URL_RE.search(ln)]
        signed_in = False
        error = None
        if code == 0:
            st = self.verify()
            signed_in = bool(st.get("signed_in"))
            error = (
                None
                if signed_in
                else st.get("error") or "Sign-in finished but Claude Code is still signed out."
            )
        else:
            error = (tail[-1] if tail else f"Sign-in failed (exit code {code}).")[:300]
        with self._lock:
            if proc is self._proc:
                self._state.update(status="signed_in" if signed_in else "failed", error=error)
        if signed_in and self.on_done:
            self.on_done()


# ---------------------------------------------------------------------- Anthropic API key

API_KEY_ENV = "ANTHROPIC_API_KEY"


def api_key_path() -> Path:
    """Return the owner-only file for a saved Anthropic API key (DL-7).

    Returns
    -------
    pathlib.Path
        ``~/.hephaestus/anthropic_api_key``.
    """
    from hephaestus.core.config import user_dir

    return user_dir() / "anthropic_api_key"


def _stored_key() -> str | None:
    """Read the saved key.

    Returns
    -------
    str or None
        The key, or ``None`` when none is saved.
    """
    try:
        return api_key_path().read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def load_api_key() -> bool:
    """Put a saved key into the environment unless one is already set (startup of server and CLI).

    Returns
    -------
    bool
        True when a saved key was loaded.
    """
    if os.environ.get(API_KEY_ENV) or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return False
    key = _stored_key()
    if key:
        os.environ[API_KEY_ENV] = key
    return bool(key)


def check_api_key(key: str, timeout_s: float = 15.0) -> str | None:
    """Ask the API whether a key works (one cheap request).

    Parameters
    ----------
    key : str
        Anthropic API key.
    timeout_s : float
        Request timeout, s.

    Returns
    -------
    str or None
        Why the key was rejected, or ``None`` when it works or the API could not be reached.
    """
    import anthropic

    try:
        anthropic.Anthropic(api_key=key, timeout=timeout_s, max_retries=0).models.list(limit=1)
    except anthropic.AuthenticationError:
        return "The API rejected this key."
    except anthropic.PermissionDeniedError:
        return "This key is not allowed to use the API."
    except Exception:
        return None  # offline or a transient error: keep the key, a run will report problems
    return None


def save_api_key(key: str, check: Callable[[str], str | None] | None = check_api_key) -> dict[str, Any]:
    """Check, save (owner-only) and activate an Anthropic API key (OR-11, DL-7).

    Parameters
    ----------
    key : str
        The key.
    check : callable or None
        Returns a rejection reason or ``None``; ``None`` skips the check.

    Returns
    -------
    dict
        :func:`api_key_status` after saving.

    Raises
    ------
    ValueError
        If the key is malformed or rejected.
    """
    key = key.strip()
    if not key or any(c.isspace() for c in key) or not key.startswith("sk-"):
        raise ValueError("That does not look like an Anthropic API key (they start with sk-ant-).")
    if check is not None and (reason := check(key)):
        raise ValueError(reason)
    path = api_key_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(key)
    os.chmod(path, 0o600)
    os.environ[API_KEY_ENV] = key
    return api_key_status()


def forget_api_key() -> dict[str, Any]:
    """Delete the saved key and drop it from the environment if it came from the file.

    Returns
    -------
    dict
        :func:`api_key_status` afterwards.
    """
    key = _stored_key()
    if key and os.environ.get(API_KEY_ENV) == key:
        os.environ.pop(API_KEY_ENV, None)
    api_key_path().unlink(missing_ok=True)
    return api_key_status()


def api_key_status() -> dict[str, Any]:
    """Whether an API key is available, and where from; never the key itself.

    Returns
    -------
    dict
        ``{present, source, hint}``: ``source`` is ``saved``, ``environment`` or ``None``, and
        ``hint`` shows only the key's last four characters.
    """
    key = os.environ.get(API_KEY_ENV)
    stored = _stored_key()
    if not key and os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return {"present": True, "source": "environment", "hint": "auth token"}
    if not key:
        return {"present": False, "source": None, "hint": None}
    return {"present": True, "source": "saved" if key == stored else "environment", "hint": f"sk-…{key[-4:]}"}
