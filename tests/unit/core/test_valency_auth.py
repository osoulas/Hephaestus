"""Valency OAuth: token store, expiry restore, rejected refresh and the loopback sign-in (no network)."""

import os
import stat
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import anyio
import httpx2
import pytest
from mcp.shared.auth import OAuthToken

from hephaestus.core import valency_auth
from hephaestus.core.valency_auth import (
    LoopbackCallback,
    SignInRequired,
    TokenStore,
    build_oauth_provider,
    start_signin,
    status,
)

TOKEN = OAuthToken(access_token="secret-access", refresh_token="secret-refresh", expires_in=3600)


@pytest.fixture
def store(tmp_path: Path) -> TokenStore:
    """Return a token store in a temporary home.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Per-test directory.

    Returns
    -------
    TokenStore
        Empty store.
    """
    return TokenStore(tmp_path / "home" / "valency_tokens.json")


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_dl7_token_file_owner_only(store: TokenStore) -> None:
    """The token file and its folder are readable by the owner only."""
    anyio.run(store.set_tokens, TOKEN)
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700


def test_tokens_round_trip_with_absolute_expiry(store: TokenStore) -> None:
    """Stored tokens reload, and the absolute expiry is recorded."""
    before = time.time()
    anyio.run(store.set_tokens, TOKEN)
    assert anyio.run(store.get_tokens) == TOKEN
    assert before + 3600 <= (store.expires_at() or 0) <= time.time() + 3600
    assert status(store)["signed_in"] is True


def test_lm2_expiry_restored_on_load(store: TokenStore) -> None:
    """An expired stored token is seen as expired but refreshable, so no browser opens (LM-2)."""
    anyio.run(store.set_tokens, TOKEN)
    data = store._read()
    data["expires_at"] = time.time() - 10
    data["client_info"] = {"client_id": "hephaestus", "redirect_uris": ["http://127.0.0.1:1/callback"]}
    store._write(data)
    provider = build_oauth_provider(store, interactive=False)
    anyio.run(provider._initialize)
    assert provider.context.token_expiry_time is not None
    assert not provider.context.is_token_valid()
    assert provider.context.can_refresh_token()
    assert status(store)["access_token_expired"] is True


def test_rejected_refresh_marks_needs_signin(store: TokenStore) -> None:
    """A refused refresh flags the token, so the UI asks for a new sign-in (UI-11)."""
    anyio.run(store.set_tokens, TOKEN)
    provider = build_oauth_provider(store, interactive=False)
    anyio.run(provider._initialize)
    ok = anyio.run(provider._handle_refresh_response, httpx2.Response(400, json={"error": "invalid_grant"}))
    assert ok is False
    assert status(store)["needs_signin"] is True
    assert status(store)["signed_in"] is False
    assert anyio.run(store.get_tokens) is None


def test_new_tokens_clear_rejection(store: TokenStore) -> None:
    """Signing in again clears the rejection flag."""
    store.mark_rejected()
    anyio.run(store.set_tokens, TOKEN)
    assert status(store)["needs_signin"] is False


def test_lm2_noninteractive_connect_never_opens_browser(
    store: TokenStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without permission to sign in, the redirect fails with a hint instead of opening a browser."""
    opened: list[str] = []
    monkeypatch.setattr(valency_auth.webbrowser, "open", opened.append)
    provider = build_oauth_provider(store, interactive=False)
    assert provider.context.redirect_handler is not None
    with pytest.raises(SignInRequired, match="python -m hephaestus signin"):
        anyio.run(provider.context.redirect_handler, "https://auth.example/authorize?state=s")
    assert opened == []


def _get(url: str) -> tuple[int, str]:
    """GET a local URL.

    Parameters
    ----------
    url : str
        URL on 127.0.0.1.

    Returns
    -------
    tuple of (int, str)
        Status code and body.
    """
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8")


def test_callback_rejects_state_mismatch_then_returns_code() -> None:
    """A redirect with the wrong state is refused and ignored; the right one completes the sign-in."""
    callback = LoopbackCallback(port=0)
    callback.start("good-state")
    base = callback.redirect_uri
    assert _get(f"{base}?code=evil&state=bad-state")[0] == 400
    assert _get(base.replace("/callback", "/favicon.ico"))[0] == 404
    assert _get(f"{base}?code=abc&state=good-state")[0] == 200
    result = callback.wait(timeout_s=5)
    assert (result.code, result.state) == ("abc", "good-state")


def test_callback_reports_provider_error() -> None:
    """An error redirect from Valency fails the sign-in with its description."""
    callback = LoopbackCallback(port=0)
    callback.start("s")
    _get(f"{callback.redirect_uri}?error=access_denied&error_description=user+cancelled&state=s")
    with pytest.raises(RuntimeError, match="user cancelled"):
        callback.wait(timeout_s=5)


def test_callback_times_out() -> None:
    """A sign-in nobody finishes times out with a clear message."""
    callback = LoopbackCallback(port=0)
    callback.start("s")
    with pytest.raises(RuntimeError, match="timed out"):
        callback.wait(timeout_s=0.05)


def test_interactive_redirect_opens_browser_and_publishes_url(
    store: TokenStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interactive sign-in starts the callback, reports the URL and opens the browser."""
    opened: list[str] = []
    seen: list[str] = []
    monkeypatch.setattr(valency_auth.webbrowser, "open", opened.append)
    callback = LoopbackCallback(port=0)
    provider = build_oauth_provider(store, interactive=True, on_url=seen.append, callback=callback)
    url = "https://auth.example/authorize?state=xyz"
    assert provider.context.redirect_handler is not None
    anyio.run(provider.context.redirect_handler, url)
    assert opened == seen == [url]
    assert callback.expected_state == "xyz"
    _get(f"{callback.redirect_uri}?code=c&state=xyz")
    assert callback.wait(timeout_s=5).code == "c"


def test_start_signin_reports_pending_and_url(store: TokenStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """The background sign-in reports pending and the URL, then finishes (``/auth/valency``)."""
    release = threading.Event()

    def fake_signin(store_arg: Any, on_url: Any) -> dict[str, bool]:
        """Publish a URL, then wait to be released.

        Parameters
        ----------
        store_arg : Any
            Ignored.
        on_url : callable
            URL callback.

        Returns
        -------
        dict
            Empty status.
        """
        on_url("https://auth.example/authorize")
        release.wait(5)
        return {}

    monkeypatch.setattr(valency_auth, "signin", fake_signin)
    start_signin(store)
    deadline = time.time() + 5
    while status(store)["url"] is None and time.time() < deadline:
        time.sleep(0.01)
    current = status(store)
    assert current["pending"] is True
    assert current["url"] == "https://auth.example/authorize"
    release.set()
    while status(store)["pending"] and time.time() < deadline:
        time.sleep(0.01)
    assert status(store)["pending"] is False
    assert status(store)["error"] is None


def test_sys10_tokens_not_in_status(store: TokenStore) -> None:
    """Status reports flags only, never token values."""
    anyio.run(store.set_tokens, TOKEN)
    assert "secret" not in repr(status(store))
