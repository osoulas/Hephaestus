"""Valency Bond OAuth: token store, browser sign-in and the authenticated MCP client (LM-2, DL-7, UI-11).

Lives in ``core`` so ``tools/literature.py`` can open a session without importing ``llm``.
Tokens are kept in ``~/.hephaestus/valency_tokens.json``, owner-only. Token values and
authorisation codes are never logged (SYS-10).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from contextlib import AsyncExitStack
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import anyio
import httpx2
from mcp import Client
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import (
    AuthorizationCodeResult,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)

VALENCY_URL = "https://mcp.valency.io/"
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = int(os.environ.get("HEPHAESTUS_VALENCY_CALLBACK_PORT", "33418"))
SIGNIN_TIMEOUT_S = 300.0
SIGNIN_HINT = "Valency sign-in required: run `python -m hephaestus signin`"


class SignInRequired(RuntimeError):
    """No usable Valency token, and the caller did not allow a browser sign-in (LM-2)."""


def default_token_path() -> Path:
    """Return the Valency token file path.

    Returns
    -------
    pathlib.Path
        ``$HEPHAESTUS_HOME/valency_tokens.json``, default ``~/.hephaestus/valency_tokens.json``.
    """
    home = os.environ.get("HEPHAESTUS_HOME")
    return (Path(home) if home else Path.home() / ".hephaestus") / "valency_tokens.json"


def _replace_with_retry(src: Path, dst: Path, attempts: int = 20) -> None:
    """Atomically replace ``dst`` with ``src``, retrying while Windows holds the file (DL-12).

    Parameters
    ----------
    src : pathlib.Path
        Temporary file to move into place.
    dst : pathlib.Path
        Destination file.
    attempts : int
        Number of tries, 50 ms apart.

    Raises
    ------
    PermissionError
        If the file is still locked after the last try.
    """
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.05)


class TokenStore:
    """Owner-only JSON store for Valency tokens and the registered client (DL-7).

    Implements the MCP SDK ``TokenStorage`` protocol, and also records the absolute token expiry
    so it can be restored on load (LM-2) and a ``needs_signin`` flag set when a refresh is
    rejected (UI-11).

    Parameters
    ----------
    path : pathlib.Path or None
        Token file; defaults to :func:`default_token_path`.
    """

    def __init__(self, path: Path | None = None) -> None:
        """Create a store backed by ``path``.

        Parameters
        ----------
        path : pathlib.Path or None
            Token file; defaults to :func:`default_token_path`.
        """
        self.path = path or default_token_path()

    def _read(self) -> dict[str, Any]:
        """Return the stored JSON, or an empty dict if the file is missing or unreadable.

        Returns
        -------
        dict
            Keys ``tokens``, ``expires_at``, ``client_info`` and ``needs_signin`` when present.
        """
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict[str, Any]) -> None:
        """Write ``data`` atomically to an owner-only file (DL-7, DL-12).

        Parameters
        ----------
        data : dict
            JSON-serialisable store contents.
        """
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.chmod(tmp, 0o600)
        _replace_with_retry(tmp, self.path)

    async def get_tokens(self) -> OAuthToken | None:
        """Return the stored tokens, or ``None`` if absent or marked rejected.

        Returns
        -------
        OAuthToken or None
            Tokens to present to Valency.
        """
        data = self._read()
        if data.get("needs_signin") or not data.get("tokens"):
            return None
        return OAuthToken.model_validate(data["tokens"])

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Store new tokens with their absolute expiry, and clear any rejection.

        Parameters
        ----------
        tokens : OAuthToken
            Tokens from a sign-in or refresh.
        """
        data = self._read()
        data["tokens"] = tokens.model_dump(mode="json")
        data["expires_at"] = time.time() + tokens.expires_in if tokens.expires_in else None
        data["needs_signin"] = False
        self._write(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """Return the dynamically registered client, so registration happens once.

        Returns
        -------
        OAuthClientInformationFull or None
            Stored registration.
        """
        info = self._read().get("client_info")
        return OAuthClientInformationFull.model_validate(info) if info else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Store the client registration.

        Parameters
        ----------
        client_info : OAuthClientInformationFull
            Registration returned by Valency.
        """
        data = self._read()
        data["client_info"] = client_info.model_dump(mode="json", exclude_none=True)
        self._write(data)

    def expires_at(self) -> float | None:
        """Return the stored access-token expiry.

        Returns
        -------
        float or None
            Unix time in seconds, or ``None`` if unknown.
        """
        value = self._read().get("expires_at")
        return float(value) if value is not None else None

    def mark_rejected(self) -> None:
        """Flag the stored token as rejected, so the next connect asks for a new sign-in (UI-11)."""
        data = self._read()
        data["needs_signin"] = True
        self._write(data)

    def status(self) -> dict[str, bool]:
        """Report sign-in state without exposing any token.

        Returns
        -------
        dict
            ``signed_in``, ``needs_signin`` and ``access_token_expired`` flags.
        """
        data = self._read()
        expires_at = data.get("expires_at")
        needs_signin = bool(data.get("needs_signin"))
        return {
            "signed_in": bool(data.get("tokens")) and not needs_signin,
            "needs_signin": needs_signin,
            "access_token_expired": expires_at is not None and time.time() > float(expires_at),
        }


class ValencyOAuth(OAuthClientProvider):
    """MCP SDK OAuth provider with the two Valency fixes from ``CLAUDE.md``.

    - On load, restore the stored token expiry; the SDK does not, so it never refreshes an
      expired token and falls back to a browser sign-in (LM-2).
    - When a refresh is rejected, mark the token so the UI asks for a new sign-in (UI-11).

    Parameters
    ----------
    store : TokenStore
        Token storage.
    **kwargs
        Passed to ``OAuthClientProvider``.
    """

    def __init__(self, store: TokenStore, **kwargs: Any) -> None:
        """Create the provider.

        Parameters
        ----------
        store : TokenStore
            Token storage, also passed to the SDK as ``storage``.
        **kwargs
            Passed to ``OAuthClientProvider``.
        """
        super().__init__(storage=store, **kwargs)
        self._store = store

    async def _initialize(self) -> None:
        """Load stored tokens and client info, then restore the token expiry (LM-2)."""
        await super()._initialize()
        self.context.token_expiry_time = self._store.expires_at()

    async def _handle_refresh_response(self, response: httpx2.Response) -> bool:
        """Handle a refresh response, marking the token rejected on failure (UI-11).

        Parameters
        ----------
        response : httpx2.Response
            Token endpoint response to the refresh request.

        Returns
        -------
        bool
            Whether the refresh succeeded.
        """
        ok = await super()._handle_refresh_response(response)
        if not ok:
            self._store.mark_rejected()
        return ok


class LoopbackCallback:
    """One-shot HTTP server on 127.0.0.1 that receives the OAuth redirect.

    Requests whose ``state`` does not match the expected value are answered with 400 and
    ignored, so a stray or forged request cannot complete the sign-in.

    Parameters
    ----------
    host : str
        Interface to bind; loopback only.
    port : int
        Port registered in the redirect URI; 0 picks a free port (tests).
    """

    def __init__(self, host: str = CALLBACK_HOST, port: int = CALLBACK_PORT) -> None:
        """Create the server without starting it.

        Parameters
        ----------
        host : str
            Interface to bind.
        port : int
            Port to listen on.
        """
        self.host, self.port = host, port
        self.expected_state: str | None = None
        self._result: AuthorizationCodeResult | None = None
        self._error: str | None = None
        self._done = threading.Event()
        self._server: HTTPServer | None = None

    @property
    def redirect_uri(self) -> str:
        """Return the redirect URI registered with Valency.

        Returns
        -------
        str
            ``http://127.0.0.1:<port>/callback``.
        """
        return f"http://{self.host}:{self.port}/callback"

    def start(self, expected_state: str | None) -> None:
        """Start listening in a background thread.

        Parameters
        ----------
        expected_state : str or None
            The ``state`` from the authorisation URL; ``None`` accepts any state.
        """
        self.expected_state = expected_state
        callback = self

        class Handler(BaseHTTPRequestHandler):
            """Request handler that records the authorisation code."""

            def do_GET(self) -> None:
                """Accept the redirect, or reject a mismatched or unrelated request."""
                params = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
                if "code" not in params and "error" not in params:
                    self._reply(404, "Not found.")
                    return
                state = params.get("state")
                if callback.expected_state is not None and state != callback.expected_state:
                    self._reply(400, "Sign-in state did not match; ignored.")
                    return
                if "error" in params:
                    callback._error = params.get("error_description") or params["error"]
                    self._reply(400, "Valency sign-in failed. You can close this tab.")
                else:
                    callback._result = AuthorizationCodeResult(
                        code=params["code"], state=state, iss=params.get("iss")
                    )
                    self._reply(200, "Signed in to Valency. You can close this tab.")
                callback._done.set()

            def _reply(self, status: int, text: str) -> None:
                """Send a short plain-text response.

                Parameters
                ----------
                status : int
                    HTTP status code.
                text : str
                    Body text.
                """
                body = text.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                """Silence request logging; the query string carries the authorisation code (SYS-10).

                Parameters
                ----------
                format : str
                    Ignored.
                *args : Any
                    Ignored.
                """

        self._server = HTTPServer((self.host, self.port), Handler)
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def wait(self, timeout_s: float = SIGNIN_TIMEOUT_S) -> AuthorizationCodeResult:
        """Block until the redirect arrives, then stop the server.

        Parameters
        ----------
        timeout_s : float
            Seconds to wait for the user to finish signing in.

        Returns
        -------
        AuthorizationCodeResult
            Code, state and issuer from the redirect.

        Raises
        ------
        RuntimeError
            If the sign-in timed out or Valency returned an error.
        """
        try:
            if not self._done.wait(timeout_s):
                raise RuntimeError(f"Valency sign-in timed out after {timeout_s:.0f} s")
            if self._error or self._result is None:
                raise RuntimeError(f"Valency sign-in failed: {self._error}")
            return self._result
        finally:
            if self._server is not None:
                self._server.shutdown()
                self._server.server_close()


def build_oauth_provider(
    store: TokenStore,
    *,
    interactive: bool,
    on_url: Callable[[str], None] | None = None,
    callback: LoopbackCallback | None = None,
) -> ValencyOAuth:
    """Build the OAuth provider for Valency (LM-2).

    Parameters
    ----------
    store : TokenStore
        Token storage.
    interactive : bool
        If true, a missing or rejected token opens the browser sign-in. If false, it raises
        :class:`SignInRequired` instead, so a normal run never hangs on a sign-in nobody sees.
    on_url : callable or None
        Called with the authorisation URL before the browser opens (for the UI).
    callback : LoopbackCallback or None
        Redirect receiver; a default one on 127.0.0.1 is made if omitted.

    Returns
    -------
    ValencyOAuth
        Provider to pass as ``auth`` to the HTTP client.
    """
    receiver = callback or LoopbackCallback()

    async def redirect_handler(url: str) -> None:
        """Open the sign-in page, or refuse when not interactive.

        Parameters
        ----------
        url : str
            Authorisation URL built by the SDK.

        Raises
        ------
        SignInRequired
            When ``interactive`` is false.
        """
        if not interactive:
            raise SignInRequired(SIGNIN_HINT)
        receiver.start(parse_qs(urlparse(url).query).get("state", [None])[0])
        if on_url:
            on_url(url)
        print(f"Open this URL to sign in to Valency:\n{url}", file=sys.stderr)
        webbrowser.open(url)

    async def callback_handler() -> AuthorizationCodeResult:
        """Wait for the redirect without blocking the event loop.

        Returns
        -------
        AuthorizationCodeResult
            Code and state from the redirect.
        """
        return await anyio.to_thread.run_sync(receiver.wait)

    metadata = OAuthClientMetadata(
        client_name="Hephaestus",
        redirect_uris=[receiver.redirect_uri],  # type: ignore[list-item]
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
    )
    return ValencyOAuth(
        store,
        server_url=VALENCY_URL,
        client_metadata=metadata,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )


async def open_client(
    stack: AsyncExitStack,
    *,
    interactive: bool = False,
    store: TokenStore | None = None,
    on_url: Callable[[str], None] | None = None,
) -> Client:
    """Open an authenticated MCP session to Valency (``valency_connect``, LM-2).

    Parameters
    ----------
    stack : contextlib.AsyncExitStack
        Owns the HTTP client and session; closing it closes both.
    interactive : bool
        Allow a browser sign-in if no usable token is stored.
    store : TokenStore or None
        Token storage; the default file if omitted.
    on_url : callable or None
        Called with the authorisation URL during an interactive sign-in.

    Returns
    -------
    mcp.Client
        Connected session.

    Raises
    ------
    SignInRequired
        If no usable token is stored and ``interactive`` is false (possibly inside an
        exception group from the SDK's task group).
    """
    provider = build_oauth_provider(store or TokenStore(), interactive=interactive, on_url=on_url)
    http = await stack.enter_async_context(
        httpx2.AsyncClient(auth=provider, timeout=httpx2.Timeout(30.0, read=300.0))
    )
    return await stack.enter_async_context(Client(streamable_http_client(VALENCY_URL, http_client=http)))


def find_exception(exc: BaseException, kind: type[BaseException]) -> BaseException | None:
    """Find an exception of ``kind`` inside ``exc``, looking through groups and causes.

    Parameters
    ----------
    exc : BaseException
        Raised exception, possibly an ``ExceptionGroup``.
    kind : type
        Exception type to find.

    Returns
    -------
    BaseException or None
        The first match, or ``None``.
    """
    if isinstance(exc, kind):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        for inner in exc.exceptions:
            found = find_exception(inner, kind)
            if found is not None:
                return found
    for linked in (exc.__cause__, exc.__context__):
        if linked is not None and linked is not exc:
            found = find_exception(linked, kind)
            if found is not None:
                return found
    return None


def signin(store: TokenStore | None = None, on_url: Callable[[str], None] | None = None) -> dict[str, bool]:
    """Sign in to Valency in the browser if needed, and confirm the session works (LM-2).

    Blocks until done. With a usable stored token it opens no browser. Used by
    ``python -m hephaestus signin``.

    Parameters
    ----------
    store : TokenStore or None
        Token storage; the default file if omitted.
    on_url : callable or None
        Called with the authorisation URL before the browser opens.

    Returns
    -------
    dict
        :meth:`TokenStore.status` after signing in.
    """
    store = store or TokenStore()

    async def go() -> None:
        """Connect interactively and list tools once."""
        async with AsyncExitStack() as stack:
            client = await open_client(stack, interactive=True, store=store, on_url=on_url)
            await client.list_tools()

    anyio.run(go)
    return store.status()


_signin_lock = threading.Lock()
_signin_state: dict[str, Any] = {"pending": False, "url": None, "error": None}


def start_signin(store: TokenStore | None = None) -> dict[str, Any]:
    """Start a browser sign-in in the background and return at once (``/auth/valency?start=true``).

    Parameters
    ----------
    store : TokenStore or None
        Token storage; the default file if omitted.

    Returns
    -------
    dict
        :func:`status` just after starting.
    """
    with _signin_lock:
        if _signin_state["pending"]:
            return status(store)
        _signin_state.update(pending=True, url=None, error=None)

    def on_url(url: str) -> None:
        """Publish the authorisation URL for the UI.

        Parameters
        ----------
        url : str
            Authorisation URL.
        """
        with _signin_lock:
            _signin_state["url"] = url

    def worker() -> None:
        """Run the sign-in and record its outcome."""
        error = None
        try:
            signin(store, on_url=on_url)
        except BaseException as exc:  # reported through status(), never raised
            error = str(find_exception(exc, RuntimeError) or exc) or type(exc).__name__
        with _signin_lock:
            _signin_state.update(pending=False, error=error)

    threading.Thread(target=worker, daemon=True, name="valency-signin").start()
    return status(store)


def status(store: TokenStore | None = None) -> dict[str, Any]:
    """Report Valency sign-in status for ``GET /auth/valency`` and ``/health`` (UI-11).

    Parameters
    ----------
    store : TokenStore or None
        Token storage; the default file if omitted.

    Returns
    -------
    dict
        ``signed_in``, ``needs_signin``, ``access_token_expired``, ``pending``, ``url`` and ``error``.
    """
    result: dict[str, Any] = dict((store or TokenStore()).status())
    with _signin_lock:
        result.update(_signin_state)
    return result
