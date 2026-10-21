"""Shared pytest configuration: blocks network access unless a test is marked ``live``."""

import socket
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the directory of JSON fixtures.

    Returns
    -------
    pathlib.Path
        ``tests/fixtures``.
    """
    return FIXTURES


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any outbound connection from a test not marked ``live``.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The requesting test, checked for the ``live`` marker.
    monkeypatch : pytest.MonkeyPatch
        Used to patch ``socket.socket.connect`` for the duration of the test.
    """
    if request.node.get_closest_marker("live"):
        return
    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: object) -> None:
        """Allow local and Unix-socket connections; refuse everything else.

        Parameters
        ----------
        self : socket.socket
            The socket being connected.
        address : object
            Address tuple or Unix socket path.

        Raises
        ------
        RuntimeError
            If the address is not local.
        """
        host = address[0] if isinstance(address, tuple) else None
        if host is not None and host not in _LOCAL_HOSTS:
            raise RuntimeError(
                f"Network access blocked in the default suite: {address!r}. Mark the test live."
            )
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
