"""Check that the default suite blocks the network."""

import socket

import pytest


def test_default_suite_blocks_network() -> None:
    """An outbound connection fails outside ``-m live``."""
    with pytest.raises(RuntimeError, match="Network access blocked"):
        socket.create_connection(("192.0.2.1", 80), timeout=0.1)
