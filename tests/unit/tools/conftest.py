"""Test doubles for the literature tools: a fake Valency backend and an empty workspace."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from factories import workspace

from hephaestus.core.types import Paper
from hephaestus.tools import literature
from hephaestus.tools.registry import Workspace

VALENCY_FIXTURES = Path(__file__).parents[2] / "fixtures" / "valency"


class FakeValency:
    """Replays recorded Valency responses by tool name and counts the calls.

    Parameters
    ----------
    failures : int
        Number of initial calls that raise, to exercise the retry (LM-10).
    """

    def __init__(self, failures: int = 0) -> None:
        """Load every fixture in ``tests/fixtures/valency``.

        Parameters
        ----------
        failures : int
            Number of initial calls that raise.
        """
        self.responses = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in VALENCY_FIXTURES.glob("*.json")
        }
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.failures = failures

    def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """Return the recorded response for ``tool``.

        Parameters
        ----------
        tool : str
            Valency tool name.
        args : dict
            Tool arguments, recorded.

        Returns
        -------
        dict
            Recorded response.

        Raises
        ------
        ConnectionError
            While ``failures`` remain.
        """
        self.calls.append((tool, args))
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionError("simulated Valency outage")
        return self.responses[tool]


class MemoryCache:
    """In-memory stand-in for ``core.store.cached`` (DL-3)."""

    def __init__(self) -> None:
        """Start empty."""
        self.data: dict[str, Any] = {}

    def __call__(self, namespace: str, key_inputs: dict[str, Any], compute_fn: Any) -> Any:
        """Return a cached result, computing it on a miss.

        Parameters
        ----------
        namespace : str
            Cache namespace.
        key_inputs : dict
            Exact inputs.
        compute_fn : callable
            Produces the result on a miss.

        Returns
        -------
        Any
            Cached or computed result.
        """
        key = namespace + json.dumps(key_inputs, sort_keys=True)
        if key not in self.data:
            self.data[key] = compute_fn()
        return self.data[key]


@pytest.fixture
def valency() -> Iterator[FakeValency]:
    """Wire the literature module to a fresh fake backend.

    Yields
    ------
    FakeValency
        The backend in use.
    """
    fake = FakeValency()
    literature.configure(backend=fake)
    yield fake
    literature.reset()


@pytest.fixture
def ws() -> Workspace:
    """Return an empty run workspace.

    Returns
    -------
    Workspace
        The real ``Workspace``, so snapshots and resume see what the handlers store.
    """
    return workspace()


@pytest.fixture
def make_paper() -> Any:
    """Return a factory for papers with sensible defaults.

    Returns
    -------
    callable
        ``make_paper(**overrides) -> Paper``.
    """

    def factory(**overrides: Any) -> Paper:
        """Build a paper.

        Parameters
        ----------
        **overrides : Any
            Fields to set.

        Returns
        -------
        Paper
            Paper record.
        """
        fields: dict[str, Any] = {
            "paper_id": "p1",
            "title": "A paper",
            "authors": ["Ann Author"],
            "source": "arxiv",
            "abstract": "",
            "relevance": 0.5,
            "found_via": "semantic",
        }
        fields.update(overrides)
        return Paper(**fields)

    return factory
