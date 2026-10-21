"""Locks for running tool calls in parallel safely: a read-write lock and per-key locks."""

from __future__ import annotations

import threading
from collections.abc import Hashable, Iterator
from contextlib import contextmanager


class RWLock:
    """Many readers or one writer; a waiting writer blocks new readers, so it cannot starve."""

    def __init__(self) -> None:
        """Create an unlocked lock."""
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        """Hold the lock in shared mode.

        Yields
        ------
        None
            While held.
        """
        with self._cond:
            while self._writer or self._waiting_writers:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        """Hold the lock exclusively.

        Yields
        ------
        None
            While held.
        """
        with self._cond:
            self._waiting_writers += 1
            while self._writer or self._readers:
                self._cond.wait()
            self._waiting_writers -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


class KeyedLocks:
    """One lock per key, created on demand."""

    def __init__(self) -> None:
        """Create an empty set of locks."""
        self._guard = threading.Lock()
        self._locks: dict[Hashable, threading.Lock] = {}

    def get(self, key: Hashable) -> threading.Lock:
        """Return the lock for ``key``.

        Parameters
        ----------
        key : hashable
            Key, e.g. a candidate ID or a cache key.

        Returns
        -------
        threading.Lock
            The key's lock.
        """
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    @contextmanager
    def hold(self, key: Hashable | None) -> Iterator[None]:
        """Hold the lock for ``key``; ``None`` holds nothing.

        Parameters
        ----------
        key : hashable or None
            Key.

        Yields
        ------
        None
            While held.
        """
        if key is None:
            yield
            return
        with self.get(key):
            yield
