"""``ProgressEvent`` bus, event history on disk, and replay (OR-7, DL-10, UI-9).

One process-wide ``EventBus`` carries every run's events. Each event is appended to
``runs/<run_id>/events.jsonl`` and fanned out to subscribers (the HTTP SSE stream, the CLI
renderer). When a process first publishes to a run, the run's history is loaded from disk so
a resumed run keeps its earlier events.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from hephaestus.core.config import runs_root
from hephaestus.core.store import append_jsonl, read_jsonl
from hephaestus.core.types import EventStatus, ProgressEvent

Listener = Callable[[ProgressEvent], None]
_CLOSED = object()


class Subscription:
    """A queue of one run's events; iterate it, or call ``get``."""

    def __init__(self, bus: EventBus, run_id: str) -> None:
        """Create an empty subscription; ``EventBus.subscribe`` fills it.

        Parameters
        ----------
        bus : EventBus
            Bus to detach from on ``close``.
        run_id : str
            Run whose events arrive here.
        """
        self.bus = bus
        self.run_id = run_id
        self.queue: queue.Queue[Any] = queue.Queue()

    def get(self, timeout: float | None = None) -> ProgressEvent | None:
        """Return the next event, or ``None`` on timeout or close.

        Parameters
        ----------
        timeout : float, optional
            Seconds to wait; ``None`` waits forever.

        Returns
        -------
        ProgressEvent or None
            The next event.
        """
        try:
            item = self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
        return None if item is _CLOSED else item

    def __iter__(self) -> Iterator[ProgressEvent]:
        """Yield events until the subscription is closed.

        Yields
        ------
        ProgressEvent
            Events in publication order.
        """
        while True:
            item = self.queue.get()
            if item is _CLOSED:
                return
            yield item

    def close(self) -> None:
        """Detach from the bus and end iteration."""
        self.bus._detach(self)
        self.queue.put(_CLOSED)


class EventBus:
    """Thread-safe fan-out of ``ProgressEvent``s with history on disk."""

    def __init__(self, root: Path | None = None) -> None:
        """Create a bus writing under ``root``.

        Parameters
        ----------
        root : pathlib.Path, optional
            Runs folder; defaults to ``runs_root()`` at publish time.
        """
        self._root = root
        self._lock = threading.RLock()
        self._history: dict[str, list[ProgressEvent]] = {}
        self._subs: dict[str, list[Subscription]] = {}
        self._listeners: list[Listener] = []

    @property
    def root(self) -> Path:
        """Runs folder in use.

        Returns
        -------
        pathlib.Path
            The folder holding ``<run_id>/events.jsonl``.
        """
        return self._root or runs_root()

    def _path(self, run_id: str) -> Path:
        """Return the events file of a run.

        Parameters
        ----------
        run_id : str
            Run ID.

        Returns
        -------
        pathlib.Path
            ``<root>/<run_id>/events.jsonl``.
        """
        return self.root / run_id / "events.jsonl"

    def history(self, run_id: str) -> list[ProgressEvent]:
        """Return every event of a run, loading it from disk the first time (DL-10).

        Parameters
        ----------
        run_id : str
            Run ID.

        Returns
        -------
        list of ProgressEvent
            A copy of the run's events in order.
        """
        with self._lock:
            if run_id not in self._history:
                self._history[run_id] = [
                    ProgressEvent.model_validate(r) for r in read_jsonl(self._path(run_id))
                ]
            return list(self._history[run_id])

    def emit(
        self,
        run_id: str,
        section: str,
        status: EventStatus,
        payload: dict[str, Any] | None = None,
        iteration: int | None = None,
        persist: bool = True,
    ) -> ProgressEvent:
        """Build, store and publish one event.

        Parameters
        ----------
        run_id : str
            Run ID.
        section : str
            ``tool``, ``agent``, ``orchestrator``, ``goal_parser``, ``final_calculation``,
            ``brief_generator`` or ``run``.
        status : {"started", "finished", "failed"}
            Event status.
        payload : dict, optional
            Small JSON summary; see *Event payloads* in ``docs/TOOLS.md``.
        iteration : int, optional
            Review round in which the step started.
        persist : bool
            Append to ``events.jsonl``; replays set this to False.

        Returns
        -------
        ProgressEvent
            The published event, with its sequence number.
        """
        with self._lock:
            seq = len(self.history(run_id))
            event = ProgressEvent(
                run_id=run_id,
                seq=seq,
                section=section,
                status=status,
                iteration=iteration,
                payload=payload or {},
            )
            self._publish(event, persist)
            return event

    def _publish(self, event: ProgressEvent, persist: bool) -> None:
        """Record an event and deliver it to subscribers and listeners.

        Parameters
        ----------
        event : ProgressEvent
            Event to deliver.
        persist : bool
            Append it to the run's ``events.jsonl``.
        """
        with self._lock:
            self.history(event.run_id)
            self._history[event.run_id].append(event)
            if persist:
                append_jsonl(self._path(event.run_id), event)
            subs = list(self._subs.get(event.run_id, []))
            listeners = list(self._listeners)
        for sub in subs:
            sub.queue.put(event)
        for listener in listeners:
            listener(event)

    def subscribe(self, run_id: str, include_history: bool = True) -> Subscription:
        """Subscribe to a run's events, optionally starting with its history.

        Parameters
        ----------
        run_id : str
            Run ID.
        include_history : bool
            Queue every earlier event first.

        Returns
        -------
        Subscription
            Close it when done.
        """
        sub = Subscription(self, run_id)
        with self._lock:
            if include_history:
                for event in self.history(run_id):
                    sub.queue.put(event)
            self._subs.setdefault(run_id, []).append(sub)
        return sub

    def _detach(self, sub: Subscription) -> None:
        """Remove a subscription.

        Parameters
        ----------
        sub : Subscription
            Subscription to remove.
        """
        with self._lock:
            subs = self._subs.get(sub.run_id, [])
            if sub in subs:
                subs.remove(sub)

    def close_run(self, run_id: str) -> None:
        """End every subscription to a run (the run finished).

        Parameters
        ----------
        run_id : str
            Run ID.
        """
        with self._lock:
            subs = list(self._subs.pop(run_id, []))
        for sub in subs:
            sub.queue.put(_CLOSED)

    def add_listener(self, listener: Listener) -> Callable[[], None]:
        """Call ``listener`` for every event of every run.

        Parameters
        ----------
        listener : callable
            Receives each ``ProgressEvent``; must not block.

        Returns
        -------
        callable
            Call it to remove the listener.
        """
        with self._lock:
            self._listeners.append(listener)

        def remove() -> None:
            """Remove the listener."""
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return remove

    def replay(
        self,
        run_id: str,
        replay_id: str,
        speed: float | Callable[[], float] = 1.0,
        max_gap_s: float = 2.0,
        stop: threading.Event | None = None,
    ) -> list[ProgressEvent]:
        """Re-emit a stored run's events under ``replay_id`` with their original pacing (UI-9).

        Parameters
        ----------
        run_id : str
            Stored run to replay.
        replay_id : str
            Run ID the replayed events are published under (not written to disk).
        speed : float or callable
            Playback speed; 2 plays twice as fast, zero or less plays with no pauses. A callable is
            read during each pause, so the speed can change while the replay runs.
        max_gap_s : float
            Longest pause between two events, in seconds, after scaling.
        stop : threading.Event, optional
            Set it to end the replay early.

        Returns
        -------
        list of ProgressEvent
            The replayed events.
        """
        speed_now = speed if callable(speed) else (lambda: speed)
        source = self.history(run_id)
        out = []
        for i, event in enumerate(source):
            if i:
                self._pause((event.time - source[i - 1].time).total_seconds(), speed_now, max_gap_s, stop)
            if stop is not None and stop.is_set():
                break
            copy = event.model_copy(update={"run_id": replay_id})
            self._publish(copy, persist=False)
            out.append(copy)
        self.close_run(replay_id)
        return out

    @staticmethod
    def _pause(
        gap_s: float, speed: Callable[[], float], max_gap_s: float, stop: threading.Event | None
    ) -> None:
        """Wait out one gap between replayed events, re-reading the speed as it goes.

        Parameters
        ----------
        gap_s : float
            Original gap, in s.
        speed : callable
            Current playback speed.
        max_gap_s : float
            Longest pause after scaling, in s.
        stop : threading.Event or None
            Ends the wait early when set.
        """
        waited = 0.0
        while True:
            s = speed()
            need = 0.0 if s <= 0 else min(max(gap_s, 0.0) / s, max_gap_s)
            if waited >= need or (stop is not None and stop.is_set()):
                return
            step = min(0.1, need - waited)
            time.sleep(step)
            waited += step


BUS = EventBus()
"""The process-wide bus used by the harness and interfaces."""
