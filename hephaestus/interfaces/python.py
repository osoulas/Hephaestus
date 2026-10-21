"""``Experiment``: the public Python API that the HTTP API and the CLI build on (OR-1, OR-6, OR-8, UI-9).

One ``Experiment`` is one run folder. It parses a goal, runs the campaign (blocking with
``run`` or in a background thread with ``start``), streams the run's ``ProgressEvent``s,
stops, resumes, loads and replays runs, and can drive the tools one call at a time through
the same budgets, events and gates the orchestrator uses. Nothing here re-implements a gate.

Examples
--------
>>> exp = Experiment("a lead-free oxide perovskite with a smaller band gap than SrTiO3")
>>> parsed = exp.parse()                # spec and clarifying questions
>>> exp.start()                         # background thread
>>> for event in exp.events():          # history, then live, until the run ends
...     print(event.section, event.status)
>>> exp.state.best, exp.brief
"""

from __future__ import annotations

import importlib.util
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hephaestus.core.config import MODELS, RunConfig, runs_root
from hephaestus.core.events import BUS, EventBus
from hephaestus.core.store import load_model, new_run_id
from hephaestus.core.types import ProgressEvent, RunState, SearchSpec
from hephaestus.harness import runner
from hephaestus.harness.agent import Campaign
from hephaestus.harness.goal import parse_goal, validate_spec
from hephaestus.harness.state import RunFiles, load_for_resume, save_snapshot
from hephaestus.llm.base import LLM, api_credentials_present, make_llm, resolve_backend
from hephaestus.tools.registry import Workspace, load_default_handlers

_BUSES: dict[Path, EventBus] = {}
_BUSES_LOCK = threading.Lock()
_POLL_S = 0.25


class ExperimentError(RuntimeError):
    """An ``Experiment`` request cannot be done (no such run, already running, finished, no spec)."""


@dataclass
class ParseResult:
    """The goal parser's answer (GP-5, UI-2)."""

    spec: SearchSpec | None
    questions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    parser: dict[str, str] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        """Whether the spec can start a run.

        Returns
        -------
        bool
            True when there is a spec, no open question and no validation error.
        """
        return self.spec is not None and not self.questions and not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON body of ``POST /parse``.

        Returns
        -------
        dict
            ``{spec, questions, errors, parser}``.
        """
        return {
            "spec": self.spec.model_dump(mode="json") if self.spec else None,
            "questions": self.questions,
            "errors": self.errors,
            "parser": self.parser,
        }


def bus_for(root: Path | None) -> EventBus:
    """Return the event bus for a runs folder: the process-wide ``BUS`` for the default one.

    Parameters
    ----------
    root : pathlib.Path or None
        Runs folder; ``None`` means ``runs_root()``.

    Returns
    -------
    EventBus
        One bus per folder, shared by every ``Experiment`` in this process.
    """
    if root is None:
        return BUS
    key = Path(root).resolve()
    with _BUSES_LOCK:
        return _BUSES.setdefault(key, EventBus(key))


class Experiment:
    """One run: parse, run or step, stream events, stop, resume, replay (Section 8)."""

    def __init__(
        self,
        goal: str = "",
        config: RunConfig | None = None,
        *,
        spec: SearchSpec | None = None,
        answers: dict[str, str] | None = None,
        run_id: str | None = None,
        root: Path | None = None,
        deps: runner.Deps | None = None,
        **options: Any,
    ) -> None:
        """Describe a run; nothing is written until ``parse``, ``run``, ``start`` or ``tool``.

        Parameters
        ----------
        goal : str
            The goal in plain English; may be empty when ``spec`` is given.
        config : RunConfig, optional
            Options; defaults to ``RunConfig()``.
        spec : SearchSpec, optional
            An already-parsed spec; skips the goal parser.
        answers : dict, optional
            Answers to the parser's clarifying questions.
        run_id : str, optional
            Run ID; a new one by default.
        root : pathlib.Path, optional
            Runs folder; defaults to ``runs_root()``.
        deps : runner.Deps, optional
            Injected models, handlers and final check (tests, offline demos). Its ``root`` and
            ``bus`` are set from ``root`` when left at their defaults.
        **options
            ``RunConfig`` fields overriding ``config``, e.g. ``llm_backend="api"``.
        """
        cfg = config or RunConfig()
        self.config = cfg.model_copy(update=options) if options else cfg
        RunConfig.model_validate(self.config.model_dump())  # reject bad overrides now
        self.goal = goal or (spec.raw_goal if spec else "")
        self.spec = spec
        self.answers = answers
        self.run_id = run_id or new_run_id()
        self.root = Path(root) if root is not None else (deps.root if deps and deps.root else None)
        self.deps = deps or runner.Deps()
        if self.deps.root is None:
            self.deps.root = self.root
        if self.deps.bus is BUS:
            self.deps.bus = bus_for(self.root)
        self._thread: threading.Thread | None = None
        self._result: RunState | None = None
        self._error: BaseException | None = None
        self._campaign: Campaign | None = None
        self._source: str | None = None  # stored run behind a replay
        self.replay_control: dict[str, Any] | None = None  # speed and stop of a replay handle

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, run_id: str, root: Path | None = None, deps: runner.Deps | None = None) -> Experiment:
        """Open a stored run (DL-1).

        Parameters
        ----------
        run_id : str
            Run ID.
        root : pathlib.Path, optional
            Runs folder.
        deps : runner.Deps, optional
            Dependencies for a later ``resume`` or ``tool`` call.

        Returns
        -------
        Experiment
            The run, with goal, config and spec from ``state.json``.

        Raises
        ------
        ExperimentError
            If the run does not exist.
        """
        path = (Path(root) if root else runs_root()) / run_id / "state.json"
        if not path.exists():
            raise ExperimentError(f"No run {run_id}.")
        state = load_model(path, RunState)
        return cls(state.raw_goal, state.config, spec=state.spec, run_id=run_id, root=root, deps=deps)

    @property
    def files(self) -> RunFiles:
        """Paths inside the run folder.

        Returns
        -------
        RunFiles
            Paths; the folder is created on first use.
        """
        return RunFiles.for_run(self._source or self.run_id, self.root)

    @property
    def exists(self) -> bool:
        """Whether the run has a ``state.json`` on disk.

        Returns
        -------
        bool
            True once the run has been created.
        """
        return ((self.root or runs_root()) / (self._source or self.run_id) / "state.json").exists()

    @property
    def state(self) -> RunState:
        """The run's state as last saved (rewritten after every tool call).

        Returns
        -------
        RunState
            ``state.json``; a fresh ``created`` state when nothing is saved yet.
        """
        if not self.exists:
            return RunState(run_id=self.run_id, raw_goal=self.goal, config=self.config, spec=self.spec)
        return load_model(self.files.state, RunState)

    @property
    def running(self) -> bool:
        """Whether the run (or replay) is executing in this process.

        Returns
        -------
        bool
            True while its thread is alive or the runner holds it.
        """
        return bool(self._thread and self._thread.is_alive()) or runner.is_active(self.run_id)

    @property
    def brief(self) -> str | None:
        """The brief's Markdown, once written (Section 10).

        Returns
        -------
        str or None
            ``brief.md``, or ``None`` before the brief exists.
        """
        path = self.files.brief_md
        return path.read_text(encoding="utf-8") if path.exists() else None

    @property
    def brief_pdf(self) -> Path | None:
        """The brief's PDF, rendered now if the run has a brief but no PDF yet (EB-10).

        Returns
        -------
        pathlib.Path or None
            ``brief.pdf``, or ``None`` before the brief exists.
        """
        path = self.files.brief_pdf
        md = self.files.brief_md
        if md.exists() and (not path.exists() or path.stat().st_mtime < md.stat().st_mtime):
            from hephaestus.harness.brief_pdf import export_pdf

            state = self.state
            export_pdf(md.read_text(encoding="utf-8"), path, state.run_id, state.raw_goal)
        return path if path.exists() else None

    def structure(self, candidate_id: str) -> str | None:
        """CIF of a candidate's current structure, relaxed when available (UI-7).

        Parameters
        ----------
        candidate_id : str
            Candidate ID such as ``"c1-2"``.

        Returns
        -------
        str or None
            CIF text, or ``None`` when the run has no structure for that candidate.
        """
        if self.files.workspace.exists():
            cs = load_model(self.files.workspace, Workspace).candidates.get(candidate_id)
            if cs is not None:
                cif = _to_cif(cs.structure) or (cs.record.cif if cs.record is not None else None)
                if _has_atoms(cif):
                    return cif
        if not self.exists:
            return None
        for it in reversed(self.state.iterations):  # runs stored without workspace.json
            for sim in it.sims:
                if sim.candidate_id == candidate_id and sim.relaxed is not None:
                    cif = _to_cif(sim.relaxed.pmg_dict) or sim.relaxed.cif
                    if _has_atoms(cif):
                        return cif
            for rec in it.structures:
                if rec.candidate_id == candidate_id and _has_atoms(rec.cif):
                    return rec.cif
        return None

    # ------------------------------------------------------------------ parsing

    def _llm(self) -> LLM:
        """Return the goal parser's model: the injected one, else the run's backend and model (OR-11).

        Returns
        -------
        LLM
            Model for structured calls.
        """
        return self.deps.llm or make_llm(self.config.llm_backend, self.config.llm_model)

    def parse(self, answers: dict[str, str] | None = None) -> ParseResult:
        """Parse the goal into a spec and clarifying questions, without starting a run (GP-5, UI-2).

        A ready spec is kept on the ``Experiment``, so the next ``run`` or ``start`` skips the parser.

        Parameters
        ----------
        answers : dict, optional
            Answers to the questions of an earlier ``parse``.

        Returns
        -------
        ParseResult
            Spec, questions, validation errors and the parser's backend and model.
        """
        if answers is not None:
            self.answers = answers
        llm = self._llm()
        spec, questions, errors = parse_goal(llm, self.goal, self.config, self.answers)
        messages = [e.message for e in errors]
        if spec is not None:
            messages += [m for m in validate_spec(spec) if m not in messages]
        result = ParseResult(spec, questions, messages, {"backend": llm.backend, "model": llm.model})
        if result.ready:
            self.spec = spec
        return result

    # ------------------------------------------------------------------ running

    def run(self) -> RunState:
        """Run to the end in this thread: parse, campaign, final re-check, brief (OR-1).

        Continues from earlier ``tool`` calls when there were any.

        Returns
        -------
        RunState
            Final state; ``status`` is ``needs_input`` when the parser has questions.

        Raises
        ------
        ExperimentError
            If the run is already running or finished.
        """
        try:
            if self.files.workspace.exists():
                self._release_steps()
                return runner.resume(self.run_id, self.deps)
            if self.exists and self.state.status not in ("created", "needs_input"):
                raise ExperimentError(f"Run {self.run_id} already started; use resume().")
            return runner.run(self.goal, self.config, self.spec, self.answers, self.deps, self.run_id)
        except runner.RunRefused as exc:
            raise ExperimentError(str(exc)) from exc
        finally:
            self.deps.bus.close_run(self.run_id)

    def start(self, resume: bool = False) -> Experiment:
        """Start ``run`` (or ``resume``) in a background thread and return at once.

        Subscribe with ``events`` straight after; the history replays first, so nothing is missed.

        Parameters
        ----------
        resume : bool
            Resume an interrupted run instead; refusals are raised here, before the thread starts.

        Returns
        -------
        Experiment
            ``self``, for chaining.

        Raises
        ------
        ExperimentError
            If the run is already running in this process, or cannot be resumed.
        """
        if resume:
            self._check_resumable()
            return self._background(self._resume)
        return self._background(self.run)

    def resume(self) -> RunState:
        """Resume an interrupted run with a fresh orchestrator told what was done (OR-6, UI-16).

        Returns
        -------
        RunState
            Final state.

        Raises
        ------
        ExperimentError
            If the run does not exist, is running or already finished.
        """
        self._check_resumable()
        return self._resume()

    def _check_resumable(self) -> None:
        """Refuse a resume that cannot work.

        Raises
        ------
        ExperimentError
            If the run does not exist, is running or already finished.
        """
        if not self.exists:
            raise ExperimentError(f"No run {self.run_id}.")
        if self.running:
            raise ExperimentError(f"Run {self.run_id} is already running.")
        if self.state.status == "finished":
            raise ExperimentError(f"Run {self.run_id} already finished.")

    def _resume(self) -> RunState:
        """Resume through the runner.

        Returns
        -------
        RunState
            Final state.

        Raises
        ------
        ExperimentError
            If the runner refuses.
        """
        self._release_steps()
        try:
            return runner.resume(self.run_id, self.deps)
        except runner.RunRefused as exc:
            raise ExperimentError(str(exc)) from exc
        finally:
            self.deps.bus.close_run(self.run_id)

    def _background(self, target: Any) -> Experiment:
        """Run ``target`` in a daemon thread, keeping its result or error.

        Parameters
        ----------
        target : callable
            Returns the final ``RunState``.

        Returns
        -------
        Experiment
            ``self``.

        Raises
        ------
        ExperimentError
            If a run is already executing.
        """
        if self.running:
            raise ExperimentError(f"Run {self.run_id} is already running.")
        self._result, self._error = None, None

        def work() -> None:
            """Execute and store the outcome; errors are kept for ``wait``."""
            try:
                self._result = target()
            except BaseException as exc:
                self._error = exc

        self._thread = threading.Thread(target=work, daemon=True, name=f"run-{self.run_id}")
        self._thread.start()
        return self

    def wait(self, timeout: float | None = None) -> RunState | None:
        """Wait for a background run to end.

        Parameters
        ----------
        timeout : float, optional
            Seconds to wait; ``None`` waits until it ends.

        Returns
        -------
        RunState or None
            Final state, or ``None`` if it is still running after ``timeout``.

        Raises
        ------
        BaseException
            Whatever the background run raised.
        """
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                return None
        if self._error is not None:
            raise self._error
        return self._result if self._result is not None else (self.state if self.exists else None)

    def set_replay_speed(self, speed: float) -> None:
        """Change a running replay's speed (UI-9).

        Parameters
        ----------
        speed : float
            New speed; 0 plays the rest with no pauses.

        Raises
        ------
        ExperimentError
            If this is not a replay.
        """
        if self.replay_control is None:
            raise ExperimentError(f"{self.run_id} is not a replay.")
        self.replay_control["speed"] = float(speed)

    def stop(self) -> bool:
        """Ask the campaign to stop and submit; the run still ends with a brief (OR-8, UI-10).

        For a replay handle, ends the replay instead.

        Returns
        -------
        bool
            True if the run is active in this process.
        """
        if self.replay_control is not None:
            self.replay_control["stop"].set()
            return self.running
        if self._campaign is not None:
            self._campaign.request_stop()
        return runner.request_stop(self.run_id)

    # ------------------------------------------------------------------ events

    def events(self, include_history: bool = True, follow: bool = True) -> Iterator[ProgressEvent]:
        """Yield the run's events: history first, then live ones until the run ends (OR-7).

        Parameters
        ----------
        include_history : bool
            Start with every earlier event (from disk after a restart, DL-10).
        follow : bool
            Keep waiting for new events while the run executes in this process.

        Yields
        ------
        ProgressEvent
            Events in order.
        """
        sub = self.deps.bus.subscribe(self.run_id, include_history)
        try:
            while True:
                event = sub.get(timeout=_POLL_S)
                if event is not None:
                    yield event
                elif not (follow and self.running) and sub.queue.empty():
                    return
        finally:
            sub.close()

    # ------------------------------------------------------------------ replay

    def replay(self, speed: float = 1.0, replay_id: str | None = None) -> Experiment:
        """Re-emit this run's stored events at demo pace, with no network (UI-9).

        Parameters
        ----------
        speed : float
            Playback speed; 2 plays twice as fast, 0 plays with no pauses.
        replay_id : str, optional
            ID the events are published under; a new one by default.

        Returns
        -------
        Experiment
            A replay handle: its ``run_id`` is the replay ID, ``events`` streams the replay, and
            ``state`` and ``brief`` read the stored run.

        Raises
        ------
        ExperimentError
            If the run has no stored events.
        """
        if not self.deps.bus.history(self.run_id):
            raise ExperimentError(f"No stored events for run {self.run_id}.")
        rid = replay_id or f"replay-{new_run_id()}"
        handle = Experiment(self.goal, self.config, spec=self.spec, run_id=rid, root=self.root)
        handle.deps.bus = self.deps.bus
        handle._source = self.run_id
        handle.replay_control = {
            "source": self.run_id,
            "total_events": len(self.deps.bus.history(self.run_id)),
            "speed": float(speed),
            "stop": threading.Event(),
        }
        handle._play()
        return handle

    def _play(self, start: int = 0) -> None:
        """Publish the stored run's events under this replay's ID in a thread.

        Parameters
        ----------
        start : int
            Index of the first event.
        """
        ctl = self.replay_control
        assert ctl is not None
        ctl["stop"] = threading.Event()

        def play() -> None:
            """Publish the stored events under the replay ID."""
            self.deps.bus.replay(
                ctl["source"], self.run_id, lambda: ctl["speed"], stop=ctl["stop"], start=start
            )

        self._thread = threading.Thread(target=play, daemon=True, name=f"replay-{self.run_id}")
        self._thread.start()

    def resume_replay(self) -> int:
        """Continue a stopped replay from the first event it has not shown (UI-9).

        Returns
        -------
        int
            Index of the event it resumes from.

        Raises
        ------
        ExperimentError
            If this is not a replay, it is still playing, or it already showed every event.
        """
        ctl = self.replay_control
        if ctl is None:
            raise ExperimentError(f"{self.run_id} is not a replay.")
        if self.running:
            raise ExperimentError(f"Replay {self.run_id} is still playing.")
        start = len(self.deps.bus.history(self.run_id))
        if start >= ctl["total_events"]:
            raise ExperimentError(f"Replay {self.run_id} already finished.")
        self._play(start)
        return start

    # ------------------------------------------------------------------ step by step

    def tool(self, name: str, **args: Any) -> dict[str, Any]:
        """Call one tool through the campaign's budget, events, gates and snapshot (OR-10, SC-1).

        The first call creates the run folder and, for a relative target, computes the
        reference value with the screening model (GP-11). ``run`` or ``resume`` afterwards hands
        the run to the orchestrator, which is told what was already done.

        Parameters
        ----------
        name : str
            Tool name from ``docs/TOOLS.md``, e.g. ``"propose_candidate"`` or ``"submit_final"``.
        **args
            Tool input.

        Returns
        -------
        dict
            The tool's result; failures come back as ``{"error": ...}``, never raised.

        Raises
        ------
        ExperimentError
            If there is no ready spec, or the run is executing in the background.
        """
        if self.running:
            raise ExperimentError(f"Run {self.run_id} is running; stop it before calling tools.")
        result, _ = self._steps().call_tool(name, args)
        return result

    def _steps(self) -> Campaign:
        """Return the campaign that ``tool`` calls go through, created on first use.

        Returns
        -------
        Campaign
            Campaign over the restored or a fresh workspace.

        Raises
        ------
        ExperimentError
            If there is no ready spec.
        """
        if self._campaign is not None:
            return self._campaign
        files = self.files
        if files.workspace.exists():
            state, ws, critiques = load_for_resume(files)
        else:
            if self.spec is None:
                parsed = self.parse()
                if not parsed.ready:
                    raise ExperimentError(
                        "The goal needs a ready spec first: "
                        + "; ".join(parsed.questions + parsed.errors or ["parser failed"])
                    )
            assert self.spec is not None
            state, files = runner.new_run(self.goal, self.config, self.spec, self.root, self.run_id)
            ws, critiques = Workspace(run_id=self.run_id, spec=self.spec, cfg=self.config), []
        handlers = dict(self.deps.handlers) if self.deps.handlers is not None else load_default_handlers()
        if ws.spec.reference_material and ws.spec.reference_value is None:
            spec, errors = runner.resolve_reference(ws, handlers)
            ws.spec = state.spec = self.spec = spec
            state.errors += errors
        llm = self._llm()
        campaign = Campaign(
            ws,
            llm,
            reviewer_llm=self.deps.reviewer_llm or llm,
            handlers=handlers,
            bus=self.deps.bus,
            critiques=critiques,
            novelty_fn=self.deps.novelty_fn,
            snapshot=lambda c: save_snapshot(files, state, c.ws, c.critiques),
        )
        save_snapshot(files, state, ws, critiques)
        self._campaign = campaign
        return campaign

    def _release_steps(self) -> None:
        """Drop the step-by-step campaign so the runner restores the workspace from disk."""
        self._campaign = None

    def workspace(self) -> Workspace | None:
        """Return the run's workspace: candidates, papers, evidence and budgets (``docs/TOOLS.md``).

        Returns
        -------
        Workspace or None
            Live during step-by-step calls, else ``workspace.json``; ``None`` before the first call.
        """
        if self._campaign is not None:
            return self._campaign.ws
        path = self.files.workspace
        return load_model(path, Workspace) if path.exists() else None


# ---------------------------------------------------------------------- module helpers


def _to_cif(pmg_dict: dict[str, Any] | None) -> str | None:
    """Render a pymatgen structure dict as CIF.

    Parameters
    ----------
    pmg_dict : dict or None
        ``Structure.as_dict()`` output.

    Returns
    -------
    str or None
        CIF text, or ``None`` when the dict is empty or unreadable.
    """
    if not pmg_dict:
        return None
    try:
        from pymatgen.core import Structure

        return str(Structure.from_dict(pmg_dict).to(fmt="cif"))
    except Exception:
        return None


def _has_atoms(cif: str | None) -> bool:
    """Whether a CIF lists atom sites (stub records hold only a ``data_`` line).

    Parameters
    ----------
    cif : str or None
        CIF text.

    Returns
    -------
    bool
        True when the CIF has an ``_atom_site_`` loop.
    """
    return bool(cif) and "_atom_site_" in cif  # type: ignore[operator]


def list_runs(root: Path | None = None) -> list[dict[str, Any]]:
    """Summaries of the stored runs, newest first (UI-9, UI-16).

    Parameters
    ----------
    root : pathlib.Path, optional
        Runs folder.

    Returns
    -------
    list of dict
        ``{run_id, raw_goal, status, best, stop_reason, created_at, running}`` per run.
    """
    base = Path(root) if root else runs_root()
    out = []
    for path in base.glob("*/state.json"):
        try:
            s = load_model(path, RunState)
        except Exception:
            continue
        out.append(
            {
                "run_id": s.run_id,
                "raw_goal": s.raw_goal,
                "status": s.status,
                "best": s.best,
                "stop_reason": s.stop_reason,
                "created_at": s.created_at.isoformat(),
                "running": runner.is_active(s.run_id),
            }
        )
    return sorted(out, key=lambda r: r["created_at"], reverse=True)


def backend_status(check_claude_code: bool = True) -> dict[str, Any]:
    """Readiness of each LLM backend, for ``/health`` and ``hephaestus backends`` (OR-11, UI-12).

    A signed-out Claude Code CLI hangs instead of failing, so it is checked here first.

    Parameters
    ----------
    check_claude_code : bool
        Run ``claude auth status`` (can take seconds); otherwise report it as unchecked.

    Returns
    -------
    dict
        ``{api: {ready, error}, claude-code: {ready, error, auth_method}, auto: {resolves_to, ready}}``.
    """
    api_ok = api_credentials_present()
    api = {"ready": api_ok, "error": None if api_ok else "ANTHROPIC_API_KEY is not set."}
    if check_claude_code:
        from hephaestus.llm.auth import claude_code_status

        cc = claude_code_status()
        claude = {"ready": cc["signed_in"], "error": cc["error"], "auth_method": cc["auth_method"]}
    else:
        claude = {"ready": None, "error": "Not checked.", "auth_method": None}
    auto = resolve_backend("auto")
    ready = api["ready"] if auto == "api" else claude["ready"]
    return {"api": api, "claude-code": claude, "auto": {"resolves_to": auto, "ready": ready}}


def simulation_status() -> dict[str, Any]:
    """Whether the simulation stack and Materials Project access are available, without importing them.

    Returns
    -------
    dict
        ``{mace, matgl, torch, mp_key}`` booleans.
    """
    have = lambda mod: importlib.util.find_spec(mod) is not None  # noqa: E731
    return {
        "mace": have("mace"),
        "matgl": have("matgl"),
        "torch": have("torch"),
        "mp_key": bool(os.environ.get("MP_API_KEY")),
    }


def run_blockers(sim: dict[str, Any] | None = None) -> list[str]:
    """List what stops any run from working: missing MACE or Materials Project key.

    Every run builds and relaxes its reference and candidates with MACE, and fetches competing
    phases for the hull from Materials Project, so without either it fails at the first step.

    Parameters
    ----------
    sim : dict, optional
        A ``simulation_status()`` result; computed when omitted.

    Returns
    -------
    list of str
        One sentence per problem, with the fix; empty when runs can start.
    """
    sim = sim or simulation_status()
    out = []
    if not sim["mace"]:
        out.append('MACE is not installed in the server\'s Python: pip install -e ".[sim]".')
    if not sim["mp_key"]:
        out.append("No Materials Project API key: add one under Connections, or set MP_API_KEY.")
    return out


def backend_ready(backend: str, status: dict[str, Any] | None = None) -> tuple[bool, str | None]:
    """Whether a run on ``backend`` can start now (``POST /runs`` answers 503 otherwise).

    Parameters
    ----------
    backend : {"auto", "api", "claude-code"}
        Requested backend.
    status : dict, optional
        A ``backend_status`` result to reuse; computed when omitted.

    Returns
    -------
    ready : bool
        True when the backend can be used.
    error : str or None
        Why not.
    """
    try:
        resolved = resolve_backend(backend)
    except ValueError as exc:
        return False, str(exc)
    st = (status or backend_status(check_claude_code=resolved == "claude-code"))[resolved]
    return bool(st["ready"]), st["error"]


def models() -> dict[str, Any]:
    """List the selectable Claude models and the default (OR-11, UI-12).

    Returns
    -------
    dict
        ``{models, default}``.
    """
    return {"models": list(MODELS), "default": RunConfig().llm_model}


def device_status() -> dict[str, Any]:
    """Whether torch and a CUDA GPU are available, without importing torch when it is absent.

    Returns
    -------
    dict
        ``{torch, cuda, device}``.
    """
    if importlib.util.find_spec("torch") is None:
        return {"torch": False, "cuda": False, "device": "cpu"}
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    return {"torch": True, "cuda": cuda, "device": "cuda" if cuda else "cpu"}


def run(goal: str, config: RunConfig | None = None, **options: Any) -> RunState:
    """Run a goal end to end; shorthand for ``Experiment(goal, config, **options).run()``.

    Parameters
    ----------
    goal : str
        The goal.
    config : RunConfig, optional
        Options.
    **options
        ``Experiment`` keyword arguments or ``RunConfig`` fields.

    Returns
    -------
    RunState
        Final state.
    """
    return Experiment(goal, config, **options).run()


def load(run_id: str, root: Path | None = None) -> Experiment:
    """Open a stored run; shorthand for ``Experiment.load``.

    Parameters
    ----------
    run_id : str
        Run ID.
    root : pathlib.Path, optional
        Runs folder.

    Returns
    -------
    Experiment
        The run.
    """
    return Experiment.load(run_id, root)


def resume(run_id: str, root: Path | None = None) -> RunState:
    """Resume an interrupted run to the end; shorthand for ``Experiment.load(...).resume()``.

    Parameters
    ----------
    run_id : str
        Run ID.
    root : pathlib.Path, optional
        Runs folder.

    Returns
    -------
    RunState
        Final state.
    """
    return Experiment.load(run_id, root).resume()


__all__ = [
    "Experiment",
    "ExperimentError",
    "ParseResult",
    "backend_ready",
    "backend_status",
    "bus_for",
    "device_status",
    "list_runs",
    "load",
    "models",
    "resume",
    "run",
    "run_blockers",
    "simulation_status",
]
