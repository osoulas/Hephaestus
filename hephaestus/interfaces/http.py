"""HTTP API: FastAPI over ``Experiment``, with server-sent events (REQUIREMENTS § Backend endpoints).

Every endpoint is a thin call into ``interfaces/python.py``; nothing here decides a gate, a
budget or a stop. Runs execute in background threads inside this process, one at a time,
because MACE and MEGNet share one GPU lock. A restart kills them; ``/runs/{id}/resume``
continues from the last snapshot, and event history is read back from disk (DL-10, UI-16).

Run it with ``python -m hephaestus serve`` or ``uvicorn hephaestus.interfaces.http:app``.
"""

from __future__ import annotations

import faulthandler
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from hephaestus.core import credentials, valency_auth
from hephaestus.core.config import PROFILES, RunConfig
from hephaestus.core.types import RunState, SearchSpec
from hephaestus.harness import runner
from hephaestus.interfaces.python import (
    Experiment,
    ExperimentError,
    approve_tool,
    backend_ready,
    backend_status,
    device_status,
    list_runs,
    models,
    remove_tool,
    run_blockers,
    simulation_status,
    tool_library,
)
from hephaestus.llm.auth import (
    ClaudeLogin,
    api_key_status,
    check_api_key,
    forget_api_key,
    load_api_key,
    save_api_key,
)
from hephaestus.tools.materials_project import check_mp_key

WEB_DIR = Path(__file__).parent / "web"
STATUS_TTL_S = 60.0

DepsFactory = Callable[[], runner.Deps]


class RevalidatedStaticFiles(StaticFiles):
    """Static files the browser must revalidate on every load.

    The UI is plain ES modules: a browser that keeps an old module beside a new one that imports
    from it fails to start, leaving a blank page. ``no-cache`` still lets it reuse a file whose
    ETag has not changed, so offline replays stay fast.
    """

    def file_response(self, *args: Any, **kwargs: Any) -> Any:
        """Serve a file with ``Cache-Control: no-cache``.

        Parameters
        ----------
        *args, **kwargs
            Passed to ``StaticFiles.file_response``.

        Returns
        -------
        starlette.responses.Response
            The file response.
        """
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


class ParseRequest(BaseModel):
    """Body of ``POST /parse`` (UI-2)."""

    model_config = ConfigDict(extra="forbid")

    raw_goal: str
    answers: dict[str, str] | None = None
    llm_backend: str | None = None
    llm_model: str | None = None
    require_novel: bool | None = None
    band_gap_model: str | None = None
    band_gap_fidelity: str | None = None  # older clients; mapped to band_gap_model
    run_hull: bool | None = None
    e_hull_max_eV_per_atom: float | None = None
    run_phonons: bool | None = None
    custom_tools: str | None = None
    disabled_capabilities: list[str] | None = None


class RunRequest(BaseModel):
    """Body of ``POST /runs``: a parsed spec, or a goal to parse first."""

    model_config = ConfigDict(extra="forbid")

    spec: SearchSpec | None = None
    config: RunConfig = Field(default_factory=RunConfig)
    raw_goal: str = ""
    answers: dict[str, str] | None = None


class ToolDecisionRequest(BaseModel):
    """Body of ``POST /runs/{run_id}/tool-decisions`` (AT-7, AT-13)."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    action: str
    message: str | None = None


class ApproveToolRequest(BaseModel):
    """Body of ``POST /tools/{name}/approve``: the fingerprint of the files reviewed (AT-21)."""

    model_config = ConfigDict(extra="forbid")

    sha256: str


class CodeRequest(BaseModel):
    """Body of ``POST /auth/claude/code``."""

    model_config = ConfigDict(extra="forbid")

    code: str


class ApiKeyRequest(BaseModel):
    """Body of ``POST /auth/anthropic``."""

    model_config = ConfigDict(extra="forbid")

    api_key: str


class TitleRequest(BaseModel):
    """Body of ``PATCH /runs/{run_id}``."""

    model_config = ConfigDict(extra="forbid")

    title: str


class ReplayRequest(BaseModel):
    """Body of ``POST /replay/{run_id}`` (UI-9)."""

    model_config = ConfigDict(extra="forbid")

    speed: float = 1.0


class StatusCache:
    """Backend and device readiness, refreshed in the background so ``/health`` never blocks."""

    def __init__(self, ttl_s: float = STATUS_TTL_S) -> None:
        """Start empty; the first ``get`` triggers a refresh.

        Parameters
        ----------
        ttl_s : float
            Seconds before a result counts as stale, s.
        """
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        self._value: dict[str, Any] | None = None
        self._at = 0.0
        self._refreshing = False

    def get(self) -> dict[str, Any] | None:
        """Return the last result, starting a refresh when it is missing or stale.

        Returns
        -------
        dict or None
            ``{backends, device}``, or ``None`` before the first refresh finishes.
        """
        with self._lock:
            stale = time.monotonic() - self._at > self.ttl_s
            if (self._value is None or stale) and not self._refreshing:
                self._refreshing = True
                threading.Thread(target=self._refresh, daemon=True, name="status-refresh").start()
            return self._value

    def _refresh(self) -> None:
        """Compute readiness (runs ``claude auth status``, imports torch when installed)."""
        value: dict[str, Any]
        try:
            value = {"backends": backend_status(), "device": device_status()}
        except Exception as exc:
            value = {"backends": None, "device": None, "error": str(exc)[:300]}
        with self._lock:
            self._value, self._at, self._refreshing = value, time.monotonic(), False

    def invalidate(self) -> None:
        """Force the next ``get`` to refresh (after a sign-in)."""
        with self._lock:
            self._at = 0.0


def sse(events: Iterator[Any]) -> Iterator[str]:
    """Format ``ProgressEvent``s as server-sent events, ending with an ``end`` event (OR-7).

    Parameters
    ----------
    events : iterator of ProgressEvent
        Events to send.

    Yields
    ------
    str
        ``id``/``event``/``data`` frames; ``data`` is the event's JSON.
    """
    for event in events:
        yield f"id: {event.seq}\nevent: progress\ndata: {event.model_dump_json()}\n\n"
    yield "event: end\ndata: {}\n\n"


def create_app(
    root: Path | None = None,
    deps_factory: DepsFactory | None = None,
    check_backends: bool | None = None,
    claude_login: ClaudeLogin | None = None,
    key_check: Callable[[str], str | None] | None = check_api_key,
    mp_check: Callable[[str], str | None] | None = check_mp_key,
    library_dir: Path | None = None,
) -> FastAPI:
    """Build the API (tests pass a runs folder and scripted dependencies).

    Parameters
    ----------
    root : pathlib.Path, optional
        Runs folder; defaults to ``runs_root()``.
    deps_factory : callable, optional
        Returns fresh ``runner.Deps`` for each run or parse; real dependencies by default.
    check_backends : bool, optional
        Refuse runs on a backend that is not ready (503). Defaults to True unless
        ``deps_factory`` is given.
    claude_login : ClaudeLogin, optional
        Claude Code sign-in runner (tests inject a fake CLI).
    key_check : callable or None
        Checks an Anthropic API key before saving; ``None`` skips the check (tests).
    mp_check : callable or None
        Checks a Materials Project key before saving; ``None`` skips the check (tests).
    library_dir : pathlib.Path, optional
        Tool library folder; ``RunConfig().tool_library_dir`` by default.

    Returns
    -------
    fastapi.FastAPI
        The application.
    """
    app = FastAPI(title="Hephaestus", description="Self-designing experiment agent for materials discovery.")
    experiments: dict[str, Experiment] = {}
    lock = threading.Lock()
    status = StatusCache()
    load_api_key()  # keys saved from the UI earlier
    credentials.load_secret("materials_project")
    login = claude_login or ClaudeLogin()
    if login.on_done is None:
        login.on_done = status.invalidate
    check = deps_factory is None if check_backends is None else check_backends

    def make_deps() -> runner.Deps:
        """Fresh dependencies for one run.

        Returns
        -------
        runner.Deps
            Injected or real dependencies.
        """
        return deps_factory() if deps_factory else runner.Deps()

    def require_ready(backend: str) -> None:
        """Answer 503 when the backend cannot serve a call (a signed-out Claude Code hangs, OR-11).

        Parameters
        ----------
        backend : str
            Requested backend.

        Raises
        ------
        fastapi.HTTPException
            503 with the reason.
        """
        if not check:
            return
        st = (status.get() or {}).get("backends") or backend_status()  # first request: check now
        ok, error = backend_ready(backend, st)
        if not ok:
            raise HTTPException(503, f"LLM backend {backend!r} is not ready: {error}")

    def require_runnable() -> None:
        """Answer 503 when no run could get past its first step (no MACE, no MP key).

        Raises
        ------
        fastapi.HTTPException
            503 listing what is missing and how to fix it.
        """
        if not check:
            return
        blockers = run_blockers(simulation_status())
        if blockers:
            raise HTTPException(503, " ".join(blockers))

    def get_experiment(run_id: str) -> Experiment:
        """Return the live ``Experiment`` for a run, or the stored run.

        Parameters
        ----------
        run_id : str
            Run or replay ID.

        Returns
        -------
        Experiment
            The run.

        Raises
        ------
        fastapi.HTTPException
            404 when there is no such run.
        """
        with lock:
            exp = experiments.get(run_id)
        if exp is not None:
            return exp
        try:
            return Experiment.load(run_id, root, deps=make_deps())
        except ExperimentError as exc:
            raise HTTPException(404, str(exc)) from exc

    def refuse_if_busy() -> None:
        """Answer 409 while another run executes: one GPU job at a time.

        Raises
        ------
        fastapi.HTTPException
            409 naming the running run.
        """
        with lock:
            busy = [rid for rid, e in experiments.items() if e.running and not rid.startswith("replay-")]
        if busy:
            raise HTTPException(409, f"Run {busy[0]} is still running; stop it or wait.")

    # ------------------------------------------------------------------ status

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Backend, Valency and device status; never blocks (slow checks run in the background).

        Returns
        -------
        dict
            ``{ok, backends, device, simulation, valency, running}``; ``backends`` is ``None`` until
            checked, and ``simulation.blockers`` lists what stops any run.
        """
        st = status.get() or {}
        try:
            valency = valency_auth.status()
        except Exception as exc:
            valency = {"signed_in": False, "error": str(exc)[:200]}
        with lock:
            running = [rid for rid, e in experiments.items() if e.running]
        return {
            "ok": True,
            "backends": st.get("backends"),
            "device": st.get("device"),
            "simulation": {**simulation_status(), "blockers": run_blockers()},
            "valency": valency,
            "running": running,
        }

    @app.get("/models")
    def get_models() -> dict[str, Any]:
        """List the selectable Claude models (UI-12).

        Returns
        -------
        dict
            ``{models, default}``.
        """
        return models()

    @app.get("/config/defaults")
    def config_defaults() -> dict[str, Any]:
        """``RunConfig`` defaults, for the UI's controls.

        Returns
        -------
        dict
            ``RunConfig().model_dump()``.
        """
        return RunConfig().model_dump(mode="json")

    @app.get("/config/profiles")
    def config_profiles() -> list[dict[str, Any]]:
        """Return the setting profiles: ``{id, name, hint, values}``, where ``values`` are ``RunConfig`` fields (OR-4).

        Returns
        -------
        list of dict
            ``PROFILES``, quick to thorough.
        """
        return PROFILES

    @app.get("/config/schema")
    def config_schema() -> dict[str, Any]:
        """JSON schema of ``RunConfig``, so the UI can offer every option (OR-4).

        Returns
        -------
        dict
            ``RunConfig.model_json_schema()``.
        """
        return RunConfig.model_json_schema()

    # ------------------------------------------------------------------ parse and runs

    @app.post("/parse")
    def parse(req: ParseRequest) -> dict[str, Any]:
        """Parse a goal into a spec and clarifying questions (UI-2).

        Parameters
        ----------
        req : ParseRequest
            Goal, answers and LLM choice.

        Returns
        -------
        dict
            ``{spec, questions, errors, parser}``.
        """
        opts = {k: v for k, v in req.model_dump(exclude={"raw_goal", "answers"}).items() if v is not None}
        try:
            cfg = RunConfig(**opts)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        require_ready(cfg.llm_backend)
        exp = Experiment(req.raw_goal, cfg, root=root, deps=make_deps())
        return exp.parse(req.answers).to_dict()

    @app.post("/runs", status_code=201)
    def start_run(req: RunRequest) -> dict[str, Any]:
        """Start a run in the background (OR-1).

        Parameters
        ----------
        req : RunRequest
            Spec (or goal) and config.

        Returns
        -------
        dict
            ``{run_id}``; stream ``/runs/{run_id}/events`` next.
        """
        if req.spec is None and not req.raw_goal.strip():
            raise HTTPException(422, "Give a spec or a raw_goal.")
        require_ready(req.config.llm_backend)
        require_runnable()
        refuse_if_busy()
        exp = Experiment(
            req.raw_goal, req.config, spec=req.spec, answers=req.answers, root=root, deps=make_deps()
        )
        with lock:
            experiments[exp.run_id] = exp
        exp.start()
        return {"run_id": exp.run_id}

    @app.get("/runs")
    def runs() -> list[dict[str, Any]]:
        """List stored runs, newest first (UI-9, UI-16).

        Returns
        -------
        list of dict
            Run summaries.
        """
        return list_runs(root)

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        """Full state of a run, for a refresh (OR-5).

        Parameters
        ----------
        run_id : str
            Run ID.

        Returns
        -------
        dict
            ``RunState`` as JSON, plus ``running``.
        """
        exp = get_experiment(run_id)
        if not exp.exists:
            state = RunState(run_id=run_id, raw_goal=exp.goal, config=exp.config, status="created")
        else:
            state = exp.state
        out = {
            **state.model_dump(mode="json"),
            "running": exp.running,
            "title": exp.title if exp.exists else "",
        }
        if exp.replay_control is not None:
            ctl = exp.replay_control
            out["replay"] = {
                "source": ctl["source"],
                "total_events": ctl["total_events"],
                "speed": ctl["speed"],
            }
        return out

    @app.patch("/runs/{run_id}")
    def rename_run(run_id: str, req: TitleRequest) -> dict[str, Any]:
        """Rename a run, also while it runs.

        Parameters
        ----------
        run_id : str
            Run ID.
        req : TitleRequest
            New title.

        Returns
        -------
        dict
            ``{run_id, title}``; 422 for an empty or over-long title, 409 for a replay.
        """
        exp = get_experiment(run_id)
        try:
            return {"run_id": run_id, "title": exp.set_title(req.title)}
        except ExperimentError as exc:
            code = 409 if "replay" in str(exc) else 404 if str(exc).startswith("No run") else 422
            raise HTTPException(code, str(exc)) from exc

    @app.get("/runs/{run_id}/events")
    def events(run_id: str, history: bool = True) -> StreamingResponse:
        """Stream a run's events as server-sent events: history, then live (OR-7, UI-3, UI-16).

        Parameters
        ----------
        run_id : str
            Run or replay ID.
        history : bool
            Send earlier events first.

        Returns
        -------
        fastapi.responses.StreamingResponse
            ``text/event-stream``; ends with an ``end`` event once the run stops.
        """
        exp = get_experiment(run_id)
        return StreamingResponse(
            sse(exp.events(include_history=history)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/runs/{run_id}/structure/{candidate_id}", response_class=PlainTextResponse)
    def structure(run_id: str, candidate_id: str) -> str:
        """CIF of a candidate's current structure (UI-7).

        Parameters
        ----------
        run_id : str
            Run ID.
        candidate_id : str
            Candidate ID.

        Returns
        -------
        str
            CIF text.
        """
        cif = get_experiment(run_id).structure(candidate_id)
        if cif is None:
            raise HTTPException(404, f"No structure for {candidate_id} in run {run_id}.")
        return cif

    @app.get("/runs/{run_id}/brief", response_model=None)
    def brief(
        run_id: str,
        format: str = Query("md", pattern="^(md|pdf)$"),
        kind: str = Query("lab", pattern="^(lab|compute)$"),
    ) -> Any:
        """Return the lab or computational brief as Markdown or PDF (UI-8).

        Parameters
        ----------
        run_id : str
            Run ID.
        format : {"md", "pdf"}
            File type.
        kind : {"lab", "compute"}
            The lab brief, or the computational brief (DFT settings and workflow).

        Returns
        -------
        fastapi.Response
            ``text/markdown`` or ``application/pdf``.
        """
        exp = get_experiment(run_id)
        name = f"{run_id}_{'compute_' if kind == 'compute' else ''}brief"
        if format == "pdf":
            try:
                pdf = exp.compute_brief_pdf if kind == "compute" else exp.brief_pdf
            except Exception as exc:
                raise HTTPException(500, f"Could not render the PDF: {exc}") from exc
            if pdf is None:
                raise HTTPException(404, f"Run {run_id} has no PDF brief.")
            return FileResponse(pdf, media_type="application/pdf", filename=f"{name}.pdf")
        try:
            md = exp.compute_brief if kind == "compute" else exp.brief
        except Exception as exc:
            raise HTTPException(500, f"Could not build the brief: {exc}") from exc
        if md is None:
            raise HTTPException(404, f"Run {run_id} has no brief yet.")
        return PlainTextResponse(
            md,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'inline; filename="{name}.md"'},
        )

    @app.post("/runs/{run_id}/stop")
    def stop(run_id: str, mode: str = Query("campaign", pattern="^(campaign|all)$")) -> dict[str, Any]:
        """Stop a run (OR-8, UI-10).

        Parameters
        ----------
        run_id : str
            Run ID.
        mode : {"campaign", "all"}
            ``campaign`` ends the search and still runs the final re-check and the brief; ``all``
            ends the run after the current calculation, with neither.

        Returns
        -------
        dict
            ``{stopping, mode}``: ``stopping`` is False when the run is not executing here.
        """
        return {"stopping": get_experiment(run_id).stop(mode), "mode": mode}

    @app.post("/runs/{run_id}/resume")
    def resume(run_id: str) -> dict[str, Any]:
        """Resume an interrupted run in the background (OR-6, UI-16).

        Parameters
        ----------
        run_id : str
            Run ID.

        Returns
        -------
        dict
            ``{run_id, resumed}``; 409 when finished or already running.
        """
        exp = get_experiment(run_id)
        require_ready(exp.config.llm_backend)
        require_runnable()
        refuse_if_busy()
        try:
            exp.start(resume=True)
        except ExperimentError as exc:
            raise HTTPException(409, str(exc)) from exc
        with lock:
            experiments[run_id] = exp
        return {"run_id": run_id, "resumed": True}

    # ------------------------------------------------------------------ agent-built tools (Section 14)

    @app.get("/runs/{run_id}/tool-proposals")
    def tool_proposals(run_id: str) -> list[dict[str, Any]]:
        """Proposals of a run's capability phase, with source at the code stage (AT-13).

        Parameters
        ----------
        run_id : str
            Run ID.

        Returns
        -------
        list of dict
            ``ToolProposal`` fields plus ``waiting`` and ``allowed``.
        """
        return get_experiment(run_id).tool_proposals()

    @app.post("/runs/{run_id}/tool-decisions")
    def tool_decision(run_id: str, req: ToolDecisionRequest) -> dict[str, Any]:
        """Deliver the user's decision on a waiting proposal (AT-7, AT-13).

        Parameters
        ----------
        run_id : str
            Run ID.
        req : ToolDecisionRequest
            Proposal, action and message.

        Returns
        -------
        dict
            ``{accepted: true, decision}``; 409 when the proposal is not waiting or the action is
            not offered.
        """
        try:
            decision = get_experiment(run_id).decide(req.proposal_id, req.action, req.message)
        except (ExperimentError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"accepted": True, "decision": decision.model_dump(mode="json")}

    @app.get("/tools")
    def tools() -> dict[str, Any]:
        """Return the tool library: approved tools and files waiting for review (AT-20, AT-21).

        Returns
        -------
        dict
            ``{dir, approved, unapproved}``.
        """
        return tool_library(library_dir)

    @app.post("/tools/{name}/approve")
    def tools_approve(name: str, req: ApproveToolRequest) -> dict[str, Any]:
        """Approve a library file exactly as it was shown (AT-21).

        Parameters
        ----------
        name : str
            Tool name.
        req : ApproveToolRequest
            Fingerprint of the reviewed files.

        Returns
        -------
        dict
            The approved ``CustomTool``; 409 when the files changed or fail the static check.
        """
        try:
            return approve_tool(name, req.sha256, library_dir).model_dump(mode="json")
        except ExperimentError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.delete("/tools/{name}")
    def tools_remove(name: str) -> dict[str, Any]:
        """Delete a tool and its tests from the library.

        Parameters
        ----------
        name : str
            Tool name.

        Returns
        -------
        dict
            ``{removed: true}``; 404 when there was nothing to remove.
        """
        if not remove_tool(name, library_dir):
            raise HTTPException(404, f"No tool {name}.")
        return {"removed": True}

    @app.post("/replay/{run_id}")
    def replay(run_id: str, req: ReplayRequest | None = None) -> dict[str, Any]:
        """Re-emit a stored run's events at demo pace with no network (UI-9).

        Parameters
        ----------
        run_id : str
            Stored run.
        req : ReplayRequest, optional
            Playback speed.

        Returns
        -------
        dict
            ``{replay_id}``; stream ``/runs/{replay_id}/events``.
        """
        exp = get_experiment(run_id)
        try:
            handle = exp.replay((req or ReplayRequest()).speed)
        except ExperimentError as exc:
            raise HTTPException(404, str(exc)) from exc
        with lock:
            experiments[handle.run_id] = handle
        return {"replay_id": handle.run_id, "total_events": handle.replay_control["total_events"]}

    @app.post("/replay/{replay_id}/resume")
    def replay_resume(replay_id: str) -> dict[str, Any]:
        """Continue a stopped replay where it ended (UI-9).

        Parameters
        ----------
        replay_id : str
            Replay ID.

        Returns
        -------
        dict
            ``{replay_id, resumed_at}``; reconnect to ``/runs/{replay_id}/events``.
        """
        exp = get_experiment(replay_id)
        try:
            start = exp.resume_replay()
        except ExperimentError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"replay_id": replay_id, "resumed_at": start}

    @app.post("/replay/{replay_id}/speed")
    def replay_speed(replay_id: str, req: ReplayRequest) -> dict[str, Any]:
        """Change a running replay's speed (UI-9).

        Parameters
        ----------
        replay_id : str
            Replay ID.
        req : ReplayRequest
            New speed; 0 plays the rest with no pauses.

        Returns
        -------
        dict
            ``{replay_id, speed}``.
        """
        exp = get_experiment(replay_id)
        try:
            exp.set_replay_speed(req.speed)
        except ExperimentError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"replay_id": replay_id, "speed": req.speed}

    # ------------------------------------------------------------------ sign-in

    @app.get("/auth/valency")
    def auth_valency(start: bool = False) -> dict[str, Any]:
        """Valency sign-in status; ``?start=true`` opens the browser sign-in (UI-11).

        Parameters
        ----------
        start : bool
            Start a sign-in in the background.

        Returns
        -------
        dict
            ``{signed_in, needs_signin, access_token_expired, pending, url, error}``.
        """
        return valency_auth.start_signin() if start else valency_auth.status()

    @app.get("/auth/claude")
    def auth_claude() -> dict[str, Any]:
        """Claude Code sign-in: the running sign-in's progress, else the cached readiness (UI-12).

        Returns
        -------
        dict
            ``{status, signed_in, url, code_prompt, error}``. ``status`` is ``checking`` until the
            first readiness check ends, then ``signed_in`` or ``signed_out``, or the running
            sign-in's ``starting``, ``waiting``, ``verifying``, ``failed`` or ``cancelled``.
        """
        flow = login.state()
        st = (status.get() or {}).get("backends")
        cc = (st or {}).get("claude-code") or {}
        signed_in = bool(cc.get("ready")) or flow["status"] == "signed_in"
        if flow["status"] in ("starting", "waiting", "verifying", "failed", "cancelled"):
            return {**flow, "signed_in": signed_in}
        if st is None and flow["status"] != "signed_in":
            return {
                "status": "checking",
                "signed_in": False,
                "url": None,
                "code_prompt": False,
                "error": None,
            }
        return {
            "status": "signed_in" if signed_in else "signed_out",
            "signed_in": signed_in,
            "url": None,
            "code_prompt": False,
            "error": None if signed_in else cc.get("error"),
        }

    @app.post("/auth/claude/start")
    def auth_claude_start() -> dict[str, Any]:
        """Start ``claude auth login --claudeai``; the CLI opens the Claude sign-in page (UI-12).

        Returns
        -------
        dict
            Sign-in state; ``url`` is the fallback page once the CLI prints it.
        """
        return {**login.start(), "signed_in": False}

    @app.post("/auth/claude/code")
    def auth_claude_code(req: CodeRequest) -> dict[str, Any]:
        """Pass the code shown on the fallback sign-in page to the CLI.

        Parameters
        ----------
        req : CodeRequest
            The code.

        Returns
        -------
        dict
            Sign-in state.
        """
        try:
            return {**login.submit_code(req.code), "signed_in": False}
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/auth/claude/cancel")
    def auth_claude_cancel() -> dict[str, Any]:
        """Stop a running sign-in.

        Returns
        -------
        dict
            Sign-in state.
        """
        return {**login.cancel(), "signed_in": False}

    @app.get("/auth/anthropic")
    def auth_anthropic() -> dict[str, Any]:
        """Whether an Anthropic API key is available; never returns the key (OR-11).

        Returns
        -------
        dict
            ``{present, source, hint}``.
        """
        return api_key_status()

    @app.post("/auth/anthropic")
    def auth_anthropic_save(req: ApiKeyRequest) -> dict[str, Any]:
        """Check and save an API key owner-only in ``~/.hephaestus/`` (DL-7).

        Parameters
        ----------
        req : ApiKeyRequest
            The key.

        Returns
        -------
        dict
            ``{present, source, hint}``; 422 when the key is malformed or rejected.
        """
        try:
            out = save_api_key(req.api_key, check=key_check)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        status.invalidate()
        return out

    @app.delete("/auth/anthropic")
    def auth_anthropic_forget() -> dict[str, Any]:
        """Delete the saved API key.

        Returns
        -------
        dict
            ``{present, source, hint}`` afterwards (a key set in the environment stays).
        """
        out = forget_api_key()
        status.invalidate()
        return out

    @app.get("/auth/materials-project")
    def auth_mp() -> dict[str, Any]:
        """Whether a Materials Project API key is available; never returns the key.

        Returns
        -------
        dict
            ``{present, source, hint}``.
        """
        return credentials.secret_status("materials_project")

    @app.post("/auth/materials-project")
    def auth_mp_save(req: ApiKeyRequest) -> dict[str, Any]:
        """Check and save a Materials Project API key owner-only in ``~/.hephaestus/`` (DL-7).

        Parameters
        ----------
        req : ApiKeyRequest
            The key.

        Returns
        -------
        dict
            ``{present, source, hint}``; 422 when the key is malformed or rejected.
        """
        key = req.api_key.strip()
        if not key.isalnum() or len(key) < 16:
            raise HTTPException(
                422, "That does not look like a Materials Project API key (32 letters and digits)."
            )
        if mp_check is not None and (reason := mp_check(key)):
            raise HTTPException(422, reason)
        credentials.write_secret("materials_project", key)
        return credentials.secret_status("materials_project")

    @app.delete("/auth/materials-project")
    def auth_mp_forget() -> dict[str, Any]:
        """Delete the saved Materials Project key.

        Returns
        -------
        dict
            ``{present, source, hint}`` afterwards.
        """
        credentials.delete_secret("materials_project")
        return credentials.secret_status("materials_project")

    # ------------------------------------------------------------------ static UI

    if (WEB_DIR / "index.html").exists():
        app.mount("/", RevalidatedStaticFiles(directory=WEB_DIR, html=True), name="web")
    else:

        @app.get("/", response_class=HTMLResponse)
        def index() -> str:
            """Serve a placeholder page until ``interfaces/web/index.html`` exists.

            Returns
            -------
            str
                HTML pointing at the API docs.
            """
            return '<p>Hephaestus API. The web UI is not built yet; see <a href="/docs">/docs</a>.</p>'

    app.state.experiments = experiments
    app.state.status = status
    return app


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve the API and UI with uvicorn, with ``faulthandler`` on (CUDA crashes are native).

    Parameters
    ----------
    host : str
        Interface to bind.
    port : int
        Port.
    """
    import uvicorn

    faulthandler.enable()
    uvicorn.run(create_app(), host=host, port=port)


app = create_app()
"""Default application for ``uvicorn hephaestus.interfaces.http:app``."""
