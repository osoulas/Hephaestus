"""Orchestrator loop for the API and Claude Code backends (Section 8).

``Campaign.call_tool`` is the single path every tool call takes, whichever backend
drives the conversation: budget, events, run-control tools, gates and snapshots all live
there. ``Campaign.run`` drives backends that return one turn at a time (API, scripted); the
Claude Code backend drives its own loop and calls back into ``call_tool``.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from hephaestus.core.concurrency import KeyedLocks, RWLock
from hephaestus.core.config import budget_limit
from hephaestus.core.events import BUS, EventBus
from hephaestus.core.types import Critique, ErrorRecord, Submission
from hephaestus.harness import gates as G
from hephaestus.harness.prompts import ORCHESTRATOR_SYSTEM, kickoff_message, resume_message
from hephaestus.harness.reviewer import review_round
from hephaestus.harness.state import resume_summary
from hephaestus.llm.base import LLM, LLMError, UsageLimitError
from hephaestus.tools.registry import (
    EXCLUSIVE_TOOLS,
    HANDLERS,
    RUN_CONTROL_TOOLS,
    TOOLS,
    Handler,
    ToolError,
    Workspace,
    execute,
    lock_key,
)

NoveltyFn = Callable[[Workspace, str], dict[str, Any]]
HARD_EXTRA_CALLS = 8
HARD_TIME_FACTOR = 1.25
# After the user stops the campaign, the orchestrator may only submit; if it has not done so
# within this time, or keeps asking for calculations, code picks the answer (OR-3, OR-8).
STOP_GRACE_S = 90.0
STOP_REFUSALS = 3
STOP_ALLOWED = {"submit_final", "candidate_status"}
MAX_NUDGES = 3
LITERATURE_TOOLS = {"search_literature", "related_papers", "get_paper", "publication_trend", "note_evidence"}


def default_novelty_fn() -> NoveltyFn | None:
    """Return C's ``check_novelty`` when the literature module provides it.

    Returns
    -------
    callable or None
        ``f(ws, composition) -> {hits, paper_ids}``.
    """
    try:
        from hephaestus.tools import literature

        fn = getattr(literature, "check_novelty", None)
    except Exception:
        return None
    if fn is None:
        return None
    return lambda ws, comp: fn(ws, comp, ws.spec.target_property)


def shorten(obj: Any, max_str: int = 400, max_list: int = 12) -> Any:
    """Shorten a JSON value for the event stream.

    Parameters
    ----------
    obj : Any
        JSON value.
    max_str : int
        Longest string kept, in characters.
    max_list : int
        Longest list kept.

    Returns
    -------
    Any
        Shortened copy.
    """
    if isinstance(obj, str):
        return obj if len(obj) <= max_str else obj[:max_str] + "…"
    if isinstance(obj, list):
        out = [shorten(x, max_str, max_list) for x in obj[:max_list]]
        if len(obj) > max_list:
            out.append(f"… {len(obj) - max_list} more")
        return out
    if isinstance(obj, dict):
        return {k: shorten(v, max_str, max_list) for k, v in obj.items()}
    return obj


class Campaign:
    """One agent campaign over a workspace (Section 8)."""

    def __init__(
        self,
        ws: Workspace,
        llm: LLM,
        reviewer_llm: LLM | None = None,
        handlers: Mapping[str, Handler] | None = None,
        bus: EventBus = BUS,
        critiques: list[Critique] | None = None,
        novelty_fn: NoveltyFn | None = None,
        snapshot: Callable[[Campaign], None] | None = None,
        elapsed_offset_s: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        """Set up a campaign.

        Parameters
        ----------
        ws : Workspace
            Run workspace (fresh or restored).
        llm : LLM
            Orchestrator model.
        reviewer_llm : LLM, optional
            Reviewer model; defaults to ``llm``.
        handlers : mapping, optional
            Tool handlers; defaults to the registry's.
        bus : EventBus
            Where events go.
        critiques : list of Critique, optional
            Reviews already done (resume).
        novelty_fn : callable, optional
            Novelty search; defaults to the literature module's.
        snapshot : callable, optional
            Called after every tool call to persist the run.
        elapsed_offset_s : float
            Wall time already spent before a resume, in seconds.
        clock : callable
            Monotonic clock, injectable for tests.
        tools : list of dict, optional
            Tool definitions the model sees; ``TOOLS`` by default. Runs with agent-built tools or
            disabled capabilities pass their own list (AT-16, AT-26).
        """
        self.ws = ws
        self.tools = list(TOOLS if tools is None else tools)
        self.llm = llm
        self.reviewer_llm = reviewer_llm or llm
        self.handlers = dict(HANDLERS if handlers is None else handlers)
        self.bus = bus
        self.critiques: list[Critique] = list(critiques or [])
        self.best_checks: list[dict[str, bool | None]] = []
        for c in self.critiques:
            self.best_checks.append(
                {
                    "target_met_central": c.target_met_central,
                    "target_met_conservative": c.target_met_conservative,
                    "thermo_stable": c.thermo_stable,
                    "dyn_stable": c.dyn_stable,
                    "all_passed": c.target_met_central
                    and c.thermo_stable is not False
                    and c.dyn_stable is not False,
                }
            )
        self.novelty_fn = novelty_fn if novelty_fn is not None else default_novelty_fn()
        self._novelty: dict[str, dict[str, Any]] = {}
        self.snapshot = snapshot
        self.clock = clock
        self._t0 = clock() - elapsed_offset_s
        self.stop_event = threading.Event()
        self.abort_event = threading.Event()
        self._stop_at: float | None = None
        self._refused = 0
        self._rw = RWLock()  # shared: per-candidate calls; exclusive: run control, snapshots
        self._cand_locks = KeyedLocks()
        self._book = threading.Lock()  # budget counter, call IDs, submit reason
        self._call_seq = 0
        self.submit_reason: str | None = None
        self.stop_rule: str | None = None
        self.errors: list[ErrorRecord] = []
        self.usage: dict[str, int] = {}
        self.messages: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ budget

    @property
    def elapsed_s(self) -> float:
        """Wall time of the campaign so far, including time before a resume.

        Returns
        -------
        float
            Seconds.
        """
        return self.clock() - self._t0

    def request_stop(self, abort: bool = False) -> None:
        """Stop the campaign (OR-8).

        No new calculation starts: the orchestrator may only submit. If it does not within
        ``STOP_GRACE_S``, or keeps asking for calculations, the hard limit ends the campaign and
        code picks the answer.

        Parameters
        ----------
        abort : bool
            End the campaign at once, without waiting for the orchestrator to submit.
        """
        with self._book:
            if self._stop_at is None:
                self._stop_at = self.elapsed_s
            self.stop_event.set()
            if abort:
                self.abort_event.set()
            self._update_submit_reason()  # so the next submission is accepted even if it fails the gate

    def hard_limit_hit(self) -> bool:
        """Whether the hard limit is reached: budget + 8 calls, 125 % of the wall time, or a stop.

        A stop ends the campaign at once when aborting, else after ``STOP_GRACE_S`` or
        ``STOP_REFUSALS`` refused calls.

        Returns
        -------
        bool
            True when the campaign must end now.
        """
        cfg = self.ws.cfg
        stopped = self._stop_at is not None and (
            self.abort_event.is_set()
            or self._refused >= STOP_REFUSALS
            or self.elapsed_s - self._stop_at >= STOP_GRACE_S
        )
        over_calls = self.ws.tool_calls >= budget_limit(cfg.max_tool_calls) + HARD_EXTRA_CALLS
        over_time = self.elapsed_s >= HARD_TIME_FACTOR * budget_limit(cfg.time_budget_s)
        return stopped or over_calls or over_time

    def _update_submit_reason(self) -> None:
        """Set ``submit_reason`` when the user stopped, or a budget ran out (OR-8 to OR-10)."""
        if self.submit_reason:
            return
        if self.stop_event.is_set():
            self.submit_reason = "user_stop"
        elif self.ws.tool_calls >= budget_limit(self.ws.cfg.max_tool_calls):
            self.submit_reason = "tool_budget"
        elif self.elapsed_s >= budget_limit(self.ws.cfg.time_budget_s):
            self.submit_reason = "wall_time"
        if self.submit_reason and not self.stop_rule:
            self.stop_rule = self.submit_reason

    # ------------------------------------------------------------------ events

    def emit(self, section: str, status: str, payload: dict[str, Any]) -> None:
        """Publish an event for this run.

        Parameters
        ----------
        section : str
            Event section.
        status : {"started", "finished", "failed"}
            Status.
        payload : dict
            Payload.
        """
        self.bus.emit(self.ws.run_id, section, status, payload, iteration=self.ws.round)  # type: ignore[arg-type]

    def emit_agent(self, kind: str, text: str) -> None:
        """Publish the orchestrator's text or thinking.

        Parameters
        ----------
        kind : {"text", "thinking"}
            Block kind.
        text : str
            Content.
        """
        if text:
            self.emit("agent", "finished", {"kind": kind, "text": text})

    def candidate_fields(self, cid: str | None) -> dict[str, Any]:
        """Flat candidate fields for the UI (``docs/TOOLS.md``, *Event payloads*).

        Parameters
        ----------
        cid : str or None
            Candidate ID.

        Returns
        -------
        dict
            Empty when the candidate is unknown.
        """
        if not cid or cid not in self.ws.candidates:
            return {}
        cs = self.ws.candidates[cid]
        f = G.facts(self.ws, cid)
        return {
            "candidate_id": cid,
            "composition": cs.candidate.composition,
            "cand_round": cs.round,
            "e_hull": cs.e_hull,
            "thermo_stable": f["thermo_stable"],
            "dyn_stable": f["dyn_stable"],
            "min_freq": f["min_freq"],
            "verdict": f["phonon_verdict"],
            "space_group": cs.space_group,
            "value": f["value"],
            "uncertainty": f["uncertainty"],
            "novelty": cs.candidate.novelty,
            "rationale": cs.candidate.rationale,
            "evidence_ids": cs.candidate.evidence_ids,
            "targets": f.get("targets"),  # per-target results when there is more than one target
        }

    # ------------------------------------------------------------------ tools

    def call_tool(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Run one tool call with budget, events, gates and a snapshot. Never raises. Thread-safe.

        Concurrency rules, whichever backend issues the calls and from whichever thread:

        - calls on the same candidate run one at a time;
        - calls on different candidates, and literature and database lookups, run in parallel;
        - ``propose_candidate``, ``candidate_status``, ``request_review`` and ``submit_final`` run
          with no other call in flight, because they read or change every candidate;
        - snapshots are taken with no call in flight, so they are never torn.

        Identical expensive work (the same structure relaxed twice) is computed once, by the
        cache's per-key lock.

        Parameters
        ----------
        name : str
            Tool name.
        args : dict
            Tool input.

        Returns
        -------
        result : dict
            Tool result, with an ``instruction`` added when the orchestrator must submit.
        is_error : bool
            Whether the call failed.
        """
        if name in EXCLUSIVE_TOOLS:
            with self._rw.write():
                out = self._call_tool(name, args)
                self._snapshot()
            return out
        with self._rw.read(), self._cand_locks.hold(lock_key(name, args)):
            out = self._call_tool(name, args)
        with self._rw.write():
            self._snapshot()
        return out

    def _snapshot(self) -> None:
        """Persist the run; the caller holds the exclusive lock."""
        if self.snapshot:
            try:
                self.snapshot(self)
            except Exception as exc:
                self.errors.append(ErrorRecord(step="snapshot", code="write_failed", message=str(exc)[:300]))

    def _call_tool(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Run one tool call; the caller holds the right locks.

        Parameters
        ----------
        name : str
            Tool name.
        args : dict
            Tool input.

        Returns
        -------
        result : dict
            Tool result.
        is_error : bool
            Whether the call failed.
        """
        with self._book:
            if self.ws.submitted:
                return {
                    "error": "A submission was already accepted; the campaign is over. Make no more calls."
                }, True
            if self.hard_limit_hit():
                return {"error": "Hard limit reached; the campaign is ending."}, True
            if self.stop_event.is_set() and name not in STOP_ALLOWED:
                self._refused += 1
                return {
                    "error": "The user stopped the campaign: no more calculations. "
                    "Call submit_final now with your best candidate and a runner-up."
                }, True
            self.ws.tool_calls += 1
            self._call_seq += 1
            call_id = self._call_seq
        round_at_start = self.ws.round
        started = {
            "tool": name,
            "call_id": call_id,
            "input": args,
            **self.candidate_fields(args.get("candidate_id")),
        }
        self.bus.emit(self.ws.run_id, "tool", "started", started, iteration=round_at_start)
        if name in RUN_CONTROL_TOOLS:
            result, is_error = self._run_control(name, args)
        else:
            result, is_error = execute(self.ws, name, args, self.handlers)
        with self._book:
            self._update_submit_reason()
            if self.submit_reason and name != "submit_final" and not self.ws.submitted:
                result = {**result, "instruction": self._submit_instruction()}
        cid = args.get("candidate_id") or (result.get("candidate_id") if isinstance(result, dict) else None)
        payload: dict[str, Any] = {
            "tool": name,
            "call_id": call_id,
            "input": args,
            **self.candidate_fields(cid),
        }
        if name == "request_review" and not is_error:
            payload["review"] = result
        else:
            payload["result"] = shorten(result)
        if name in LITERATURE_TOOLS and isinstance(result.get("papers"), list):
            payload["mode"] = args.get("mode") or args.get("relation")
            payload["papers"] = [
                {k: p.get(k) for k in ("paper_id", "title", "year", "citations")} for p in result["papers"]
            ]
        self.bus.emit(
            self.ws.run_id, "tool", "failed" if is_error else "finished", payload, iteration=round_at_start
        )
        return result, is_error

    def run_batch(self, calls: list[tuple[str, dict[str, Any]]]) -> list[tuple[dict[str, Any], bool]]:
        """Run one turn's tool calls, in parallel where the concurrency rules allow.

        Calls are split at each exclusive tool, which runs alone in its place. Between those,
        calls are grouped by candidate: a group runs in the order given (so build, relax and hull
        for one candidate may share a turn), and different groups run at the same time, up to
        ``cfg.max_parallel_tools``. Lookups without a candidate each form their own group.

        Parameters
        ----------
        calls : list of tuple
            ``(name, args)`` in the order the model gave them.

        Returns
        -------
        list of tuple
            ``(result, is_error)`` for each call, in the same order.
        """
        results: list[tuple[dict[str, Any], bool] | None] = [None] * len(calls)
        segment: list[int] = []

        def run_group(idx: list[int]) -> None:
            """Run one group of calls in order.

            Parameters
            ----------
            idx : list of int
                Call indices.
            """
            for i in idx:
                results[i] = self.call_tool(*calls[i])

        def flush() -> None:
            """Run the pending non-exclusive calls, grouped by candidate."""
            groups: dict[Any, list[int]] = {}
            for i in segment:
                key = lock_key(*calls[i])
                groups.setdefault(key if key is not None else ("lookup", i), []).append(i)
            workers = max(1, min(self.ws.cfg.max_parallel_tools, len(groups)))
            if workers == 1:
                for idx in groups.values():
                    run_group(idx)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    list(pool.map(run_group, groups.values()))
            segment.clear()

        for i, (name, _) in enumerate(calls):
            if name in EXCLUSIVE_TOOLS:
                flush()
                results[i] = self.call_tool(*calls[i])
            else:
                segment.append(i)
        flush()
        return [r for r in results if r is not None]

    def nudge(self) -> str:
        """Message for a model that stopped calling tools before submitting.

        Returns
        -------
        str
            A submit instruction when one is due, else a prompt to continue.
        """
        self._update_submit_reason()
        if self.submit_reason:
            return self._submit_instruction()
        return "Continue the campaign with tool calls. When you are done, call submit_final."

    def _submit_instruction(self) -> str:
        """Tell the orchestrator to submit now.

        Returns
        -------
        str
            Instruction text naming the reason.
        """
        reasons = {
            "user_stop": "The user asked to stop.",
            "tool_budget": "The tool-call budget is used up.",
            "wall_time": "The wall-time budget is used up.",
        }
        why = reasons.get(self.submit_reason or "", f"Stop rule fired: {self.submit_reason}.")
        return f"{why} Call submit_final now with your best candidate and a runner-up."

    def _run_control(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Dispatch a run-control tool, turning ``ToolError`` into an error result.

        Parameters
        ----------
        name : str
            ``candidate_status``, ``request_review`` or ``submit_final``.
        args : dict
            Tool input.

        Returns
        -------
        result : dict
            Tool result.
        is_error : bool
            Whether the call failed.
        """
        fn = {
            "candidate_status": self._candidate_status,
            "request_review": self._request_review,
            "submit_final": self._submit_final,
        }[name]
        try:
            return fn(**args), False
        except ToolError as exc:
            return {"error": str(exc)}, True
        except TypeError as exc:
            return {"error": f"Bad arguments for {name}: {exc}"}, True

    def novelty(self, composition: str) -> dict[str, Any]:
        """Novelty search for a composition, cached per run (SC-11).

        Parameters
        ----------
        composition : str
            Formula.

        Returns
        -------
        dict
            ``{hits, paper_ids}``; ``hits`` is ``None`` when the search is unavailable.
        """
        if composition not in self._novelty:
            if self.novelty_fn is None:
                out: dict[str, Any] = {"hits": None, "paper_ids": [], "error": "novelty search unavailable"}
            else:
                try:
                    out = dict(self.novelty_fn(self.ws, composition))
                except Exception as exc:
                    out = {"hits": None, "paper_ids": [], "error": str(exc)[:200]}
                    self.errors.append(
                        ErrorRecord(step="novelty", code="search_failed", message=out["error"])
                    )
            self._novelty[composition] = out
        return self._novelty[composition]

    def _candidate_status(self) -> dict[str, Any]:
        """Code facts for every candidate, and the budget.

        Returns
        -------
        dict
            ``{round, candidates, budget}``.
        """
        return {
            "round": self.ws.round,
            "candidates": [G.facts(self.ws, cid) for cid in self.ws.candidates],
            "budget": {
                "tool_calls_used": self.ws.tool_calls,
                "max_tool_calls": self.ws.cfg.max_tool_calls or "no limit",
                "compute_s": round(self.ws.compute_s, 1),
                "elapsed_s": round(self.elapsed_s, 1),
            },
        }

    def _request_review(self, summary: str) -> dict[str, Any]:
        """End the round: code facts, independent review, stop rules (Section 7, OR-2).

        Parameters
        ----------
        summary : str
            The orchestrator's summary of the round.

        Returns
        -------
        dict
            Review result with a stop instruction or the next round.

        Raises
        ------
        ToolError
            If no candidate was proposed this round.
        """
        ws = self.ws
        ids = [cid for cid, cs in ws.candidates.items() if cs.round == ws.round]
        if not ids:
            raise ToolError(f"No candidates in round {ws.round}; propose and evaluate candidates first.")
        round_facts = [G.facts(ws, cid) for cid in ids]
        best = G.select_best(round_facts, ws.spec)
        assert best is not None
        cs = ws.candidates[best]
        checks = G.check_target(cs, ws.spec)
        nov = self.novelty(cs.candidate.composition)
        route = G.has_synthesis_route(cs.candidate.composition, ws.insights)
        score, breakdown = G.score_rubric(cs, ws.spec, nov.get("hits"), route)
        critique, errors = review_round(
            self.reviewer_llm, ws, summary, round_facts, best, checks, score, breakdown, nov, self.critiques
        )
        self.errors.extend(errors)
        self.critiques.append(critique)
        self.best_checks.append(checks)
        decision = G.should_stop(
            self.critiques, self.best_checks, ws.cfg, self.elapsed_s, user_stop=self.stop_event.is_set()
        )
        result: dict[str, Any] = {
            "round": ws.round,
            "best_of_round": best,
            "score": score,
            "score_breakdown": breakdown,
            "checks": checks,
            "findings": [f.model_dump() for f in critique.findings],
            "recommended_next": critique.next_composition,
            "why": critique.next_rationale,
            "reviewer_recommends_stop": critique.stop_recommended,
            "stop_reason": critique.stop_reason,
            "llm_confidence": critique.llm_confidence,
            "llm_confidence_reason": critique.llm_confidence_reason,
        }
        if decision.gate_note:
            result["gate_note"] = decision.gate_note
        self.emit(
            "orchestrator",
            "finished",
            {
                "round": ws.round,
                "best": best,
                "stop": decision.stop,
                "rule": decision.rule,
                "gate_note": decision.gate_note,
                "scores": [c.score for c in self.critiques],
                "llm_scores": [c.llm_confidence for c in self.critiques],
                "usage": dict(self.usage),
            },
        )
        if decision.stop:
            self.stop_rule = decision.rule
            self.submit_reason = decision.rule
            result["stop_rule"] = decision.rule
            result["instruction"] = "call submit_final"
        else:
            ws.round += 1
            result["next_round"] = ws.round
            nxt = critique.next_composition
            result["instruction"] = (
                f"Start round {ws.round}. Address the recommended composition {nxt}: try it or say why not."
                if nxt
                else f"Start round {ws.round}."
            )
        return result

    def _submit_final(
        self, candidate_id: str, runner_up_id: str | None, justification: str, accept_not_met: bool
    ) -> dict[str, Any]:
        """Apply the submission gate (``submit_final``).

        Parameters
        ----------
        candidate_id : str
            Submitted candidate.
        runner_up_id : str or None
            Runner-up for the final re-check.
        justification : str
            The orchestrator's reasons.
        accept_not_met : bool
            Report the best attempt even though the gate fails.

        Returns
        -------
        dict
            ``{accepted, gate_passed, note}`` and ``reasons`` when refused.

        Raises
        ------
        ToolError
            For unknown candidates or a runner-up equal to the winner.
        """
        ws = self.ws
        if ws.submitted:
            raise ToolError("A submission was already accepted; the campaign is over.")
        ws.get(candidate_id)
        if runner_up_id is not None:
            ws.get(runner_up_id)
            if runner_up_id == candidate_id:
                raise ToolError("runner_up_id must differ from candidate_id.")
        hits = (
            self.novelty(ws.candidates[candidate_id].candidate.composition).get("hits")
            if ws.cfg.require_novel
            else None
        )
        passed, reasons = G.submission_gate(ws, candidate_id, hits)
        forced = self.submit_reason in {"tool_budget", "wall_time", "user_stop"} or bool(self.stop_rule)
        if not passed and not accept_not_met and not forced:
            return {
                "accepted": False,
                "gate_passed": False,
                "reasons": reasons,
                "note": "Improve the candidate, or resubmit with accept_not_met=true to report the best attempt.",
            }
        ws.submitted = Submission(
            candidate_id=candidate_id,
            runner_up_id=runner_up_id,
            justification=justification,
            gate_passed=passed,
            accept_not_met=accept_not_met and not passed,
        )
        note = "Gate passed." if passed else "Accepted as best attempt; target not met: " + "; ".join(reasons)
        return {"accepted": True, "gate_passed": passed, "note": note}

    def code_pick(self) -> Submission | None:
        """Pick the answer in code when no submission was accepted (OR-3).

        Returns
        -------
        Submission or None
            Best and runner-up by checks then distance; ``None`` with no candidates.
        """
        all_facts = [G.facts(self.ws, cid) for cid in self.ws.candidates]
        best = G.select_best(all_facts, self.ws.spec)
        if best is None:
            return None
        rest = [f for f in all_facts if f["candidate_id"] != best]
        runner = G.select_best(rest, self.ws.spec)
        passed, reasons = G.submission_gate(self.ws, best)
        sub = Submission(
            candidate_id=best,
            runner_up_id=runner,
            justification="Picked by code: no submission was accepted. " + "; ".join(reasons),
            gate_passed=passed,
            picked_by="code",
        )
        self.ws.submitted = sub
        return sub

    # ------------------------------------------------------------------ loop

    def kickoff(self, resume: bool = False) -> str:
        """First user message: the goal, or what a resumed run already did.

        Parameters
        ----------
        resume : bool
            Whether this is a resumed run.

        Returns
        -------
        str
            Message text.
        """
        if not resume:
            return kickoff_message(self.ws.spec, self.ws.cfg)
        facts = [G.facts(self.ws, cid) for cid in self.ws.candidates]
        return resume_message(self.ws.spec, self.ws.cfg, resume_summary(self.ws, self.critiques, facts))

    def run(self, resume: bool = False) -> Submission | None:
        """Run the campaign until a submission is accepted or a hard limit ends it.

        Parameters
        ----------
        resume : bool
            Start a fresh conversation told what was already done (OR-6).

        Returns
        -------
        Submission or None
            The accepted submission, or the code pick; ``None`` when nothing was proposed.
        """
        self.emit(
            "orchestrator",
            "started",
            {"resume": resume, "backend": self.llm.backend, "model": self.llm.model},
        )
        if not self.ws.submitted:
            driver = getattr(self.llm, "run_campaign", None)
            if callable(driver):
                try:
                    driver(ORCHESTRATOR_SYSTEM, self.kickoff(resume), self.tools, self)
                except UsageLimitError:
                    raise  # the runner pauses the run (Claude usage limit)
                except LLMError as exc:
                    self.errors.append(
                        ErrorRecord(step="orchestrator", code="llm_failed", message=str(exc)[:500])
                    )
            else:
                self._loop(resume)
        sub = self.ws.submitted or self.code_pick()
        if self.stop_rule is None:
            self.stop_rule = "submitted" if sub and sub.picked_by == "orchestrator" else "hard_limit"
        self.emit(
            "orchestrator",
            "finished",
            {"submitted": sub.model_dump() if sub else None, "rule": self.stop_rule, "usage": self.usage},
        )
        return sub

    def _loop(self, resume: bool) -> None:
        """Turn-by-turn loop for backends that return one turn at a time.

        Parameters
        ----------
        resume : bool
            Whether this is a resumed run.
        """
        self.messages = [{"role": "user", "content": self.kickoff(resume)}]
        nudges = 0
        failures = 0
        while not self.ws.submitted and not self.hard_limit_hit():
            try:
                turn = self.llm.step(ORCHESTRATOR_SYSTEM, self.messages, self.tools)
            except UsageLimitError:
                raise
            except LLMError as exc:
                failures += 1
                self.errors.append(
                    ErrorRecord(step="orchestrator", code="llm_failed", message=str(exc)[:500])
                )
                if failures >= 2:
                    return
                continue
            failures = 0
            for k, v in turn.usage.items():
                self.usage[k] = self.usage.get(k, 0) + int(v or 0)
            for t in turn.thinking:
                self.emit_agent("thinking", t)
            for t in turn.texts:
                self.emit_agent("text", t)
            self.messages.append({"role": "assistant", "content": turn.content})
            if turn.stop_reason == "refusal":
                self.errors.append(
                    ErrorRecord(step="orchestrator", code="refusal", message="Model declined.")
                )
                return
            if not turn.tool_calls:
                nudges += 1
                if nudges > MAX_NUDGES:
                    return
                self.messages.append({"role": "user", "content": self.nudge()})
                continue
            nudges = 0
            blocks = []
            outcomes = self.run_batch([(c.name, c.input) for c in turn.tool_calls])
            for call, (result, is_error) in zip(turn.tool_calls, outcomes, strict=True):
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
                if is_error:
                    block["is_error"] = True
                blocks.append(block)
            self.messages.append({"role": "user", "content": blocks})
