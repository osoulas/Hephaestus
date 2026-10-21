"""``Workspace`` to ``RunState`` sync, snapshots and resume (OR-5, OR-6, DL-1, DL-2, DL-10)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hephaestus.core.config import runs_root
from hephaestus.core.store import atomic_write_text, load_model, save_json
from hephaestus.core.types import (
    Critique,
    IterationRecord,
    LiteratureContext,
    PropertyPrediction,
    RunState,
)
from hephaestus.tools.registry import CandidateState, Workspace

COVERAGE_NOTE = (
    "Literature comes from Valency Bond, which indexes arXiv, bioRxiv, medRxiv and PubMed records; "
    "journal-only papers with no preprint are missing (LM-19)."
)


@dataclass(frozen=True)
class RunFiles:
    """Paths inside one run folder (DL-1)."""

    run_id: str
    root: Path

    @classmethod
    def for_run(cls, run_id: str, root: Path | None = None) -> RunFiles:
        """Return the paths for a run, creating its folder.

        Parameters
        ----------
        run_id : str
            Run ID.
        root : pathlib.Path, optional
            Runs folder; defaults to ``runs_root()``.

        Returns
        -------
        RunFiles
            Paths.
        """
        files = cls(run_id, (root or runs_root()) / run_id)
        files.root.mkdir(parents=True, exist_ok=True)
        return files

    def __getattr__(self, name: str) -> Path:
        """Map ``files.state`` to ``state.json``, ``files.llm_log`` to ``llm_log.jsonl`` and so on.

        Parameters
        ----------
        name : str
            File stem.

        Returns
        -------
        pathlib.Path
            Path inside the run folder.

        Raises
        ------
        AttributeError
            For names that are not run files.
        """
        names = {
            "state": "state.json",
            "spec": "spec.json",
            "workspace": "workspace.json",
            "progress": "progress.json",
            "submission": "submission.json",
            "literature": "literature.json",
            "events": "events.jsonl",
            "llm_log": "llm_log.jsonl",
            "calls": "calls.jsonl",
            "brief_md": "brief.md",
            "brief_pdf": "brief.pdf",
        }
        if name in names:
            return self.root / names[name]
        raise AttributeError(name)


def literature_context(ws: Workspace) -> LiteratureContext:
    """Aggregate the run's papers and evidence into a ``LiteratureContext``.

    Parameters
    ----------
    ws : Workspace
        Run workspace.

    Returns
    -------
    LiteratureContext
        Papers, insights, failed approaches, mechanisms, synthesis routes, trend and coverage.
    """
    ins = ws.insights
    return LiteratureContext(
        papers=list(ws.papers.values()),
        insights=list(ins),
        failed_approaches=[i.summary for i in ins if i.finding_type == "negative"],
        mechanisms=[i.summary for i in ins if i.finding_type == "mechanism"],
        synthesis_routes=[i.summary for i in ins if i.finding_type == "synthesis"],
        field_trend=dict(ws.trend),
        coverage_note=COVERAGE_NOTE,
    )


def sync_state(state: RunState, ws: Workspace, critiques: list[Critique]) -> RunState:
    """Mirror the workspace into ``RunState``: one ``IterationRecord`` per review round (OR-5).

    Parameters
    ----------
    state : RunState
        State to update in place.
    ws : Workspace
        Run workspace.
    critiques : list of Critique
        Reviews so far, oldest first.

    Returns
    -------
    RunState
        ``state``, updated.
    """
    n_rounds = max([ws.round, len(critiques), *[cs.round for cs in ws.candidates.values()]])
    iterations = []
    for r in range(1, n_rounds + 1):
        members = [(cid, cs) for cid, cs in ws.candidates.items() if cs.round == r]
        it = IterationRecord(
            candidates=[cs.candidate for _, cs in members],
            structures=[cs.record for _, cs in members if cs.record is not None],
            sims=[ws.sim_result(cid) for cid, cs in members if cs.energy_eV_per_atom is not None],
            predictions=[p for _, cs in members for p in cs.predictions.values()],
            critique=critiques[r - 1] if r <= len(critiques) else None,
        )
        iterations.append(it)
    state.iterations = iterations
    state.literature = literature_context(ws)
    state.submission = ws.submitted
    if ws.submitted:
        state.best = ws.submitted.candidate_id
    elif critiques:
        state.best = critiques[-1].candidate_id
    return state


def save_snapshot(files: RunFiles, state: RunState, ws: Workspace, critiques: list[Critique]) -> None:
    """Write ``workspace.json`` and ``state.json`` after a tool call (OR-6, DL-10).

    Parameters
    ----------
    files : RunFiles
        Run paths.
    state : RunState
        Run state; synced before writing.
    ws : Workspace
        Run workspace.
    critiques : list of Critique
        Reviews so far.
    """
    save_json(files.workspace, ws)
    save_json(files.state, sync_state(state, ws, critiques))


def write_artifacts(files: RunFiles, state: RunState) -> None:
    """Write the per-round files: candidates, critique, and per-candidate CIF and JSON (DL-1, DL-2).

    Parameters
    ----------
    files : RunFiles
        Run paths.
    state : RunState
        Synced run state.
    """
    if state.spec:
        save_json(files.spec, state.spec)
    if state.literature:
        save_json(files.literature, state.literature)
    if state.submission:
        save_json(files.submission, state.submission)
    for n, it in enumerate(state.iterations, start=1):
        d = files.root / f"iter_{n}"
        save_json(d / "candidates.json", [c.model_dump(mode="json") for c in it.candidates])
        if it.critique:
            save_json(d / "critique.json", it.critique)
        for rec in it.structures:
            atomic_write_text(d / rec.candidate_id / "start.cif", rec.cif)
        for sim in it.sims:
            save_json(d / sim.candidate_id / "sim.json", sim)
            if sim.relaxed is not None:
                atomic_write_text(
                    d / sim.candidate_id / "relaxed.cif", _cif(sim.relaxed.pmg_dict) or sim.relaxed.cif
                )
        by_cand: dict[str, list[PropertyPrediction]] = {}
        for p in it.predictions:
            by_cand.setdefault(p.candidate_id, []).append(p)
        for cid, preds in by_cand.items():
            save_json(d / cid / "prop.json", [p.model_dump(mode="json") for p in preds])


def _cif(pmg_dict: dict[str, Any]) -> str | None:
    """Render a pymatgen structure dict as CIF, or ``None`` when it cannot be read.

    Parameters
    ----------
    pmg_dict : dict
        ``Structure.as_dict()`` output.

    Returns
    -------
    str or None
        CIF text.
    """
    try:
        from pymatgen.core import Structure

        return str(Structure.from_dict(pmg_dict).to(fmt="cif"))
    except Exception:
        return None


def workspace_from_state(state: RunState) -> Workspace:
    """Rebuild a workspace from ``state.json`` alone, when ``workspace.json`` is missing (OR-6).

    Parameters
    ----------
    state : RunState
        Saved state with a spec.

    Returns
    -------
    Workspace
        Candidates, structures, results, papers and evidence; budget counters restart.

    Raises
    ------
    ValueError
        If the state has no spec.
    """
    if state.spec is None:
        raise ValueError("Cannot resume a run that has no parsed specification.")
    ws = Workspace(run_id=state.run_id, spec=state.spec, cfg=state.config)
    for r, it in enumerate(state.iterations, start=1):
        recs = {s.candidate_id: s for s in it.structures}
        sims = {s.candidate_id: s for s in it.sims}
        for cand in it.candidates:
            cs = CandidateState(candidate=cand, round=r, record=recs.get(cand.candidate_id))
            sim = sims.get(cand.candidate_id)
            if sim is not None:
                cs.structure = sim.relaxed.pmg_dict if sim.relaxed else None
                cs.energy_eV_per_atom = sim.energy_eV_per_atom
                cs.converged = sim.converged
                cs.potential = sim.potential
                cs.space_group = sim.space_group
                cs.lattice = sim.lattice
                cs.density_g_cm3 = sim.density_g_cm3
                cs.domain_note = sim.domain_note
                cs.e_hull = sim.energy_above_hull_eV_per_atom
                cs.e_form = sim.formation_energy_eV_per_atom
                cs.phonon = sim.phonons
                cs.errors = list(sim.errors)
            for p in it.predictions:
                if p.candidate_id == cand.candidate_id:
                    cs.predictions[p.property] = p
            ws.candidates[cand.candidate_id] = cs
    if state.literature:
        ws.papers = {p.paper_id: p for p in state.literature.papers}
        ws.insights = list(state.literature.insights)
        ws.trend = dict(state.literature.field_trend)
    critiques = [it.critique for it in state.iterations if it.critique]
    ws.round = len(critiques) + 1
    ws.submitted = state.submission
    return ws


def load_for_resume(files: RunFiles) -> tuple[RunState, Workspace, list[Critique]]:
    """Load a run's state, workspace and reviews for resuming (OR-6).

    Parameters
    ----------
    files : RunFiles
        Run paths.

    Returns
    -------
    state : RunState
        Saved state.
    ws : Workspace
        ``workspace.json`` when present, else rebuilt from ``state.json``.
    critiques : list of Critique
        Reviews so far.
    """
    state = load_model(files.state, RunState)
    ws = load_model(files.workspace, Workspace) if files.workspace.exists() else workspace_from_state(state)
    critiques = [it.critique for it in state.iterations if it.critique]
    return state, ws, critiques


def resume_summary(ws: Workspace, critiques: list[Critique], facts: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe what a resumed orchestrator has already done (OR-6).

    Parameters
    ----------
    ws : Workspace
        Restored workspace.
    critiques : list of Critique
        Reviews so far.
    facts : list of dict
        Code facts for every candidate.

    Returns
    -------
    dict
        Round, candidates with facts, reviews, papers, evidence count and budget used.
    """
    return {
        "round": ws.round,
        "candidates": facts,
        "reviews": [
            {
                "round": c.iteration,
                "best": c.candidate_id,
                "score": c.score,
                "llm_confidence": c.llm_confidence,
                "recommended_next": c.next_composition,
            }
            for c in critiques
        ],
        "papers": [{"paper_id": p.paper_id, "title": p.title} for p in ws.papers.values()],
        "n_evidence_notes": len(ws.insights),
        "tool_calls_used": ws.tool_calls,
    }
