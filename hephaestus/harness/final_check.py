"""Higher-fidelity re-check of the winner and runner-up, and the verdict rule (Section 9).

The re-check reuses the tool handlers on a deep copy of the workspace, so the screening results
stay as they were and the two can be compared. Each finalist is re-relaxed with the final
potential at tight force convergence, then gets full phonons with mode mapping, a fresh hull and
the target property (FC-1, FC-2). A relative target's reference is recomputed with the final
potential too, so candidate and reference stay on the same scale (GP-11).

Within the separate time budget (FC-9), work runs in priority order: the winner's Must steps,
the Should items (disorder spread FC-4, runner-up FC-6), then the Could items (MD FC-5, second
potential FC-3). The Must steps always run, since there is no verdict without them; each optional
item starts only if its estimated cost, from the step times just measured, fits in what remains.
Items that do not fit, or that a stop request cuts, are skipped and named in the notes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pymatgen.core import Composition

from hephaestus.core.config import budget_limit
from hephaestus.core.types import SearchSpec, Submission, ValidationCheck, ValidationReport
from hephaestus.harness.gates import pessimistic, target_met
from hephaestus.tools.registry import CandidateState, Handler, Workspace, execute, load_default_handlers

FINAL_FMAX = 0.005  # eV/Å (FC-1)
MD_PS = 2.0  # ps (FC-5)
DISORDER_SEEDS = (1, 2)  # extra configurations besides the finalist's own (FC-4)
EHULL_WEAKEN = 0.02  # eV/atom: a rise this large weakens a result that still passes
DISORDER_WEAKEN = 0.01  # eV/atom: an energy spread this large weakens the result (FC-4)
OVERTURN_CHECKS = ("target_met", "thermo_stable", "dyn_stable", "md_stable")
MD_COST_FACTOR = 10.0  # MD for MD_PS takes about this many final relaxations (FC-9 estimate)

StepHook = Callable[[str, str, dict[str, Any]], None]
"""``on_step(status, tool, info)``: status is ``started``, ``finished`` or ``failed``."""


def _timed(
    handlers: Mapping[str, Handler],
    times: dict[str, float],
    on_step: StepHook | None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Handler]:
    """Wrap handlers to record each tool's latest duration and report progress.

    Parameters
    ----------
    handlers : mapping
        Tool handlers.
    times : dict
        Filled with tool name to the latest wall time, in s.
    on_step : callable or None
        Progress callback.
    clock : callable
        Monotonic clock in s.

    Returns
    -------
    dict
        Wrapped handlers.
    """

    def wrap(name: str, fn: Handler) -> Handler:
        """Time and report one handler.

        Parameters
        ----------
        name : str
            Tool name.
        fn : callable
            Handler.

        Returns
        -------
        callable
            Wrapped handler.
        """

        def run(ws: Workspace, **args: Any) -> dict[str, Any]:
            """Call the handler, timing it and reporting start and end.

            Parameters
            ----------
            ws : Workspace
                Final-check workspace.
            **args
                Tool input.

            Returns
            -------
            dict
                The handler's result.
            """
            info = {
                "candidate_id": args.get("candidate_id"),
                "potential": args.get("potential"),
                "mode": args.get("mode"),
            }
            if on_step:
                on_step("started", name, info)
            t0 = clock()
            try:
                out = fn(ws, **args)
            except Exception:
                times[name] = clock() - t0
                if on_step:
                    on_step("failed", name, {**info, "wall_s": round(times[name], 1)})
                raise
            times[name] = clock() - t0
            if on_step:
                on_step("finished", name, {**info, "wall_s": round(times[name], 1)})
            return out

        return run

    return {name: wrap(name, fn) for name, fn in handlers.items()}


@dataclass
class _Finalist:
    """Work record for one finalist during the re-check."""

    cid: str
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    extra: list[ValidationCheck] = field(default_factory=list)
    weak: list[str] = field(default_factory=list)


def _call(
    fws: Workspace, handlers: Mapping[str, Handler], name: str, args: dict[str, Any]
) -> tuple[dict[str, Any] | None, str]:
    """Run one tool handler on the final workspace.

    Parameters
    ----------
    fws : Workspace
        Final-check workspace.
    handlers : mapping
        Tool handlers.
    name : str
        Tool name.
    args : dict
        Tool input.

    Returns
    -------
    result : dict or None
        The result, or ``None`` when the call failed.
    error : str
        The error message, or ``""``.
    """
    result, is_error = execute(fws, name, args, handlers)
    if is_error:
        return None, str(result.get("error"))[:200]
    return result, ""


def _must_steps(fws: Workspace, handlers: Mapping[str, Handler], fin: _Finalist) -> None:
    """Re-relax, run full phonons, rebuild the hull and predict the target (FC-1, FC-2).

    Phonons run before the hull and the prediction, so a distortion found by mode mapping is
    reflected in both.

    Parameters
    ----------
    fws : Workspace
        Final-check workspace.
    handlers : mapping
        Tool handlers.
    fin : _Finalist
        Finalist; failures and notes are added to it.
    """
    cfg, cid = fws.cfg, fin.cid
    steps: list[tuple[str, dict[str, Any]]] = [
        ("relax", {"candidate_id": cid, "potential": cfg.final_potential, "fmax": FINAL_FMAX})
    ]
    if cfg.run_phonons:
        steps.append(("phonons", {"candidate_id": cid, "mode": "full"}))
    if cfg.run_hull:
        steps.append(("energy_above_hull", {"candidate_id": cid}))
    steps.append(("predict_property", {"candidate_id": cid, "property": fws.spec.target_property}))
    supercell = None
    for name, args in steps:
        out, err = _call(fws, handlers, name, args)
        if out is None:
            fin.failures.append(f"{name}: {err}")
            if name == "relax":
                return
            continue
        if name == "phonons":
            supercell = out.get("supercell")
            if out.get("note"):
                fin.notes.append(f"Final phonons: {out['note']}")
    phonon_txt = f"full phonons (supercell {supercell})" if supercell else "no phonons"
    fin.notes.insert(
        0,
        f"Final settings: {cfg.final_potential} ({cfg.final_dtype}), fmax {FINAL_FMAX} eV/Å, {phonon_txt} (FC-1, FC-2).",
    )


def _clone(fws: Workspace, cid: str, tag: str) -> str:
    """Add a temporary copy of a candidate to the final workspace.

    Parameters
    ----------
    fws : Workspace
        Final-check workspace.
    cid : str
        Candidate to copy.
    tag : str
        Suffix for the copy's ID.

    Returns
    -------
    str
        The copy's ID.
    """
    new = f"{cid}/{tag}"
    cs = fws.candidates[cid].model_copy(deep=True)
    cs.candidate = cs.candidate.model_copy(update={"candidate_id": new})
    fws.candidates[new] = cs
    return new


def is_disordered(cs: CandidateState) -> bool:
    """Whether a candidate is a solid solution built from a template, so disorder configurations differ.

    Parameters
    ----------
    cs : CandidateState
        Candidate state.

    Returns
    -------
    bool
        True for a template build of a composition with fractional amounts.
    """
    if cs.record is None or not cs.record.source.startswith("template:"):
        return False
    amounts = Composition(cs.candidate.composition).get_el_amt_dict().values()
    return any(abs(n - round(n)) > 1e-6 for n in amounts)


def disorder_spread(fws: Workspace, handlers: Mapping[str, Handler], fin: _Finalist) -> None:
    """Relax further disorder configurations and report the energy spread (FC-4).

    Parameters
    ----------
    fws : Workspace
        Final-check workspace; temporary candidates are removed afterwards.
    handlers : mapping
        Tool handlers.
    fin : _Finalist
        Finalist; the spread check and notes are added to it.
    """
    cs = fws.candidates[fin.cid]
    if not is_disordered(cs):
        fin.notes.append("Ordered compound: no disorder configurations to compare (FC-4).")
        return
    energies = [cs.energy_eV_per_atom] if cs.energy_eV_per_atom is not None else []
    for seed in DISORDER_SEEDS:
        tmp = _clone(fws, fin.cid, f"seed{seed}")
        steps: list[tuple[str, dict[str, Any]]] = [
            ("build_structure", {"candidate_id": tmp, "source": "template", "mp_id": None, "seed": seed}),
            ("relax", {"candidate_id": tmp, "potential": fws.cfg.final_potential, "fmax": FINAL_FMAX}),
        ]
        try:
            for name, args in steps:
                out, err = _call(fws, handlers, name, args)
                if out is None:
                    fin.notes.append(f"Disorder seed {seed}: {name} failed: {err}")
                    break
            else:
                energies.append(fws.candidates[tmp].energy_eV_per_atom)
        finally:
            fws.candidates.pop(tmp, None)
    values = [e for e in energies if e is not None]
    if len(values) < 2:
        fin.notes.append("Disorder spread not available: fewer than two configurations relaxed (FC-4).")
        return
    spread = max(values) - min(values)
    fin.extra.append(ValidationCheck(name="disorder_spread_eV_per_atom", final_value=round(spread, 5)))
    fin.notes.append(
        f"{len(values)} disorder configurations, energy spread {spread * 1000:.1f} meV/atom (FC-4)."
    )
    if spread > DISORDER_WEAKEN:
        fin.weak.append(f"disorder spread {spread:.3f} eV/atom above {DISORDER_WEAKEN}")


def run_md(fws: Workspace, handlers: Mapping[str, Handler], fin: _Finalist) -> None:
    """MD at the operating temperature on the final structure (FC-5).

    Parameters
    ----------
    fws : Workspace
        Final-check workspace.
    handlers : mapping
        Tool handlers.
    fin : _Finalist
        Finalist; a failure becomes a note.
    """
    out, err = _call(fws, handlers, "run_md", {"candidate_id": fin.cid, "duration_ps": MD_PS})
    if out is None:
        fin.notes.append(f"Final MD failed: {err}")


def cross_check(fws: Workspace, handlers: Mapping[str, Handler], fin: _Finalist, potential: str) -> None:
    """Relax and hull-check with a second potential and compare (FC-3).

    Parameters
    ----------
    fws : Workspace
        Final-check workspace; the temporary candidate is removed afterwards.
    handlers : mapping
        Tool handlers.
    fin : _Finalist
        Finalist; the comparison is added to it.
    potential : str
        Second potential, e.g. ``"chgnet"``.
    """
    tmp = _clone(fws, fin.cid, potential)
    try:
        for name, args in (
            ("relax", {"candidate_id": tmp, "potential": potential, "fmax": FINAL_FMAX}),
            ("energy_above_hull", {"candidate_id": tmp}),
        ):
            out, err = _call(fws, handlers, name, args)
            if out is None:
                fin.notes.append(f"Cross-check with {potential}: {name} failed: {err}")
                return
        main_ehull = fws.candidates[fin.cid].e_hull
        other = fws.candidates[tmp].e_hull
        fin.extra.append(
            ValidationCheck(
                name=f"e_hull_eV_per_atom ({fws.cfg.final_potential} vs {potential})",
                screening_value=main_ehull,
                final_value=other,
                delta=None if main_ehull is None or other is None else round(other - main_ehull, 5),
            )
        )
        thr = fws.spec.stability_thresholds.e_hull_max_eV_per_atom
        if main_ehull is not None and other is not None and (main_ehull <= thr) != (other <= thr):
            fin.weak.append(f"{potential} disagrees on thermodynamic stability")
        fin.notes.append(f"Second potential {potential}: E_hull {other:.3f} eV/atom (FC-3).")
    finally:
        fws.candidates.pop(tmp, None)


def final_reference(fws: Workspace, handlers: Mapping[str, Handler]) -> tuple[SearchSpec, list[str]]:
    """Recompute a relative target's reference with the final potential (GP-11).

    Parameters
    ----------
    fws : Workspace
        Final-check workspace.
    handlers : mapping
        Tool handlers.

    Returns
    -------
    spec : SearchSpec
        Spec whose ``target_value`` is the reference at final fidelity; the screening spec when
        the target is absolute or the reference failed.
    notes : list of str
        What happened.
    """
    spec = fws.spec
    if not spec.reference_material:
        return spec, []
    from hephaestus.harness.runner import resolve_reference

    probe = fws.model_copy(update={"spec": spec.model_copy(update={"reference_value": None})})
    new, errors = resolve_reference(probe, handlers, potential=fws.cfg.final_potential, fmax=FINAL_FMAX)
    if errors or new.reference_value is None:
        reason = errors[0].message if errors else "no value"
        return spec, [
            f"Reference {spec.reference_material} not recomputed at final fidelity ({reason}); screening target used."
        ]
    old = spec.reference_value.value if spec.reference_value else spec.target_value
    return spec.model_copy(
        update={"reference_value": new.reference_value, "target_value": new.target_value}
    ), [
        f"Reference {spec.reference_material} recomputed with {fws.cfg.final_potential}: "
        f"{old:.3g} → {new.reference_value.value:.3g} {new.reference_value.units}."
    ]


def facts(cs: CandidateState | None, spec: SearchSpec) -> dict[str, Any]:
    """Collect the quantities compared between screening and final; ``None`` where a step did not run.

    Parameters
    ----------
    cs : CandidateState or None
        Candidate state.
    spec : SearchSpec
        Target and thresholds for this fidelity.

    Returns
    -------
    dict
        ``target``, ``target_met``, ``target_met_conservative``, ``e_hull``, ``thermo_stable``,
        ``dyn_stable``, ``min_frequency_THz``, ``md_stable`` and ``space_group``.
    """
    if cs is None:
        return {}
    pred = cs.predictions.get(spec.target_property)
    e_hull = None if cs.e_hull is None or cs.hull_stale else cs.e_hull
    thr = spec.stability_thresholds.e_hull_max_eV_per_atom
    return {
        "target": None if pred is None else pred.value.value,
        "target_met": None if pred is None else target_met(pred.value.value, spec),
        "target_met_conservative": None if pred is None else target_met(pessimistic(pred.value, spec), spec),
        "e_hull": e_hull,
        "thermo_stable": None if e_hull is None else e_hull <= thr,
        "dyn_stable": None if cs.phonon is None else cs.phonon.stable,
        "min_frequency_THz": None if cs.phonon is None else cs.phonon.min_frequency_THz,
        "md_stable": None if cs.md is None else bool(cs.md.get("stable")),
        "space_group": cs.space_group,
    }


def verdict(
    screening: CandidateState,
    final: CandidateState | None,
    spec: SearchSpec,
    final_spec: SearchSpec,
    failures: list[str] | None = None,
    weak: list[str] | None = None,
) -> tuple[str, list[ValidationCheck], list[str]]:
    """Apply the verdict rule to one finalist (FC-7).

    - **overturned**: a check the screening run passed (target met, thermodynamically or
      dynamically stable, MD stable) fails in the final run.
    - **weakened**: nothing overturned, but the conservative target reading was lost, E_hull rose
      by more than ``EHULL_WEAKEN``, a final step failed, or an extra check raised a concern.
    - **confirmed**: otherwise.

    A check the final run could not compute never overturns; it weakens instead.

    Parameters
    ----------
    screening : CandidateState
        Screening results.
    final : CandidateState or None
        Final results; ``None`` when the re-relaxation failed.
    spec : SearchSpec
        Screening target.
    final_spec : SearchSpec
        Target at final fidelity (reference recomputed with the final potential).
    failures : list of str, optional
        Final steps that failed.
    weak : list of str, optional
        Concerns from the extra checks (disorder spread, second potential).

    Returns
    -------
    verdict : {"confirmed", "weakened", "overturned"}
        The verdict.
    checks : list of ValidationCheck
        Screening and final values.
    reasons : list of str
        Why the verdict is not "confirmed".
    """
    s, f = facts(screening, spec), facts(final, final_spec)
    checks = []
    for key, name in (
        ("target", spec.target_property),
        ("target_met", "target_met"),
        ("target_met_conservative", "target_met_conservative"),
        ("e_hull", "e_hull_eV_per_atom"),
        ("thermo_stable", "thermo_stable"),
        ("dyn_stable", "dyn_stable"),
        ("min_frequency_THz", "min_frequency_THz"),
        ("md_stable", "md_stable"),
    ):
        a, b = s.get(key), f.get(key)
        if a is None and b is None:
            continue
        delta = None
        if isinstance(a, float | int) and isinstance(b, float | int) and not isinstance(a, bool):
            delta = round(float(b) - float(a), 5)
        checks.append(ValidationCheck(name=name, screening_value=a, final_value=b, delta=delta))
    lost = [k for k in OVERTURN_CHECKS if s.get(k) is True and f.get(k) is False]
    if lost:
        return "overturned", checks, [f"{k} passed in screening but fails in the final run" for k in lost]
    reasons = [f"final step failed: {x}" for x in failures or []]
    reasons += [
        f"{k} not computed in the final run" for k in OVERTURN_CHECKS if s.get(k) is True and f.get(k) is None
    ]
    if s.get("target_met_conservative") is True and f.get("target_met_conservative") is False:
        reasons.append("the target is no longer met after shifting by the uncertainty")
    if (
        s.get("e_hull") is not None
        and f.get("e_hull") is not None
        and f["e_hull"] - s["e_hull"] > EHULL_WEAKEN
    ):
        reasons.append(f"E_hull rose by {f['e_hull'] - s['e_hull']:.3f} eV/atom")
    reasons += weak or []
    return ("weakened" if reasons else "confirmed"), checks, reasons


def validate_finalists(
    ws: Workspace,
    submission: Submission,
    handlers: Mapping[str, Handler] | None = None,
    time_budget_s: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    on_step: StepHook | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[ValidationReport]:
    """Re-check the winner and the runner-up at higher fidelity (FC-1 to FC-7, FC-9).

    Parameters
    ----------
    ws : Workspace
        Workspace after the campaign; left unchanged.
    submission : Submission
        Names the winner and the runner-up.
    handlers : mapping, optional
        Tool handlers; defaults to the registered ones.
    time_budget_s : float, optional
        Budget in s; defaults to ``ws.cfg.final_time_budget_s``. The winner's Must steps always
        run; each optional item runs only if its estimated cost fits in what remains (FC-9).
    clock : callable
        Monotonic clock in s, injectable for tests.
    on_step : callable, optional
        Called before and after each tool call, for live progress.
    should_stop : callable, optional
        Returns True when the user asked the run to stop; optional items are then skipped.

    Returns
    -------
    list of ValidationReport
        Winner first, then the runner-up when it was checked.
    """
    times: dict[str, float] = {}
    handlers = _timed(handlers if handlers is not None else load_default_handlers(), times, on_step, clock)
    budget = budget_limit(ws.cfg.final_time_budget_s if time_budget_s is None else time_budget_s)
    start = clock()
    remaining = lambda: budget - (clock() - start)  # noqa: E731
    fws = ws.model_copy(deep=True)
    final_spec, ref_notes = final_reference(fws, handlers)

    winner = _Finalist(submission.candidate_id)
    _must_steps(fws, handlers, winner)
    must_cost = clock() - start
    relaxed = not any(x.startswith("relax:") for x in winner.failures)
    skipped: list[str] = []

    def optional(label: str, run: Callable[[], None], cost: float = 0.0) -> None:
        """Run an optional item if the winner relaxed and the budget allows (FC-9).

        Parameters
        ----------
        label : str
            Name for the skipped list.
        run : callable
            The item.
        cost : float
            Time it needs, in s; the item runs only if at least this much remains.
        """
        if not relaxed:
            skipped.append(f"{label}, as the final relaxation failed")
        elif should_stop is not None and should_stop():
            skipped.append(f"{label}, as the run was stopped")
        elif (left := remaining()) > 0 and left >= cost:
            run()
        else:
            skipped.append(f"{label} (needs about {cost:.0f} s, {max(left, 0):.0f} s left)")

    # Should items first, then Could items, so a short budget drops the Could items (FC-9).
    relax_s = times.get("relax", 0.0)
    disordered = winner.cid in fws.candidates and is_disordered(fws.candidates[winner.cid])
    optional(
        "disorder spread (FC-4)",
        lambda: disorder_spread(fws, handlers, winner),
        cost=len(DISORDER_SEEDS) * relax_s if disordered else 0.0,
    )
    finalists = [winner]
    rid = submission.runner_up_id
    if rid and rid in ws.candidates and rid != winner.cid:
        runner_up = _Finalist(rid)

        def check_runner_up() -> None:
            """Run the Must steps on the runner-up (FC-6)."""
            _must_steps(fws, handlers, runner_up)
            finalists.append(runner_up)

        optional(f"runner-up {rid} (FC-6)", check_runner_up, cost=must_cost)
    optional("MD (FC-5)", lambda: run_md(fws, handlers, winner), cost=MD_COST_FACTOR * relax_s)
    second = ws.cfg.cross_check_potential
    if second and ws.cfg.run_hull:
        optional(
            "second potential (FC-3)",
            lambda: cross_check(fws, handlers, winner, second),
            cost=relax_s + times.get("energy_above_hull", 0.0),
        )

    reports = []
    for fin in finalists:
        done = not any(x.startswith("relax:") for x in fin.failures)
        v, checks, reasons = verdict(
            ws.candidates[fin.cid],
            fws.candidates[fin.cid] if done else None,
            ws.spec,
            final_spec,
            fin.failures,
            fin.weak,
        )
        notes = [*fin.notes, *ref_notes, *(f"{v.capitalize()}: {r}." for r in reasons)]
        if fin is winner and skipped:
            notes.append(f"Skipped (budget {budget:.0f} s): {'; '.join(skipped)} (FC-9).")
        reports.append(
            ValidationReport(candidate_id=fin.cid, checks=checks + fin.extra, verdict=v, notes=notes)
        )
    return reports
