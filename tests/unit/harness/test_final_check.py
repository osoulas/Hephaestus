"""Final re-check: the verdict rule and the finalist pipeline through stub tools (Section 9)."""

from typing import Any

from factories import candidate_state, workspace

from hephaestus.core.types import PhononResult, Submission
from hephaestus.harness import final_check
from hephaestus.harness.final_check import validate_finalists, verdict
from hephaestus.tools import stubs
from hephaestus.tools.registry import Workspace


def _spec(ws: Workspace) -> Any:
    """Return the demo spec: smaller band gap than 3.12 eV, E_hull <= 0.05, dynamically stable.

    Parameters
    ----------
    ws : Workspace
        Workspace.

    Returns
    -------
    SearchSpec
        Spec.
    """
    return ws.spec


def test_fc7_confirmed_when_final_agrees() -> None:
    """Same results at both fidelities give "confirmed", with screening and final values listed."""
    ws = workspace()
    v, checks, reasons = verdict(candidate_state(), candidate_state(gap=2.1), _spec(ws), _spec(ws))
    assert v == "confirmed" and reasons == []
    gap = next(c for c in checks if c.name == "band_gap")
    assert (gap.screening_value, gap.final_value, gap.delta) == (2.0, 2.1, 0.1)


def test_fc7_overturned_when_dynamic_stability_lost() -> None:
    """Passing phonons in screening but failing them in the final run overturns the result."""
    ws = workspace()
    v, _, reasons = verdict(candidate_state(), candidate_state(dyn=False), _spec(ws), _spec(ws))
    assert v == "overturned"
    assert reasons == ["dyn_stable passed in screening but fails in the final run"]


def test_fc7_overturned_when_target_or_hull_lost() -> None:
    """Missing the target or leaving the hull window at final fidelity overturns the result."""
    ws = workspace()
    assert verdict(candidate_state(), candidate_state(gap=3.3), _spec(ws), _spec(ws))[0] == "overturned"
    assert verdict(candidate_state(), candidate_state(e_hull=0.07), _spec(ws), _spec(ws))[0] == "overturned"


def test_fc7_final_reference_moves_the_target() -> None:
    """The target is judged against the reference recomputed at final fidelity (GP-11)."""
    ws = workspace()
    lower = ws.spec.model_copy(update={"target_value": 1.9})
    assert verdict(candidate_state(), candidate_state(), ws.spec, lower)[0] == "overturned"


def test_fc7_weakened_when_conservative_reading_lost() -> None:
    """Still past the target, but no longer once shifted by the uncertainty: weakened."""
    ws = workspace()
    v, _, reasons = verdict(candidate_state(gap=2.0), candidate_state(gap=2.6), _spec(ws), _spec(ws))
    assert v == "weakened"
    assert "the target is no longer met after shifting by the uncertainty" in reasons


def test_fc7_weakened_when_hull_rises() -> None:
    """A large E_hull rise that stays inside the window weakens the result."""
    ws = workspace()
    v, _, reasons = verdict(candidate_state(e_hull=0.0), candidate_state(e_hull=0.04), _spec(ws), _spec(ws))
    assert v == "weakened" and reasons == ["E_hull rose by 0.040 eV/atom"]


def test_fc7_missing_final_result_weakens_never_overturns() -> None:
    """A check the final run could not compute weakens instead of overturning."""
    ws = workspace()
    v, _, reasons = verdict(
        candidate_state(), candidate_state(dyn=None), _spec(ws), _spec(ws), ["phonons: boom"]
    )
    assert v == "weakened"
    assert "final step failed: phonons: boom" in reasons
    assert "dyn_stable not computed in the final run" in reasons
    assert verdict(candidate_state(), None, _spec(ws), _spec(ws), ["relax: boom"])[0] == "weakened"


def _campaign_ws(**cfg: Any) -> Workspace:
    """Return a workspace with two screened candidates from the stub tools.

    Parameters
    ----------
    **cfg
        ``RunConfig`` overrides.

    Returns
    -------
    Workspace
        Candidates ``c1-1`` (SrTi0.875Fe0.125O3) and ``c1-2`` (SrTi0.75Fe0.25O3), fully screened.
    """
    ws = workspace(**cfg)
    ws.spec = ws.spec.model_copy(update={"reference_material": None})
    handlers = stubs.stub_handlers()
    for cid, comp in (("c1-1", "SrTi0.875Fe0.125O3"), ("c1-2", "SrTi0.75Fe0.25O3")):
        ws.candidates[cid] = candidate_state(cid, comp, gap=None, e_hull=None, dyn=None)
        handlers["relax"](ws, cid, "mace-mp-0-small", 0.02)
        handlers["energy_above_hull"](ws, cid)
        handlers["phonons"](ws, cid, "fast")
        handlers["predict_property"](ws, cid, "band_gap")
    return ws


def test_fc1_fc6_winner_and_runner_up_validated_without_touching_screening() -> None:
    """Both finalists are re-checked with the final settings; the screening workspace is unchanged."""
    ws = _campaign_ws()
    before = ws.model_dump_json()
    reports = validate_finalists(
        ws, Submission(candidate_id="c1-1", runner_up_id="c1-2", gate_passed=True), stubs.stub_handlers()
    )
    assert [r.candidate_id for r in reports] == ["c1-1", "c1-2"]
    assert all(r.verdict == "confirmed" for r in reports)
    assert (
        reports[0]
        .notes[0]
        .startswith("Final settings: mace-mp-0-medium (float64), fmax 0.005 eV/Å, full phonons")
    )
    assert ws.model_dump_json() == before


def test_fc4_disorder_spread_reported_for_solid_solutions() -> None:
    """A template-built solid solution gets two more configurations and an energy spread."""
    ws = _campaign_ws()
    reports = validate_finalists(ws, Submission(candidate_id="c1-1", gate_passed=True), stubs.stub_handlers())
    spread = next(c for c in reports[0].checks if c.name == "disorder_spread_eV_per_atom")
    assert spread.final_value == 0.0
    assert any("3 disorder configurations" in n for n in reports[0].notes)


def test_fc7_overturn_detected_through_tools() -> None:
    """A full-phonon instability that the fast screening missed overturns the winner."""
    ws = _campaign_ws()
    handlers = stubs.stub_handlers()
    fast = handlers["phonons"]

    def full_unstable(ws: Workspace, candidate_id: str, mode: str) -> dict[str, Any]:
        """Run the stub, then report an instability for full phonons.

        Parameters
        ----------
        ws : Workspace
            Workspace.
        candidate_id : str
            Candidate.
        mode : str
            Phonon mode.

        Returns
        -------
        dict
            Stub result.
        """
        out = fast(ws, candidate_id, mode)
        if mode == "full":
            ws.candidates[candidate_id].phonon = PhononResult(
                mode="full", stable=False, min_frequency_THz=-1.5
            )
        return out

    handlers["phonons"] = full_unstable
    report = validate_finalists(ws, Submission(candidate_id="c1-1", gate_passed=True), handlers)[0]
    assert report.verdict == "overturned"


def test_fc9_short_budget_keeps_must_steps_and_drops_optional_items() -> None:
    """With no time left after the winner's Must steps, the Should and Could items are skipped and named."""
    ws = _campaign_ws()
    ticks = iter(range(0, 1000, 100))
    reports = validate_finalists(
        ws,
        Submission(candidate_id="c1-1", runner_up_id="c1-2", gate_passed=True),
        stubs.stub_handlers(),
        time_budget_s=50,
        clock=lambda: float(next(ticks)),
    )
    assert [r.candidate_id for r in reports] == ["c1-1"]
    winner = reports[0]
    assert winner.verdict == "confirmed"
    skipped = next(n for n in winner.notes if n.startswith("Skipped"))
    assert (
        "disorder spread (FC-4)" in skipped and "runner-up c1-2 (FC-6)" in skipped and "MD (FC-5)" in skipped
    )


def test_fc9_should_items_run_before_could_items() -> None:
    """When time runs out part-way, the runner-up (Should) is checked and MD (Could) is dropped."""
    ws = _campaign_ws()
    t = {"now": 0.0}
    handlers = stubs.stub_handlers()
    relax = handlers["relax"]

    def slow_relax(ws: Workspace, candidate_id: str, potential: str, fmax: float) -> dict[str, Any]:
        """Take 10 s per relaxation.

        Parameters
        ----------
        ws : Workspace
            Workspace.
        candidate_id : str
            Candidate.
        potential : str
            Potential.
        fmax : float
            Force threshold, eV/Å.

        Returns
        -------
        dict
            Stub result.
        """
        t["now"] += 10.0
        return relax(ws, candidate_id, potential, fmax)

    handlers["relax"] = slow_relax
    # winner 10 s, two disorder seeds 20 s, runner-up 10 s: exactly the 40 s budget, none left for MD.
    reports = validate_finalists(
        ws,
        Submission(candidate_id="c1-1", runner_up_id="c1-2", gate_passed=True),
        handlers,
        time_budget_s=40,
        clock=lambda: t["now"],
    )
    assert [r.candidate_id for r in reports] == ["c1-1", "c1-2"]
    skipped = next(n for n in reports[0].notes if n.startswith("Skipped"))
    assert "MD (FC-5)" in skipped and "FC-6" not in skipped and "FC-4" not in skipped


def test_gp11_relative_target_recomputed_with_final_potential(monkeypatch: Any) -> None:
    """For a relative target, the reference is rebuilt and relaxed with the final potential."""
    ws = _campaign_ws()
    ws.spec = workspace().spec
    used: list[str] = []
    handlers = stubs.stub_handlers()
    relax = handlers["relax"]

    def spy(ws: Workspace, candidate_id: str, potential: str, fmax: float) -> dict[str, Any]:
        """Record the reference's relax settings.

        Parameters
        ----------
        ws : Workspace
            Workspace.
        candidate_id : str
            Candidate.
        potential : str
            Potential.
        fmax : float
            Force threshold, eV/Å.

        Returns
        -------
        dict
            Stub result.
        """
        if candidate_id == "ref":
            used.append(f"{potential}@{fmax}")
        return relax(ws, candidate_id, potential, fmax)

    handlers["relax"] = spy
    report = validate_finalists(ws, Submission(candidate_id="c1-1", gate_passed=True), handlers)[0]
    assert used == ["mace-mp-0-medium@0.005"]
    assert any(n.startswith("Reference SrTiO3 recomputed with mace-mp-0-medium") for n in report.notes)
    assert final_check.FINAL_FMAX == 0.005
