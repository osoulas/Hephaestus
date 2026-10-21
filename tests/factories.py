"""Builders for workspaces and candidate states used across tests."""

from pathlib import Path

from hephaestus.core.config import RunConfig
from hephaestus.core.types import (
    Candidate,
    PhononResult,
    PropertyPrediction,
    Quantity,
    SearchSpec,
    StructureRecord,
)
from hephaestus.tools.registry import CandidateState, Workspace

FIXTURES = Path(__file__).parent / "fixtures"


def demo_spec() -> SearchSpec:
    """Return the demo spec: band gap below 3.12 eV (computed SrTiO3), stable.

    Returns
    -------
    SearchSpec
        Spec from ``tests/fixtures/search_spec.json``.
    """
    return SearchSpec.model_validate_json((FIXTURES / "search_spec.json").read_text(encoding="utf-8"))


def workspace(**cfg: object) -> Workspace:
    """Return an empty workspace for the demo spec.

    Parameters
    ----------
    **cfg
        ``RunConfig`` overrides.

    Returns
    -------
    Workspace
        Fresh workspace with run ID ``r-test``.
    """
    return Workspace(run_id="r-test", spec=demo_spec(), cfg=RunConfig(**cfg))  # type: ignore[arg-type]


def candidate_state(
    cid: str = "c1-1",
    composition: str = "SrTi0.875Fe0.125O3",
    gap: float | None = 2.0,
    unc: float = 0.65,
    e_hull: float | None = 0.01,
    dyn: bool | None = True,
    round: int = 1,
    converged: bool = True,
    in_domain: bool = True,
) -> CandidateState:
    """Return a candidate state with the given results; ``None`` means the step has not run.

    Parameters
    ----------
    cid : str
        Candidate ID.
    composition : str
        Formula.
    gap : float or None
        Predicted band gap, eV.
    unc : float
        Its uncertainty, eV.
    e_hull : float or None
        Energy above hull, eV/atom.
    dyn : bool or None
        Phonon verdict.
    round : int
        Review round.
    converged : bool
        Relaxation converged.
    in_domain : bool
        Inside the potential's and the property model's domains.

    Returns
    -------
    CandidateState
        The candidate state.
    """
    cand = Candidate(
        candidate_id=cid,
        iteration=round,
        composition=composition,
        structure_type="perovskite_cubic",
        rationale="test",
        evidence_ids=["mp:5229"],
    )
    rec = StructureRecord(
        candidate_id=cid, cif="", pmg_dict={}, source="template:perovskite_cubic", n_atoms=40
    )
    cs = CandidateState(
        candidate=cand,
        round=round,
        record=rec,
        structure={},
        energy_eV_per_atom=-7.0,
        converged=converged,
        potential="mace-mp-0-small",
        potential_in_domain=in_domain,
        e_hull=e_hull,
    )
    if dyn is not None:
        cs.phonon = PhononResult(
            mode="fast",
            stable=dyn,
            min_frequency_THz=0.1 if dyn else -2.0,
            instability_at=None if dyn else "R",
        )
        cs.phonon_details = {"verdict": "stable" if dyn else "unstable"}
    if gap is not None:
        cs.predictions["band_gap"] = PropertyPrediction(
            candidate_id=cid,
            property="band_gap",
            model="MEGNet",
            value=Quantity(value=gap, uncertainty=unc, units="eV", source="MEGNet"),
            in_domain=in_domain,
        )
    return cs
