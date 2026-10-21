"""Shared data types: the frozen contracts between sections (SYS-5).

Pydantic v2 models that round-trip through JSON with no loss. Agreed by the whole team in
hour 0; see the *Shared data types* table in ``docs/REQUIREMENTS.md``. Changing a type needs
all three people to agree.

Conventions: units are in field names where ambiguous, compositions are pymatgen formula
strings, times are ISO 8601 UTC, and errors are ``ErrorRecord`` objects collected in lists.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hephaestus.core.config import RunConfig

Direction = Literal["maximise", "minimise", "hit"]
FindingType = Literal["positive", "negative", "mechanism", "synthesis"]
Novelty = Literal["known", "variant", "novel"]
Feasibility = Literal["easy", "moderate", "difficult", "unknown"]
Severity = Literal["critical", "major", "minor", "info"]
EventStatus = Literal["started", "finished", "failed"]
RunStatus = Literal[
    "created", "needs_input", "running", "final_check", "writing_brief", "finished", "failed", "stopped"
]


def utcnow() -> datetime:
    """Return the current time, timezone-aware in UTC.

    Returns
    -------
    datetime.datetime
        Now, with ``tzinfo=UTC``.
    """
    return datetime.now(UTC)


class Record(BaseModel):
    """Base class for every shared record: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


class ErrorRecord(Record):
    """A recorded, non-fatal failure (SYS-6)."""

    step: str
    code: str
    message: str


class Quantity(Record):
    """A number with its uncertainty, units and source (SYS-4).

    ``uncertainty`` is ``None`` only when no uncertainty is available, which must be said.
    """

    value: float
    uncertainty: float | None
    units: str
    source: str


class Conditions(Record):
    """Operating conditions (GP-7 defaults)."""

    temperature_K: float = 300.0
    pressure_GPa: float = 0.0001


class StabilityThresholds(Record):
    """Thresholds for the code stability verdicts (SE-13, GP-12)."""

    e_hull_max_eV_per_atom: float = 0.05
    phonon_min_THz: float = -0.1


class SearchSpec(Record):
    """Structured search specification produced by the goal parser (Section 1)."""

    raw_goal: str
    title: str = ""  # short run title proposed by the parser, e.g. "Lead-free perovskite below SrTiO3's gap"
    material_family: str
    elements: list[str] = Field(default_factory=list)
    excluded_elements: list[str] = Field(default_factory=list)
    target_property: str
    target_value: float | None = None
    target_units: str
    direction: Direction
    target_tolerance: float | None = None  # only for direction "hit"
    reference_material: str | None = None
    reference_value: Quantity | None = None
    conditions: Conditions = Field(default_factory=Conditions)
    hard_constraints: list[str] = Field(default_factory=list)
    soft_preferences: list[str] = Field(default_factory=list)
    stability_thresholds: StabilityThresholds = Field(default_factory=StabilityThresholds)
    assumptions: list[str] = Field(default_factory=list)


class Paper(Record):
    """One literature record from Valency Bond (Section 2)."""

    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    source: str = "arxiv"
    abstract: str = ""
    citation_count: int | None = None
    relevance: float = 0.0
    found_via: str = "semantic"


class Insight(Record):
    """A finding recorded with ``note_evidence`` (LM-13)."""

    composition: str | None = None
    property: str
    value: Quantity | None = None
    conditions: str = ""
    finding_type: FindingType
    summary: str
    paper_id: str


class LiteratureContext(Record):
    """Literature aggregated over a run, for the reviewer and the brief."""

    papers: list[Paper] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)
    property_map: dict[str, Quantity] = Field(default_factory=dict)
    failed_approaches: list[str] = Field(default_factory=list)
    mechanisms: list[str] = Field(default_factory=list)
    synthesis_routes: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    field_trend: dict[int, int] = Field(default_factory=dict)
    coverage_note: str = ""


class Candidate(Record):
    """A proposed composition plus structure type (Section 3)."""

    candidate_id: str
    iteration: int
    composition: str
    structure_type: str
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)
    prior_estimate: Quantity | None = None
    novelty: Novelty = "variant"
    synthesis_feasibility: Feasibility = "unknown"
    risks: list[str] = Field(default_factory=list)


class StructureRecord(Record):
    """An atomic structure, stored as CIF and as a pymatgen dict (Section 4)."""

    candidate_id: str
    cif: str
    pmg_dict: dict[str, Any]
    source: str
    supercell: list[int] = Field(default_factory=lambda: [1, 1, 1])
    n_atoms: int
    disorder_seed: int | None = None
    composition_error: float = 0.0


class PhononResult(Record):
    """Phonon stability result (SE-6 to SE-8, SE-13)."""

    mode: Literal["fast", "full"]
    stable: bool
    min_frequency_THz: float
    instability_at: str | None = None
    supercell: list[int] = Field(default_factory=lambda: [1, 1, 1])


class Lattice(Record):
    """Lattice parameters in Å and degrees."""

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float


class SimulationResult(Record):
    """Everything the Simulation Engine knows about one candidate (Section 5).

    Fields are ``None`` when the step has not run or failed; the failure is in ``errors``.
    """

    candidate_id: str
    potential: str | None = None
    converged: bool = False
    relaxed: StructureRecord | None = None
    space_group: str | None = None
    lattice: Lattice | None = None
    density_g_cm3: float | None = None
    energy_eV_per_atom: float | None = None
    formation_energy_eV_per_atom: float | None = None
    energy_above_hull_eV_per_atom: float | None = None
    thermo_stable: bool = False
    phonons: PhononResult | None = None
    md_stable: bool | None = None
    domain_note: str = ""
    wall_time_s: float = 0.0
    errors: list[ErrorRecord] = Field(default_factory=list)


class PropertyPrediction(Record):
    """A predicted property with its uncertainty and domain flag (Section 6)."""

    candidate_id: str
    property: str
    value: Quantity
    model: str
    in_domain: bool = True
    literature_value: Quantity | None = None


class Finding(Record):
    """One reviewer finding (Section 7)."""

    severity: Severity
    category: str
    text: str
    suggestion: str = ""
    paper_ids: list[str] = Field(default_factory=list)


class Critique(Record):
    """The result of one review round (Section 7). Checks and score are code; the rest is the reviewer."""

    iteration: int
    candidate_id: str | None
    target_met_central: bool = False
    target_met_conservative: bool = False
    thermo_stable: bool = False
    dyn_stable: bool = False
    score: int = 0
    score_breakdown: dict[str, int] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    next_composition: str = ""
    next_rationale: str = ""
    stop_recommended: bool = False
    stop_reason: str | None = None
    llm_confidence: int | None = None
    llm_confidence_reason: str | None = None


class ValidationCheck(Record):
    """One quantity compared between screening and the final re-check."""

    name: str
    screening_value: float | bool | None = None
    final_value: float | bool | None = None
    delta: float | None = None


class ValidationReport(Record):
    """Final re-check of one finalist (Section 9)."""

    candidate_id: str
    checks: list[ValidationCheck] = Field(default_factory=list)
    verdict: Literal["confirmed", "weakened", "overturned"]
    notes: list[str] = Field(default_factory=list)


class Brief(Record):
    """The experimental brief (Section 10)."""

    run_id: str
    markdown: str
    pdf_path: str | None = None
    cited_paper_ids: list[str] = Field(default_factory=list)


class ProgressEvent(Record):
    """One event on the run's stream (OR-7). See *Event payloads* in ``docs/TOOLS.md``."""

    run_id: str
    seq: int = 0
    time: datetime = Field(default_factory=utcnow)
    section: str
    status: EventStatus
    iteration: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class Submission(Record):
    """The orchestrator's accepted ``submit_final`` (OR-3)."""

    candidate_id: str
    runner_up_id: str | None = None
    justification: str = ""
    gate_passed: bool
    accept_not_met: bool = False
    picked_by: Literal["orchestrator", "code"] = "orchestrator"


class IterationRecord(Record):
    """Everything from one review round (OR-5)."""

    candidates: list[Candidate] = Field(default_factory=list)
    structures: list[StructureRecord] = Field(default_factory=list)
    sims: list[SimulationResult] = Field(default_factory=list)
    predictions: list[PropertyPrediction] = Field(default_factory=list)
    critique: Critique | None = None


class RunState(Record):
    """The full record of one run, rewritten after every tool call (``state.json``)."""

    run_id: str
    raw_goal: str = ""
    status: RunStatus = "created"
    created_at: datetime = Field(default_factory=utcnow)
    config: RunConfig = Field(default_factory=RunConfig)
    spec: SearchSpec | None = None
    literature: LiteratureContext | None = None
    iterations: list[IterationRecord] = Field(default_factory=list)
    best: str | None = None
    submission: Submission | None = None
    stop_reason: str | None = None
    validation: list[ValidationReport] = Field(default_factory=list)
    brief: Brief | None = None
    errors: list[ErrorRecord] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    versions: dict[str, str] = Field(default_factory=dict)
