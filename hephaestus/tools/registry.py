"""Tool schemas (``TOOLS``), dispatch, ``ToolError`` and the per-run ``Workspace``.

The contract between the harness and the tools; see ``docs/TOOLS.md``. Every handler is
``handler(ws: Workspace, **args) -> dict``. Tool modules register handlers by name with
``@register("name")``; the harness registers the run-control tools itself.
"""

from __future__ import annotations

import importlib
import traceback
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hephaestus.core.config import RunConfig
from hephaestus.core.types import (
    Candidate,
    ErrorRecord,
    Insight,
    Lattice,
    Paper,
    PhononResult,
    PropertyPrediction,
    SearchSpec,
    SimulationResult,
    StructureRecord,
    Submission,
)
from hephaestus.tools.prototypes import MP_STRUCTURE_TYPE, PROTOTYPES

Handler = Callable[..., dict[str, Any]]
HANDLERS: dict[str, Handler] = {}
TOOL_MODULES = ["literature", "materials_project", "structures", "simulation", "properties"]


class ToolError(Exception):
    """A tool failure reported to the LLM as ``{"error": message}``, never raised past dispatch."""


def register(name: str) -> Callable[[Handler], Handler]:
    """Register a handler under a tool name.

    Parameters
    ----------
    name : str
        Tool name, as in ``TOOLS``.

    Returns
    -------
    callable
        Decorator returning the handler unchanged.
    """

    def deco(fn: Handler) -> Handler:
        """Store ``fn`` in ``HANDLERS``.

        Parameters
        ----------
        fn : callable
            Handler ``fn(ws, **args) -> dict``.

        Returns
        -------
        callable
            ``fn``.
        """
        HANDLERS[name] = fn
        return fn

    return deco


def load_default_handlers() -> dict[str, Handler]:
    """Import every tool module so its handlers register, and return the registry.

    Returns
    -------
    dict
        Tool name to handler.
    """
    for mod in TOOL_MODULES:
        importlib.import_module(f"hephaestus.tools.{mod}")
    return dict(HANDLERS)


class CandidateState(BaseModel):
    """Everything known about one candidate during a run (``docs/TOOLS.md``)."""

    model_config = ConfigDict(extra="forbid")

    candidate: Candidate
    round: int
    record: StructureRecord | None = None
    structure: dict[str, Any] | None = None  # current pymatgen dict: relaxed, or distorted after mode mapping
    energy_eV_per_atom: float | None = None
    converged: bool = False
    potential: str | None = None
    space_group: str | None = None
    lattice: Lattice | None = None
    density_g_cm3: float | None = None
    domain_note: str = ""
    potential_in_domain: bool | None = None
    e_hull: float | None = None
    e_form: float | None = None
    decomposes_to: list[str] = Field(default_factory=list)
    hull_stale: bool = False
    phonon: PhononResult | None = None
    phonon_details: dict[str, Any] = Field(default_factory=dict)
    predictions: dict[str, PropertyPrediction] = Field(default_factory=dict)
    md: dict[str, Any] | None = None
    wall_time_s: float = 0.0
    errors: list[ErrorRecord] = Field(default_factory=list)


class Workspace(BaseModel):
    """Per-run state shared by every tool; snapshotted to ``workspace.json`` after each call (OR-6)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    spec: SearchSpec
    cfg: RunConfig
    round: int = 1
    candidates: dict[str, CandidateState] = Field(default_factory=dict)
    papers: dict[str, Paper] = Field(default_factory=dict)
    insights: list[Insight] = Field(default_factory=list)
    trend: dict[int, int] = Field(default_factory=dict)
    queries: list[str] = Field(default_factory=list)
    tool_calls: int = 0
    compute_s: float = 0.0
    submitted: Submission | None = None

    def get(self, candidate_id: str) -> CandidateState:
        """Return a candidate's state or raise a ``ToolError`` for the LLM.

        Parameters
        ----------
        candidate_id : str
            Candidate ID such as ``"c1-2"``.

        Returns
        -------
        CandidateState
            The candidate's state.

        Raises
        ------
        ToolError
            If no such candidate was proposed.
        """
        try:
            return self.candidates[candidate_id]
        except KeyError:
            known = ", ".join(self.candidates) or "none"
            raise ToolError(f"Unknown candidate_id {candidate_id!r}; known: {known}.") from None

    def sim_result(self, candidate_id: str) -> SimulationResult:
        """Assemble a ``SimulationResult`` from a candidate's state.

        Parameters
        ----------
        candidate_id : str
            Candidate ID.

        Returns
        -------
        SimulationResult
            Current results; ``thermo_stable`` uses the spec threshold and a fresh hull.
        """
        cs = self.get(candidate_id)
        relaxed = None
        if cs.record is not None and cs.structure is not None:
            relaxed = cs.record.model_copy(update={"pmg_dict": cs.structure})
        thr = self.spec.stability_thresholds.e_hull_max_eV_per_atom
        return SimulationResult(
            candidate_id=candidate_id,
            potential=cs.potential,
            converged=cs.converged,
            relaxed=relaxed,
            space_group=cs.space_group,
            lattice=cs.lattice,
            density_g_cm3=cs.density_g_cm3,
            energy_eV_per_atom=cs.energy_eV_per_atom,
            formation_energy_eV_per_atom=cs.e_form,
            energy_above_hull_eV_per_atom=cs.e_hull,
            thermo_stable=cs.e_hull is not None and not cs.hull_stale and cs.e_hull <= thr,
            phonons=cs.phonon,
            md_stable=None if cs.md is None else bool(cs.md.get("stable")),
            domain_note=cs.domain_note,
            wall_time_s=cs.wall_time_s,
            errors=list(cs.errors),
        )


def execute(
    ws: Workspace, name: str, args: dict[str, Any], handlers: Mapping[str, Handler] | None = None
) -> tuple[dict[str, Any], bool]:
    """Run one tool call; never raises (``docs/TOOLS.md``: a tool never crashes the run).

    Parameters
    ----------
    ws : Workspace
        The run's workspace.
    name : str
        Tool name.
    args : dict
        Tool input from the LLM.
    handlers : mapping, optional
        Name to handler; defaults to ``HANDLERS``.

    Returns
    -------
    result : dict
        The handler's result, or ``{"error": message}``.
    is_error : bool
        True when the call failed.
    """
    table = HANDLERS if handlers is None else handlers
    fn = table.get(name)
    if fn is None:
        return {"error": f"Unknown or unavailable tool {name!r}."}, True
    try:
        return fn(ws, **args), False
    except ToolError as exc:
        return {"error": str(exc)}, True
    except TypeError as exc:
        if "argument" in str(exc):
            return {"error": f"Bad arguments for {name}: {exc}"}, True
        return {"error": f"{type(exc).__name__}: {exc}", "trace": _short_trace()}, True
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "trace": _short_trace()}, True


def _short_trace() -> str:
    """Return the last few lines of the current traceback.

    Returns
    -------
    str
        At most 600 characters of traceback.
    """
    return traceback.format_exc()[-600:]


# ---------------------------------------------------------------------------
# Schemas. strict: every property required; optional values are nullable.
# ---------------------------------------------------------------------------


def _obj(props: dict[str, Any]) -> dict[str, Any]:
    """Build a strict object schema in which every property is required.

    Parameters
    ----------
    props : dict
        Property name to JSON schema.

    Returns
    -------
    dict
        JSON schema with ``additionalProperties: false``.
    """
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _tool(name: str, description: str, props: dict[str, Any]) -> dict[str, Any]:
    """Build one tool definition.

    Parameters
    ----------
    name : str
        Tool name.
    description : str
        What the tool does, for the LLM.
    props : dict
        Input properties.

    Returns
    -------
    dict
        ``{name, description, input_schema, strict}``.
    """
    return {"name": name, "description": description, "input_schema": _obj(props), "strict": True}


STR = {"type": "string"}
STR_NULL = {"type": ["string", "null"]}
NUM_NULL = {"type": ["number", "null"]}
INT = {"type": "integer"}
BOOL = {"type": "boolean"}
STR_LIST = {"type": "array", "items": {"type": "string"}}
CID = {"type": "string", "description": "Candidate ID, e.g. c1-2."}


def _enum(*values: str) -> dict[str, Any]:
    """Build a string enum schema.

    Parameters
    ----------
    *values : str
        Allowed values.

    Returns
    -------
    dict
        JSON schema.
    """
    return {"type": "string", "enum": list(values)}


POTENTIALS = ["mace-mp-0-small", "mace-mp-0-medium", "mace-mpa-0", "mace-omat-0", "chgnet"]

TOOLS: list[dict[str, Any]] = [
    # Literature (C)
    _tool(
        "search_literature",
        "Search Valency Bond. mode=semantic searches by concept; mode=abstract is keyword search in "
        "abstracts (use it for failed attempts: secondary phases, solubility limits, no improvement); "
        "mode=citations finds highly cited foundational work. limit is capped at 15.",
        {"query": STR, "mode": _enum("semantic", "abstract", "citations"), "limit": INT},
    ),
    _tool(
        "related_papers",
        "Papers similar to, or citing, a paper. Citing papers catch later revisions of headline numbers. "
        "limit is capped at 10.",
        {"paper_id": STR, "relation": _enum("similar", "citing"), "limit": INT},
    ),
    _tool(
        "get_paper",
        "Full record for one paper, with an abstract of up to 3000 characters.",
        {"paper_id": STR},
    ),
    _tool(
        "publication_trend", "Papers per year for a keyword query; missing years are zero.", {"query": STR}
    ),
    _tool(
        "note_evidence",
        "Record a finding from a retrieved paper for the reviewer and the brief. A value is dropped "
        "unless that number appears in the paper's abstract.",
        {
            "paper_id": STR,
            "composition": STR_NULL,
            "property": STR,
            "value": NUM_NULL,
            "units": STR_NULL,
            "conditions": STR,
            "finding_type": _enum("positive", "negative", "mechanism", "synthesis"),
            "summary": STR,
        },
    ),
    # Materials Project (C)
    _tool(
        "lookup_materials_project",
        "All Materials Project polymorphs of an exact composition, with the default (most-reported "
        "experimentally observed near the hull) polymorph, PBE E_hull, formation energy and band gap, "
        "ICSD report counts, and MP's own phonon minimum where available (information only: the "
        "phonons tool decides dynamic stability). Solid solutions are not in MP.",
        {"composition": STR},
    ),
    # Candidates and structures (BOOL)
    _tool(
        "propose_candidate",
        "Register a candidate before any calculation. structure_type is a prototype (perovskites, "
        "pyrochlore, spinel, fluorite and antifluorite, rock salt, rutile, olivine, Heuslers ...), or "
        "'mp' to take the structure from Materials Project as-is when no prototype fits (the exact "
        "composition must be in MP; there is no prototype fallback). "
        "Code checks hard constraints, charge balance under common oxidation states (declare a risk "
        "'off-stoichiometric: <reason>' to waive it), that the composition fits the prototype's sites "
        "(assigned by ionic radius), repeats, evidence IDs (retrieved paper_id or mp:<id>) and, when "
        "required, novelty. Only ordered compounds for now: every site must hold a single element "
        "(no doped solid solutions or partial vacancies); for two B-site cations use an ordered "
        "prototype such as double_perovskite.",
        {
            "composition": STR,
            "structure_type": _enum(*PROTOTYPES, MP_STRUCTURE_TYPE),
            "rationale": STR,
            "evidence_ids": STR_LIST,
            "prior_estimate": NUM_NULL,
            "novelty": _enum("known", "variant", "novel"),
            "synthesis_feasibility": _enum("easy", "moderate", "difficult", "unknown"),
            "risks": STR_LIST,
        },
    ),
    _tool(
        "build_structure",
        "Build the starting structure. source=mp with mp_id builds that polymorph; source=auto uses the "
        "default polymorph when the composition is in MP, else the template; source=template forces the "
        "prototype. A note flags an MP polymorph whose space group differs from the proposed prototype. "
        "Candidates with structure_type 'mp' cannot use source=template. Resets this candidate's later "
        "results.",
        {"candidate_id": CID, "source": _enum("auto", "mp", "template"), "mp_id": STR_NULL, "seed": INT},
    ),
    # Simulation (BOOL)
    _tool(
        "relax",
        "Relax positions and cell with an ML potential. fmax in eV/Å: 0.02 for screening, 0.005 for a "
        "careful check. Resets hull, phonon and prediction results.",
        {"candidate_id": CID, "potential": _enum(*POTENTIALS), "fmax": {"type": "number"}},
    ),
    _tool(
        "energy_above_hull",
        "Energy above the convex hull against MP stable phases re-relaxed with the same potential. "
        "thermo_stable is computed by code. A new chemical system can take minutes on CPU.",
        {"candidate_id": CID},
    ),
    _tool(
        "phonons",
        "Finite-displacement phonons with mode mapping: the softest imaginary mode is frozen in and "
        "re-relaxed; only an energy drop above the threshold is a real instability, and then the "
        "candidate's structure becomes the distorted one. dyn_stable is computed by code.",
        {"candidate_id": CID, "mode": _enum("fast", "full")},
    ),
    _tool(
        "run_md",
        "Short NVT MD at the operating temperature; stable if no atom drifts more than 1 Å.",
        {"candidate_id": CID, "duration_ps": {"type": "number"}},
    ),
    # Properties (C)
    _tool(
        "predict_property",
        "Predict a property on the current structure with its uncertainty and a domain flag.",
        {
            "candidate_id": CID,
            "property": _enum("band_gap", "bulk_modulus", "shear_modulus", "formation_energy", "density"),
        },
    ),
    # Run control (A)
    _tool(
        "candidate_status",
        "Code-computed facts for every candidate (checks, stability verdicts, what is missing) and budget.",
        {},
    ),
    _tool(
        "request_review",
        "End the current round and get an independent review. Returns code checks, the rubric score, "
        "findings, one recommended next composition, the reviewer's confidence and either a stop "
        "instruction or the next round.",
        {"summary": STR},
    ),
    _tool(
        "submit_final",
        "Submit the recommended candidate. Code gate: relax, energy_above_hull (not stale), phonons and "
        "the target predict_property must have run, and every check must pass. If the gate fails you "
        "may improve the candidate, or resubmit with accept_not_met=true to report the best attempt.",
        {"candidate_id": CID, "runner_up_id": STR_NULL, "justification": STR, "accept_not_met": BOOL},
    ),
]
TOOL_NAMES = [t["name"] for t in TOOLS]
RUN_CONTROL_TOOLS = {"candidate_status", "request_review", "submit_final"}
