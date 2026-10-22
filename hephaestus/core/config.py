"""``RunConfig`` (every per-run option), paths and environment (OR-4)."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_MODEL = "claude-opus-5-5"
MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1"]


def default_model() -> str:
    """Return the default Claude model, from ``HEPHAESTUS_LLM_MODEL`` if set.

    Returns
    -------
    str
        A model ID such as ``"claude-opus-5-5"``.
    """
    return os.environ.get("HEPHAESTUS_LLM_MODEL") or DEFAULT_MODEL


def budget_limit(value: float) -> float:
    """Return a budget as a limit, where 0 (or less) means no limit (OR-9, OR-10).

    Parameters
    ----------
    value : float
        A budget from ``RunConfig``: rounds, tool calls or seconds.

    Returns
    -------
    float
        ``value``, or ``math.inf`` when it is 0.
    """
    return value if value > 0 else math.inf


def runs_root() -> Path:
    """Return the folder holding run folders (``HEPHAESTUS_RUNS_DIR``, default ``./runs``).

    Returns
    -------
    pathlib.Path
        Directory for ``runs/<run_id>/``.
    """
    return Path(os.environ.get("HEPHAESTUS_RUNS_DIR", "runs"))


def cache_root() -> Path:
    """Return the cache folder (``HEPHAESTUS_CACHE_DIR``, default ``./cache``).

    Returns
    -------
    pathlib.Path
        Directory for ``cache/{llm,valency,mp}/``.
    """
    return Path(os.environ.get("HEPHAESTUS_CACHE_DIR", "cache"))


def user_dir() -> Path:
    """Return the owner-only folder for tokens and credentials (DL-7).

    Returns
    -------
    pathlib.Path
        ``~/.hephaestus`` unless ``HEPHAESTUS_HOME`` is set.
    """
    return Path(os.environ.get("HEPHAESTUS_HOME", Path.home() / ".hephaestus"))


BAND_GAP_MODELS: dict[str, dict[str, Any]] = {
    "megnet-hse": {
        "label": "MEGNet multi-fidelity · HSE",
        "fidelity": "hse",
        "description": "Hybrid-functional scale; closest to measured gaps (±0.65 eV on 15 perovskites).",
    },
    "megnet-pbe": {
        "label": "MEGNet multi-fidelity · PBE",
        "fidelity": "pbe",
        "description": "Materials Project PBE scale; runs about 1.6 eV below experiment (±0.57 eV vs PBE).",
    },
    "megnet-gllb-sc": {
        "label": "MEGNet multi-fidelity · GLLB-SC",
        "fidelity": "gllb-sc",
        "description": "GLLB-SC scale; not benchmarked against measured gaps here, so no uncertainty is claimed.",
    },
    "megnet-scan": {
        "label": "MEGNet multi-fidelity · SCAN",
        "fidelity": "scan",
        "description": "SCAN meta-GGA scale; not benchmarked against measured gaps here, so no uncertainty is claimed.",
    },
}
"""Band gap models a run can choose: one pretrained MEGNet model with four output fidelities."""


class RunConfig(BaseModel):
    """Every per-run option (OR-4). Stored in ``progress.json`` and inside ``RunState``."""

    model_config = ConfigDict(extra="forbid")

    # Campaign limits and thresholds (OR-2, OR-9, OR-10); a budget of 0 means no limit
    max_iterations: int = Field(default=3, ge=0)
    confidence_threshold: int = 72
    max_tool_calls: int = Field(default=90, ge=0)
    time_budget_s: float = Field(default=900.0, ge=0)
    max_parallel_tools: int = 4  # tool calls on different candidates run at once
    # LLM access (OR-11)
    llm_backend: Literal["auto", "api", "claude-code"] = "auto"
    llm_model: str = Field(default_factory=default_model)
    # Science options
    band_gap_model: Literal["megnet-hse", "megnet-pbe", "megnet-gllb-sc", "megnet-scan"] = "megnet-hse"
    require_novel: bool = False
    potential: str = "mace-mp-0-small"
    final_potential: str = "mace-mp-0-medium"
    cross_check_potential: str | None = None
    phonon_mode: Literal["fast", "full"] = "fast"
    phonon_gate: Literal["mode_map", "strict"] = "mode_map"
    distortion_threshold_meV: float = 2.0
    run_phonons: bool = True
    # Thermodynamic stability (SE-5, SE-13): whether to compute E_hull, and a threshold that
    # replaces the goal's (None keeps the goal's, which defaults to 0.05 eV/atom)
    run_hull: bool = True
    e_hull_max_eV_per_atom: float | None = Field(default=None, ge=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["float32", "float64"] = "float32"
    final_dtype: Literal["float32", "float64"] = "float64"
    max_atoms: int = 80
    # Final re-check (Section 9)
    run_final: bool = True
    final_time_budget_s: float = Field(default=600.0, ge=0)
    # Reproducibility and offline replay (SYS-7, SYS-8)
    seed: int = 0
    offline: bool = False
    # Agent-built tools (Section 14). "ask": a capability gap lets the agent propose a tool for
    # the user to approve; "off": a gap ends the run before the campaign (AT-3)
    custom_tools: Literal["off", "ask"] = "off"
    capability_time_budget_s: float = Field(default=600.0, ge=0)  # agent time only, not waiting
    max_custom_tools: int = Field(default=3, ge=1)
    max_tool_revisions: int = Field(default=3, ge=0)
    custom_tool_timeout_s: float = Field(default=300.0, gt=0)
    tool_library_dir: str = "hephaestus_tools"  # relative to the working directory
    disabled_capabilities: list[Literal["elastic"]] = Field(default_factory=list)  # demos and tests

    @model_validator(mode="before")
    @classmethod
    def _from_fidelity(cls, data: Any) -> Any:
        """Read configs saved before ``band_gap_model`` existed, which stored ``band_gap_fidelity``.

        Parameters
        ----------
        data : Any
            Raw input.

        Returns
        -------
        Any
            Input with ``band_gap_fidelity`` turned into ``band_gap_model``.
        """
        if isinstance(data, dict) and "band_gap_fidelity" in data:
            data = dict(data)
            fid = data.pop("band_gap_fidelity")
            data.setdefault("band_gap_model", f"megnet-{fid}")
        return data

    @property
    def band_gap_fidelity(self) -> str:
        """Fidelity of the chosen band gap model, e.g. ``"hse"``.

        Returns
        -------
        str
            The MEGNet output fidelity.
        """
        return str(BAND_GAP_MODELS[self.band_gap_model]["fidelity"])


PROFILES: list[dict[str, Any]] = [
    {
        "id": "quick",
        "name": "Quick look",
        "hint": "Two short rounds; useful to check a goal makes sense.",
        "values": {
            "max_iterations": 2,
            "max_tool_calls": 50,
            "time_budget_s": 480,
            "final_time_budget_s": 300,
            "phonon_mode": "fast",
        },
    },
    {
        "id": "standard",
        "name": "Standard",
        "hint": "The defaults: three rounds and a 10-minute re-check.",
        "values": {},
    },
    {
        "id": "thorough",
        "name": "Thorough",
        "hint": "Five rounds, full phonons and more time for the re-check.",
        "values": {
            "max_iterations": 5,
            "max_tool_calls": 150,
            "time_budget_s": 1800,
            "final_time_budget_s": 1200,
            "phonon_mode": "full",
        },
    },
]
"""Setting profiles: one choice sets the budgets and depth of checking (OR-4, UI-12).

``values`` are ``RunConfig`` fields; ``standard`` keeps the defaults. The web composer keeps a copy
in ``web/js/spec.js`` (it loads without a request); ``tests/unit/interfaces/test_profiles.py``
keeps the two equal.
"""
