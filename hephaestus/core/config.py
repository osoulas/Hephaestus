"""``RunConfig`` (every per-run option), paths and environment (OR-4)."""

from __future__ import annotations

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

    # Campaign limits and thresholds (OR-2, OR-9, OR-10)
    max_iterations: int = 3
    confidence_threshold: int = 72
    max_tool_calls: int = 90
    time_budget_s: float = 900.0
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
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["float32", "float64"] = "float32"
    final_dtype: Literal["float32", "float64"] = "float64"
    max_atoms: int = 80
    # Final re-check (Section 9)
    run_final: bool = True
    final_time_budget_s: float = 600.0
    # Reproducibility and offline replay (SYS-7, SYS-8)
    seed: int = 0
    offline: bool = False

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
