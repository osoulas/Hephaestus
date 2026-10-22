"""The marimo run notebook stays valid and offline-checkable (notebooks/run_goal.py)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "run_goal.py"


@pytest.mark.skipif(importlib.util.find_spec("marimo") is None, reason="marimo is not installed")
def test_notebook_passes_marimo_check() -> None:
    """Cell variables are unique, and no cell returns early: marimo's checks pass."""
    out = subprocess.run(
        [sys.executable, "-m", "marimo", "check", str(NOTEBOOK)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr
