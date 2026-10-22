"""A toy toolkit and agent-built tool files for the capability tests (Section 14)."""

from typing import Any

from hephaestus.tools.custom import Toolkit

GAP = "thermal_expansion"

TOOL_SOURCE = '''"""Toy thermal expansion: twice the energy from a single point."""

import math

TOOL = {
    "name": "toy_expansion",
    "property": "thermal_expansion",
    "units": "1e-6/K",
    "description": "Twice the single-point energy.",
    "method": "Toy method for tests.",
    "citations": [],
    "inputs": {"type": "object", "properties": {"scale": {"type": "number", "default": 2.0}}},
    "building_blocks": ["single_point"],
    "trial_range": [1.0, 100.0],
    "author": "agent",
}


def run(structure, inputs, hp):
    """Return scale times the single-point energy."""
    energy = hp.single_point(structure)["energy_eV"]
    hp.log("toy expansion computed")
    return {"value": math.fabs(inputs.get("scale", 2.0) * energy), "units": "1e-6/K", "details": {"energy_eV": energy}}
'''

TEST_SOURCE = '''"""Tests for the toy tool."""


def test_positive(run, hp, structure):
    """The value is positive."""
    assert run(structure, {}, hp)["value"] > 0


def test_header(tool):
    """The header names the property."""
    assert tool["property"] == "thermal_expansion"
'''

BAD_TEST_SOURCE = '''"""A failing test."""


def test_always_fails(run, hp, structure):
    """Fail."""
    assert run(structure, {}, hp)["value"] < 0, "value should be negative"
'''

ENERGIES = {"SrTiO3": 5.0, "SrTi0.875Fe0.125O3": 4.0, "SrTi0.75Fe0.25O3": 3.0}
"""Toy single-point energies keyed by the structure's ``formula`` (stubs carry no atoms)."""


class FakeToolkit(Toolkit):
    """Serves ``single_point`` from ``ENERGIES``; records every call."""

    def single_point(self, structure: dict[str, Any]) -> dict[str, Any]:
        """Return a toy energy.

        Parameters
        ----------
        structure : dict
            ``{"formula": ...}`` from the test relax handler, or empty.

        Returns
        -------
        dict
            ``{energy_eV}``.
        """
        return {"energy_eV": ENERGIES.get(structure.get("formula", ""), 6.0)}


def fake_factory(ws: Any, potential: str, dtype: str) -> FakeToolkit:
    """Build a toy toolkit (``ToolkitFactory``).

    Parameters
    ----------
    ws : Workspace
        Workspace.
    potential : str
        Potential name.
    dtype : str
        Precision.

    Returns
    -------
    FakeToolkit
        Toolkit.
    """
    return FakeToolkit(potential, dtype=dtype)


def plan(name: str = "toy_expansion", prop: str = GAP) -> dict[str, Any]:
    """Return ``propose_tool`` input.

    Parameters
    ----------
    name : str
        Tool name.
    prop : str
        Property.

    Returns
    -------
    dict
        Plan.
    """
    return {
        "name": name,
        "property": prop,
        "units": "1e-6/K",
        "description": "Twice the single-point energy.",
        "method": "Toy method for tests.",
        "citations": [],
        "inputs": {"type": "object", "properties": {"scale": {"type": "number", "default": 2.0}}},
        "building_blocks": ["single_point"],
        "expected_runtime_s": 1.0,
        "supplies_target": True,
        "if_cancelled": "",
    }
