"""Tool registry: strict schemas, dispatch that never raises, and the Workspace."""

from pathlib import Path

import pytest

from hephaestus.core.types import Candidate, SearchSpec
from hephaestus.tools import registry as R
from hephaestus.tools.registry import ToolError, Workspace

DOC_TOOLS = [
    "search_literature",
    "related_papers",
    "get_paper",
    "publication_trend",
    "note_evidence",
    "lookup_materials_project",
    "propose_candidate",
    "build_structure",
    "relax",
    "energy_above_hull",
    "phonons",
    "run_md",
    "predict_property",
    "candidate_status",
    "request_review",
    "submit_final",
]


@pytest.fixture
def ws(fixtures_dir: Path) -> Workspace:
    """Return an empty workspace for the demo spec.

    Parameters
    ----------
    fixtures_dir : pathlib.Path
        ``tests/fixtures``.

    Returns
    -------
    Workspace
        Fresh workspace.
    """
    spec = SearchSpec.model_validate_json((fixtures_dir / "search_spec.json").read_text(encoding="utf-8"))
    return Workspace(run_id="r-test", spec=spec, cfg=R.RunConfig())


def test_tool_names_match_docs() -> None:
    """``TOOLS`` lists exactly the tools in ``docs/TOOLS.md``."""
    assert R.TOOL_NAMES == DOC_TOOLS


@pytest.mark.parametrize("tool", R.TOOLS, ids=lambda t: t["name"])
def test_schemas_are_strict(tool: dict) -> None:
    """Every property is required and no extra properties are allowed.

    Parameters
    ----------
    tool : dict
        Tool definition.
    """
    schema = tool["input_schema"]
    assert tool["strict"] is True
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == sorted(schema["properties"])


def test_execute_turns_tool_error_into_error_result(ws: Workspace) -> None:
    """A ``ToolError`` becomes ``{"error": ...}`` with ``is_error``."""

    def boom(ws: Workspace) -> dict:
        """Fail like a tool.

        Parameters
        ----------
        ws : Workspace
            Unused.

        Returns
        -------
        dict
            Never returns.
        """
        raise ToolError("no structure yet")

    result, is_error = R.execute(ws, "x", {}, {"x": boom})
    assert is_error and result == {"error": "no structure yet"}


def test_execute_never_raises(ws: Workspace) -> None:
    """Unexpected exceptions, bad arguments and unknown tools are all error results."""

    def crash(ws: Workspace) -> dict:
        """Crash unexpectedly.

        Parameters
        ----------
        ws : Workspace
            Unused.

        Returns
        -------
        dict
            Never returns.
        """
        return {"x": 1 / 0}

    for name, args in [("crash", {}), ("crash", {"bogus": 1}), ("missing", {})]:
        result, is_error = R.execute(ws, name, args, {"crash": crash})
        assert is_error and "error" in result


def test_workspace_get_unknown_candidate(ws: Workspace) -> None:
    """Unknown candidate IDs are a ToolError naming the known ones."""
    with pytest.raises(ToolError, match="known: none"):
        ws.get("c9-9")


def test_sim_result_thermo_verdict_ignores_stale_hull(ws: Workspace) -> None:
    """A hull computed before mode mapping changed the structure does not count."""
    cand = Candidate(
        candidate_id="c1-1",
        iteration=1,
        composition="SrTiO3",
        structure_type="perovskite_cubic",
        rationale="r",
    )
    ws.candidates["c1-1"] = R.CandidateState(candidate=cand, round=1, e_hull=0.01)
    assert ws.sim_result("c1-1").thermo_stable
    ws.candidates["c1-1"].hull_stale = True
    assert not ws.sim_result("c1-1").thermo_stable


def test_workspace_round_trips(ws: Workspace) -> None:
    """The workspace snapshot is lossless JSON (OR-6)."""
    assert Workspace.model_validate_json(ws.model_dump_json()) == ws
