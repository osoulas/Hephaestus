"""Scripted campaigns with the real structure and Materials Project handlers in place of the stubs."""

from pathlib import Path
from typing import Any

import pytest
import scripts
from factories import workspace
from mp_fake import install_fake_mp

from hephaestus.core.events import EventBus
from hephaestus.harness.agent import Campaign
from hephaestus.llm.scripted import ScriptedLLM
from hephaestus.tools import materials_project, structures
from hephaestus.tools.stubs import check_novelty, stub_handlers


def _campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, turns: list[Any], reviews: list[Any]
) -> tuple[Campaign, EventBus]:
    """Build a campaign whose structure and MP tools are real (MP served from fixtures).

    Parameters
    ----------
    tmp_path : pathlib.Path
        Folder for the cache and events.
    monkeypatch : pytest.MonkeyPatch
        Used to install the MP fake.
    turns : list
        Scripted orchestrator turns.
    reviews : list
        Scripted reviewer answers.

    Returns
    -------
    tuple
        ``(campaign, bus)``.
    """
    install_fake_mp(monkeypatch, tmp_path)
    handlers = stub_handlers() | {
        "propose_candidate": structures.propose_candidate,
        "build_structure": structures.build_structure,
        "lookup_materials_project": materials_project.lookup_materials_project,
    }
    bus = EventBus(tmp_path / "runs")
    llm = ScriptedLLM(turns=turns, structured={"reviewer": reviews})
    return Campaign(workspace(), llm, handlers=handlers, bus=bus, novelty_fn=check_novelty), bus


def _finished(bus: EventBus, tool: str) -> list[dict[str, Any]]:
    """Results of every finished call of one tool.

    Parameters
    ----------
    bus : EventBus
        The campaign's event bus.
    tool : str
        Tool name.

    Returns
    -------
    list of dict
        Tool results in call order.
    """
    return [
        e.payload["result"]
        for e in bus.history("r-test")
        if e.status == "finished" and e.payload.get("tool") == tool
    ]


def test_doped_candidates_are_refused_without_breaking_the_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stub demo's Fe-doped SrTiO3 proposals are refused as disordered; the campaign still ends cleanly."""
    turns: list[Any] = [*scripts.ROUND_1, *scripts.ROUND_2]
    camp, bus = _campaign(tmp_path, monkeypatch, turns, list(scripts.REVIEWS))
    camp.run()
    proposals = _finished(bus, "propose_candidate")
    assert len(proposals) == 3 and not any(p["accepted"] for p in proposals)
    assert all("Disordered solid solutions are not supported yet" in p["reasons"][0] for p in proposals)
    assert _finished(bus, "build_structure") == []


def test_ordered_candidate_builds_from_mp_and_is_evaluated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ordered candidate (BaSnO3) is accepted, built from its MP polymorph and carried to submission."""
    cid = "c1-1"
    evaluate = [
        {
            "tool": "build_structure",
            "input": {"candidate_id": cid, "source": "auto", "mp_id": None, "seed": 0},
        },
        {"tool": "relax", "input": {"candidate_id": cid, "potential": "mace-mp-0-small", "fmax": 0.02}},
        {"tool": "energy_above_hull", "input": {"candidate_id": cid}},
        {"tool": "phonons", "input": {"candidate_id": cid, "mode": "fast"}},
        {"tool": "predict_property", "input": {"candidate_id": cid, "property": "band_gap"}},
    ]
    propose = scripts.propose("BaSnO3", evidence="mp:mp-3163")
    turns: list[Any] = [
        [{"tool": "lookup_materials_project", "input": {"composition": "BaSnO3"}}, propose],
        evaluate,
        [{"tool": "request_review", "input": {"summary": "BaSnO3 evaluated."}}],
        [scripts.submit(cid, accept_not_met=True)],
    ]
    camp, bus = _campaign(tmp_path, monkeypatch, turns, [scripts.review("SrSnO3", 60)])
    sub = camp.run()
    assert _finished(bus, "propose_candidate")[0]["accepted"]
    (build,) = _finished(bus, "build_structure")
    assert build["source"].startswith("mp:") and build["polymorph"]["space_group"] == "Pm-3m"
    assert sub is not None and sub.candidate_id == cid
