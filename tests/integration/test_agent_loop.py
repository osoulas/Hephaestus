"""Scripted model + real gates through stub tools: a full campaign (round-1 done-when for A)."""

from pathlib import Path

import scripts
from factories import workspace

from hephaestus.core.events import EventBus
from hephaestus.harness.agent import Campaign
from hephaestus.llm.scripted import ScriptedLLM
from hephaestus.tools.stubs import check_novelty, stub_handlers


def campaign(tmp_path: Path, turns: list, reviews: list, **cfg: object) -> tuple[Campaign, EventBus]:
    """Build a campaign over stub tools with a scripted model.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Runs folder for events.
    turns : list
        Scripted orchestrator turns.
    reviews : list
        Scripted reviewer answers.
    **cfg
        ``RunConfig`` overrides.

    Returns
    -------
    tuple
        ``(campaign, bus)``.
    """
    bus = EventBus(tmp_path)
    llm = ScriptedLLM(turns=list(turns), structured={"reviewer": list(reviews)})
    camp = Campaign(workspace(**cfg), llm, handlers=stub_handlers(), bus=bus, novelty_fn=check_novelty)
    return camp, bus


def test_full_campaign_reaches_target_and_submits(tmp_path: Path) -> None:
    """Two rounds: blocked stop (low confidence), refused bad submission, target met, accepted submission."""
    camp, bus = campaign(tmp_path, scripts.ROUND_1 + scripts.ROUND_2, scripts.REVIEWS)
    sub = camp.run()
    assert sub is not None and sub.candidate_id == "c2-1" and sub.gate_passed
    assert sub.runner_up_id == "c1-1" and sub.picked_by == "orchestrator"
    assert camp.stop_rule == "target_met"
    assert [c.llm_confidence for c in camp.critiques] == [60, 80]
    assert camp.critiques[0].candidate_id == "c1-1"  # best of round 1 chosen by code

    events = bus.history("r-test")
    reviews = [
        e.payload["review"]
        for e in events
        if e.payload.get("tool") == "request_review" and e.status == "finished"
    ]
    assert "reviewer confidence 60" in reviews[0]["gate_note"]  # OR-12
    assert reviews[0]["next_round"] == 2 and reviews[1]["stop_rule"] == "target_met"
    submits = [
        e.payload["result"]
        for e in events
        if e.payload.get("tool") == "submit_final" and e.status == "finished"
    ]
    assert submits[0]["accepted"] is False and "not thermodynamically stable" in submits[0]["reasons"]
    assert submits[1] == {"accepted": True, "gate_passed": True, "note": "Gate passed."}
    assert any(e.section == "agent" and e.payload["kind"] == "thinking" for e in events)
    lit = next(e for e in events if e.payload.get("tool") == "search_literature" and e.status == "finished")
    assert lit.payload["papers"][0]["paper_id"] == "stub:fe-sto"


def test_or7_every_tool_call_has_started_and_finished_events(tmp_path: Path) -> None:
    """Each tool call emits a started and a finished-or-failed event, with flat candidate fields."""
    camp, bus = campaign(tmp_path, scripts.ROUND_1 + scripts.ROUND_2, scripts.REVIEWS)
    camp.run()
    tool_events = [e for e in bus.history("r-test") if e.section == "tool"]
    assert len(tool_events) == 2 * camp.ws.tool_calls
    relax = next(e for e in tool_events if e.payload["tool"] == "relax" and e.status == "finished")
    assert relax.payload["composition"] == scripts.FE1 and relax.payload["cand_round"] == 1


def test_or10_tool_budget_forces_submission(tmp_path: Path) -> None:
    """When the budget runs out, results carry an instruction and a failing gate is accepted."""
    turns = [*scripts.ROUND_1[:3], [scripts.submit("c1-2")]]
    camp, _ = campaign(tmp_path, turns, scripts.REVIEWS, max_tool_calls=10)
    sub = camp.run()
    assert sub is not None and sub.candidate_id == "c1-2" and not sub.gate_passed
    assert camp.stop_rule == "tool_budget"


def test_or3_code_picks_best_without_submission(tmp_path: Path) -> None:
    """A model that stops talking gets nudged, then code picks the best candidate."""
    camp, _ = campaign(tmp_path, scripts.ROUND_1[:3], [])
    sub = camp.run()
    assert sub is not None and sub.picked_by == "code" and sub.candidate_id == "c1-1"
    assert sub.runner_up_id == "c1-2"


def test_or8_user_stop_tells_orchestrator_to_submit(tmp_path: Path) -> None:
    """After a user stop, the next tool result says to submit, and a failing gate is accepted."""
    camp, _ = campaign(tmp_path, [*scripts.ROUND_1[:3], [scripts.submit("c1-2")]], [])
    camp.request_stop()
    sub = camp.run()
    assert sub is not None and sub.candidate_id == "c1-2" and camp.stop_rule == "user_stop"


def test_or2_max_rounds_with_reviewer_down(tmp_path: Path) -> None:
    """One round allowed and the reviewer down: minimal critique (SC-10), stop on max rounds."""
    turns = [*scripts.ROUND_1, [scripts.submit("c1-1")]]
    camp, _ = campaign(tmp_path, turns, [RuntimeError("down"), RuntimeError("down")], max_iterations=1)
    sub = camp.run()
    assert camp.stop_rule == "max_rounds" and camp.critiques[0].llm_confidence is None
    assert sub is not None and sub.candidate_id == "c1-1"
    assert any(e.code == "llm_failed" for e in camp.errors)
