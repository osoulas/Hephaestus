"""Agent-built tools end to end: gap, plan and code review, sandbox, campaign, resume (Section 14)."""

import json
import threading
from pathlib import Path
from typing import Any

import pytest
import scripts
from toolkit_fake import BAD_TEST_SOURCE, GAP, TEST_SOURCE, TOOL_SOURCE, fake_factory, plan

from hephaestus.core.config import RunConfig
from hephaestus.core.events import EventBus
from hephaestus.core.types import ToolDecision, ToolProposal, ValidationReport
from hephaestus.harness import runner
from hephaestus.harness.capability import HUB, CapabilityPhase, find_gap
from hephaestus.llm.scripted import ScriptedLLM
from hephaestus.tools import custom, stubs
from hephaestus.tools.registry import Workspace

PARSED_GAP = scripts.PARSED | {
    "target_property": "unsupported",
    "property_name": "Thermal expansion",
    "target_units": "1e-6/K",
    "title": "Low thermal expansion perovskite",
}
GOAL = "a lead-free oxide perovskite with lower thermal expansion than SrTiO3, stable on the hull"


@pytest.fixture(autouse=True)
def _cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the cache inside the test's folder.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test folder.
    monkeypatch : pytest.MonkeyPatch
        Sets ``HEPHAESTUS_CACHE_DIR``.
    """
    monkeypatch.setenv("HEPHAESTUS_CACHE_DIR", str(tmp_path / "cache"))


def handlers() -> dict[str, Any]:
    """Stub handlers whose relax leaves the formula in the structure, for the toy toolkit.

    Returns
    -------
    dict
        Handlers.
    """
    table = stubs.stub_handlers()
    relax = table["relax"]

    def relax_with_formula(ws: Workspace, candidate_id: str, potential: str, fmax: float) -> dict[str, Any]:
        """Stub relax, then record the composition as the structure.

        Parameters
        ----------
        ws : Workspace
            Workspace.
        candidate_id : str
            Candidate.
        potential : str
            Potential.
        fmax : float
            Force threshold.

        Returns
        -------
        dict
            Stub result.
        """
        out = relax(ws, candidate_id, potential, fmax)
        cs = ws.candidates[candidate_id]
        cs.structure = {"formula": cs.candidate.composition}
        return out

    table["relax"] = relax_with_formula
    return table


def cfg(tmp_path: Path, **kw: Any) -> RunConfig:
    """Run options with agent-built tools on and the library in the test folder.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test folder.
    **kw
        Overrides.

    Returns
    -------
    RunConfig
        Options.
    """
    return RunConfig(**({"custom_tools": "ask", "tool_library_dir": str(tmp_path / "lib")} | kw))


def write_turn(source: str = TOOL_SOURCE, tests: str = TEST_SOURCE) -> list[dict[str, Any]]:
    """Return a scripted ``write_tool`` turn for proposal t1.

    Parameters
    ----------
    source, tests : str
        Files.

    Returns
    -------
    list of dict
        Turn.
    """
    return [{"tool": "write_tool", "input": {"proposal_id": "t1", "source": source, "tests": tests}}]


PHASE_TURNS = [
    [{"text": "Researching the method."}, {"tool": "read_tool_source", "input": {"function": None}}],
    [{"tool": "propose_tool", "input": plan()}],
    write_turn(),
]


def decider(*answers: Any) -> tuple[Any, list[ToolProposal]]:
    """Return a decision callback answering in order, and the proposals it saw.

    Parameters
    ----------
    *answers
        ``ToolDecision``-like dicts or action strings.

    Returns
    -------
    tuple
        ``(callback, seen)``.
    """
    queue, seen = list(answers), []

    def decide(prop: ToolProposal) -> Any:
        """Answer the next decision.

        Parameters
        ----------
        prop : ToolProposal
            Proposal.

        Returns
        -------
        Any
            The next answer.
        """
        seen.append(prop)
        answer = queue.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    return decide, seen


def final_check(ws: Any, sub: Any) -> list[ValidationReport]:
    """Confirm the winner.

    Parameters
    ----------
    ws : Workspace
        Workspace.
    sub : Submission
        Submission.

    Returns
    -------
    list of ValidationReport
        One confirmed report.
    """
    return [ValidationReport(candidate_id=sub.candidate_id, verdict="confirmed")]


def deps(tmp_path: Path, turns: list, decide: Any, reviews: list | None = None) -> runner.Deps:
    """Return offline dependencies with a scripted model and the toy toolkit.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Runs folder.
    turns : list
        Capability then campaign turns.
    decide : callable
        Decision callback.
    reviews : list, optional
        Reviewer answers.

    Returns
    -------
    runner.Deps
        Dependencies.
    """
    llm = ScriptedLLM(
        turns=list(turns),
        structured={
            "goal_parser": [PARSED_GAP],
            "reviewer": list(reviews or [scripts.review(scripts.FE2, 90)]),
        },
    )
    return runner.Deps(
        llm=llm,
        handlers=handlers(),
        novelty_fn=stubs.check_novelty,
        final_check=final_check,
        bus=EventBus(tmp_path),
        root=tmp_path,
        decide=decide,
        toolkit_factory=fake_factory,
    )


def campaign_turns(tool: str = "custom_toy_expansion") -> list[list[dict[str, Any]]]:
    """Return one round that evaluates one candidate with the agent-built tool, then submits.

    Parameters
    ----------
    tool : str
        Tool computing the target.

    Returns
    -------
    list
        Turns.
    """
    evaluate = [s for s in scripts.evaluate("c1-1") if s["tool"] != "predict_property"]
    return [
        scripts.ROUND_1[0],
        [scripts.propose(scripts.FE1)],
        [*evaluate, {"tool": tool, "input": {"candidate_id": "c1-1"}}],
        [{"tool": "request_review", "input": {"summary": "One candidate."}}],
        [scripts.submit("c1-1")],
    ]


def events(tmp_path: Path, run_id: str) -> list[dict[str, Any]]:
    """Read a run's events.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Runs folder.
    run_id : str
        Run.

    Returns
    -------
    list of dict
        Events.
    """
    lines = (tmp_path / run_id / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


# ---------------------------------------------------------------- end to end


def test_at1_to_at17_gap_to_brief_with_an_agent_built_tool(tmp_path: Path) -> None:
    """A thermal expansion goal: plan accepted, code accepted, tool used for reference and target."""
    decide, seen = decider("accept", {"action": "accept"})
    d = deps(tmp_path, PHASE_TURNS + campaign_turns(), decide)
    state = runner.run(GOAL, cfg=cfg(tmp_path), deps=d, run_id="r1")
    assert state.status == "finished", state.errors
    spec = state.spec
    assert spec and spec.target_property == GAP and spec.target_tool == "custom_toy_expansion"
    assert spec.reference_value and spec.reference_value.value == pytest.approx(10.0)  # 2 x 5 for SrTiO3
    assert spec.reference_value.uncertainty is None
    assert [p.stage for p in seen] == ["plan", "code"]
    code = seen[1]
    assert code.source == TOOL_SOURCE and all(c.passed for c in code.checks.values())
    assert (
        code.trial and code.trial["value"] == pytest.approx(10.0) and code.toolkit_calls == ["single_point"]
    )
    approved, _ = custom.scan_library(tmp_path / "lib")  # AT-20
    assert [t.name for t in approved] == ["toy_expansion"] and approved[0].run_id == "r1"
    pred = state.iterations[0].predictions[0]
    assert (
        pred.property == GAP
        and pred.value.value == pytest.approx(8.0)
        and pred.model.startswith("agent-built:")
    )
    assert state.submission and state.submission.gate_passed  # 8 < 10, stable
    kinds = [e["payload"].get("kind") for e in events(tmp_path, "r1") if e["section"] == "capability"]
    assert kinds[0] == "gap" and "plan" in kinds and "code" in kinds and "registered" in kinds
    order = [e["section"] for e in events(tmp_path, "r1")]
    assert order.index("capability") < order.index("orchestrator")  # OR-1
    md = (tmp_path / "r1" / "brief.md").read_text(encoding="utf-8")
    assert "agent-built tool" in md
    camp_tools = {
        t["name"]
        for t in custom.campaign_tools(runner.load_for_resume(runner.RunFiles.for_run("r1", tmp_path))[1])
    }
    assert "custom_toy_expansion" in camp_tools  # AT-16
    assert (tmp_path / "r1" / "capability" / "decisions.json").exists()


def test_at3_gap_with_custom_tools_off_ends_before_the_campaign(tmp_path: Path) -> None:
    """With custom tools off the parser flags the property and nothing runs."""
    d = deps(tmp_path, [], decider()[0])
    state = runner.run(GOAL, cfg=cfg(tmp_path, custom_tools="off"), deps=d, run_id="r1")
    assert state.status == "needs_input"
    assert any("'thermal_expansion' is not supported" in e.message for e in state.errors)
    assert not (tmp_path / "r1" / "iter_1").exists()


def test_at3_given_spec_with_tools_off_fails_with_the_property_named(tmp_path: Path) -> None:
    """A spec handed in directly still cannot start a campaign with a gap."""
    from factories import demo_spec

    spec = demo_spec().model_copy(update={"target_property": GAP, "target_units": "1e-6/K"})
    state = runner.run(
        spec=spec, cfg=cfg(tmp_path, custom_tools="off"), deps=deps(tmp_path, [], None), run_id="r1"
    )
    assert state.status == "failed" and "'thermal_expansion'" in state.errors[-1].message


def test_at7_revision_then_accept_and_the_revision_cap(tmp_path: Path) -> None:
    """A change request reaches the agent; revise is not offered once revisions run out."""
    told: list[str] = []

    def revise(messages: list) -> list:
        """Record what the agent was told, then propose again.

        Parameters
        ----------
        messages : list
            Conversation.

        Returns
        -------
        list
            Turn.
        """
        told.append(messages[-1]["content"][-1]["text"])
        return [{"tool": "propose_tool", "input": plan() | {"method": "Equation-of-state fit."}}]

    turns = [[{"tool": "propose_tool", "input": plan()}], revise, write_turn()]
    decide, seen = decider({"action": "revise", "message": "use an EOS fit"}, "accept", "accept")
    d = deps(tmp_path, turns + campaign_turns(), decide)
    state = runner.run(GOAL, cfg=cfg(tmp_path, max_tool_revisions=1), deps=d, run_id="r1")
    assert state.spec and state.spec.target_tool == "custom_toy_expansion"
    assert "use an EOS fit" in told[0]
    assert [p.revision for p in seen] == [1, 2, 2] and seen[1].revisions_left == 0
    assert seen[1].method == "Equation-of-state fit."
    cap = json.loads((tmp_path / "r1" / "capability" / "state.json").read_text(encoding="utf-8"))
    assert [x["action"] for x in cap["decisions"]] == ["revise", "accept", "accept"]
    plan_events = [e for e in events(tmp_path, "r1") if e["payload"].get("kind") == "plan"]
    assert plan_events[-1]["payload"]["allowed"] == ["accept", "cancel_run", "cancel_tool"]


def test_at8_cancelling_the_target_tool_screens_stability_only(tmp_path: Path) -> None:
    """The run continues, target met never fires, and the brief names the missing property."""
    turns = [
        [{"tool": "propose_tool", "input": plan()}],
        *campaign_turns()[:4],
        [scripts.submit("c1-1", accept_not_met=True)],
    ]
    turns[3] = [s for s in turns[3] if not s.get("tool", "").startswith("custom_")]
    state = runner.run(
        GOAL, cfg=cfg(tmp_path), deps=deps(tmp_path, turns, decider("cancel_tool")[0]), run_id="r1"
    )
    assert state.status == "finished" and state.spec and not state.spec.target_evaluable
    assert state.stop_reason != "target_met"
    assert state.submission and not state.submission.gate_passed
    md = (tmp_path / "r1" / "brief.md").read_text(encoding="utf-8")
    assert "thermal_expansion was not evaluated" in md or "was not evaluated" in md
    assert not custom.scan_library(tmp_path / "lib")[0]


def test_at9_cancel_run_ends_before_the_campaign(tmp_path: Path) -> None:
    """Cancel the run stops it with the user's reason and no round folder."""
    turns = [[{"tool": "propose_tool", "input": plan()}]]
    state = runner.run(
        GOAL, cfg=cfg(tmp_path), deps=deps(tmp_path, turns, decider("cancel_run")[0]), run_id="r1"
    )
    assert state.status == "stopped" and state.stop_reason == "cancelled by user"
    assert not (tmp_path / "r1" / "iter_1").exists() and state.brief is None


def test_at11_failed_checks_go_back_to_the_agent_and_cancel_after_three(tmp_path: Path) -> None:
    """A failing test is reported to the agent; three failures cancel the tool."""
    bad = write_turn(tests=BAD_TEST_SOURCE)
    turns = [[{"tool": "propose_tool", "input": plan()}], bad, bad, bad]
    decide, seen = decider("accept")
    state = runner.run(
        GOAL, cfg=cfg(tmp_path), deps=deps(tmp_path, turns + campaign_turns(), decide), run_id="r1"
    )
    assert [p.stage for p in seen] == ["plan"]  # the code never reached the user
    assert state.spec and not state.spec.target_evaluable
    checks = [e["payload"] for e in events(tmp_path, "r1") if e["payload"].get("kind") == "checks"]
    assert len(checks) == 3 and "value should be negative" in checks[0]["checks"]["tests"]["report"]


def test_at22_an_approved_library_tool_is_reused_without_a_proposal(tmp_path: Path) -> None:
    """A second run with the same gap proposes nothing."""
    custom.save_tool(tmp_path / "lib", "toy_expansion", TOOL_SOURCE, TEST_SOURCE)
    state = runner.run(
        GOAL, cfg=cfg(tmp_path), deps=deps(tmp_path, campaign_turns(), decider()[0]), run_id="r1"
    )
    assert state.status == "finished" and state.spec and state.spec.target_tool == "custom_toy_expansion"
    kinds = [e["payload"].get("kind") for e in events(tmp_path, "r1") if e["section"] == "capability"]
    assert kinds == ["library"]


def test_at21_a_hand_written_file_is_offered_for_review_first(tmp_path: Path) -> None:
    """An unapproved library file for the gap is shown as a code review; accepting it fills the gap."""
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "toy_expansion.py").write_text(
        TOOL_SOURCE.replace('"author": "agent"', '"author": "user"'), encoding="utf-8"
    )
    (lib / "test_toy_expansion.py").write_text(TEST_SOURCE, encoding="utf-8")
    decide, seen = decider("accept")
    state = runner.run(GOAL, cfg=cfg(tmp_path), deps=deps(tmp_path, campaign_turns(), decide), run_id="r1")
    assert [p.author for p in seen] == ["user"] and seen[0].stage == "code"
    assert state.spec and state.spec.target_tool == "custom_toy_expansion"
    assert custom.scan_library(lib)[0][0].author == "user"


class Killed(BaseException):
    """Simulates the process dying while waiting for the user."""


def test_at24_resume_shows_the_waiting_card_again(tmp_path: Path) -> None:
    """Killed at the code review, a resumed run asks again and finishes without repeating work."""
    decide, _ = decider("accept", Killed())
    with pytest.raises(Killed):
        runner.run(GOAL, cfg=cfg(tmp_path), deps=deps(tmp_path, PHASE_TURNS, decide), run_id="r1")
    saved = json.loads((tmp_path / "r1" / "capability" / "state.json").read_text(encoding="utf-8"))
    assert saved["proposals"]["t1"]["stage"] == "code" and saved["proposals"]["t1"]["status"] == "waiting"
    decide2, seen2 = decider("accept")
    state = runner.resume("r1", deps=deps(tmp_path, campaign_turns(), decide2))
    assert [p.stage for p in seen2] == ["code"] and state.status == "finished"
    assert state.spec and state.spec.target_tool == "custom_toy_expansion"


def test_at25_waiting_for_the_user_counts_against_no_budget(tmp_path: Path) -> None:
    """A long wait for the decision does not use up the capability phase's time budget."""
    now = [0.0]

    def slow(prop: ToolProposal) -> str:
        """Take an hour to decide.

        Parameters
        ----------
        prop : ToolProposal
            Proposal.

        Returns
        -------
        str
            ``accept``.
        """
        now[0] += 3600.0
        return "accept"

    from factories import workspace

    ws = workspace(custom_tools="ask", tool_library_dir=str(tmp_path / "lib"), capability_time_budget_s=60)
    ws.spec = ws.spec.model_copy(update={"target_property": GAP, "target_units": "1e-6/K"})
    llm = ScriptedLLM(turns=PHASE_TURNS[1:])
    phase = CapabilityPhase(
        ws,
        llm,
        handlers(),
        EventBus(tmp_path),
        tmp_path / "cap",
        GAP,
        decide=slow,
        toolkit_factory=fake_factory,
        clock=lambda: now[0],
    )
    assert phase.run() == "filled" and phase.agent_s < 60


def test_at1_find_gap_is_code() -> None:
    """A supported property is no gap; an unknown one is; disabling elastic opens one."""
    from factories import demo_spec

    spec = demo_spec()
    assert find_gap(spec, RunConfig()) is None
    assert (
        find_gap(spec.model_copy(update={"target_property": GAP}), RunConfig(tool_library_dir="/nonexistent"))
        == GAP
    )
    bulk = spec.model_copy(update={"target_property": "bulk_modulus"})
    assert find_gap(bulk, RunConfig()) is None
    assert (
        find_gap(bulk, RunConfig(disabled_capabilities=["elastic"], tool_library_dir="/nonexistent"))
        == "bulk_modulus"
    )


def test_at24_decisions_through_the_hub(tmp_path: Path) -> None:
    """Without a callback, a decision submitted from another thread (the HTTP API) reaches the phase."""
    from factories import workspace

    ws = workspace(custom_tools="ask", tool_library_dir=str(tmp_path / "lib"))
    ws.spec = ws.spec.model_copy(update={"target_property": GAP, "target_units": "1e-6/K"})
    phase = CapabilityPhase(
        ws,
        ScriptedLLM(turns=PHASE_TURNS[1:]),
        handlers(),
        EventBus(tmp_path),
        tmp_path / "cap",
        GAP,
        toolkit_factory=fake_factory,
    )
    result: list[str] = []
    t = threading.Thread(target=lambda: result.append(phase.run()))
    t.start()
    for stage in ("plan", "code"):
        for _ in range(200):
            if any(w["stage"] == stage for w in HUB.waiting(ws.run_id)):
                break
            threading.Event().wait(0.05)
        with pytest.raises(ValueError, match="not waiting"):
            HUB.submit(ws.run_id, ToolDecision(proposal_id="t9", stage=stage, action="accept"))  # type: ignore[arg-type]
        HUB.submit(ws.run_id, ToolDecision(proposal_id="t1", stage=stage, action="accept"))  # type: ignore[arg-type]
    t.join(10)
    assert result == ["filled"] and "toy_expansion" in ws.custom_tools


def test_at2_capability_tools_are_not_in_the_campaign() -> None:
    """propose_tool and write_tool exist only in the capability phase."""
    from factories import workspace

    names = {t["name"] for t in custom.campaign_tools(workspace())}
    assert "propose_tool" not in names and "write_tool" not in names and "read_tool_source" not in names


# ---------------------------------------------------------------- asking the user (no dead ends)

PHONOPY_TOOL = TOOL_SOURCE.replace("import math\n", "import math\n\nimport phonopy\n").replace(
    '"author": "agent",', '"packages": ["phonopy"],\n    "author": "agent",'
)


def phase(tmp_path: Path, turns: list, decide: Any) -> CapabilityPhase:
    """Return a capability phase for the thermal expansion gap.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test folder.
    turns : list
        Scripted turns.
    decide : callable
        Decision callback.

    Returns
    -------
    CapabilityPhase
        The phase.
    """
    from factories import workspace

    ws = workspace(custom_tools="ask", tool_library_dir=str(tmp_path / "lib"))
    ws.spec = ws.spec.model_copy(update={"target_property": GAP, "target_units": "1e-6/K"})
    return CapabilityPhase(
        ws,
        ScriptedLLM(turns=turns),
        handlers(),
        EventBus(tmp_path),
        tmp_path / "cap",
        GAP,
        decide=decide,
        toolkit_factory=fake_factory,
    )


def test_a_code_change_needing_a_package_is_asked_for_not_a_dead_end(tmp_path: Path) -> None:
    """The run that got stuck: "use phonopy" at code review. The agent asks, is allowed, and delivers."""
    turns = [
        [{"tool": "propose_tool", "input": plan()}],
        write_turn(),
        [
            {
                "tool": "write_tool",
                "input": {"proposal_id": "t1", "source": PHONOPY_TOOL, "tests": TEST_SOURCE},
            }
        ],
        [
            {
                "tool": "request_package",
                "input": {"module": "phonopy", "pip_name": None, "reason": "Force constants."},
            }
        ],
        [
            {
                "tool": "write_tool",
                "input": {"proposal_id": "t1", "source": PHONOPY_TOOL, "tests": TEST_SOURCE},
            }
        ],
    ]
    decide, seen = decider("accept", {"action": "revise", "message": "use phonopy"}, "allow", "accept")
    ph = phase(tmp_path, turns, decide)
    assert ph.run() == "filled"
    assert [getattr(x, "kind", getattr(x, "stage", None)) for x in seen] == [
        "plan",
        "code",
        "package",
        "code",
    ]
    pkg = seen[2]
    assert (
        pkg.package == "phonopy" and pkg.installed is True
    )  # already in the environment: nothing to install
    tool = ph.ws.custom_tools["toy_expansion"]
    assert tool.packages == ["phonopy"]
    checks = [
        e
        for e in map(json.loads, (tmp_path / "r-test" / "events.jsonl").read_text().splitlines())
        if e["payload"].get("kind") == "checks"
    ]
    first_try = checks[1]["payload"]["checks"]["static"]["report"]
    assert "not allowed" in first_try and "request_package" in first_try  # the way out is named
    assert custom.scan_library(tmp_path / "lib")[0][0].packages == ["phonopy"]


def test_a_code_change_can_go_back_to_a_new_plan(tmp_path: Path) -> None:
    """After a change request on the code, the agent may propose a revised plan instead of rewriting."""
    turns = [
        [{"tool": "propose_tool", "input": plan()}],
        write_turn(),
        [{"tool": "propose_tool", "input": plan() | {"method": "Equation-of-state fit."}}],
        write_turn(),
    ]
    decide, seen = decider("accept", {"action": "revise", "message": "different method"}, "accept", "accept")
    ph = phase(tmp_path, turns, decide)
    assert ph.run() == "filled"
    assert [p.stage for p in seen] == ["plan", "code", "plan", "code"] and seen[
        2
    ].method == "Equation-of-state fit."


def test_a_stuck_agent_ends_the_phase_instead_of_looping(tmp_path: Path) -> None:
    """Text-only turns with nothing waiting end the phase after a few turns, with the reason."""
    turns = [[{"tool": "propose_tool", "input": plan()}]] + [[{"text": "Please choose A, B or C."}]] * 20
    ph = phase(tmp_path, turns, decider("accept")[0])
    assert ph.run() == "failed"
    assert "no progress" in ph.state.end_reason
    assert len(ph.llm.turns) >= 20 - 4  # stopped after a handful of idle turns, not twenty


def test_ask_user_reaches_the_user_and_the_answer_reaches_the_agent(tmp_path: Path) -> None:
    """A question is answered by a plain string; denying a package says so to the agent."""
    told: list[str] = []

    def after(messages: list) -> list:
        """Record the answer, then give up.

        Parameters
        ----------
        messages : list
            Conversation.

        Returns
        -------
        list
            Turn.
        """
        told.append(json.dumps(messages[-1]["content"]))
        return [{"tool": "request_package", "input": {"module": "phonopy", "pip_name": None, "reason": "r"}}]

    def last(messages: list) -> list:
        """Record the denial, then give up.

        Parameters
        ----------
        messages : list
            Conversation.

        Returns
        -------
        list
            Turn.
        """
        told.append(json.dumps(messages[-1]["content"]))
        return [{"tool": "give_up", "input": {"reason": "needs phonopy"}}]

    turns = [
        [{"tool": "ask_user", "input": {"question": "Strain or EOS?", "options": ["Strain", "EOS"]}}],
        after,
        last,
    ]
    decide, seen = decider("EOS", "deny")
    ph = phase(tmp_path, turns, decide)
    assert ph.run() == "failed" and "gave up: needs phonopy" in ph.state.end_reason
    assert seen[0].kind == "question" and seen[0].options == ["Strain", "EOS"]
    assert "EOS" in told[0] and "did not allow phonopy" in told[1]


def test_a_package_request_waits_through_the_hub(tmp_path: Path) -> None:
    """Without a callback the request waits for allow or deny from the API."""
    turns = [
        [{"tool": "request_package", "input": {"module": "phonopy", "pip_name": None, "reason": "r"}}],
        [{"tool": "give_up", "input": {"reason": "done"}}],
    ]
    ph = phase(tmp_path, turns, None)
    out: list[str] = []
    t = threading.Thread(target=lambda: out.append(ph.run()))
    t.start()
    for _ in range(200):
        if HUB.waiting(ph.ws.run_id):
            break
        threading.Event().wait(0.05)
    w = HUB.waiting(ph.ws.run_id)[0]
    assert w["stage"] == "package" and w["allowed"] == ["allow", "cancel_run", "deny"]
    with pytest.raises(ValueError):
        HUB.submit(ph.ws.run_id, ToolDecision(proposal_id=w["proposal_id"], stage="package", action="accept"))  # type: ignore[arg-type]
    HUB.submit(ph.ws.run_id, ToolDecision(proposal_id=w["proposal_id"], stage="package", action="allow"))  # type: ignore[arg-type]
    t.join(10)
    assert out == ["failed"] and ph.state.approved_packages == ["phonopy"]
