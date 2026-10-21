"""CLI renderer: the recorded demo run renders completely and never decides a verdict (OR-7, UI-14)."""

import io
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from hephaestus.core.types import ProgressEvent
from hephaestus.interfaces.cli.render import Renderer
from hephaestus.interfaces.cli.theme import fmt_quantity, make_console, sparkline

DEMO = Path(__file__).parents[2] / "fixtures" / "events" / "demo" / "events.jsonl"


def demo_events() -> list[ProgressEvent]:
    """Return the recorded demo run's events.

    Returns
    -------
    list of ProgressEvent
        Events in order.
    """
    return [ProgressEvent.model_validate_json(line) for line in DEMO.read_text(encoding="utf-8").splitlines()]


def render(events: list[ProgressEvent], renderer: Renderer | None = None, width: int = 140) -> str:
    """Feed events through a renderer and return the plain text printed.

    Parameters
    ----------
    events : list of ProgressEvent
        Events to feed.
    renderer : Renderer, optional
        Renderer to use; a fresh one by default.
    width : int
        Console width.

    Returns
    -------
    str
        Printed text without styles.
    """
    r = renderer or Renderer()
    console = make_console(file=io.StringIO(), width=width, record=True)
    for e in events:
        for item in r.feed(e):
            console.print(item)
    return console.export_text()


def event(
    section: str,
    status: Literal["started", "finished", "failed"],
    payload: dict[str, Any],
    seconds: float = 0.0,
) -> ProgressEvent:
    """Build one event.

    Parameters
    ----------
    section, status : str
        Event section and status.
    payload : dict
        Payload.
    seconds : float
        Offset from a fixed start time.

    Returns
    -------
    ProgressEvent
        The event.
    """
    t = datetime(2026, 10, 21, tzinfo=UTC) + timedelta(seconds=seconds)
    return ProgressEvent(run_id="r-t", section=section, status=status, payload=payload, time=t)


def test_ui14_every_finished_tool_call_is_printed_once() -> None:
    """One call line per finished call, and the call count matches the events."""
    events = demo_events()
    r = Renderer()
    text = render(events, r)
    finished = [e for e in events if e.section == "tool" and e.status in ("finished", "failed")]
    assert r.tool_calls == len(finished)
    assert text.count("▸ relax  ") == sum(1 for e in finished if e.payload["tool"] == "relax")
    assert text.count("▸ request_review") == 2
    assert not r.calls  # every started call was paired with its end


def test_parallel_calls_pair_by_call_id() -> None:
    """Interleaved starts and ends still print each result under its own call."""
    a = {"tool": "relax", "call_id": 1, "input": {"candidate_id": "c1-1"}, "candidate_id": "c1-1"}
    b = {"tool": "relax", "call_id": 2, "input": {"candidate_id": "c1-2"}, "candidate_id": "c1-2"}
    res_a = {**a, "result": {"converged": True, "energy_eV_per_atom": -1.0}}
    res_b = {**b, "result": {"converged": False, "energy_eV_per_atom": -2.0}}
    text = render(
        [
            event("tool", "started", a),
            event("tool", "started", b, 1),
            event("tool", "finished", res_b, 2),
            event("tool", "finished", res_a, 5),
        ]
    )
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines[0].startswith("▸ relax  c1-2") and "not converged" in lines[1] and "\u22122.000" in lines[1]
    assert lines[2].startswith("▸ relax  c1-1") and "\u22121.000" in lines[3]
    assert "5s" in lines[2]  # duration from the event timestamps


def test_sys4_values_carry_their_uncertainty() -> None:
    """Predicted values print as value ± uncertainty."""
    text = render(demo_events())
    assert "band gap 2.60 ± 0.65 eV" in text
    assert fmt_quantity(2.6, None, "eV") == "2.60 eV"


def test_review_panel_shows_rubric_reviewer_and_gate_note() -> None:
    """The review panel carries the code rubric, the reviewer's confidence and why the stop was blocked."""
    text = render(demo_events())
    assert "REVIEW · ROUND 1" in text and "75/100" in text and "60/100" in text
    assert "reviewer confidence 60 < 72" in text
    assert text.index("REVIEW · ROUND 1") < text.index("Round 1 · best c1-1")  # round line after its review


def test_ui15_confidence_trend_and_tables() -> None:
    """Rounds show sparklines; the tables list candidates and rounds."""
    r = Renderer()
    text = render(demo_events(), r)
    assert sparkline([75, 100]) in text
    console = make_console(file=io.StringIO(), width=140, record=True)
    console.print(r.candidates_table())
    console.print(r.rounds_table())
    tables = console.export_text()
    assert "SrTi0.75Fe0.25O3" in tables and "2.20 ± 0.65" in tables and "stop · target_met" in tables


def test_final_check_brief_and_summary() -> None:
    """Validation verdict, brief line and the end-of-run summary."""
    r = Renderer()
    text = render(demo_events(), r)
    assert "FINAL RE-CHECK · CONFIRMED" in text and "Brief written · cites 2 papers" in text
    console = make_console(file=io.StringIO(), width=140, record=True)
    console.print(r.summary(brief_path="runs/r-demo/brief.md"))
    summary = console.export_text()
    assert "finished · target_met" in summary and "c2-1 SrTi0.75Fe0.25O3" in summary
    assert "hephaestus replay r-demo" in summary and "runs/r-demo/brief.md" in summary


def test_marks_follow_the_event_flags_not_the_numbers() -> None:
    """The renderer shows the harness's verdict even when it disagrees with a naive reading."""
    base = {"tool": "energy_above_hull", "input": {"candidate_id": "c1-1"}, "candidate_id": "c1-1"}
    stable = render(
        [
            event(
                "tool",
                "finished",
                {**base, "call_id": 1, "result": {"e_hull_eV_per_atom": 0.3, "thermo_stable": True}},
            )
        ]
    )
    unstable = render(
        [
            event(
                "tool",
                "finished",
                {**base, "call_id": 2, "result": {"e_hull_eV_per_atom": 0.01, "thermo_stable": False}},
            )
        ]
    )
    assert "✓ thermodynamically stable" in stable
    assert "✗ thermodynamically stable" in unstable


def test_failed_and_unknown_tools() -> None:
    """A failed call shows its error; a tool without a summary uses the fallback."""
    text = render(
        [
            event(
                "tool",
                "failed",
                {
                    "tool": "relax",
                    "call_id": 1,
                    "input": {"candidate_id": "c9"},
                    "result": {"error": "Call build_structure first."},
                },
            ),
            event(
                "tool",
                "finished",
                {"tool": "mystery", "call_id": 2, "input": {"x": 1}, "result": {"answer": 42}},
            ),
        ]
    )
    assert "Error: Call build_structure first." in text
    assert "▸ mystery  1" in text and "answer 42" in text


def test_thinking_is_shortened_unless_verbose() -> None:
    """Long thinking is cut to two lines by default."""
    long = "\n".join(f"step {i}" for i in range(6))
    short = render([event("agent", "finished", {"kind": "thinking", "text": long})])
    full = render([event("agent", "finished", {"kind": "thinking", "text": long})], Renderer(verbose=True))
    assert "step 1" in short and "step 4" not in short and "4 more lines" in short
    assert "step 5" in full


def test_status_line_names_the_running_tool() -> None:
    """While a call runs, the status line says what is happening, with round and budget."""
    now = [100.0]
    r = Renderer(clock=lambda: now[0])
    r.feed(event("run", "started", {"config": {"max_iterations": 3, "max_tool_calls": 90}, "goal": "g"}))
    r.feed(
        event(
            "tool",
            "started",
            {"tool": "phonons", "call_id": 7, "input": {"candidate_id": "c1-1"}, "candidate_id": "c1-1"},
        )
    )
    now[0] = 112.0
    console = make_console(file=io.StringIO(), width=140, record=True)
    console.print(r.status())
    line = console.export_text()
    assert "Computing phonons for c1-1" in line and "round 1/3" in line and "0/90 tools" in line
    assert "▹ phonons c1-1 (12s)" in line and "^C stop" in line
    r.stopping = True
    console.print(r.status())
    assert "Stopping" in console.export_text()


def test_parser_questions_panel() -> None:
    """Open questions from the parser are shown for the user to answer."""
    text = render(
        [
            event(
                "goal_parser",
                "failed",
                {"questions": ["Which band gap target, in eV?"], "errors": [], "spec": None},
            )
        ]
    )
    assert "NEEDS INPUT" in text and "Which band gap target, in eV?" in text


def test_logo_block_letters_or_wordmark() -> None:
    """Wide terminals get six rows of block letters; narrow ones a one-line wordmark."""
    from hephaestus.interfaces.cli.theme import LOGO_WIDTH, logo

    wide = logo(LOGO_WIDTH + 10).plain.splitlines()
    assert len(wide) == 6 and all("█" in row for row in wide[:5])
    narrow = logo(60).plain
    assert "\n" not in narrow and "H E P H A E S T U S" in narrow


def test_gutter_runs_down_every_line_of_a_campaign() -> None:
    """Inside a campaign, every printed line, panels included, starts with the timeline bar."""
    events = demo_events()
    start = next(i for i, e in enumerate(events) if e.section == "orchestrator" and e.status == "started")
    review = next(
        i
        for i, e in enumerate(events)
        if e.payload.get("tool") == "request_review" and e.status == "finished"
    )
    text = render(events[: review + 1])
    campaign = text[text.index("━━ ROUND 1") :].splitlines()[1:]
    assert campaign and all(line.startswith(("┃", "━━ ROUND")) for line in campaign if line.strip())
    assert "━━ ROUND 2" in text  # rules open each round, outside the bar
    assert start < review


def test_fc9_final_steps_stream_under_the_re_check() -> None:
    """Each finished re-check step prints once; the running one is named in the status line."""
    now = [0.0]
    r = Renderer(clock=lambda: now[0])
    events = [
        event("orchestrator", "started", {"backend": "api", "model": "m"}),
        event("final_calculation", "started", {"candidates": ["c2-1", "c1-1"]}, 1),
        event(
            "final_step",
            "started",
            {"tool": "relax", "candidate_id": "c2-1", "potential": "mace-mp-0-medium"},
            2,
        ),
        event(
            "final_step",
            "finished",
            {"tool": "relax", "candidate_id": "c2-1", "potential": "mace-mp-0-medium", "wall_s": 12.3},
            14,
        ),
        event(
            "final_step",
            "started",
            {"tool": "phonons", "candidate_id": "ref", "potential": "mace-mp-0-medium", "mode": "full"},
            15,
        ),
    ]
    text = render(events, r)
    assert "▸ Re-relax  c2-1 · mace-mp-0-medium" in text and "12s" in text
    assert "Full phonons" not in text  # still running
    console = make_console(file=io.StringIO(), width=140, record=True)
    console.print(r.status())
    assert "Re-checking: Full phonons for the reference · mace-mp-0-medium" in console.export_text()
    text = render(
        [
            event(
                "final_step",
                "failed",
                {"tool": "phonons", "candidate_id": "ref", "potential": "p", "mode": "full", "wall_s": 3},
            )
        ],
        r,
    )
    assert "Full phonons  reference · p · full" in text and "failed" in text and not r.final_steps


def test_or9_campaign_and_final_time_against_budgets() -> None:
    """Campaign and re-check time come from event timestamps and are shown against their budgets."""
    cfg = {"time_budget_s": 900, "final_time_budget_s": 200, "max_iterations": 3, "max_tool_calls": 90}
    r = Renderer()
    render(
        [
            event("run", "started", {"config": cfg, "goal": "g"}),
            event("orchestrator", "started", {"backend": "api", "model": "m"}, 10),
            event(
                "orchestrator",
                "finished",
                {"submitted": {"candidate_id": "c1-1"}, "rule": "x", "usage": {}},
                250,
            ),
            event("final_calculation", "started", {"candidates": ["c1-1"]}, 260),
            event("final_step", "started", {"tool": "relax", "candidate_id": "c1-1"}, 261),
            event("final_step", "finished", {"tool": "relax", "candidate_id": "c1-1", "wall_s": 160}, 430),
        ],
        r,
    )
    console = make_console(file=io.StringIO(), width=140, record=True)
    console.print(r.budget_text())
    text = console.export_text()
    assert "campaign 4m 00s / 15m 00s" in text  # 10 s to 250 s
    assert "re-check 2m 50s / 3m 20s" in text  # 260 s to the last event, still open (not live)
    assert r.budget_text().spans[-2].style == "warn"  # 170 of 200 s is above 80 %


def test_ui5_best_so_far_after_each_round() -> None:
    """Each round closes with the best candidate's code-computed facts."""
    text = render(demo_events())
    assert "best so far c1-1 SrTi0.875Fe0.125O3 · band gap 2.60 ± 0.65 eV · E_hull 0.030 eV/atom ✓" in text
