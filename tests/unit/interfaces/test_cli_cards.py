"""CLI rendering of agent-built tools and several targets (AT-8, AT-17, AT-24, AT-28, GP-1)."""

import io
from typing import Any

from test_cli_render import event, render

from hephaestus.interfaces.cli.cards import cancelled_text, library_table
from hephaestus.interfaces.cli.render import Renderer, spec_panel, target_fact
from hephaestus.interfaces.cli.theme import make_console, strip_codes


def proposal(revision: int = 1, **kw: Any) -> dict[str, Any]:
    """Return a tool proposal payload.

    Parameters
    ----------
    revision : int
        Revision number.
    **kw
        Overrides.

    Returns
    -------
    dict
        Proposal.
    """
    return {
        "proposal_id": "t1",
        "name": "toy_expansion",
        "property": "thermal_expansion",
        "units": "1e-6/K",
        "description": "Twice the single-point energy.",
        "method": "Toy method." if revision == 1 else "Equation-of-state fit.",
        "citations": [],
        "building_blocks": ["single_point"],
        "expected_runtime_s": 1.0,
        "supplies_target": True,
        "if_cancelled": "",
        "revision": revision,
        "stage": "plan",
        **kw,
    }


def cap(kind: str, **payload: Any) -> Any:
    """Return a ``capability`` event.

    Parameters
    ----------
    kind : str
        Payload kind.
    **payload
        Other payload fields.

    Returns
    -------
    ProgressEvent
        Event.
    """
    return event("capability", "finished", {"kind": kind, "property": "thermal_expansion", **payload})


def text(renderable: Any) -> str:
    """Print a renderable and return its plain text.

    Parameters
    ----------
    renderable : Any
        Rich renderable.

    Returns
    -------
    str
        Text.
    """
    console = make_console(file=io.StringIO(), width=140, record=True)
    console.print(renderable)
    return console.export_text()


def test_at28_plan_card_and_the_open_card_bookkeeping() -> None:
    """A revised plan replaces the first; a decision closes it; answered cards are not offered again."""
    r = Renderer(live=True)
    allowed = ["accept", "revise", "cancel_tool", "cancel_run"]
    out = render(
        [
            cap("gap", units="1e-6/K"),
            cap("plan", proposal=proposal(), allowed=allowed),
        ],
        r,
    )
    assert "BUILD TOOL" in out and "No tool computes thermal expansion (1e-6/K)" in out
    assert "TOOL PLAN · REVISION 1" in out and "Toy method." in out and "needs your decision" in out
    card = r.open_card()
    assert card is not None and (card["id"], card["stage"], card["revision"]) == ("t1", "plan", 1)
    revise = {"proposal_id": "t1", "stage": "plan", "action": "revise", "message": "use an EOS fit"}
    out = render(
        [
            cap("decision", decision=revise, name="toy_expansion"),
            cap("plan", proposal=proposal(2), allowed=allowed),
        ],
        r,
    )
    assert "use an EOS fit" in out and "REVISION 2" in out
    card = r.open_card()
    assert card is not None and card["revision"] == 2 and len(r.cards) == 1
    r.answered.add(("t1", "plan", 2))
    assert r.open_card() is None  # answered from here, before the decision event arrives
    accept = {"proposal_id": "t1", "stage": "plan", "action": "accept"}
    out = render([cap("decision", decision=accept, name="toy_expansion")], r)
    assert "You accepted the plan for toy_expansion" in out


def test_at28_code_card_checks_withdrawal_and_registration() -> None:
    """The code card shows each check and the trial; a withdrawal closes it; registration is announced."""
    checks = {
        "static": {"passed": True, "report": "passed"},
        "tests": {"passed": True, "report": "2 test(s) passed"},
        "trial": {"passed": True, "report": "10 1e-6/K"},
    }
    code = proposal(stage="code", source="def run(s, i, hp):\n    return 1\n", tests="", checks=checks)
    code["trial"] = {"material": "SrTiO3", "value": 10.0, "units": "1e-6/K", "range": [1.0, 100.0]}
    r = Renderer(live=True)
    out = render([cap("gap"), cap("code", proposal=code, allowed=["accept", "cancel_tool", "cancel_run"])], r)
    assert "CODE REVIEW" in out and "Static check" in out and "2 test(s) passed" in out
    assert "Trial run on SrTiO3" in out
    assert r.open_card() is not None
    out = render([cap("withdrawn", proposal_id="t1", reason="found a unit bug")], r)
    assert "withdrew its code" in out and r.open_card() is None
    out = render([cap("registered", tool={"name": "toy_expansion"})], r)
    assert "saved to the tool library" in out


def test_ui9_a_replay_never_has_a_waiting_card_label() -> None:
    """Replayed cards are not marked as waiting for a decision."""
    out = render([cap("gap"), cap("plan", proposal=proposal(), allowed=["accept"])], Renderer(replaying=True))
    assert "needs your decision" not in out


def test_requests_and_cancellations() -> None:
    """Package, question and give-up cards, and the four end-of-tool texts."""
    r = Renderer(live=True)
    out = render(
        [
            cap("gap"),
            cap(
                "question",
                request={
                    "request_id": "q1",
                    "kind": "question",
                    "question": "Strain or EOS?",
                    "options": ["Strain", "EOS"],
                },
                allowed=["answer", "cancel_run"],
            ),
        ],
        r,
    )
    assert "Strain or EOS?" in out and "Strain" in out
    card = r.open_card()
    assert card is not None and card["stage"] == "question"
    assert cancelled_text({"scope": "run"}) == "You cancelled the run."
    assert "ran out of time" in cancelled_text({"scope": "tool", "reason": "budget"})
    assert "screens stability only" in cancelled_text({"scope": "tool", "reason": "failed"})


def test_at17_agent_built_values_have_no_error_bar_and_targets_list() -> None:
    """A target from an agent-built tool shows no ± and says so; an unevaluated one says not evaluated."""
    built = {
        "property": "thermal_expansion",
        "value": 8.0,
        "uncertainty": None,
        "units": "1e-6/K",
        "met_central": True,
        "agent_built": True,
    }
    assert "(agent-built)" in target_fact(built) and "±" not in target_fact(built)
    gap = {
        "property": "band_gap",
        "value": 2.2,
        "uncertainty": 0.65,
        "units": "eV",
        "met_central": True,
        "met_conservative": False,
    }
    assert "± 0.65" in strip_codes(target_fact(gap)) or "±0.65" in target_fact(gap)
    assert "not evaluated" in target_fact({"property": "x", "evaluable": False})


def test_gp1_spec_panel_lists_every_target_and_the_gap() -> None:
    """Several targets are listed as all required; a gap note explains the review that follows."""
    spec = {
        "title": "Narrow gap, stiff",
        "material_family": "oxide perovskite",
        "target_property": "band_gap",
        "target_direction": "decrease",
        "reference_material": "SrTiO3",
        "extra_targets": [
            {
                "property": "bulk_modulus",
                "direction": "increase",
                "reference_material": "SrTiO3",
                "units": "GPa",
                "evaluable": True,
            }
        ],
        "stability_thresholds": {"e_hull_max_eV_per_atom": 0.05, "thermodynamic": True, "dynamic": False},
    }
    out = text(spec_panel(spec, gap="thermal_expansion"))
    assert "Narrow gap, stiff" in out and "bulk modulus" in out and "all must be met" in out
    assert "No tool computes thermal expansion yet" in out and "phonons off" in out


def test_requirement_codes_are_stripped_from_engine_text() -> None:
    """Engine messages lose their requirement codes, as in the web UI."""
    assert strip_codes("Refused (GP-7): no reference.") == "Refused: no reference."
    assert "SE-5" not in strip_codes("Hull check SE-5 \u2013 SE-13 skipped")


def test_at20_library_table() -> None:
    """Approved tools and files waiting for review are listed separately."""
    lib = {
        "dir": "/tmp/lib",
        "approved": [
            {
                "name": "toy_expansion",
                "property": "thermal_expansion",
                "units": "1e-6/K",
                "author": "agent",
                "run_id": "r1",
            }
        ],
        "unapproved": [
            {
                "name": "hand_made",
                "header": {"property": "x"},
                "changed": True,
                "check": {"passed": True, "report": "ok"},
            }
        ],
    }
    out = text(library_table(lib))
    assert "toy_expansion" in out and "the agent" in out and "edited: review again" in out
