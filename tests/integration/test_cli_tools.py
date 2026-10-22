"""CLI on agent-built tools: decision prompts, replay, resume and the tool library (AT-3, AT-21, AT-24, AT-28)."""

import faulthandler
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import scripts
from test_cli import factory, lines
from test_custom_tools import GOAL, PHASE_TURNS, Killed, campaign_turns, cfg, decider, deps, write_turn
from toolkit_fake import TEST_SOURCE, TOOL_SOURCE, plan

from hephaestus.core.config import RunConfig
from hephaestus.harness import runner
from hephaestus.interfaces.cli import repl
from hephaestus.interfaces.cli.app import build_parser, config_from, main
from hephaestus.interfaces.cli.theme import make_console
from hephaestus.tools import custom


@pytest.fixture(autouse=True)
def _cache_and_watchdog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the cache in the test folder, and end a hung test (a run waiting for an answer) loudly.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test folder.
    monkeypatch : pytest.MonkeyPatch
        Sets ``HEPHAESTUS_CACHE_DIR``.

    Yields
    ------
    None
        While the test runs.
    """
    monkeypatch.setenv("HEPHAESTUS_CACHE_DIR", str(tmp_path / "cache"))
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def gap_run(tmp_path: Path, turns: list, answers: list[str], *extra: str) -> tuple[int, str]:
    """Run the gap goal with ``hephaestus run --custom-tools ask``, answering the cards from a script.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Runs folder (the library is ``tmp_path/lib``).
    turns : list
        Scripted orchestrator turns.
    answers : list of str
        Lines typed at the prompts, in order.
    *extra : str
        More command-line options.

    Returns
    -------
    tuple
        Exit code and the printed text.
    """
    console = make_console(file=io.StringIO(), width=140, record=True)
    argv = ["run", GOAL, "--custom-tools", "ask", "--tool-library", str(tmp_path / "lib"), *extra]
    code = main(
        argv,
        console=console,
        root=tmp_path,
        deps=lambda: deps(tmp_path, turns, None),
        read_line=lines(*answers),
    )
    return code, console.export_text()


def run_ids(tmp_path: Path) -> list[str]:
    """Return the run folders under ``tmp_path``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Runs folder.

    Returns
    -------
    list of str
        Run IDs.
    """
    return [p.parent.name for p in tmp_path.glob("*/state.json")]


def test_at28_cli_accepts_the_plan_and_the_code(tmp_path: Path) -> None:
    """The plan card, then the code card with its checks, are answered in the terminal; the tool registers."""
    code, out = gap_run(tmp_path, PHASE_TURNS + campaign_turns(), ["a", "a"])
    assert code == 0, out
    assert "No tool computes" in out  # the spec panel's gap note
    assert "toy_expansion" in out and "Toy method for tests." in out
    assert "Your decision:" in out and "approve the code" in out
    assert "def run(structure, inputs, hp)" in out  # the code is shown (printed off a terminal)
    assert "You accepted the plan" in out and "You accepted the code" in out
    assert "saved to the tool library" in out
    assert [t.name for t in custom.scan_library(tmp_path / "lib")[0]] == ["toy_expansion"]


def test_at7_cli_change_request_reaches_the_agent(tmp_path: Path) -> None:
    """Request changes asks for the message, and the agent's revised plan is a new card to answer."""
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
        told.append(json.dumps(messages[-1]["content"]))
        return [{"tool": "propose_tool", "input": plan() | {"method": "Equation-of-state fit."}}]

    turns = [[{"tool": "propose_tool", "input": plan()}], revise, write_turn(), *campaign_turns()]
    code, out = gap_run(tmp_path, turns, ["c", "use an EOS fit", "a", "a"])
    assert code == 0, out
    assert "use an EOS fit" in told[0]
    assert "Equation-of-state fit." in out and "revision 2" in out.lower()


def test_at8_cli_cancel_the_tool_screens_stability_only(tmp_path: Path) -> None:
    """Cancel the tool confirms first; the run continues without the target."""
    turns = [
        [{"tool": "propose_tool", "input": plan()}],
        *campaign_turns()[:4],
        [scripts.submit("c1-1", accept_not_met=True)],
    ]
    turns[3] = [s for s in turns[3] if not s.get("tool", "").startswith("custom_")]
    code, out = gap_run(tmp_path, turns, ["x", "y"])
    assert "cancel the tool" in out
    assert not custom.scan_library(tmp_path / "lib")[0]
    state = json.loads((tmp_path / run_ids(tmp_path)[0] / "state.json").read_text(encoding="utf-8"))
    assert state["spec"]["target_evaluable"] is False and state["status"] == "finished", code


def test_at9_cli_cancel_the_run(tmp_path: Path) -> None:
    """Cancel the run (after a declined confirmation) stops it with the stopped summary."""
    code, out = gap_run(tmp_path, [[{"tool": "propose_tool", "input": plan()}]], ["q", "n", "q", "y"])
    assert code == 1
    assert out.count("Your decision:") == 2  # the first cancellation was declined
    assert "cancelled by user" in out


def test_cli_question_package_and_give_up(tmp_path: Path) -> None:
    """A numbered answer, a denied package and agreeing to stop all go through the prompts."""
    told: list[str] = []

    def after(messages: list) -> list:
        """Record the answer, then ask for a package.

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

    camp = campaign_turns()
    camp[2] = [s for s in camp[2] if not s.get("tool", "").startswith("custom_")]
    camp[4] = [scripts.submit("c1-1", accept_not_met=True)]
    turns = [
        [{"tool": "ask_user", "input": {"question": "Strain or EOS?", "options": ["Strain", "EOS"]}}],
        after,
        last,
        *camp,
    ]
    code, out = gap_run(tmp_path, turns, ["2", "d", "s"])
    assert "Strain or EOS?" in out and "[1] Strain" in out
    assert "EOS" in told[0] and "did not allow phonopy" in told[1]
    assert "needs phonopy" in out and "let it stop" in out
    assert code in (0, 1)


def test_at3_cli_ask_refused_when_nobody_can_answer(tmp_path: Path) -> None:
    """``run --custom-tools ask --yes`` is refused before anything is parsed or started."""
    code, out = gap_run(tmp_path, PHASE_TURNS, [], "--yes")
    assert code == 1 and "need someone to review" in out
    assert not run_ids(tmp_path)


def test_at3_cli_run_defaults_to_off_and_the_session_to_ask() -> None:
    """``run`` leaves agent-built tools off; the interactive session turns them to ``ask``."""
    args = build_parser().parse_args(["run", GOAL])
    assert config_from(args).custom_tools == "off"
    assert config_from(build_parser().parse_args([]), custom_tools="ask").custom_tools == "ask"
    off = build_parser().parse_args(["--custom-tools", "off"])
    assert config_from(off, custom_tools="ask").custom_tools == "off"


def test_at24_cli_resume_asks_the_waiting_card_again(tmp_path: Path) -> None:
    """A run killed at the code review asks for it again on ``resume``, then finishes."""
    decide, _ = decider("accept", Killed())
    with pytest.raises(Killed):
        runner.run(GOAL, cfg=cfg(tmp_path), deps=deps(tmp_path, PHASE_TURNS, decide), run_id="r1")
    console = make_console(file=io.StringIO(), width=140, record=True)
    code = main(
        ["resume", "r1"],
        console=console,
        root=tmp_path,
        deps=lambda: deps(tmp_path, campaign_turns(), None),
        read_line=lines("a"),
    )
    out = console.export_text()
    assert code == 0, out
    assert out.count("Your decision:") == 1 and "You accepted the code" in out


def test_ui9_cli_replay_shows_decisions_without_asking(tmp_path: Path) -> None:
    """A replay of a gap run shows the cards and the decisions, and never prompts."""
    code, _ = gap_run(tmp_path, PHASE_TURNS + campaign_turns(), ["a", "a"])
    assert code == 0
    rid = run_ids(tmp_path)[0]
    console = make_console(file=io.StringIO(), width=140, record=True)
    code = main(["replay", rid, "--speed", "0"], console=console, root=tmp_path, read_line=lines())
    out = console.export_text()
    assert code == 0 and "You accepted the plan" in out and "Your decision:" not in out


# ---------------------------------------------------------------- tool library


def library(tmp_path: Path) -> Path:
    """Write a hand-made tool waiting for review.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test folder.

    Returns
    -------
    pathlib.Path
        Library folder.
    """
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "toy_expansion.py").write_text(
        TOOL_SOURCE.replace('"author": "agent"', '"author": "user"'), encoding="utf-8"
    )
    (lib / "test_toy_expansion.py").write_text(TEST_SOURCE, encoding="utf-8")
    return lib


def tools(tmp_path: Path, *argv: str, answers: Any = None) -> tuple[int, str]:
    """Run ``hephaestus tools ...`` on the test library.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test folder.
    *argv : str
        Action and name.
    answers : callable, optional
        Line reader.

    Returns
    -------
    tuple
        Exit code and the printed text.
    """
    console = make_console(file=io.StringIO(), width=140, record=True)
    code = main(
        ["tools", *argv, "--dir", str(tmp_path / "lib")],
        console=console,
        root=tmp_path,
        read_line=answers or lines(),
    )
    return code, console.export_text()


def test_at21_tools_list_review_approve_show_remove(tmp_path: Path) -> None:
    """A waiting file is listed, reviewed (code shown) and approved by its fingerprint, shown, then removed."""
    library(tmp_path)
    code, out = tools(tmp_path, "list")
    assert code == 0 and "waiting for review" in out and "needs review" in out
    code, out = tools(tmp_path, "review", answers=lines("a"))
    assert code == 0 and "def run(structure, inputs, hp)" in out and "Approved" in out
    assert [t.name for t in custom.scan_library(tmp_path / "lib")[0]] == ["toy_expansion"]
    code, out = tools(tmp_path, "show", "toy_expansion")
    assert code == 0 and "approved" in out and "def test_positive" in out
    code, out = tools(tmp_path, "remove", "toy_expansion", answers=lines("y"))
    assert code == 0 and "Removed" in out
    assert not (tmp_path / "lib" / "toy_expansion.py").exists()


def test_at21_tools_review_refuses_a_file_edited_after_it_was_shown(tmp_path: Path) -> None:
    """Approval carries the fingerprint shown: a file changed meanwhile is refused by the engine."""
    lib = library(tmp_path)

    def edit_then_approve(prompt: str) -> str:
        """Edit the tool while the user reads it, then approve.

        Parameters
        ----------
        prompt : str
            Ignored.

        Returns
        -------
        str
            ``a``.
        """
        path = lib / "toy_expansion.py"
        path.write_text(path.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
        return "a"

    code, out = tools(tmp_path, "review", answers=edit_then_approve)
    assert "Approved" not in out
    assert not custom.scan_library(lib)[0]
    assert code == 0


def test_tools_remove_needs_confirmation_and_a_known_name(tmp_path: Path) -> None:
    """Remove asks first; an unknown name is an error."""
    library(tmp_path)
    code, out = tools(tmp_path, "remove", "toy_expansion", answers=lines("n"))
    assert code == 0 and "Kept" in out and (tmp_path / "lib" / "toy_expansion.py").exists()
    code, out = tools(tmp_path, "remove", "nothing_here")
    assert code == 1 and "No tool called nothing_here" in out


# ---------------------------------------------------------------- settings, titles, briefs


def test_or4_profiles_and_new_options() -> None:
    """Profiles set budgets that explicit options override; the new switches reach ``RunConfig``."""
    args = build_parser().parse_args(
        [
            "run",
            "g",
            "--profile",
            "quick",
            "--max-rounds",
            "4",
            "--no-hull",
            "--no-elastic",
            "--e-hull-max",
            "0.1",
        ]
    )
    c = config_from(args)
    assert c.max_iterations == 4 and c.max_tool_calls == 50 and c.time_budget_s == 480
    assert c.run_hull is False and c.disabled_capabilities == ["elastic"] and c.e_hull_max_eV_per_atom == 0.1
    assert (
        config_from(build_parser().parse_args(["run", "g", "--fidelity", "pbe"])).band_gap_model
        == "megnet-pbe"
    )
    model = build_parser().parse_args(["run", "g", "--band-gap-model", "megnet-scan"])
    assert config_from(model).band_gap_model == "megnet-scan"
    assert config_from(build_parser().parse_args(["run", "g", "--profile", "thorough"])).phonon_mode == "full"


def test_or4_config_lists_every_option_grouped(tmp_path: Path) -> None:
    """/config covers every ``RunConfig`` field; budgets of 0 read as no limit; /profile applies."""
    assert set(repl.CONFIG_KEYS) == set(RunConfig.model_fields)
    console = make_console(file=io.StringIO(), width=140, record=True)
    s = repl.Session(console, RunConfig(max_tool_calls=0), read_line=lines(), root=tmp_path)
    s.command("/config")
    s.command("/profile thorough")
    s.command("/config custom_tools ask")
    s.command("/config disabled_capabilities elastic")
    out = console.export_text()
    assert "agent-built tools" in out and "stability checks" in out and "0 (no limit)" in out
    assert s.config.max_iterations == 5 and s.config.phonon_mode == "full"
    assert s.config.custom_tools == "ask" and s.config.disabled_capabilities == ["elastic"]
    s.command("/profile standard")
    assert s.config.max_iterations == RunConfig().max_iterations


def test_ui9_rename_and_compute_brief(tmp_path: Path) -> None:
    """A run's title can be changed and shows in ``runs``; ``brief --compute`` prints the computational brief."""
    console = make_console(file=io.StringIO(), width=140, record=True)
    assert main(["run", scripts.GOAL], console=console, root=tmp_path, deps=factory(), read_line=lines()) == 0
    out = console.export_text()
    assert "compute brief" in out
    rid = run_ids(tmp_path)[0]
    console = make_console(file=io.StringIO(), width=140, record=True)
    assert main(["rename", rid, "Narrow", "gap", "titanates"], console=console, root=tmp_path) == 0
    assert main(["runs"], console=console, root=tmp_path) == 0
    assert main(["brief", rid, "--compute"], console=console, root=tmp_path) == 0
    out = console.export_text()
    assert "Narrow gap titanates" in out and "COMPUTATIONAL BRIEF" in out
