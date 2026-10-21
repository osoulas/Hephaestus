"""LLM wrapper: logging and delegation (DL-4)."""

from pathlib import Path

from hephaestus.llm.base import LoggedLLM
from hephaestus.llm.claude_code import ClaudeCodeLLM
from hephaestus.llm.scripted import ScriptedLLM


def test_logged_llm_exposes_own_loop_driver(tmp_path: Path) -> None:
    """The wrapper keeps ``run_campaign`` visible, so claude-code drives its own loop."""
    assert callable(
        getattr(LoggedLLM(ClaudeCodeLLM("m", check_auth=False), tmp_path / "l"), "run_campaign", None)
    )
    assert getattr(LoggedLLM(ScriptedLLM(), tmp_path / "l"), "run_campaign", None) is None


def test_dl4_structured_calls_are_logged(tmp_path: Path) -> None:
    """Every structured call is appended to llm_log.jsonl with its section."""
    log = tmp_path / "llm_log.jsonl"
    LoggedLLM(ScriptedLLM(structured={"reviewer": [{"a": 1}]}), log).structured("reviewer", "s", "p", {})
    assert '"section": "reviewer"' in log.read_text(encoding="utf-8")
