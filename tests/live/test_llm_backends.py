"""Live checks of both LLM backends (pytest -m live). Costs a few cents."""

import pytest

from hephaestus.llm.api import AnthropicAPI
from hephaestus.llm.auth import claude_code_status
from hephaestus.llm.base import api_credentials_present
from hephaestus.llm.claude_code import ClaudeCodeLLM

SCHEMA = {
    "type": "object",
    "properties": {"formula": {"type": "string"}},
    "required": ["formula"],
    "additionalProperties": False,
}
PROMPT = "Give the formula of strontium titanate."


@pytest.mark.live
def test_live_api_structured() -> None:
    """The API backend returns schema-shaped JSON."""
    if not api_credentials_present():
        pytest.skip("no API credentials")
    out = AnthropicAPI("claude-haiku-5-5").structured("goal_parser", "Answer in JSON.", PROMPT, SCHEMA)
    assert "Sr" in out["formula"]


@pytest.mark.live
def test_live_claude_code_structured() -> None:
    """The Claude Code backend returns schema-shaped JSON and reuses its session."""
    if not claude_code_status()["signed_in"]:
        pytest.skip("Claude Code not signed in")
    llm = ClaudeCodeLLM("claude-haiku-5-5")
    try:
        for _ in range(2):
            assert "Sr" in llm.structured("goal_parser", "Answer in JSON.", PROMPT, SCHEMA)["formula"]
    finally:
        llm.close()


@pytest.mark.live
def test_live_claude_code_campaign_on_stub_tools(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A real model drives a one-round campaign through the in-process MCP tools to a submission."""
    if not claude_code_status()["signed_in"]:
        pytest.skip("Claude Code not signed in")
    import scripts

    from hephaestus.core.config import RunConfig
    from hephaestus.core.events import EventBus
    from hephaestus.harness import runner
    from hephaestus.tools.stubs import check_novelty, stub_handlers

    llm = ClaudeCodeLLM("claude-haiku-5-5")
    cfg = RunConfig(
        llm_backend="claude-code",
        llm_model="claude-haiku-5-5",
        max_iterations=1,
        max_tool_calls=30,
        run_final=False,
    )
    deps = runner.Deps(
        llm=llm, handlers=stub_handlers(), novelty_fn=check_novelty, bus=EventBus(tmp_path), root=tmp_path
    )
    try:
        state = runner.run(scripts.GOAL, cfg, deps=deps, run_id="live")
    finally:
        llm.close()
    assert state.status == "finished", state.errors
    assert state.submission is not None and state.brief is not None
    tools = [e.payload["tool"] for e in EventBus(tmp_path).history("live") if e.section == "tool"]
    assert "propose_candidate" in tools and "request_review" in tools
