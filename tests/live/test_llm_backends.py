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
