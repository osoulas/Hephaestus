"""Claude Code backend: a signed-out CLI is reported, not left hanging (OR-11)."""

import pytest

from hephaestus.llm import claude_code
from hephaestus.llm.base import LLMError
from hephaestus.llm.claude_code import ClaudeCodeLLM


def test_or11_signed_out_cli_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the CLI signed out, structured calls and campaigns raise LLMError at once."""
    monkeypatch.setattr(
        claude_code,
        "claude_code_status",
        lambda: {"signed_in": False, "auth_method": None, "error": "signed out"},
    )
    llm = ClaudeCodeLLM("claude-opus-5-5")
    with pytest.raises(LLMError, match="not signed in"):
        llm.structured("reviewer", "s", "p", {})
    with pytest.raises(LLMError, match="not signed in"):
        llm.run_campaign("s", "k", [], None)  # type: ignore[arg-type]


def test_step_is_not_used() -> None:
    """The claude-code backend drives its own loop."""
    with pytest.raises(LLMError):
        ClaudeCodeLLM("claude-opus-5-5", check_auth=False).step("s", [], [])


def test_or11_sessions_see_only_hephaestus_tools() -> None:
    """Account connectors are switched off and any non-Hephaestus tool is refused."""
    import asyncio

    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    opts = ClaudeCodeLLM("claude-opus-5-5", check_auth=False)._options("system")
    assert opts.tools == [] and opts.setting_sources == [] and opts.strict_mcp_config
    assert opts.env["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"
    deny = asyncio.run(opts.can_use_tool("mcp__claude_ai_Docs__create", {}, None))
    allow = asyncio.run(opts.can_use_tool(f"mcp__{claude_code.SERVER}__relax", {}, None))
    assert isinstance(deny, PermissionResultDeny) and isinstance(allow, PermissionResultAllow)
