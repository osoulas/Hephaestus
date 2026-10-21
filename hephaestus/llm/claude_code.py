"""Claude Agent SDK backend with long-lived sessions, one per call type (OR-11).

The orchestrator runs as a Claude Code session whose only tools are ours, served from an
in-process MCP server; built-in Claude Code tools are disabled and no user settings are
loaded. Every tool call goes through ``Campaign.call_tool``, so budget, gates and events
behave exactly as on the API backend. Structured calls (reviewer, goal parser, brief) reuse
one connected session per call type: a cold start costs about 100 s, a reused one about 2 s.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import threading
from collections.abc import Coroutine
from typing import Any

from hephaestus.llm.auth import claude_code_status
from hephaestus.llm.base import Backend, CampaignHost, LLMError, Turn

SERVER = "hephaestus"
STREAM_CLOSE_TIMEOUT_MS = "300000"
MAX_NUDGES = 3
STRUCTURED_TIMEOUT_S = 600.0


async def only_hephaestus_tools(name: str, args: dict[str, Any], context: Any) -> Any:
    """Refuse any tool that is not one of ours, should the CLI offer one anyway (OR-11).

    Parameters
    ----------
    name : str
        Tool the model asked for.
    args : dict
        Its input.
    context : ToolPermissionContext
        Unused.

    Returns
    -------
    PermissionResultAllow or PermissionResultDeny
        Allow for ``mcp__hephaestus__*``; deny with a message otherwise.
    """
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    if name.startswith(f"mcp__{SERVER}__"):
        return PermissionResultAllow()
    return PermissionResultDeny(
        message=f"{name} is not available in Hephaestus; use the Hephaestus tools only."
    )


class ClaudeCodeLLM:
    """LLM backend on a signed-in Claude Code CLI via the Claude Agent SDK."""

    backend: Backend = "claude-code"

    def __init__(self, model: str, check_auth: bool = True) -> None:
        """Create the backend; sessions start lazily.

        Parameters
        ----------
        model : str
            Claude model ID.
        check_auth : bool
            Check ``claude auth status`` before the first call (a signed-out CLI hangs).
        """
        self.model = model
        self._check_auth = check_auth
        self._authed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._sessions: dict[str, Any] = {}
        self._lock = threading.Lock()
        os.environ.setdefault("CLAUDE_CODE_STREAM_CLOSE_TIMEOUT", STREAM_CLOSE_TIMEOUT_MS)

    # ------------------------------------------------------------------ plumbing

    def _ensure_auth(self) -> None:
        """Fail fast when the CLI is signed out.

        Raises
        ------
        LLMError
            If Claude Code is not signed in.
        """
        if self._authed or not self._check_auth:
            return
        status = claude_code_status()
        if not status["signed_in"]:
            raise LLMError(f"Claude Code is not signed in: {status['error']}")
        self._authed = True

    def _run(self, coro: Coroutine[Any, Any, Any], timeout: float | None = None) -> Any:
        """Run a coroutine on the backend's own event loop thread.

        Parameters
        ----------
        coro : coroutine
            Work to run.
        timeout : float, optional
            Seconds to wait.

        Returns
        -------
        Any
            The coroutine's result.
        """
        with self._lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                self._thread = threading.Thread(
                    target=self._loop.run_forever, name="claude-code", daemon=True
                )
                self._thread.start()
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def _options(self, system: str, **kw: Any) -> Any:
        """Build session options: our tools only, no user settings, adaptive thinking.

        Parameters
        ----------
        system : str
            System prompt.
        **kw
            Extra ``ClaudeAgentOptions`` fields.

        Returns
        -------
        ClaudeAgentOptions
            Options.
        """
        from claude_agent_sdk import ClaudeAgentOptions

        return ClaudeAgentOptions(
            model=self.model,
            system_prompt=system,
            tools=[],
            setting_sources=[],
            # The signed-in account's claude.ai connectors (Docs, Slack, ...) are not settings, so
            # setting_sources=[] does not hide them: switch them off and use only our MCP server.
            strict_mcp_config=True,
            can_use_tool=only_hephaestus_tools,
            thinking={"type": "adaptive", "display": "summarized"},
            env={
                "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT": STREAM_CLOSE_TIMEOUT_MS,
                "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
            },
            **kw,
        )

    def close(self) -> None:
        """Disconnect every reused session and stop the loop thread."""
        if self._loop is None:
            return

        async def _close() -> None:
            """Disconnect sessions."""
            for client in list(self._sessions.values()):
                with contextlib.suppress(Exception):
                    await client.disconnect()
            self._sessions.clear()

        try:
            self._run(_close(), timeout=30)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop = None

    # ------------------------------------------------------------------ LLM protocol

    def step(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Turn:
        """Not used: this backend drives its own loop (``run_campaign``).

        Parameters
        ----------
        system : str
            Unused.
        messages : list of dict
            Unused.
        tools : list of dict
            Unused.

        Returns
        -------
        Turn
            Never returns.

        Raises
        ------
        LLMError
            Always.
        """
        raise LLMError("The claude-code backend drives its own loop; use run_campaign.")

    def structured(self, section: str, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Answer in JSON matching ``schema``, reusing one session per call type.

        Parameters
        ----------
        section : str
            Call type: ``reviewer``, ``goal_parser``, ``brief``.
        system : str
            System prompt.
        prompt : str
            User prompt.
        schema : dict
            JSON schema of the answer.

        Returns
        -------
        dict
            Parsed answer.

        Raises
        ------
        LLMError
            On sign-in, session or parsing failure.
        """
        self._ensure_auth()
        key = (
            section
            + ":"
            + hashlib.sha256((system + json.dumps(schema, sort_keys=True)).encode()).hexdigest()[:16]
        )
        try:
            return dict(self._run(self._structured(key, system, prompt, schema), STRUCTURED_TIMEOUT_S))
        except LLMError:
            raise
        except Exception as exc:
            self._sessions.pop(key, None)
            raise LLMError(f"claude-code {section} call failed: {type(exc).__name__}: {exc}") from exc

    async def _structured(self, key: str, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Send one prompt on the reused session and read the structured result.

        Parameters
        ----------
        key : str
            Session key.
        system : str
            System prompt.
        prompt : str
            User prompt.
        schema : dict
            JSON schema.

        Returns
        -------
        dict
            Parsed answer.

        Raises
        ------
        LLMError
            When the session reports an error or no parsable result.
        """
        from claude_agent_sdk import ClaudeSDKClient, ResultMessage

        client = self._sessions.get(key)
        if client is None:
            opts = self._options(system, output_format={"type": "json_schema", "schema": schema}, max_turns=3)
            client = ClaudeSDKClient(opts)
            await client.connect()
            self._sessions[key] = client
        await client.query(prompt)
        result: Any = None
        async for msg in client.receive_response():
            if isinstance(msg, ResultMessage):
                self.last_usage = {k: int(v) for k, v in (msg.usage or {}).items() if isinstance(v, int)}
                if msg.is_error:
                    raise LLMError(f"Claude Code error: {msg.result or msg.subtype}")
                result = msg.structured_output if msg.structured_output is not None else msg.result
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError as exc:
                raise LLMError(f"Unparsable answer: {result[:200]!r}") from exc
        if not isinstance(result, dict):
            raise LLMError("No structured result.")
        return result

    # ------------------------------------------------------------------ campaign

    def run_campaign(
        self, system: str, kickoff: str, tools: list[dict[str, Any]], campaign: CampaignHost
    ) -> None:
        """Drive the orchestrator as a Claude Code session with our tools on an in-process MCP server.

        Parameters
        ----------
        system : str
            Orchestrator system prompt.
        kickoff : str
            First user message.
        tools : list of dict
            Tool definitions (``TOOLS``).
        campaign : CampaignHost
            Receives every tool call through ``call_tool``.

        Raises
        ------
        LLMError
            If Claude Code is not signed in or the session fails.
        """
        self._ensure_auth()
        try:
            self._run(self._campaign(system, kickoff, tools, campaign))
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"claude-code campaign failed: {type(exc).__name__}: {exc}") from exc

    async def _campaign(
        self, system: str, kickoff: str, tools: list[dict[str, Any]], campaign: CampaignHost
    ) -> None:
        """Run the session until a submission is accepted, nudging when the model stops early.

        Parameters
        ----------
        system : str
            System prompt.
        kickoff : str
            First message.
        tools : list of dict
            Tool definitions.
        campaign : CampaignHost
            The campaign.
        """
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeSDKClient,
            ResultMessage,
            TextBlock,
            ThinkingBlock,
            create_sdk_mcp_server,
            tool,
        )

        def make(name: str, description: str, schema: dict[str, Any]) -> Any:
            """Wrap one tool so it runs through the campaign in a worker thread.

            Parameters
            ----------
            name : str
                Tool name.
            description : str
                Tool description.
            schema : dict
                Input schema.

            Returns
            -------
            SdkMcpTool
                The MCP tool.
            """

            async def handler(args: dict[str, Any]) -> dict[str, Any]:
                """Run the tool call.

                Parameters
                ----------
                args : dict
                    Tool input.

                Returns
                -------
                dict
                    MCP tool result.
                """
                result, is_error = await asyncio.to_thread(campaign.call_tool, name, dict(args))
                text = json.dumps(result, ensure_ascii=False, default=str)
                return {"content": [{"type": "text", "text": text}], "is_error": is_error}

            return tool(name, description, schema)(handler)

        sdk_tools = [make(t["name"], t["description"], t["input_schema"]) for t in tools]
        server = create_sdk_mcp_server(SERVER, tools=sdk_tools)
        opts = self._options(
            system,
            mcp_servers={SERVER: server},
            allowed_tools=[f"mcp__{SERVER}__{t['name']}" for t in tools],
            effort="high",
            max_turns=campaign.ws.cfg.max_tool_calls + 40,
        )
        async with ClaudeSDKClient(opts) as client:
            prompt = kickoff
            for _ in range(MAX_NUDGES + 1):
                await client.query(prompt)
                interrupted = False
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, ThinkingBlock):
                                campaign.emit_agent("thinking", block.thinking)
                            elif isinstance(block, TextBlock):
                                campaign.emit_agent("text", block.text)
                    elif isinstance(msg, ResultMessage):
                        for k, v in (msg.usage or {}).items():
                            if isinstance(v, int):
                                campaign.usage[k] = campaign.usage.get(k, 0) + v
                    if not interrupted and (campaign.ws.submitted or campaign.hard_limit_hit()):
                        interrupted = True
                        await client.interrupt()
                if campaign.ws.submitted or campaign.hard_limit_hit():
                    return
                prompt = campaign.nudge()
