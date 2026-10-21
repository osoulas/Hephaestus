"""Anthropic API backend (OR-11).

Messages API with tools, adaptive thinking (summarised, so the UI can show reasoning), prompt
caching, streaming for long turns, and server-side refusal fallbacks. The harness owns the
loop; this backend runs one turn at a time and passes every content block back unchanged.
"""

from __future__ import annotations

import json
from typing import Any

from hephaestus.llm.base import Backend, LLMError, ToolCall, Turn

FALLBACK_BETA = "server-side-fallback-2026-07-01"
NO_FALLBACK_MODELS = ("claude-haiku",)  # no server-side fallback for Haiku 5.5
EFFORT = {"orchestrator": "high", "reviewer": "high", "goal_parser": "medium", "brief": "medium"}


class AnthropicAPI:
    """LLM backend on the Anthropic Messages API."""

    backend: Backend = "api"

    def __init__(self, model: str, client: Any | None = None, max_tokens: int = 32000) -> None:
        """Create the backend.

        Parameters
        ----------
        model : str
            Claude model ID, e.g. ``"claude-opus-5-5"``.
        client : anthropic.Anthropic, optional
            Injected client (tests); by default one is built from the environment.
        max_tokens : int
            Output cap per turn; turns are streamed, so large values are safe.
        """
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def _common(self, section: str) -> dict[str, Any]:
        """Request parameters shared by every call.

        Parameters
        ----------
        section : str
            Caller, which sets the effort.

        Returns
        -------
        dict
            Keyword arguments for ``client.beta.messages.stream``.
        """
        kw: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": EFFORT.get(section, "high")},
            "cache_control": {"type": "ephemeral"},
        }
        if not self.model.startswith(NO_FALLBACK_MODELS):
            kw["betas"] = [FALLBACK_BETA]
            kw["fallbacks"] = "default"
        return kw

    def _send(self, **kw: Any) -> Any:
        """Stream one request and return the final message.

        Parameters
        ----------
        **kw
            Request parameters.

        Returns
        -------
        BetaMessage
            The complete response.

        Raises
        ------
        LLMError
            On any API or connection error (after the SDK's own retries).
        """
        import anthropic

        try:
            with self.client.beta.messages.stream(**kw) as stream:
                return stream.get_final_message()
        except anthropic.APIError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

    def step(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Turn:
        """Run one orchestrator turn.

        Parameters
        ----------
        system : str
            System prompt (frozen, so it caches).
        messages : list of dict
            Conversation so far; assistant content is passed back exactly as received.
        tools : list of dict
            Tool definitions.

        Returns
        -------
        Turn
            Content blocks plus text, thinking and tool-call views.
        """
        msg = self._send(system=system, messages=messages, tools=tools, **self._common("orchestrator"))
        return to_turn(msg)

    def structured(self, section: str, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return JSON matching ``schema`` via structured outputs.

        Parameters
        ----------
        section : str
            Caller: ``reviewer``, ``goal_parser``, ``brief``.
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
            On API errors, refusals or unparsable output.
        """
        kw = self._common(section)
        kw["output_config"] = {**kw["output_config"], "format": {"type": "json_schema", "schema": schema}}
        msg = self._send(system=system, messages=[{"role": "user", "content": prompt}], **kw)
        if msg.stop_reason == "refusal":
            raise LLMError(f"Model declined the {section} request.")
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            return dict(json.loads(text))
        except (json.JSONDecodeError, TypeError) as exc:
            raise LLMError(f"Unparsable {section} answer: {text[:200]!r}") from exc


def to_turn(msg: Any) -> Turn:
    """Convert an SDK message into a ``Turn``.

    Parameters
    ----------
    msg : BetaMessage
        API response.

    Returns
    -------
    Turn
        Blocks as plain dicts (unchanged, for the history) plus views.
    """
    turn = Turn(content=[], stop_reason=msg.stop_reason or "end_turn")
    for block in msg.content:
        turn.content.append(block.to_dict() if hasattr(block, "to_dict") else dict(block))
        if block.type == "text":
            turn.texts.append(block.text)
        elif block.type == "thinking":
            turn.thinking.append(getattr(block, "thinking", "") or "")
        elif block.type == "tool_use":
            turn.tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input or {})))
    u = msg.usage
    turn.usage = {
        k: int(getattr(u, k, 0) or 0)
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    }
    return turn
