"""Anthropic API backend against a fake client (no network)."""

from types import SimpleNamespace
from typing import Any

import pytest

from hephaestus.llm.api import FALLBACK_BETA, AnthropicAPI
from hephaestus.llm.base import LLMError, resolve_backend


class Block(SimpleNamespace):
    """A content block with ``to_dict``."""

    def to_dict(self) -> dict[str, Any]:
        """Return the block as a dict.

        Returns
        -------
        dict
            Block fields.
        """
        return dict(vars(self))


class FakeClient:
    """Records requests and returns a canned message."""

    def __init__(self, content: list[Block], stop_reason: str = "end_turn") -> None:
        """Store the canned reply.

        Parameters
        ----------
        content : list of Block
            Response content.
        stop_reason : str
            Response stop reason.
        """
        self.requests: list[dict[str, Any]] = []
        msg = SimpleNamespace(
            content=content, stop_reason=stop_reason, usage=SimpleNamespace(input_tokens=10, output_tokens=5)
        )
        outer = self

        class Stream:
            """Context manager returning the canned message."""

            def __enter__(self) -> "Stream":
                """Enter.

                Returns
                -------
                Stream
                    Self.
                """
                return self

            def __exit__(self, *a: object) -> None:
                """Exit.

                Parameters
                ----------
                *a : object
                    Ignored.
                """

            def get_final_message(self) -> Any:
                """Return the canned message.

                Returns
                -------
                Any
                    Message.
                """
                return msg

        def stream(**kw: Any) -> Stream:
            """Record the request.

            Parameters
            ----------
            **kw
                Request.

            Returns
            -------
            Stream
                Context manager.
            """
            outer.requests.append(kw)
            return Stream()

        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=stream))


def test_or11_step_maps_blocks_and_requests_fallbacks() -> None:
    """Thinking, text and tool calls are mapped; Opus requests use adaptive thinking and fallbacks."""
    content = [
        Block(type="thinking", thinking="plan", signature="sig"),
        Block(type="text", text="Searching."),
        Block(
            type="tool_use",
            id="t1",
            name="search_literature",
            input={"query": "x", "mode": "semantic", "limit": 5},
        ),
    ]
    client = FakeClient(content, "tool_use")
    turn = AnthropicAPI("claude-opus-5-5", client=client).step("sys", [{"role": "user", "content": "go"}], [])
    assert turn.thinking == ["plan"] and turn.texts == ["Searching."]
    assert turn.tool_calls[0].name == "search_literature" and turn.content[0]["signature"] == "sig"
    req = client.requests[0]
    assert req["thinking"]["type"] == "adaptive" and req["fallbacks"] == "default"
    assert req["betas"] == [FALLBACK_BETA] and req["cache_control"] == {"type": "ephemeral"}


def test_haiku_has_no_fallbacks() -> None:
    """Haiku 5.5 has no server-side fallback, so none is requested."""
    client = FakeClient([Block(type="text", text="ok")])
    AnthropicAPI("claude-haiku-5-5", client=client).step("s", [], [])
    assert "fallbacks" not in client.requests[0]


def test_structured_parses_json_and_raises_on_refusal() -> None:
    """Structured output is parsed; a refusal is an LLMError (reviewer degrades, SC-10)."""
    client = FakeClient([Block(type="text", text='{"a": 1}')])
    assert AnthropicAPI("claude-opus-5-5", client=client).structured("reviewer", "s", "p", {}) == {"a": 1}
    assert client.requests[0]["output_config"]["format"]["type"] == "json_schema"
    with pytest.raises(LLMError):
        AnthropicAPI("claude-opus-5-5", client=FakeClient([], "refusal")).structured("reviewer", "s", "p", {})


def test_or11_auto_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto picks api with credentials, else claude-code."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert resolve_backend("auto") == "claude-code"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert resolve_backend("auto") == "api"
