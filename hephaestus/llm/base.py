"""``structured()`` calls, and backend and model selection (OR-11).

Every backend implements ``LLM``: ``step`` runs one orchestrator turn with tools, and
``structured`` returns JSON matching a schema (reviewer, goal parser, brief). Conversation
history uses the Anthropic Messages format (``role`` plus content blocks) in every backend.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from hephaestus.core.store import append_jsonl

Backend = Literal["api", "claude-code", "scripted"]


class LLMError(Exception):
    """An LLM call failed; callers degrade (SC-10) rather than crash."""


@dataclass
class ToolCall:
    """One tool call requested by the model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Turn:
    """One assistant turn.

    ``content`` holds the assistant blocks exactly as they must be appended to the history
    (thinking blocks unchanged); the other fields are views for the harness.
    """

    content: list[dict[str, Any]]
    tool_calls: list[ToolCall] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    thinking: list[str] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: dict[str, int] = field(default_factory=dict)


class LLM(Protocol):
    """What the harness needs from a backend."""

    backend: Backend
    model: str

    def step(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Turn:
        """Run one orchestrator turn.

        Parameters
        ----------
        system : str
            System prompt.
        messages : list of dict
            Conversation so far, Messages format.
        tools : list of dict
            Tool definitions (``TOOLS``).

        Returns
        -------
        Turn
            The assistant's turn.
        """
        ...

    def structured(self, section: str, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return JSON that matches ``schema``.

        Parameters
        ----------
        section : str
            Caller, for logs and session reuse: ``reviewer``, ``goal_parser``, ``brief``.
        system : str
            System prompt.
        prompt : str
            User prompt.
        schema : dict
            JSON schema of the answer.

        Returns
        -------
        dict
            The parsed answer.
        """
        ...


class CampaignHost(Protocol):
    """What a backend that drives its own loop needs from the harness's ``Campaign``."""

    ws: Any
    usage: dict[str, int]

    def call_tool(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Run one tool call through budget, events and gates.

        Parameters
        ----------
        name : str
            Tool name.
        args : dict
            Tool input.

        Returns
        -------
        tuple
            ``(result, is_error)``.
        """
        ...

    def emit_agent(self, kind: str, text: str) -> None:
        """Publish the orchestrator's text or thinking.

        Parameters
        ----------
        kind : {"text", "thinking"}
            Block kind.
        text : str
            Content.
        """
        ...

    def hard_limit_hit(self) -> bool:
        """Whether the campaign must end now.

        Returns
        -------
        bool
            True at the hard limit.
        """
        ...

    def nudge(self) -> str:
        """Message for a model that stopped before submitting.

        Returns
        -------
        str
            Next user message.
        """
        ...


def api_credentials_present() -> bool:
    """Whether Anthropic API credentials are in the environment.

    Returns
    -------
    bool
        True when ``ANTHROPIC_API_KEY`` or ``ANTHROPIC_AUTH_TOKEN`` is set.
    """
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def resolve_backend(choice: str) -> Backend:
    """Resolve ``auto`` to a concrete backend (OR-11).

    Parameters
    ----------
    choice : {"auto", "api", "claude-code"}
        Requested backend.

    Returns
    -------
    {"api", "claude-code"}
        ``api`` when API credentials exist, else ``claude-code``, for ``auto``.

    Raises
    ------
    ValueError
        For an unknown backend name.
    """
    if choice == "auto":
        return "api" if api_credentials_present() else "claude-code"
    if choice in ("api", "claude-code"):
        return choice  # type: ignore[return-value]
    raise ValueError(f"Unknown LLM backend {choice!r}; use auto, api or claude-code.")


def make_llm(backend: str, model: str) -> LLM:
    """Create the LLM for a run.

    Parameters
    ----------
    backend : {"auto", "api", "claude-code"}
        Requested backend.
    model : str
        Claude model ID.

    Returns
    -------
    LLM
        A ready backend.
    """
    resolved = resolve_backend(backend)
    if resolved == "api":
        from hephaestus.llm.api import AnthropicAPI

        return AnthropicAPI(model)
    from hephaestus.llm.claude_code import ClaudeCodeLLM

    return ClaudeCodeLLM(model)


class LoggedLLM:
    """Wrap an LLM so every call is appended to ``llm_log.jsonl`` (DL-4, DL-9)."""

    def __init__(self, inner: LLM, log_path: Path) -> None:
        """Wrap ``inner``.

        Parameters
        ----------
        inner : LLM
            Backend to call.
        log_path : pathlib.Path
            The run's ``llm_log.jsonl``.
        """
        self.inner = inner
        self.backend = inner.backend
        self.model = inner.model
        self.log_path = log_path
        self.usage: dict[str, int] = {}
        self.section = "orchestrator"
        self.iteration: int | None = None

    def _log(self, section: str, request: Any, response: Any, usage: dict[str, int], t0: float) -> None:
        """Append one call to the log and add its usage to the totals.

        Parameters
        ----------
        section : str
            Caller.
        request : Any
            Prompt or last message.
        response : Any
            Answer or assistant content.
        usage : dict
            Token counts.
        t0 : float
            Start time from ``time.monotonic``.
        """
        for k, v in usage.items():
            self.usage[k] = self.usage.get(k, 0) + int(v or 0)
        append_jsonl(
            self.log_path,
            {
                "section": section,
                "iteration": self.iteration,
                "backend": self.backend,
                "model": self.model,
                "duration_s": round(time.monotonic() - t0, 3),
                "request": request,
                "response": response,
                "usage": usage,
            },
        )

    def step(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Turn:
        """Run and log one orchestrator turn.

        Parameters
        ----------
        system : str
            System prompt.
        messages : list of dict
            Conversation so far.
        tools : list of dict
            Tool definitions.

        Returns
        -------
        Turn
            The assistant's turn.
        """
        t0 = time.monotonic()
        turn = self.inner.step(system, messages, tools)
        self._log("orchestrator", messages[-1] if messages else None, turn.content, turn.usage, t0)
        return turn

    def structured(self, section: str, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Run and log one structured call.

        Parameters
        ----------
        section : str
            Caller.
        system : str
            System prompt.
        prompt : str
            User prompt.
        schema : dict
            JSON schema of the answer.

        Returns
        -------
        dict
            The parsed answer.
        """
        t0 = time.monotonic()
        try:
            out = self.inner.structured(section, system, prompt, schema)
        except Exception as exc:
            self._log(section, prompt, {"error": str(exc)}, {}, t0)
            raise
        self._log(section, prompt, out, {}, t0)
        return out
