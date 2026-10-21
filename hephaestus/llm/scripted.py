"""Scripted model: replays fixed assistant turns for tests and offline demos.

A script is a list of turns. Each turn is a list of steps: ``{"text": ...}``,
``{"thinking": ...}`` or ``{"tool": name, "input": {...}}``. A turn may also be a callable
``f(messages) -> list of steps`` to react to earlier tool results. Structured answers are
queued per section (``reviewer``, ``goal_parser``, ``brief``) as dicts or callables
``f(prompt, schema) -> dict``; an ``Exception`` in the queue is raised.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hephaestus.llm.base import Backend, LLMError, ToolCall, Turn

Step = dict[str, Any]
ScriptTurn = list[Step] | Callable[[list[dict[str, Any]]], list[Step]]
Answer = dict[str, Any] | Callable[[str, dict[str, Any]], dict[str, Any]] | Exception


class ScriptedLLM:
    """Replays a script; never touches the network."""

    backend: Backend = "scripted"

    def __init__(
        self,
        turns: list[ScriptTurn] | None = None,
        structured: dict[str, list[Answer]] | None = None,
        model: str = "scripted",
    ) -> None:
        """Create a scripted model.

        Parameters
        ----------
        turns : list, optional
            Orchestrator turns, in order.
        structured : dict, optional
            Section name to a queue of answers.
        model : str
            Name reported in logs.
        """
        self.turns = list(turns or [])
        self.answers = {k: list(v) for k, v in (structured or {}).items()}
        self.model = model
        self.calls = 0
        self.structured_calls: list[tuple[str, str]] = []

    def step(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Turn:
        """Return the next scripted turn, or a plain end turn when the script is exhausted.

        Parameters
        ----------
        system : str
            Ignored.
        messages : list of dict
            Passed to callable turns.
        tools : list of dict
            Ignored.

        Returns
        -------
        Turn
            The scripted turn.
        """
        if not self.turns:
            return Turn(
                content=[{"type": "text", "text": "(script exhausted)"}], texts=["(script exhausted)"]
            )
        spec = self.turns.pop(0)
        steps = spec(messages) if callable(spec) else spec
        self.calls += 1
        turn = Turn(content=[], usage={"input_tokens": 0, "output_tokens": 0})
        for i, step in enumerate(steps):
            if "thinking" in step:
                turn.content.append({"type": "thinking", "thinking": step["thinking"], "signature": ""})
                turn.thinking.append(step["thinking"])
            elif "text" in step:
                turn.content.append({"type": "text", "text": step["text"]})
                turn.texts.append(step["text"])
            else:
                call = ToolCall(
                    id=f"toolu_s{self.calls}_{i}", name=step["tool"], input=dict(step.get("input", {}))
                )
                turn.content.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
                )
                turn.tool_calls.append(call)
        turn.stop_reason = "tool_use" if turn.tool_calls else "end_turn"
        return turn

    def structured(self, section: str, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return the next queued answer for ``section``.

        Parameters
        ----------
        section : str
            Queue to read.
        system : str
            Ignored.
        prompt : str
            Passed to callable answers and recorded.
        schema : dict
            Passed to callable answers.

        Returns
        -------
        dict
            The answer.

        Raises
        ------
        LLMError
            When the queue is empty or the queued item is an exception.
        """
        self.structured_calls.append((section, prompt))
        queue = self.answers.get(section) or []
        if not queue:
            raise LLMError(f"No scripted answer for {section!r}.")
        answer = queue.pop(0)
        if isinstance(answer, Exception):
            raise LLMError(str(answer)) from answer
        return answer(prompt, schema) if callable(answer) else dict(answer)
