"""A Claude usage limit mid-run pauses it, and resume finishes it from the same folder (AT-30)."""

from pathlib import Path

import pytest
import scripts
from test_custom_tools import campaign_turns, cfg, deps

from hephaestus.core.store import load_model
from hephaestus.core.types import RunState
from hephaestus.harness import runner
from hephaestus.llm.base import UsageLimitError
from hephaestus.llm.scripted import ScriptedLLM


def test_at30_runner_pauses_then_resume_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A limit mid-campaign leaves the run paused with its reason; resume finishes it from the same folder."""
    monkeypatch.setenv("HEPHAESTUS_CACHE_DIR", str(tmp_path / "cache"))
    d = deps(tmp_path, campaign_turns(), None, reviews=[scripts.review(scripts.FE2, 90)])
    d.llm.answers["goal_parser"] = [scripts.PARSED]
    calls = {"n": 0}
    real_step = ScriptedLLM.step

    def limited(self, system, messages, tools):  # type: ignore[no-untyped-def]
        """Raise a usage limit on the second turn, once.

        Parameters
        ----------
        self, system, messages, tools
            As for ``step``.

        Returns
        -------
        Turn
            The real turn otherwise.
        """
        calls["n"] += 1
        if calls["n"] == 2:
            raise UsageLimitError("Claude usage limit reached", 1800000000)
        return real_step(self, system, messages, tools)

    d.llm.__class__ = type("Limited", (ScriptedLLM,), {"step": limited})  # one turn raises the limit
    state = runner.run(scripts.GOAL, cfg=cfg(tmp_path), deps=d, run_id="r1")
    assert state.status == "paused" and state.pause["reason"] == "usage_limit"
    assert state.pause["resets_at"] == 1800000000 and state.stop_reason is None
    events = (tmp_path / "r1" / "events.jsonl").read_text(encoding="utf-8")
    assert '"status":"paused"' in events and '"reason":"usage_limit"' in events
    assert not (tmp_path / "r1" / "brief.md").exists()  # nothing written as if finished

    d.llm.__class__ = ScriptedLLM  # the limit has reset
    d.llm.turns = list(campaign_turns())
    state = runner.resume("r1", deps=d)
    assert state.status == "finished" and state.pause is None
    assert load_model(tmp_path / "r1" / "state.json", RunState).status == "finished"
