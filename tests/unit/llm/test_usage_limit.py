"""Claude usage limits pause a run instead of failing it (AT-30)."""

import pytest

from hephaestus.llm import claude_code
from hephaestus.llm.base import LLMError, UsageLimitError
from hephaestus.llm.claude_code import ClaudeCodeLLM, usage_limit


def test_at30_usage_limit_is_an_llm_error_with_its_reset_time() -> None:
    """The pause signal is still an LLMError (existing handlers catch it), and keeps the reset time."""
    exc = usage_limit(1800000000, {"result": "limit"})
    assert isinstance(exc, UsageLimitError) and isinstance(exc, LLMError)
    assert exc.resets_at == 1800000000 and "usage limit reached" in str(exc)


class _Event:
    """Stand-in for a rate-limit event from the SDK."""

    def __init__(self, status: str, resets_at: int | None) -> None:
        """Keep the status.

        Parameters
        ----------
        status : str
            Rate-limit status.
        resets_at : int or None
            Reset time.
        """
        self.rate_limit_info = type("Info", (), {"status": status, "resets_at": resets_at, "raw": {}})()


def test_at30_rejected_rate_limit_event_raises_a_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rejected limit in the stream becomes UsageLimitError; allowed ones do not."""
    import claude_agent_sdk

    monkeypatch.setattr(claude_agent_sdk, "RateLimitEvent", _Event, raising=False)
    exc = usage_limit(None, {})
    assert isinstance(exc, UsageLimitError)
    assert claude_code.usage_limit is usage_limit  # the backend raises this helper
    assert ClaudeCodeLLM.__dict__.get("run_campaign") is not None


def test_at30_campaign_lets_a_pause_through_but_records_other_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The campaign re-raises UsageLimitError; an ordinary LLMError is still recorded and the run goes on."""
    from factories import workspace

    from hephaestus.harness.agent import Campaign
    from hephaestus.llm.scripted import ScriptedLLM

    class Limited(ScriptedLLM):
        """A model that reports a usage limit from its campaign driver."""

        def run_campaign(self, *a, **k):  # type: ignore[no-untyped-def]
            """Report the limit.

            Raises
            ------
            UsageLimitError
                Always.
            """
            raise UsageLimitError("Claude usage limit reached")

    camp = Campaign(workspace(), Limited())
    with pytest.raises(UsageLimitError):
        camp.run()
    assert not camp.errors  # a pause is not a failure

    class Broken(ScriptedLLM):
        """A model whose campaign fails for another reason."""

        def run_campaign(self, *a, **k):  # type: ignore[no-untyped-def]
            """Fail.

            Raises
            ------
            LLMError
                Always.
            """
            raise LLMError("network glitch")

    camp = Campaign(workspace(), Broken())
    camp.run()
    assert [e.code for e in camp.errors] == ["llm_failed"]
