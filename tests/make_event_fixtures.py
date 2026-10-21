"""Regenerate the recorded event fixtures for the CLI and UI: ``python tests/make_event_fixtures.py``.

Runs the scripted demo campaign over stub tools and copies its ``events.jsonl``,
``state.json`` and ``brief.md`` into ``tests/fixtures/events/demo/``. Rerun after any change
to the event payloads, and tell the CLI and UI owners.
"""

import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import scripts  # noqa: E402

from hephaestus.core.events import EventBus  # noqa: E402
from hephaestus.core.types import ValidationCheck, ValidationReport  # noqa: E402
from hephaestus.harness import runner  # noqa: E402
from hephaestus.llm.scripted import ScriptedLLM  # noqa: E402
from hephaestus.tools.stubs import check_novelty, stub_handlers  # noqa: E402

OUT = HERE / "fixtures" / "events" / "demo"


def confirm(ws: object, sub: object) -> list[ValidationReport]:
    """Stand-in final check that confirms the winner.

    Parameters
    ----------
    ws : Workspace
        Unused.
    sub : Submission
        The submission.

    Returns
    -------
    list of ValidationReport
        One confirmed report.
    """
    cid = sub.candidate_id  # type: ignore[attr-defined]
    checks = [
        ValidationCheck(name="e_hull_eV_per_atom", screening_value=0.02, final_value=0.024, delta=0.004)
    ]
    return [
        ValidationReport(candidate_id=cid, verdict="confirmed", checks=checks, notes=["stub final check"])
    ]


def main() -> None:
    """Run the demo and copy its files into the fixtures folder."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        llm = ScriptedLLM(
            turns=scripts.ROUND_1 + scripts.ROUND_2,
            structured={"goal_parser": [scripts.PARSED], "reviewer": list(scripts.REVIEWS)},
        )
        deps = runner.Deps(
            llm=llm,
            handlers=stub_handlers(),
            novelty_fn=check_novelty,
            final_check=confirm,
            bus=EventBus(root),
            root=root,
        )
        runner.run(scripts.GOAL, deps=deps, run_id="r-demo")
        OUT.mkdir(parents=True, exist_ok=True)
        for name in ("events.jsonl", "state.json", "brief.md"):
            shutil.copy(root / "r-demo" / name, OUT / name)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
