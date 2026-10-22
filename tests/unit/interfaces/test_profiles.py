"""Setting profiles: one list in the Python API, the same in the web composer and the CLI (OR-4, UI-12)."""

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from hephaestus.core.config import PROFILES, RunConfig
from hephaestus.interfaces import http
from hephaestus.interfaces.cli import actions

SPEC_JS = Path(__file__).parents[3] / "hephaestus" / "interfaces" / "web" / "js" / "spec.js"


def web_profiles() -> dict[str, dict[str, object]]:
    """Read the web composer's profile values from ``spec.js``.

    Returns
    -------
    dict
        ``{id: values}``.
    """
    text = SPEC_JS.read_text(encoding="utf-8")
    block = text[text.index("export const PROFILES") :]
    block = block[: block.index("];")]
    out = {}
    for m in re.finditer(r'id: "(\w+)".*?values: (\{[^}]*\})', block, re.S):
        values = re.sub(r"(\w+):", r'"\1":', m.group(2))
        out[m.group(1)] = json.loads(values)
    return out


def test_or4_profiles_are_valid_run_options() -> None:
    """Every profile's values build a valid ``RunConfig``."""
    assert [p["id"] for p in PROFILES] == ["quick", "standard", "thorough"]
    for p in PROFILES:
        RunConfig(**p["values"])


def test_ui12_web_composer_matches_the_python_profiles() -> None:
    """The web UI's copy in ``spec.js`` has the same values as the Python list."""
    assert web_profiles() == {p["id"]: p["values"] for p in PROFILES}


def test_cli_and_http_use_the_python_profiles(tmp_path: Path) -> None:
    """The CLI's presets and ``GET /config/profiles`` come from the same list."""
    assert {p["id"]: p["values"] for p in PROFILES} == actions.PROFILES
    client = TestClient(http.create_app(tmp_path))
    assert client.get("/config/profiles").json() == PROFILES
