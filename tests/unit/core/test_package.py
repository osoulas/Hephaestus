"""Package-level checks: version, entry point and layering."""

import ast
from importlib.metadata import version
from pathlib import Path

import pytest

import hephaestus
from hephaestus.interfaces.cli.app import main

PACKAGE = Path(hephaestus.__file__).parent
# Layers may only import layers listed for them (CLAUDE.md: dependencies point down only).
ALLOWED = {
    "core": {"core"},
    "llm": {"llm", "core"},
    "tools": {"tools", "core"},
    "harness": {"harness", "tools", "llm", "core"},
    "interfaces": {"interfaces", "harness", "tools", "llm", "core"},
}


def _hephaestus_imports(path: Path) -> set[str]:
    """Return the top-level hephaestus subpackages a module imports.

    Parameters
    ----------
    path : pathlib.Path
        Python source file inside the package.

    Returns
    -------
    set of str
        Subpackage names such as ``"core"`` or ``"tools"``.
    """
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for name in names:
            parts = name.split(".")
            if parts[0] == "hephaestus" and len(parts) > 1:
                found.add(parts[1])
    return found


def test_version_matches_distribution_metadata() -> None:
    """The runtime version stays in sync with package metadata."""
    assert hephaestus.__version__ == version("hephaestus")


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_layers_import_only_downwards(layer: str) -> None:
    """A lower layer never imports a higher one.

    Parameters
    ----------
    layer : str
        Subpackage under test.
    """
    for path in (PACKAGE / layer).rglob("*.py"):
        bad = _hephaestus_imports(path) - ALLOWED[layer]
        assert not bad, f"{path.relative_to(PACKAGE)} imports {sorted(bad)}"


def test_cli_entry_point_runs() -> None:
    """``python -m hephaestus`` parses its subcommands."""
    assert main(["replay"]) == 2


def test_or11_cli_login_saves_an_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``login --api-key`` stores the key without echoing it; ``--forget-api-key`` removes it."""
    from hephaestus.llm import auth

    monkeypatch.setenv("HEPHAESTUS_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(auth, "check_api_key", lambda key: None)
    monkeypatch.setattr("getpass.getpass", lambda prompt: "sk-ant-api03-test-WXYZ")
    assert main(["login", "--api-key"]) == 0
    assert (tmp_path / "anthropic_api_key").read_text(encoding="utf-8") == "sk-ant-api03-test-WXYZ"
    assert main(["login", "--forget-api-key"]) == 0
    assert not (tmp_path / "anthropic_api_key").exists()
