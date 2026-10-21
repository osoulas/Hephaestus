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
    assert main(["backends"]) == 2
