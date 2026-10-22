"""Agent-built tools: static check, sandbox, toolkit, tool library and campaign handlers (Section 14).

A tool is one Python file with a literal ``TOOL`` header and a ``run(structure, inputs, hp)``
function, plus a ``test_<name>.py`` file (``docs/TOOLS.md``, *The tool file*). Files live in
the tool library, ``RunConfig.tool_library_dir`` in the working directory, and load only when
their hash matches ``manifest.json`` (AT-20, AT-21).

Every call runs in a worker process (``sandbox_worker.py``). The worker asks this process for
building blocks (``Toolkit``) over a pipe, so model calls run here, under the GPU lock and the
cache (AT-14, AT-15). Results of agent-built tools never feed a stability verdict (AT-18).
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import math
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hephaestus.core.config import RunConfig
from hephaestus.core.store import atomic_write_text, cached
from hephaestus.core.types import CustomTool, PropertyPrediction, Quantity, utcnow
from hephaestus.tools.registry import PROPERTY_KEYS, TOOLS, Handler, ToolError, Workspace, target_info

WORKER = Path(__file__).with_name("sandbox_worker.py")
MANIFEST = "manifest.json"
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
MAX_LINES = 500
MEMORY_MB = 4096
AGENT_LABEL = "agent-built tool"

ALLOWED_IMPORTS = [
    "math",
    "cmath",
    "statistics",
    "itertools",
    "functools",
    "collections",
    "typing",
    "dataclasses",
    "numpy.*",
    "scipy.*",
    "pymatgen.core.*",
    "pymatgen.analysis.*",
    "pymatgen.symmetry.*",
    "ase",
    "ase.atoms",
    "ase.build.*",
    "ase.units",
    "ase.eos",
]
"""Modules a tool may import; ``pkg.*`` allows the package and its submodules (AT-11)."""

BANNED_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "input",
    "breakpoint",
    "help",
    "exit",
    "quit",
    "memoryview",
}
BANNED_ATTRS = {
    # file and process access through library objects
    "to",
    "from_file",
    "save",
    "savez",
    "savez_compressed",
    "savetxt",
    "tofile",
    "fromfile",
    "load",
    "loadtxt",
    "genfromtxt",
    "memmap",
    "write",
    "read",
    "io",
    "os",
    "sys",
    "subprocess",
    "system",
    "popen",
    "ctypeslib",
}
TOOLKIT_CALLS = ("single_point", "relax", "phonons", "md")

DISABLEABLE_PROPERTIES = {
    "elastic": [k for k in PROPERTY_KEYS if k not in ("band_gap", "formation_energy", "density")]
}
"""Properties each capability provides, removed from a run by ``RunConfig.disabled_capabilities`` (AT-26)."""

HIDDEN_SOURCE = {
    "elastic": {
        "strain_matrix",
        "elastic_tensor",
        "elastic_properties",
        "elastic_cached",
        "from_moduli",
        "predict_property",
        "property_registry",
    }
}
"""Functions ``read_tool_source`` hides when a capability is disabled, so a rebuild is not a copy (AT-5)."""

SOURCE_MODULES = ("simulation", "structures")


# ---------------------------------------------------------------------------
# Static check (AT-11)
# ---------------------------------------------------------------------------


def allowlist(extra: tuple[str, ...] | list[str] = ()) -> list[str]:
    """Return the import allowlist, plus packages the user approved for this tool.

    Parameters
    ----------
    extra : sequence of str
        Approved top-level packages, e.g. ``["phonopy"]``; each allows its submodules too.

    Returns
    -------
    list of str
        Entries in ``ALLOWED_IMPORTS`` form.
    """
    return [*ALLOWED_IMPORTS, *(f"{p}.*" for p in extra)]


def import_allowed(name: str, extra: tuple[str, ...] | list[str] = ()) -> bool:
    """Whether a tool may import a module.

    Parameters
    ----------
    name : str
        Dotted module name.
    extra : sequence of str
        Packages the user approved for this tool.

    Returns
    -------
    bool
        True when ``name`` matches the allowlist.
    """
    for entry in allowlist(extra):
        if entry.endswith(".*"):
            root = entry[:-2]
            if name == root or name.startswith(root + "."):
                return True
        elif name == entry:
            return True
    return False


def read_header(source: str) -> dict[str, Any]:
    """Read a tool's ``TOOL`` header without running the file.

    Parameters
    ----------
    source : str
        Tool source.

    Returns
    -------
    dict
        The header.

    Raises
    ------
    ValueError
        If the file does not parse or has no literal ``TOOL`` dict.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"syntax error on line {exc.lineno}: {exc.msg}") from None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "TOOL" for t in node.targets
        ):
            try:
                header = ast.literal_eval(node.value)
            except ValueError:
                raise ValueError("TOOL must be a literal dict (no names or calls inside it)") from None
            if not isinstance(header, dict):
                raise ValueError("TOOL must be a dict")
            return header
    raise ValueError("no TOOL header")


def static_check(source: str, kind: str = "tool", extra: tuple[str, ...] | list[str] = ()) -> dict[str, Any]:
    """Check a tool or test file without importing it (AT-11).

    Parameters
    ----------
    source : str
        File contents.
    kind : {"tool", "test"}
        A tool needs a valid ``TOOL`` header and ``run(structure, inputs, hp)``; a test file
        needs at least one ``test_*`` function.
    extra : sequence of str
        Packages beyond the allowlist that the user approved (``TOOL["packages"]``).

    Returns
    -------
    dict
        ``{passed, report}``; ``report`` names every problem, one per line.
    """
    problems: list[str] = []
    lines = source.count("\n") + 1
    if lines > MAX_LINES:
        problems.append(f"{lines} lines; at most {MAX_LINES}")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"passed": False, "report": f"syntax error on line {exc.lineno}: {exc.msg}"}
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not import_allowed(alias.name, extra):
                    problems.append(f"line {line}: import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                problems.append(f"line {line}: relative imports are not allowed")
            elif not import_allowed(node.module or "", extra):
                problems.append(f"line {line}: import from {node.module!r} is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES or node.id.startswith("__"):
                problems.append(f"line {line}: {node.id} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                problems.append(f"line {line}: attribute {node.attr!r} is not allowed (starts with _)")
            elif node.attr in BANNED_ATTRS:
                problems.append(
                    f"line {line}: attribute {node.attr!r} is not allowed (file or process access)"
                )
        elif isinstance(node, ast.Global | ast.Nonlocal):
            problems.append(f"line {line}: global and nonlocal are not allowed")
    if kind == "tool":
        try:
            problems += header_problems(read_header(source))
        except ValueError as exc:
            problems.append(str(exc))
        run = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"), None)
        if run is None:
            problems.append("no top-level run(structure, inputs, hp) function")
        elif [a.arg for a in run.args.args] != ["structure", "inputs", "hp"]:
            problems.append("run must take exactly (structure, inputs, hp)")
    elif not any(isinstance(n, ast.FunctionDef) and n.name.startswith("test_") for n in tree.body):
        problems.append("no test_* function")
    return {"passed": not problems, "report": "\n".join(problems) or "passed"}


PIP_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}(==[0-9][A-Za-z0-9.]{0,30})?$")
MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,60}$")


def header_packages(header: dict[str, Any]) -> list[str]:
    """Return the top-level packages a tool declares beyond the allowlist (``TOOL["packages"]``).

    Parameters
    ----------
    header : dict
        The ``TOOL`` header.

    Returns
    -------
    list of str
        Valid package names; anything else is dropped.
    """
    raw = header.get("packages") or []
    return [str(p) for p in raw if isinstance(p, str) and MODULE_RE.match(p)] if isinstance(raw, list) else []


def header_trial_range(header: dict[str, Any]) -> tuple[float, float] | None:
    """Return the plausible trial value a tool declares (``TOOL["trial_range"]``), if valid.

    Parameters
    ----------
    header : dict
        The ``TOOL`` header.

    Returns
    -------
    tuple of float or None
        ``(min, max)`` with min <= max; ``None`` when missing or malformed.
    """
    raw = header.get("trial_range")
    if not isinstance(raw, list | tuple) or len(raw) != 2:
        return None
    try:
        lo, hi = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    return (lo, hi) if math.isfinite(lo) and math.isfinite(hi) and lo <= hi else None


def importable(module: str) -> bool:
    """Whether the server's Python (which the sandbox uses) can import a package.

    Parameters
    ----------
    module : str
        Top-level import name.

    Returns
    -------
    bool
        True when it is installed.
    """
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def install_package(pip_name: str, timeout_s: float = 600.0) -> tuple[bool, str]:
    """Install a package into the server's Python with pip, after the user allowed it.

    Parameters
    ----------
    pip_name : str
        Distribution name, optionally pinned (``name==1.2``); no URLs, paths or flags.
    timeout_s : float
        Limit on the install, s.

    Returns
    -------
    ok : bool
        Whether pip succeeded.
    log : str
        The end of pip's output.
    """
    if not PIP_RE.match(pip_name):
        return False, f"{pip_name!r} is not a plain package name."
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", pip_name],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)[:500]
    return out.returncode == 0, (out.stdout + out.stderr)[-1500:]


def header_problems(header: dict[str, Any]) -> list[str]:
    """Problems with a ``TOOL`` header.

    Parameters
    ----------
    header : dict
        The header.

    Returns
    -------
    list of str
        Empty when valid.
    """
    out = []
    for key, typ in (("name", str), ("property", str), ("units", str), ("description", str)):
        if not isinstance(header.get(key), typ):
            out.append(f"TOOL[{key!r}] must be a string")
    if isinstance(header.get("name"), str) and not NAME_RE.match(header["name"]):
        out.append("TOOL['name'] must be snake_case, 2 to 41 characters")
    inputs = header.get("inputs", {"type": "object", "properties": {}})
    if not isinstance(inputs, dict) or not isinstance(inputs.get("properties", {}), dict):
        out.append("TOOL['inputs'] must be a JSON schema object")
    elif "candidate_id" in inputs.get("properties", {}):
        out.append("TOOL['inputs'] must not declare candidate_id; the harness adds it")
    if header.get("author", "agent") not in ("agent", "user"):
        out.append("TOOL['author'] must be 'agent' or 'user'")
    return out


# ---------------------------------------------------------------------------
# Tool library (AT-20, AT-21)
# ---------------------------------------------------------------------------


def library_dir(cfg: RunConfig) -> Path:
    """Return the tool library folder for a run.

    Parameters
    ----------
    cfg : RunConfig
        ``tool_library_dir``, relative to the working directory.

    Returns
    -------
    pathlib.Path
        The folder (not created).
    """
    return Path(cfg.tool_library_dir)


def tool_paths(lib: Path, name: str) -> tuple[Path, Path]:
    """Paths of a tool file and its tests.

    Parameters
    ----------
    lib : pathlib.Path
        Library folder.
    name : str
        Tool name.

    Returns
    -------
    tuple of pathlib.Path
        ``(<name>.py, test_<name>.py)``.
    """
    return lib / f"{name}.py", lib / f"test_{name}.py"


def fingerprint(source: str, tests: str) -> str:
    """SHA-256 of a tool and its tests: any edit to either needs a new approval (AT-20, AT-23).

    Parameters
    ----------
    source : str
        Tool file.
    tests : str
        Test file.

    Returns
    -------
    str
        Hex digest.
    """
    return hashlib.sha256((source + "\0" + tests).encode("utf-8")).hexdigest()


def _read(path: Path) -> str:
    """Read a file as UTF-8, or ``""`` when it is missing.

    Parameters
    ----------
    path : pathlib.Path
        File.

    Returns
    -------
    str
        Contents.
    """
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_manifest(lib: Path) -> dict[str, CustomTool]:
    """Approved tools by name.

    Parameters
    ----------
    lib : pathlib.Path
        Library folder.

    Returns
    -------
    dict
        Name to ``CustomTool``; empty without a manifest.
    """
    path = lib / MANIFEST
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: CustomTool.model_validate(v) for k, v in raw.get("tools", {}).items()}


def _write_manifest(lib: Path, tools: dict[str, CustomTool]) -> None:
    """Write the manifest atomically.

    Parameters
    ----------
    lib : pathlib.Path
        Library folder.
    tools : dict
        Name to ``CustomTool``.
    """
    lib.mkdir(parents=True, exist_ok=True)
    body = {"tools": {k: v.model_dump(mode="json") for k, v in sorted(tools.items())}}
    atomic_write_text(lib / MANIFEST, json.dumps(body, indent=2, ensure_ascii=False))


def scan_library(lib: Path) -> tuple[list[CustomTool], list[dict[str, Any]]]:
    """Approved tools whose files still match the manifest, and every other tool file (AT-20, AT-21).

    Unapproved files are only parsed for their header, never imported.

    Parameters
    ----------
    lib : pathlib.Path
        Library folder.

    Returns
    -------
    approved : list of CustomTool
        Tools that may load.
    unapproved : list of dict
        ``{name, file, sha256, header, error, source, tests, changed}`` for new or edited files.
    """
    if not lib.is_dir():
        return [], []
    manifest = read_manifest(lib)
    approved, unapproved = [], []
    for path in sorted(lib.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        name = path.stem
        source, tests = _read(path), _read(tool_paths(lib, name)[1])
        sha = fingerprint(source, tests)
        entry = manifest.get(name)
        if entry is not None and entry.sha256 == sha:
            approved.append(entry)
            continue
        try:
            header, error = read_header(source), ""
        except ValueError as exc:
            header, error = {}, str(exc)
        unapproved.append(
            {
                "name": name,
                "file": str(path),
                "sha256": sha,
                "header": header,
                "error": error,
                "source": source,
                "tests": tests,
                "changed": entry is not None,
            }
        )
    return approved, unapproved


def save_tool(
    lib: Path, name: str, source: str, tests: str, author: str = "agent", run_id: str | None = None
) -> CustomTool:
    """Write an approved tool and its tests to the library and record its hash (AT-20).

    Parameters
    ----------
    lib : pathlib.Path
        Library folder.
    name : str
        Tool name; must match the header.
    source, tests : str
        Files.
    author : {"agent", "user"}
        Who wrote it.
    run_id : str, optional
        The run that built it.

    Returns
    -------
    CustomTool
        The manifest entry.
    """
    header = read_header(source)
    tool_file, test_file = tool_paths(lib, name)
    lib.mkdir(parents=True, exist_ok=True)
    atomic_write_text(tool_file, source)
    atomic_write_text(test_file, tests)
    entry = CustomTool(
        name=name,
        property=header["property"],
        units=header["units"],
        file=str(tool_file),
        sha256=fingerprint(source, tests),
        author=author,  # type: ignore[arg-type]
        approved_at=utcnow(),
        run_id=run_id,
        description=header.get("description", ""),
        packages=header_packages(header),
    )
    tools = read_manifest(lib)
    tools[name] = entry
    _write_manifest(lib, tools)
    return entry


def approve_file(lib: Path, name: str, sha_shown: str) -> CustomTool:
    """Approve a tool file as it was shown to the user (AT-21).

    Parameters
    ----------
    lib : pathlib.Path
        Library folder.
    name : str
        Tool name (file stem).
    sha_shown : str
        Fingerprint of the files the user reviewed.

    Returns
    -------
    CustomTool
        The manifest entry.

    Raises
    ------
    ValueError
        If the files are missing, changed since they were shown, or fail the static check.
    """
    tool_file, test_file = tool_paths(lib, name)
    if not tool_file.exists():
        raise ValueError(f"No tool file {tool_file}.")
    source, tests = _read(tool_file), _read(test_file)
    if fingerprint(source, tests) != sha_shown:
        raise ValueError("The files changed since they were shown; review them again.")
    check = static_check(source, "tool", header_packages(read_header(source)))
    if not check["passed"]:
        raise ValueError("Static check failed: " + check["report"])
    header = read_header(source)
    if header["name"] != name:
        raise ValueError(f"TOOL['name'] is {header['name']!r} but the file is {name}.py.")
    return save_tool(lib, name, source, tests, author=header.get("author", "user"))


def remove_tool(lib: Path, name: str) -> bool:
    """Delete a tool's files and its manifest entry.

    Parameters
    ----------
    lib : pathlib.Path
        Library folder.
    name : str
        Tool name.

    Returns
    -------
    bool
        True if anything was removed.
    """
    removed = False
    for path in tool_paths(lib, name):
        if path.exists():
            path.unlink()
            removed = True
    tools = read_manifest(lib)
    if tools.pop(name, None) is not None:
        _write_manifest(lib, tools)
        removed = True
    return removed


def load_sources(tool: CustomTool) -> tuple[str, str]:
    """Read an approved tool's files, refusing any that changed since approval (AT-20).

    Parameters
    ----------
    tool : CustomTool
        Manifest entry.

    Returns
    -------
    tuple of str
        ``(source, tests)``.

    Raises
    ------
    ToolError
        If the files no longer match the approved hash.
    """
    tool_file = Path(tool.file)
    source, tests = _read(tool_file), _read(tool_file.with_name(f"test_{tool.name}.py"))
    if fingerprint(source, tests) != tool.sha256:
        raise ToolError(f"Tool {tool.name} changed since it was approved; approve it again before use.")
    return source, tests


# ---------------------------------------------------------------------------
# Property registry for a run (AT-1, AT-26)
# ---------------------------------------------------------------------------


def disabled_properties(cfg: RunConfig) -> set[str]:
    """Built-in properties removed from this run.

    Parameters
    ----------
    cfg : RunConfig
        ``disabled_capabilities``.

    Returns
    -------
    set of str
        Property keys.
    """
    return {p for cap in cfg.disabled_capabilities for p in DISABLEABLE_PROPERTIES.get(cap, [])}


def tool_entry(tool: CustomTool) -> dict[str, Any]:
    """Registry entry for a property supplied by an agent-built tool.

    Parameters
    ----------
    tool : CustomTool
        Approved tool.

    Returns
    -------
    dict
        ``{model, units, published_error, error_note, scale_note, tool}``.
    """
    return {
        "model": f"agent-built:{tool.name}@{tool.sha256[:8]}",
        "units": tool.units,
        "published_error": None,
        "error_note": AGENT_LABEL,
        "scale_note": f"{AGENT_LABEL}; compare only with values from the same tool",
        "tool": f"custom_{tool.name}",
    }


def run_registry(cfg: RunConfig, library: list[CustomTool] | None = None) -> dict[str, dict[str, Any]]:
    """Properties this run can compute: built-in minus disabled, plus approved library tools (AT-1).

    Parameters
    ----------
    cfg : RunConfig
        Band gap model, disabled capabilities and the library folder.
    library : list of CustomTool, optional
        Approved tools; read from the library folder by default.

    Returns
    -------
    dict
        Property key to registry entry.
    """
    from hephaestus.tools.simulation import property_registry

    off = disabled_properties(cfg)
    reg = {k: v for k, v in property_registry(cfg.band_gap_fidelity).items() if k not in off}
    for tool in library if library is not None else scan_library(library_dir(cfg))[0]:
        reg.setdefault(tool.property, tool_entry(tool))
    return reg


def campaign_tools(ws: Workspace) -> list[dict[str, Any]]:
    """Return the campaign's tools: ``TOOLS`` without disabled properties, plus approved tools (AT-16).

    Parameters
    ----------
    ws : Workspace
        Run workspace with ``custom_tools``.

    Returns
    -------
    list of dict
        Tool definitions.
    """
    off = disabled_properties(ws.cfg)
    out = []
    for t in TOOLS:
        if t["name"] == "predict_property" and off:
            t = json.loads(json.dumps(t))
            prop = t["input_schema"]["properties"]["property"]
            prop["enum"] = [k for k in prop["enum"] if k not in off]
            t["description"] = (
                "Predict a property on the current structure with its uncertainty and a domain flag."
            )
        out.append(t)
    for tool in ws.custom_tools.values():
        out.append(tool_schema(tool))
    return out


def tool_schema(tool: CustomTool) -> dict[str, Any]:
    """Tool definition of an approved tool, as the campaign sees it (AT-16).

    Declared inputs are optional: the tool's defaults apply when they are left out.

    Parameters
    ----------
    tool : CustomTool
        Approved tool.

    Returns
    -------
    dict
        ``{name, description, input_schema}``.
    """
    try:
        header = read_header(load_sources(tool)[0])
    except (ToolError, ValueError):
        header = {}
    props = dict((header.get("inputs") or {}).get("properties") or {})
    return {
        "name": f"custom_{tool.name}",
        "description": (
            f"Agent-built tool, approved by the user: {tool.description or header.get('description', '')} "
            f"Computes {tool.property} ({tool.units}) on the candidate's current structure. Results carry no "
            "uncertainty and never change a stability verdict."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string", "description": "Candidate ID, e.g. c1-2."},
                **props,
            },
            "required": ["candidate_id"],
            "additionalProperties": False,
        },
    }


# ---------------------------------------------------------------------------
# Our own tools, for inspiration and scope (AT-5)
# ---------------------------------------------------------------------------


def read_tool_source(cfg: RunConfig, function: str | None = None) -> dict[str, Any]:
    """Signatures and docstrings of our simulation and structure tools, or one function's source.

    Parsed from the files, never imported. Functions of disabled capabilities are hidden.

    Parameters
    ----------
    cfg : RunConfig
        ``disabled_capabilities``.
    function : str, optional
        A function to return in full.

    Returns
    -------
    dict
        ``{functions: [{module, name, signature, docstring}]}``, or ``{module, name, source}``.

    Raises
    ------
    ToolError
        For an unknown or hidden function.
    """
    hidden = {n for cap in cfg.disabled_capabilities for n in HIDDEN_SOURCE.get(cap, set())}
    listing = []
    for mod in SOURCE_MODULES:
        text = (Path(__file__).parent / f"{mod}.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name in hidden:
                continue
            if function is not None:
                if node.name == function:
                    return {
                        "module": f"tools/{mod}.py",
                        "name": node.name,
                        "source": ast.get_source_segment(text, node),
                    }
                continue
            if node.name.startswith("_"):
                continue
            args = ast.unparse(node.args)
            ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            doc = ast.get_docstring(node) or ""
            listing.append(
                {
                    "module": f"tools/{mod}.py",
                    "name": node.name,
                    "signature": f"{node.name}({args}){ret}",
                    "docstring": doc.split("\n\nParameters")[0][:600],
                }
            )
    if function is not None:
        raise ToolError(f"No readable function {function!r}.")
    return {"functions": listing}


# ---------------------------------------------------------------------------
# Toolkit: building blocks the worker asks for (AT-15)
# ---------------------------------------------------------------------------


class Toolkit:
    """Building blocks run in this process with one potential, under the GPU lock and the cache."""

    def __init__(
        self,
        potential: str,
        device: str = "auto",
        dtype: str = "float32",
        seed: int = 0,
        on_compute: Callable[[float], None] | None = None,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        """Bind the toolkit to a potential.

        Parameters
        ----------
        potential : str
            Potential name, e.g. ``"mace-mp-0-small"``.
        device : {"auto", "cpu", "cuda"}
            Device.
        dtype : {"float32", "float64"}
            Precision.
        seed : int
            MD seed.
        on_compute : callable, optional
            Receives the wall time of each call, in seconds (the run's compute counter).
        on_log : callable, optional
            Receives ``hp.log`` messages.
        """
        self.potential, self.device, self.dtype, self.seed = potential, device, dtype, seed
        self.on_compute, self.on_log = on_compute, on_log
        self.calls: list[str] = []

    def call(self, name: str, args: dict[str, Any]) -> Any:
        """Run one building block by name.

        Parameters
        ----------
        name : str
            One of ``TOOLKIT_CALLS``.
        args : dict
            Its arguments.

        Returns
        -------
        Any
            JSON-serialisable result.

        Raises
        ------
        ToolError
            For an unknown building block.
        """
        if name not in TOOLKIT_CALLS:
            raise ToolError(f"Unknown building block {name!r}; use one of {', '.join(TOOLKIT_CALLS)}.")
        self.calls.append(name)
        t0 = time.monotonic()
        try:
            return getattr(self, name)(**args)
        finally:
            if self.on_compute:
                self.on_compute(round(time.monotonic() - t0, 2))

    def log(self, message: str) -> None:
        """Forward an ``hp.log`` message.

        Parameters
        ----------
        message : str
            Text.
        """
        if self.on_log:
            self.on_log(message)

    def _calc(self) -> Any:
        """Return this thread's calculator for the potential.

        Returns
        -------
        Calculator
            ASE calculator.
        """
        from hephaestus.tools.simulation import load_potential

        return load_potential(self.potential, self.device, self.dtype)

    def _version(self) -> str:
        """Cache version of the potential.

        Returns
        -------
        str
            Potential name and package version.
        """
        from hephaestus.tools.simulation import potential_version

        return f"{potential_version(self.potential)}:{self.dtype}"

    def single_point(self, structure: dict[str, Any]) -> dict[str, Any]:
        """Energy, forces and stress.

        Parameters
        ----------
        structure : dict
            pymatgen structure dict.

        Returns
        -------
        dict
            ``{energy_eV, forces_eV_per_A, stress_GPa}``; stress is 3x3 in ASE's sign convention
            (negative under compression).
        """
        from hephaestus.tools.simulation import EV_PER_A3_TO_GPA, model_lock, structure_key, to_atoms

        atoms = to_atoms(structure)

        def compute() -> dict[str, Any]:
            """Evaluate the potential.

            Returns
            -------
            dict
                Result.
            """
            calc = self._calc()
            work = atoms.copy()
            work.calc = calc
            with model_lock(calc):
                energy = float(work.get_potential_energy())
                forces = work.get_forces().tolist()
                stress = (work.get_stress(voigt=False) * EV_PER_A3_TO_GPA).tolist()
            return {"energy_eV": energy, "forces_eV_per_A": forces, "stress_GPa": stress}

        return dict(
            cached("toolkit_single_point", {"s": structure_key(atoms)}, compute, version=self._version())
        )

    def relax(self, structure: dict[str, Any], fix_cell: bool = False, fmax: float = 0.01) -> dict[str, Any]:
        """Relax positions, and the cell unless ``fix_cell``.

        Parameters
        ----------
        structure : dict
            pymatgen structure dict.
        fix_cell : bool
            Relax positions only.
        fmax : float
            Force threshold, eV/Å, clamped to [0.002, 0.1].

        Returns
        -------
        dict
            ``{structure, energy_eV_per_atom, converged}``.
        """
        from hephaestus.tools.simulation import FMAX_RANGE, relax_atoms, structure_key, to_atoms, to_pmg

        atoms = to_atoms(structure)
        fmax = min(max(float(fmax), FMAX_RANGE[0]), FMAX_RANGE[1])

        def compute() -> dict[str, Any]:
            """Run the relaxation.

            Returns
            -------
            dict
                Result.
            """
            out, energy, converged, _ = relax_atoms(atoms, self._calc(), fmax, relax_cell=not fix_cell)
            return {"structure": to_pmg(out).as_dict(), "energy_eV_per_atom": energy, "converged": converged}

        inputs = {"s": structure_key(atoms), "fmax": fmax, "fix_cell": bool(fix_cell)}
        return dict(cached("toolkit_relax", inputs, compute, version=self._version()))

    def phonons(self, structure: dict[str, Any], supercell: list[int] | None = None) -> dict[str, Any]:
        """Phonon frequencies at the supercell's commensurate q-points.

        Parameters
        ----------
        structure : dict
            pymatgen structure dict.
        supercell : list of int, optional
            Repetitions; at least 7 Å across by default.

        Returns
        -------
        dict
            ``{frequencies_THz, qpoints, min_frequency_THz, supercell}``.
        """
        import numpy as np
        from phonopy.harmonic.dynmat_to_fc import get_commensurate_points

        from hephaestus.tools.simulation import (
            SUPERCELL_MIN_A,
            analyse_phonons,
            structure_key,
            supercell_dims,
            to_atoms,
        )

        atoms = to_atoms(structure)
        dims = [int(x) for x in supercell] if supercell else supercell_dims(atoms, SUPERCELL_MIN_A["fast"])

        def compute() -> dict[str, Any]:
            """Run the phonon calculation.

            Returns
            -------
            dict
                Result.
            """
            res = analyse_phonons(atoms, self._calc(), dims)
            return {
                "frequencies_THz": np.asarray(res["phonopy"].qpoints.frequencies).tolist(),
                "qpoints": np.asarray(get_commensurate_points(np.diag(dims))).tolist(),
                "min_frequency_THz": res["min_frequency_THz"],
                "supercell": dims,
            }

        return dict(
            cached(
                "toolkit_phonons", {"s": structure_key(atoms), "dims": dims}, compute, version=self._version()
            )
        )

    def md(self, structure: dict[str, Any], temperature_K: float, duration_ps: float) -> dict[str, Any]:
        """Short NVT Langevin MD on a supercell at least 8 Å across.

        Parameters
        ----------
        structure : dict
            pymatgen structure dict.
        temperature_K : float
            Temperature, K.
        duration_ps : float
            Simulated time, ps, clamped to [0.1, 10].

        Returns
        -------
        dict
            ``{energies_eV, temperatures_K, final_structure, max_displacement_A, stable}``; energies
            and temperatures every 10 steps.
        """
        import numpy as np
        from ase import units
        from ase.md.langevin import Langevin
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

        from hephaestus.tools.simulation import (
            MD_DRIFT_LIMIT_A,
            MD_MIN_CELL_A,
            MD_TIMESTEP_FS,
            model_lock,
            structure_key,
            supercell_dims,
            to_atoms,
            to_pmg,
        )

        atoms = to_atoms(structure)
        duration_ps = min(max(float(duration_ps), 0.1), 10.0)

        def compute() -> dict[str, Any]:
            """Run the MD.

            Returns
            -------
            dict
                Result.
            """
            calc = self._calc()
            work = atoms.repeat(supercell_dims(atoms, MD_MIN_CELL_A))
            work.calc = calc
            rng = np.random.default_rng(self.seed)
            MaxwellBoltzmannDistribution(work, temperature_K=temperature_K, rng=rng)
            start = work.get_positions().copy()
            energies: list[float] = []
            temps: list[float] = []
            steps = max(1, round(duration_ps * 1000 / MD_TIMESTEP_FS))
            with model_lock(calc):
                dyn = Langevin(
                    work, MD_TIMESTEP_FS * units.fs, temperature_K=temperature_K, friction=0.02, rng=rng
                )
                dyn.attach(
                    lambda: (
                        energies.append(float(work.get_potential_energy())),
                        temps.append(float(work.get_temperature())),
                    ),
                    interval=10,
                )
                dyn.run(steps)
            disp = work.get_positions() - start
            disp -= disp.mean(axis=0)
            max_disp = float(np.linalg.norm(disp, axis=1).max())
            return {
                "energies_eV": energies,
                "temperatures_K": temps,
                "final_structure": to_pmg(work).as_dict(),
                "max_displacement_A": round(max_disp, 3),
                "stable": max_disp <= MD_DRIFT_LIMIT_A,
            }

        inputs = {"s": structure_key(atoms), "T": temperature_K, "ps": duration_ps, "seed": self.seed}
        return dict(cached("toolkit_md", inputs, compute, version=self._version()))


ToolkitFactory = Callable[[Workspace, str, str], Toolkit]
"""``f(ws, potential, dtype) -> Toolkit``; tests inject toy toolkits."""


def default_toolkit(ws: Workspace, potential: str, dtype: str) -> Toolkit:
    """Return the real toolkit for a run.

    Parameters
    ----------
    ws : Workspace
        Run workspace; toolkit time adds to its compute counter.
    potential : str
        Potential name.
    dtype : str
        Precision.

    Returns
    -------
    Toolkit
        Toolkit.
    """
    return Toolkit(potential, ws.cfg.device, dtype, ws.cfg.seed, on_compute=ws.add_compute)


# ---------------------------------------------------------------------------
# Sandbox (AT-14)
# ---------------------------------------------------------------------------


def _clean_env() -> dict[str, str]:
    """Environment for the worker: no credentials, no Python path overrides.

    Returns
    -------
    dict
        Environment variables.
    """
    keep = (
        "PATH",
        "SYSTEMROOT",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    )
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONHASHSEED"] = "0"
    return env


def run_sandboxed(
    op: str,
    tool_file: Path,
    toolkit: Toolkit,
    structure: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    test_file: Path | None = None,
    timeout_s: float = 300.0,
    extra: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Run a tool, or its tests, in a worker process (AT-14, AT-15).

    The timeout counts the tool's own time only: building blocks run here and their time is
    not charged to it.

    Parameters
    ----------
    op : {"run", "test"}
        What to do.
    tool_file : pathlib.Path
        Tool file.
    toolkit : Toolkit
        Serves the worker's building-block calls.
    structure : dict, optional
        pymatgen structure dict passed to ``run`` (or to tests as ``structure``).
    inputs : dict, optional
        Tool inputs.
    test_file : pathlib.Path, optional
        Test file, for ``op="test"``.
    timeout_s : float
        Limit on the worker's own time, s.
    extra : sequence of str
        Packages beyond the allowlist that the user approved for this tool.

    Returns
    -------
    dict
        ``{ok, result}`` or ``{ok: False, error, trace}``, plus ``stderr`` (its tail) and
        ``toolkit_calls``.
    """
    request = {
        "op": op,
        "tool": str(Path(tool_file).resolve()),
        "tests": str(Path(test_file).resolve()) if test_file else None,
        "structure": structure or {},
        "inputs": inputs or {},
        "allowed": allowlist(extra),
        "cpu_s": timeout_s,
        "mem_mb": MEMORY_MB,
    }
    workdir = tempfile.mkdtemp(prefix="hephaestus_tool_")
    proc = subprocess.Popen(
        [sys.executable, "-I", "-B", str(WORKER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=workdir,
        env=_clean_env(),
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    lines: queue.Queue[str | None] = queue.Queue()
    err_tail: list[str] = []

    def pump_out() -> None:
        """Queue the worker's messages; ``None`` marks the end."""
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def pump_err() -> None:
        """Keep the last part of the worker's stderr."""
        assert proc.stderr is not None
        for line in proc.stderr:
            err_tail.append(line)
            del err_tail[:-60]

    threading.Thread(target=pump_out, daemon=True).start()
    threading.Thread(target=pump_err, daemon=True).start()
    calls_before = len(toolkit.calls)
    out: dict[str, Any] = {}

    def send(msg: dict[str, Any]) -> None:
        """Write one message to the worker.

        Parameters
        ----------
        msg : dict
            Message.
        """
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(msg, default=str) + "\n")
        proc.stdin.flush()

    try:
        send(request)
        budget = float(timeout_s)
        while True:
            t0 = time.monotonic()
            try:
                line = lines.get(timeout=max(0.0, budget))
            except queue.Empty:
                out = {"ok": False, "error": f"timed out after {timeout_s:g} s"}
                break
            budget -= time.monotonic() - t0
            if line is None:
                out = {"ok": False, "error": "the tool process ended without a result"}
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = msg.get("op")
            if kind == "call":
                try:
                    send(
                        {
                            "ok": True,
                            "result": toolkit.call(str(msg.get("name")), dict(msg.get("args") or {})),
                        }
                    )
                except Exception as exc:
                    send({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]})
            elif kind == "log":
                toolkit.log(str(msg.get("message", "")))
            elif kind == "done":
                out = {"ok": True, "result": msg.get("result")}
                break
            elif kind == "failed":
                out = {"ok": False, "error": msg.get("error", "failed"), "trace": msg.get("trace", "")}
                break
    except (BrokenPipeError, OSError) as exc:
        out = {"ok": False, "error": f"the tool process failed: {exc}"}
    finally:
        if proc.poll() is None:
            proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            with contextlib.suppress(OSError):
                stream.close()
    out["stderr"] = "".join(err_tail)[-2000:]
    out["toolkit_calls"] = toolkit.calls[calls_before:]
    return out


def check_result(result: Any, units: str) -> tuple[float, dict[str, Any]]:
    """Check a tool's result: a finite number in the declared units (AT-12).

    Parameters
    ----------
    result : Any
        What ``run`` returned.
    units : str
        Units from the ``TOOL`` header.

    Returns
    -------
    value : float
        The value.
    details : dict
        The tool's ``details``.

    Raises
    ------
    ToolError
        If the result is malformed.
    """
    if not isinstance(result, dict) or "value" not in result:
        raise ToolError("run must return a dict with 'value', 'units' and 'details'.")
    try:
        value = float(result["value"])
    except (TypeError, ValueError):
        raise ToolError(f"value {result['value']!r} is not a number.") from None
    if not math.isfinite(value):
        raise ToolError(f"value is {value}, not a finite number.")
    if str(result.get("units", "")) != units:
        raise ToolError(f"units {result.get('units')!r} differ from the declared {units!r}.")
    details = result.get("details") or {}
    return value, details if isinstance(details, dict) else {"details": details}


# ---------------------------------------------------------------------------
# Campaign handlers (AT-16 to AT-18, AT-23)
# ---------------------------------------------------------------------------


def make_handler(tool: CustomTool, toolkit_factory: ToolkitFactory | None = None) -> Handler:
    """Build the handler for ``custom_<name>``: run the tool on a candidate's current structure.

    The result is stored as a ``PropertyPrediction`` with no uncertainty, labelled as an
    agent-built tool. Nothing else on the candidate changes, so no stability verdict can.

    Parameters
    ----------
    tool : CustomTool
        Approved tool.
    toolkit_factory : callable, optional
        ``f(ws, potential, dtype) -> Toolkit``; the real toolkit by default.

    Returns
    -------
    callable
        ``handler(ws, candidate_id, **inputs) -> dict``.
    """
    factory = toolkit_factory or default_toolkit
    model = f"agent-built:{tool.name}@{tool.sha256[:8]}"

    def handler(ws: Workspace, candidate_id: str, **inputs: Any) -> dict[str, Any]:
        """Run the agent-built tool on one candidate.

        Parameters
        ----------
        ws : Workspace
            Run workspace.
        candidate_id : str
            Candidate.
        **inputs
            The tool's declared inputs.

        Returns
        -------
        dict
            ``{candidate_id, property, value, units, model, uncertainty, note, details}``, plus
            ``target`` for the target property.

        Raises
        ------
        ToolError
            If the candidate is not relaxed or the tool fails.
        """
        cs = ws.get(candidate_id)
        if cs.energy_eV_per_atom is None or cs.structure is None or not cs.potential:
            raise ToolError("Not relaxed yet: call relax first.")
        load_sources(tool)  # refuse a tool edited since approval
        potential = cs.potential
        dtype = ws.cfg.final_dtype if potential == ws.cfg.final_potential else ws.cfg.dtype
        structure = cs.structure

        def compute() -> dict[str, Any]:
            """Run the tool in the sandbox.

            Returns
            -------
            dict
                ``{value, details, toolkit_calls}``.

            Raises
            ------
            ToolError
                If the tool failed or returned a bad result.
            """
            out = run_sandboxed(
                "run",
                Path(tool.file),
                factory(ws, potential, dtype),
                structure=structure,
                inputs=inputs,
                timeout_s=ws.cfg.custom_tool_timeout_s,
                extra=tool.packages,
            )
            if not out["ok"]:
                raise ToolError(f"{tool.name} failed: {out['error']}")
            value, details = check_result(out["result"], tool.units)
            return {"value": value, "details": details, "toolkit_calls": out["toolkit_calls"]}

        key = {"s": structure, "inputs": inputs, "potential": potential, "dtype": dtype}
        t0 = time.monotonic()
        r = cached("custom", key, compute, version=tool.sha256)
        cs.wall_time_s += round(time.monotonic() - t0, 2)
        q = Quantity(value=r["value"], uncertainty=None, units=tool.units, source=model)
        cs.predictions[tool.property] = PropertyPrediction(
            candidate_id=candidate_id,
            property=tool.property,
            value=q,
            model=model,
            in_domain=cs.potential_in_domain is not False,
        )
        out = {
            "candidate_id": candidate_id,
            "property": tool.property,
            "value": round(r["value"], 4),
            "uncertainty": None,
            "units": tool.units,
            "model": model,
            "note": AGENT_LABEL,
            "details": r["details"],
        }
        if (target := target_info(ws.spec, tool.property)) is not None:
            out["target"] = target
        return out

    return handler


def custom_handlers(ws: Workspace, toolkit_factory: ToolkitFactory | None = None) -> dict[str, Handler]:
    """Handlers for every approved tool in the workspace.

    Parameters
    ----------
    ws : Workspace
        Run workspace.
    toolkit_factory : callable, optional
        Toolkit factory (tests).

    Returns
    -------
    dict
        ``custom_<name>`` to handler.
    """
    return {f"custom_{t.name}": make_handler(t, toolkit_factory) for t in ws.custom_tools.values()}
