"""Worker process for agent-built tools (AT-14, AT-15).

Run as ``python -I sandbox_worker.py`` by ``tools/custom.py``; it imports nothing from
Hephaestus. It reads one request from stdin, runs the tool (or its tests) and asks the parent
for building blocks over the same pipe. One JSON message per line:

- parent to worker: ``{"op": "run" | "test", "tool": path, "tests": path, "structure": {...},
  "inputs": {...}, "allowed": [...], "cpu_s": n, "mem_mb": n}``, then replies to calls,
  ``{"ok": true, "result": ...}`` or ``{"ok": false, "error": "..."}``;
- worker to parent: ``{"op": "call", "name": ..., "args": {...}}``, ``{"op": "log", ...}``,
  and finally ``{"op": "done", "result": ...}`` or ``{"op": "failed", "error": ..., "trace": ...}``.

The tool's own prints go to stderr, so they cannot corrupt the protocol. Imports made by the
tool and its tests are limited to the allowlist, sockets are disabled, and CPU (and, on Linux,
memory) are capped. This blocks accidents, not a determined attacker: the user's code review
is the real check.
"""

from __future__ import annotations

import builtins
import contextlib
import inspect
import json
import math
import os
import sys
import traceback
import types
from typing import Any

TOOL_PREFIX = "hp_tool_"


def _plain(obj: Any) -> Any:
    """Make numpy scalars and arrays JSON-serialisable.

    Parameters
    ----------
    obj : Any
        Value that ``json`` could not encode.

    Returns
    -------
    Any
        A list or a Python number.

    Raises
    ------
    TypeError
        For values with no plain form.
    """
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"{type(obj).__name__} is not JSON-serialisable")


class Channel:
    """The JSON-lines pipe to the parent process."""

    def __init__(self) -> None:
        """Keep the real stdout for messages and send everything else printed to stderr."""
        self.out = os.fdopen(os.dup(1), "w", encoding="utf-8")
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        self.inp = sys.stdin

    def send(self, msg: dict[str, Any]) -> None:
        """Write one message.

        Parameters
        ----------
        msg : dict
            JSON message.
        """
        self.out.write(json.dumps(msg, default=_plain) + "\n")
        self.out.flush()

    def recv(self) -> dict[str, Any]:
        """Read one message.

        Returns
        -------
        dict
            JSON message.

        Raises
        ------
        EOFError
            When the parent closed the pipe.
        """
        line = self.inp.readline()
        if not line:
            raise EOFError("parent closed the pipe")
        return json.loads(line)


class Toolkit:
    """Building blocks, run by the parent process under its GPU lock and cache (``hp``)."""

    def __init__(self, channel: Channel) -> None:
        """Bind the toolkit to the pipe.

        Parameters
        ----------
        channel : Channel
            Pipe to the parent.
        """
        self._channel = channel

    def _call(self, name: str, **args: Any) -> Any:
        """Ask the parent to run one building block.

        Parameters
        ----------
        name : str
            Building block.
        **args
            Its arguments.

        Returns
        -------
        Any
            Its result.

        Raises
        ------
        RuntimeError
            When the building block failed.
        """
        self._channel.send({"op": "call", "name": name, "args": args})
        reply = self._channel.recv()
        if not reply.get("ok"):
            raise RuntimeError(f"hp.{name} failed: {reply.get('error')}")
        return reply.get("result")

    def single_point(self, structure: dict[str, Any]) -> dict[str, Any]:
        """Energy, forces and stress of a structure.

        Parameters
        ----------
        structure : dict
            pymatgen ``Structure.as_dict()``.

        Returns
        -------
        dict
            ``{energy_eV, forces_eV_per_A, stress_GPa}``.
        """
        return self._call("single_point", structure=structure)

    def relax(self, structure: dict[str, Any], fix_cell: bool = False, fmax: float = 0.01) -> dict[str, Any]:
        """Relax a structure.

        Parameters
        ----------
        structure : dict
            pymatgen ``Structure.as_dict()``.
        fix_cell : bool
            Keep the cell fixed and relax positions only.
        fmax : float
            Force threshold, eV/Å.

        Returns
        -------
        dict
            ``{structure, energy_eV_per_atom, converged}``.
        """
        return self._call("relax", structure=structure, fix_cell=fix_cell, fmax=fmax)

    def phonons(self, structure: dict[str, Any], supercell: list[int] | None = None) -> dict[str, Any]:
        """Finite-displacement phonons at the supercell's commensurate q-points.

        Parameters
        ----------
        structure : dict
            pymatgen ``Structure.as_dict()``.
        supercell : list of int, optional
            Repetitions; at least 7 Å across by default.

        Returns
        -------
        dict
            ``{frequencies_THz, qpoints, min_frequency_THz, supercell}``.
        """
        return self._call("phonons", structure=structure, supercell=supercell)

    def md(self, structure: dict[str, Any], temperature_K: float, duration_ps: float) -> dict[str, Any]:
        """Short NVT molecular dynamics.

        Parameters
        ----------
        structure : dict
            pymatgen ``Structure.as_dict()``.
        temperature_K : float
            Temperature, K.
        duration_ps : float
            Simulated time, ps (at most 10).

        Returns
        -------
        dict
            ``{energies_eV, temperatures_K, final_structure, max_displacement_A, stable}``.
        """
        return self._call("md", structure=structure, temperature_K=temperature_K, duration_ps=duration_ps)

    def log(self, message: str) -> None:
        """Show a line in the orchestrator log.

        Parameters
        ----------
        message : str
            Text.
        """
        self._channel.send({"op": "log", "message": str(message)[:500]})


def allowed(name: str, allowlist: list[str]) -> bool:
    """Whether a module may be imported by a tool (same rule as the static check).

    Parameters
    ----------
    name : str
        Dotted module name.
    allowlist : list of str
        Exact names, or ``pkg.*`` for a package and everything under it.

    Returns
    -------
    bool
        True when allowed.
    """
    for entry in allowlist:
        if entry.endswith(".*"):
            root = entry[:-2]
            if name == root or name.startswith(root + "."):
                return True
        elif name == entry:
            return True
    return False


def guard_imports(allowlist: list[str]) -> None:
    """Limit imports made from tool and test modules to the allowlist.

    Imports made inside libraries (numpy importing its own modules) are not limited.

    Parameters
    ----------
    allowlist : list of str
        Allowed modules.
    """
    real = builtins.__import__

    def guarded(
        name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0
    ) -> Any:
        """Refuse a tool's import outside the allowlist.

        Parameters
        ----------
        name, globals, locals, fromlist, level
            As for ``builtins.__import__``.

        Returns
        -------
        module
            The imported module.

        Raises
        ------
        ImportError
            For a refused import.
        """
        caller = (globals or {}).get("__name__", "")
        if str(caller).startswith(TOOL_PREFIX) and (level or not allowed(name, allowlist)):
            raise ImportError(f"import of {name!r} is not allowed in a Hephaestus tool")
        return real(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded


def block_network() -> None:
    """Make every socket fail."""
    import socket

    class NoSocket(socket.socket):
        """A socket that cannot be created."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Refuse.

            Parameters
            ----------
            *args, **kwargs
                Ignored.

            Raises
            ------
            OSError
                Always.
            """
            raise OSError("network access is disabled in Hephaestus tools")

    socket.socket = NoSocket  # type: ignore[misc]
    socket.create_connection = NoSocket  # type: ignore[assignment]


def limit_resources(cpu_s: float, mem_mb: int) -> None:
    """Cap CPU time and, where the OS enforces it, memory.

    Parameters
    ----------
    cpu_s : float
        CPU seconds.
    mem_mb : int
        Address-space limit in MB; only applied on Linux.
    """
    try:
        import resource
    except ImportError:  # Windows: the parent's timeout is the only limit
        return
    cpu = max(1, math.ceil(cpu_s))
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
    if sys.platform.startswith("linux") and mem_mb > 0:
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2**20, mem_mb * 2**20))


def load(path: str, name: str) -> Any:
    """Import a tool or test file as a restricted module.

    Parameters
    ----------
    path : str
        File path.
    name : str
        Module name; starts with ``hp_tool_`` so its imports are guarded.

    Returns
    -------
    module
        The executed module.

    Raises
    ------
    ImportError
        When the file cannot be loaded.
    """
    # Compile from source every time: cached bytecode could be stale after an edit that keeps
    # the file's size and modification second (AT-20, AT-23)
    try:
        with open(path, encoding="utf-8") as fh:
            code = compile(fh.read(), path, "exec")
    except OSError as exc:
        raise ImportError(f"cannot load {path}: {exc}") from None
    mod = types.ModuleType(name)
    mod.__file__ = path
    sys.modules[name] = mod
    exec(code, mod.__dict__)
    return mod


def run_tests(tool: Any, tests: Any, hp: Toolkit, structure: dict[str, Any]) -> list[dict[str, Any]]:
    """Run every ``test_*`` function, passing the arguments it names.

    A test may take any of ``run`` (the tool's function), ``hp``, ``structure`` (the trial
    structure) and ``tool`` (the ``TOOL`` header).

    Parameters
    ----------
    tool : module
        The tool module.
    tests : module
        The test module.
    hp : Toolkit
        Building blocks.
    structure : dict
        Trial structure.

    Returns
    -------
    list of dict
        ``{name, passed, error}`` per test.
    """
    fixtures = {"run": tool.run, "hp": hp, "structure": structure, "tool": tool.TOOL}
    out = []
    for name, fn in sorted(vars(tests).items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            params = inspect.signature(fn).parameters
            missing = [p for p in params if p not in fixtures]
            if missing:
                raise TypeError(f"unknown test argument(s) {missing}; use run, hp, structure or tool")
            fn(**{p: fixtures[p] for p in params})
            out.append({"name": name, "passed": True, "error": ""})
        except Exception as exc:
            out.append({"name": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"[:400]})
    return out


def main() -> None:
    """Serve one request from the parent."""
    channel = Channel()
    try:
        req = channel.recv()
        limit_resources(float(req.get("cpu_s", 300)), int(req.get("mem_mb", 4096)))
        block_network()
        guard_imports(list(req.get("allowed", [])))
        hp = Toolkit(channel)
        tool = load(req["tool"], TOOL_PREFIX + "tool")
        if req["op"] == "test":
            tests = load(req["tests"], TOOL_PREFIX + "tests")
            result: Any = {"tests": run_tests(tool, tests, hp, req.get("structure") or {})}
        else:
            result = tool.run(req.get("structure") or {}, dict(req.get("inputs") or {}), hp)
        channel.send({"op": "done", "result": result})
    except BaseException as exc:  # report everything, including SystemExit from the tool
        with contextlib.suppress(Exception):
            channel.send(
                {
                    "op": "failed",
                    "error": f"{type(exc).__name__}: {exc}"[:1000],
                    "trace": traceback.format_exc()[-1500:],
                }
            )


if __name__ == "__main__":
    main()
