"""Agent-built tools: static check, sandbox, toolkit pipe, library, registry and handler (Section 14)."""

from pathlib import Path

import pytest
from factories import workspace
from toolkit_fake import GAP, TEST_SOURCE, TOOL_SOURCE, FakeToolkit, fake_factory

from hephaestus.core.config import RunConfig
from hephaestus.core.types import Candidate
from hephaestus.tools import custom
from hephaestus.tools.registry import CandidateState, ToolError


@pytest.fixture(autouse=True)
def _cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the cache inside the test's folder.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test folder.
    monkeypatch : pytest.MonkeyPatch
        Sets ``HEPHAESTUS_CACHE_DIR``.
    """
    monkeypatch.setenv("HEPHAESTUS_CACHE_DIR", str(tmp_path / "cache"))


def write(tmp_path: Path, source: str, name: str = "tool.py") -> Path:
    """Write a file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Folder.
    source : str
        Contents.
    name : str
        File name.

    Returns
    -------
    pathlib.Path
        The file.
    """
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


# ---------------------------------------------------------------- static check


def test_at11_static_check_accepts_a_valid_tool() -> None:
    """The toy tool and its tests pass."""
    assert custom.static_check(TOOL_SOURCE, "tool")["passed"]
    assert custom.static_check(TEST_SOURCE, "test")["passed"]


@pytest.mark.parametrize(
    ("snippet", "reason"),
    [
        ("import os", "import of 'os'"),
        ("from subprocess import run as r", "import from 'subprocess'"),
        ("x = open('f')", "open is not allowed"),
        ("x = (1).__class__", "starts with _"),
        ("x = getattr(1, 'real')", "getattr is not allowed"),
        ("import numpy as np\nnp.save('f', 1)", "'save' is not allowed"),
        ("import ase.io", "import of 'ase.io'"),
        ("from . import x", "relative imports"),
    ],
)
def test_at11_static_check_names_each_problem(snippet: str, reason: str) -> None:
    """Forbidden imports, builtins and attributes fail with a named reason."""
    out = custom.static_check(TOOL_SOURCE + "\n" + snippet + "\n", "tool")
    assert not out["passed"] and reason in out["report"]


def test_at11_static_check_needs_header_and_run() -> None:
    """A tool without a literal header or a run(structure, inputs, hp) fails."""
    out = custom.static_check("def run(a):\n    return 1\n", "tool")
    assert (
        not out["passed"]
        and "no TOOL header" in out["report"]
        and "exactly (structure, inputs, hp)" in out["report"]
    )
    assert "no test_* function" in custom.static_check("x = 1\n", "test")["report"]
    long = TOOL_SOURCE + "\n" * custom.MAX_LINES
    assert "at most 500" in custom.static_check(long, "tool")["report"]


# ---------------------------------------------------------------- sandbox


def test_at14_at15_sandbox_runs_the_tool_with_building_blocks_served_here(tmp_path: Path) -> None:
    """The worker asks this process for single_point; prints go to stderr, not the protocol."""
    tk = FakeToolkit("toy")
    logs: list[str] = []
    tk.on_log = logs.append
    out = custom.run_sandboxed(
        "run", write(tmp_path, TOOL_SOURCE), tk, structure={"formula": "SrTiO3"}, inputs={"scale": 3.0}
    )
    assert out["ok"], out
    assert out["result"]["value"] == pytest.approx(15.0) and out["toolkit_calls"] == ["single_point"]
    assert logs == ["toy expansion computed"]


def test_at12_sandbox_runs_the_tests(tmp_path: Path) -> None:
    """Every test_* runs with the arguments it names."""
    tool, tests = write(tmp_path, TOOL_SOURCE), write(tmp_path, TEST_SOURCE, "test_tool.py")
    out = custom.run_sandboxed("test", tool, FakeToolkit("toy"), structure={}, test_file=tests)
    assert out["ok"] and [t["passed"] for t in out["result"]["tests"]] == [True, True]


def test_at14_sandbox_refuses_imports_at_runtime(tmp_path: Path) -> None:
    """A tool that skipped the static check still cannot import os."""
    src = TOOL_SOURCE.replace("import math", "import math\nimport os")
    out = custom.run_sandboxed("run", write(tmp_path, src), FakeToolkit("toy"))
    assert not out["ok"] and "not allowed in a Hephaestus tool" in out["error"]


def test_at14_sandbox_kills_a_tool_at_the_timeout(tmp_path: Path) -> None:
    """An endless loop is stopped and reported, not waited on."""
    src = TOOL_SOURCE.replace(
        '    energy = hp.single_point(structure)["energy_eV"]', "    while True:\n        pass"
    )
    out = custom.run_sandboxed("run", write(tmp_path, src), FakeToolkit("toy"), timeout_s=1.5)
    assert not out["ok"] and "timed out" in out["error"]


def test_at14_worker_gets_no_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """API keys in this process's environment are not passed to the worker."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    assert "ANTHROPIC_API_KEY" not in custom._clean_env()


def test_at12_check_result_needs_a_finite_number_in_the_declared_units() -> None:
    """NaN, a missing value and wrong units are refused."""
    assert custom.check_result({"value": 2, "units": "GPa", "details": {"a": 1}}, "GPa") == (2.0, {"a": 1})
    for bad, msg in [
        ({"value": float("nan"), "units": "GPa"}, "finite"),
        ({"units": "GPa"}, "'value'"),
        ({"value": 1.0, "units": "eV"}, "differ"),
    ]:
        with pytest.raises(ToolError, match=msg):
            custom.check_result(bad, "GPa")


# ---------------------------------------------------------------- library


def test_at20_library_loads_only_files_matching_the_manifest(tmp_path: Path) -> None:
    """An approved tool loads; editing one byte makes it unapproved until approved again."""
    lib = tmp_path / "lib"
    entry = custom.save_tool(lib, "toy_expansion", TOOL_SOURCE, TEST_SOURCE, run_id="r1")
    approved, unapproved = custom.scan_library(lib)
    assert [t.name for t in approved] == ["toy_expansion"] and not unapproved
    assert entry.property == GAP and entry.units == "1e-6/K"
    (lib / "toy_expansion.py").write_text(TOOL_SOURCE + "\n", encoding="utf-8")
    approved, unapproved = custom.scan_library(lib)
    assert not approved and unapproved[0]["changed"]
    with pytest.raises(ToolError, match="changed since it was approved"):
        custom.load_sources(entry)


def test_at21_hand_written_tool_loads_after_review(tmp_path: Path) -> None:
    """A user's file is parsed, not imported, then approved as shown; a later edit is refused."""
    lib = tmp_path / "lib"
    lib.mkdir()
    write(lib, TOOL_SOURCE.replace('"author": "agent"', '"author": "user"'), "toy_expansion.py")
    write(lib, TEST_SOURCE, "test_toy_expansion.py")
    _, unapproved = custom.scan_library(lib)
    f = unapproved[0]
    assert f["header"]["property"] == GAP and not f["changed"]
    write(lib, TOOL_SOURCE + "# edited\n", "toy_expansion.py")
    with pytest.raises(ValueError, match="changed since they were shown"):
        custom.approve_file(lib, "toy_expansion", f["sha256"])
    sha = custom.scan_library(lib)[1][0]["sha256"]
    tool = custom.approve_file(lib, "toy_expansion", sha)
    assert tool.author == "agent" and custom.scan_library(lib)[0] == [tool]
    assert custom.remove_tool(lib, "toy_expansion") and custom.scan_library(lib) == ([], [])


# ---------------------------------------------------------------- registry


def test_at26_disabling_elastic_removes_every_mechanical_property(tmp_path: Path) -> None:
    """The run registry, the campaign's predict_property enum and the handler all lose them."""
    cfg = RunConfig(disabled_capabilities=["elastic"], tool_library_dir=str(tmp_path / "lib"))
    reg = custom.run_registry(cfg)
    assert "bulk_modulus" not in reg and "band_gap" in reg and "density" in reg
    ws = workspace(disabled_capabilities=["elastic"], tool_library_dir=str(tmp_path / "lib"))
    tool = next(t for t in custom.campaign_tools(ws) if t["name"] == "predict_property")
    assert "bulk_modulus" not in tool["input_schema"]["properties"]["property"]["enum"]
    from hephaestus.tools.simulation import predict_property

    with pytest.raises(ToolError, match="not available in this run"):
        predict_property(ws, "c1-1", "bulk_modulus")


def test_at1_library_tools_join_the_registry(tmp_path: Path) -> None:
    """An approved tool's property becomes part of the run's registry."""
    lib = tmp_path / "lib"
    custom.save_tool(lib, "toy_expansion", TOOL_SOURCE, TEST_SOURCE)
    reg = custom.run_registry(RunConfig(tool_library_dir=str(lib)))
    assert reg[GAP]["tool"] == "custom_toy_expansion" and reg[GAP]["published_error"] is None


def test_at5_read_tool_source_hides_disabled_capabilities() -> None:
    """With elastic disabled, the elastic functions are neither listed nor readable."""
    on = custom.read_tool_source(RunConfig())
    off = custom.read_tool_source(RunConfig(disabled_capabilities=["elastic"]))
    names_on = {f["name"] for f in on["functions"]}
    names_off = {f["name"] for f in off["functions"]}
    assert "elastic_tensor" in names_on and "elastic_tensor" not in names_off and "relax" in names_off
    assert "def relax_atoms" in custom.read_tool_source(RunConfig(), "relax_atoms")["source"]
    with pytest.raises(ToolError):
        custom.read_tool_source(RunConfig(disabled_capabilities=["elastic"]), "elastic_tensor")


# ---------------------------------------------------------------- campaign handler


def relaxed_ws(tmp_path: Path, comp: str = "SrTiO3"):  # type: ignore[no-untyped-def]
    """Return a workspace with one relaxed candidate and the toy tool registered.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Folder for the library.
    comp : str
        Composition.

    Returns
    -------
    tuple
        ``(workspace, tool)``.
    """
    ws = workspace(tool_library_dir=str(tmp_path / "lib"))
    tool = custom.save_tool(tmp_path / "lib", "toy_expansion", TOOL_SOURCE, TEST_SOURCE)
    ws.custom_tools[tool.name] = tool
    cand = Candidate(
        candidate_id="c1-1", iteration=1, composition=comp, structure_type="perovskite_cubic", rationale="r"
    )
    ws.candidates["c1-1"] = CandidateState(
        candidate=cand,
        round=1,
        structure={"formula": comp},
        energy_eV_per_atom=-7.0,
        potential="mace-mp-0-small",
        e_hull=0.01,
        converged=True,
    )
    return ws, tool


def test_at16_at18_handler_stores_an_unlabelled_uncertainty_free_prediction(tmp_path: Path) -> None:
    """The value is stored as an agent-built prediction; no stability field changes."""
    ws, tool = relaxed_ws(tmp_path)
    before = ws.candidates["c1-1"].model_dump(exclude={"predictions", "wall_time_s"})
    out = custom.make_handler(tool, fake_factory)(ws, "c1-1")
    pred = ws.candidates["c1-1"].predictions[GAP]
    assert (
        out["value"] == pytest.approx(10.0)
        and out["uncertainty"] is None
        and out["note"] == "agent-built tool"
    )
    assert pred.model.startswith("agent-built:toy_expansion@") and pred.value.uncertainty is None
    assert ws.candidates["c1-1"].model_dump(exclude={"predictions", "wall_time_s"}) == before
    schema = custom.tool_schema(tool)
    assert schema["name"] == "custom_toy_expansion" and "scale" in schema["input_schema"]["properties"]


def test_at23_cache_key_includes_the_tool_fingerprint(tmp_path: Path) -> None:
    """A second call is cached; a changed tool computes again."""
    ws, tool = relaxed_ws(tmp_path)
    calls: list[int] = []

    def counting(ws_, potential: str, dtype: str) -> FakeToolkit:
        """Count toolkits built (one per computation).

        Parameters
        ----------
        ws_ : Workspace
            Workspace.
        potential, dtype : str
            Potential and precision.

        Returns
        -------
        FakeToolkit
            Toolkit.
        """
        calls.append(1)
        return FakeToolkit(potential)

    custom.make_handler(tool, counting)(ws, "c1-1")
    custom.make_handler(tool, counting)(ws, "c1-1")
    assert len(calls) == 1
    edited = custom.save_tool(
        tmp_path / "lib", "toy_expansion", TOOL_SOURCE.replace("2.0", "4.0"), TEST_SOURCE
    )
    out = custom.make_handler(edited, counting)(ws, "c1-1")
    assert len(calls) == 2 and out["value"] == pytest.approx(20.0)


def test_at20_handler_refuses_a_tool_edited_after_approval(tmp_path: Path) -> None:
    """Editing the file on disk stops the approved handler from running it."""
    ws, tool = relaxed_ws(tmp_path)
    (tmp_path / "lib" / "toy_expansion.py").write_text(TOOL_SOURCE + "\n", encoding="utf-8")
    with pytest.raises(ToolError, match="changed since it was approved"):
        custom.make_handler(tool, fake_factory)(ws, "c1-1")


def test_approved_packages_pass_the_static_check_and_the_sandbox(tmp_path: Path) -> None:
    """A package the user allowed is importable in the tool; without approval it is refused."""
    src = TOOL_SOURCE.replace("import math\n", "import math\n\nimport phonopy\n")
    assert not custom.static_check(src, "tool")["passed"]
    assert custom.static_check(src, "tool", ["phonopy"])["passed"]
    path = write(tmp_path, src)
    assert not custom.run_sandboxed("run", path, FakeToolkit("toy"))["ok"]
    assert custom.run_sandboxed("run", path, FakeToolkit("toy"), extra=["phonopy"])["ok"]


def test_install_package_refuses_anything_but_a_plain_name() -> None:
    """URLs, paths and flags never reach pip."""
    for bad in ("https://evil.example/x.tar.gz", "../x", "--index-url=x", "x; rm -rf /"):
        ok, why = custom.install_package(bad)
        assert not ok and "not a plain package name" in why
