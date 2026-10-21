# Hephaestus: team guide for Claude Code

Hephaestus (package `hephaestus`) is our self-designing experiment agent.

OSI Hackathon, 21–22 October 2026. Three people, 24 hours, built from scratch.
Read this whole file before writing code. Then read:

- [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) — the full specification, with requirement IDs. Its change log lists what
  changed from the original plan.
- [`docs/TOOLS.md`](docs/TOOLS.md) — every tool the orchestrator can call: owner, inputs, outputs, code verdicts, errors,
  the `Workspace`, and the event payloads the UI and CLI render. **This is the contract between the harness and
  the tools.**
- [`docs/stability_check.md`](docs/stability_check.md) — why dynamic stability uses mode mapping, with the evidence.
- [`docs/valency_tools.md`](docs/valency_tools.md) — Valency Bond's live tool schemas.

## What we are building

A research agent for materials discovery. The user types a goal in plain English, for example:

> a lead-free oxide perovskite with a smaller band gap than SrTiO3 that is thermodynamically stable
> (≤ 0.05 eV/atom above hull) and dynamically stable

The package returns a recommended composition, checked by simulation, and a lab-ready brief that cites
the literature.

**How it works.** An LLM orchestrator runs the campaign. It decides what to read, which compositions and
structures to try, and which calculations each candidate needs. It acts only through tools:

- **Literature:** search Valency Bond, the only literature source; follow similar and citing papers; note evidence.
- **Database:** look up Materials Project entries and polymorphs.
- **Structures:** propose a candidate, then build its structure from an MP entry or a prototype template.
- **Simulation:** relax with MACE, compute energy above the convex hull, run phonons with mode mapping, run MD.
- **Properties:** predict the band gap with MEGNet, or elastic moduli from the potential.
- **Review and submit:** ask for an independent review, or submit a final answer.

An **independent reviewer LLM** critiques each round. After submission, a higher-fidelity re-check runs,
then the brief is written.

**What code decides, never the LLM.** This is our main pitch point:
- the stability verdicts (thermodynamic and dynamic)
- the target check
- the submission gate
- the budgets (tool calls, wall time)
- the stop rules

The LLM can read these results but cannot override them.

**Pitch line:** "LLMatDesign showed an LLM can iterate toward a target. We built the version you would
trust before spending a week in the lab." Name the prior work (LLMatDesign, PoreVoyant), then lead with
what is new: a code-enforced stability gate, live literature grounding including failed attempts, a
higher-fidelity re-check, and a cited lab brief.

## Decisions already made (do not reopen without the team)

Each one addresses a known failure mode; the evidence is in the linked docs.

- **Demo goal:** smaller band gap than SrTiO3, stable on the hull (≤ 0.05 eV/atom) and dynamically stable.
  Backup: higher bulk modulus than SrTiO3.
- **Relative targets** use the same model for the reference and every candidate. The reference value is
  computed, not quoted.
- **Band gap model:** pretrained MEGNet multi-fidelity (`matgl`, `MEGNet-BandGap-mfi-MP-2019.4.1`), **HSE
  fidelity**. Error bar ±0.65 eV (MAE against 15 measured perovskite gaps). PBE fidelity runs about 1.6 eV
  low. Do not train our own model.
- **Potentials:** MACE-MP-0 small for screening, MACE-MP-0 medium for the final re-check. The hull is built
  from Materials Project stable phases re-relaxed with the *same* potential.
- **Dynamic stability uses mode mapping.** Plain phonons fail even known-stable perovskites, and MP's own
  DFT phonons show the same soft modes. So freeze in the softest imaginary mode, re-relax, and check the
  energy change. It only counts as a real instability if the energy drops by more than 2 meV/atom; then
  carry the distorted structure forward. See `docs/stability_check.md`.
- **Polymorphs.** The 0 K DFT ground state can be a high-temperature phase; LaFeO3 is R-3c in DFT but Pnma
  at room temperature. Default to the most-reported experimentally observed MP polymorph near the hull.
  The agent confirms the phase from the literature and can choose an exact `mp_id`.
- **Stop rule "target met"** needs all four of:
  - the code checks pass
  - the code rubric is ≥ 72
  - the reviewer's own confidence is ≥ 72
  - the reviewer raised no critical finding

  This guards against model-scale wins that fail in the lab: BaSnO3 beats SrTiO3 on the model's scale,
  but not in measured gaps.
- **Optional novelty requirement** (UI checkbox): rejects compositions with an ICSD-backed MP entry, and
  requires no prior literature report at submission.
- **LLM access, chosen per run:**
  - Anthropic API key (`api`)
  - Claude Code login through the Claude Agent SDK (`claude-code`)
  - `auto`: `api` when API credentials exist, else `claude-code`

  The model is selectable (Opus 5.5 default; Sonnet 5.5, Haiku 5.5, Fable 5.1).
- **Everything external is cached** by its exact inputs (LLM, Valency, MP, relaxations), so the demo works
  offline and replays cost nothing.
- **Runs are resumable.** Save a workspace snapshot after every tool call. Resume starts a fresh
  orchestrator conversation that is told what was already done.

## The split

Before the rounds, the whole team spends **hour 0 to 1 together on the contracts** (see below).

### Round 1 (hours 1–12): engine

| Person | Owns | Done when |
| --- | --- | --- |
| **A: Harness** | `harness/`, `core/events.py`, `llm/` | A scripted model drives a full campaign through stub tools: proposals, a review, gates refusing a bad submission, stop rules, final check and brief. A killed run resumes. |
| **B: Science tools** | `tools/structures.py`, `tools/simulation.py`, `harness/final_check.py` | Build, relax, hull, phonons with mode mapping and MD work on cached tiny cases. SrTiO3, BaSnO3, LaFeO3 and a 12.5 % solid solution give the expected verdicts. |
| **C: Data tools** | `tools/literature.py`, `tools/materials_project.py`, `tools/properties.py`, `core/store.py` | Valency search, similar/citing, trends and evidence notes work and are cached. MP lookup and polymorphs work. MEGNet HSE predictions match the registry error bar. A second identical call makes zero network requests. |

Tooling is split along its dependencies: B needs structures and potentials, C needs network services and
models. Neither waits for the other: both meet only at the shared types.

### Round 2 (hours 12–20): interfaces

| Person | Owns | Done when |
| --- | --- | --- |
| **A: Python API** | `interfaces/python.py`, `interfaces/http.py` | `Experiment(goal).run()`, step-by-step tool calls, `load()` and `resume()` work. The HTTP API serves runs, live events (SSE), the brief, resume and stop. |
| **B: CLI** | `interfaces/cli/` | A Claude Code style CLI: type a goal, watch reasoning and tool calls stream, answer clarifying questions, then see the brief. Also `run`, `parse`, `replay`, `resume`, `login`, `backends` subcommands. |
| **C: UI** | `interfaces/web/` | The goal box, the model-access, model and novelty controls, live orchestrator log, rounds table, confidence chart (code rubric and reviewer), 3D structure viewer, validation, brief with exports, sign-in buttons, and replay. |

**Decision: the CLI is thin, and so is the UI. Both are renderers, not engines.** Both consume the same event stream and call the Python API.
Neither re-implements a gate, a budget or a formatting rule. They will look like different products but
behave identically, and a harness fix reaches both.

Each interface owner can start early (in round 1, if their round-1 piece finishes) against **recorded
event fixtures** in `tests/fixtures/events/`. They don't need a working engine to build the display.

### Hours 20–24: everyone

- Cache a full demo run.
- Rehearse replay with Wi-Fi off.
- Write the pitch.
- Fix only what the rehearsal breaks.

## Contracts (hour 0, all three together)

Freeze these before splitting. Changing one later needs all three people to agree, because everyone builds
against them.

1. **Shared types**, in `core/types.py` (Pydantic v2, JSON round-trip):
   - inputs and results: `Quantity`, `SearchSpec`, `Paper`, `Insight`, `LiteratureContext`, `Candidate`,
     `StructureRecord`, `PhononResult`, `SimulationResult`, `PropertyPrediction`
   - judgement and output: `Critique` (with `llm_confidence`), `ValidationReport`, `Brief`
   - run tracking: `ProgressEvent`, `RunState`
2. **Tool schemas**, in `tools/registry.py`, as specified in [`docs/TOOLS.md`](docs/TOOLS.md): names, JSON-schema
   inputs and result shapes. Every tool returns
   a JSON-serialisable dict, and errors come back as `{"error": "..."}`, never raised to the LLM.
3. **Event payloads**: what each `ProgressEvent` carries, so the UI and CLI can render a tool call, a
   review or a stop.
4. **`RunConfig`**: every per-run option, including the LLM backend, model, band gap fidelity, novelty,
   budgets and thresholds.

Add one JSON fixture per type in `tests/fixtures/` in hour 0, so every section can be tested alone.

## Package layout

Dependencies point **down only**: interfaces → harness → tools → core (and `llm` → core). A lower layer
never imports a higher one.

```
hephaestus/
  core/                 # depends on nothing else in hephaestus
    types.py            # shared data types (contracts)
    config.py           # RunConfig, paths, environment
    store.py            # run folders, cache (by exact inputs), logs
    events.py           # ProgressEvent bus, history on disk, replay
  llm/
    base.py             # structured(), backend + model selection
    api.py              # Anthropic API backend
    claude_code.py      # Claude Agent SDK backend, long-lived sessions
    auth.py             # Claude Code login, Valency OAuth token store
  tools/                # what the agent can call: schema + handler per tool
    registry.py         # TOOLS schemas, dispatch, Workspace (per-run state)
    literature.py       # Valency client, evidence notes, novelty search
    materials_project.py
    structures.py       # templates, site assignment, polymorphs, supercells
    simulation.py       # relax, hull, phonons + mode mapping, MD
    properties.py       # MEGNet band gap (HSE), elastic, property registry
  harness/
    agent.py            # orchestrator loop (API and Claude Code backends)
    reviewer.py         # independent critique + LLM confidence
    gates.py            # target check, rubric, stop rules, submit gate
    state.py            # Workspace <-> RunState, snapshot, resume
    prompts.py          # all system prompts in one place
    runner.py           # parse -> campaign -> final check -> brief
    final_check.py      # higher-fidelity re-check, verdict rule
    brief.py            # facts, number check, assembly, PDF
  interfaces/
    python.py           # Experiment: the public Python API
    http.py             # FastAPI + server-sent events
    cli/
      app.py            # subcommands
      repl.py           # interactive Claude Code style session
      render.py         # event -> terminal output
    web/                # static UI (no build step; works offline)
tests/
  conftest.py           # markers; blocks network unless -m live
  fixtures/             # one JSON per shared type, recorded events, scripted transcripts
  unit/
    core/  llm/  tools/  harness/      # mirrors the package
  integration/
    test_agent_loop.py  # scripted model + real gates through stub tools
    test_python_api.py
    test_http.py        # FastAPI TestClient on fixture runs
    test_cli.py         # CLI on fixture runs
  slow/                 # real MACE / MEGNet on cached tiny cases
  live/                 # real LLM, Valency, MP; opt-in only
docs/
  REQUIREMENTS.md  stability_check.md  valency_tools.md
```

## How we write code

### Test-driven development

1. **Write the failing test first**, named after the requirement it checks (`test_sc1_gate_is_code`,
   `test_or2_critical_finding_blocks_target_met`). Then write the least code that passes. Then tidy.
2. **What is tested how:**
   - **Deterministic code** (gates, stop rules, parsing, site assignment, supercells, cache keys,
     verdicts): plain unit tests, written first.
   - **The agent loop:** a *scripted model* that replays fixed assistant turns (tool calls included).
     Never a live LLM in the default suite.
   - **Physics:** tiny cases on cached relaxations, marked `@pytest.mark.slow`.
   - **LLM, Valency and MP:** `@pytest.mark.live`, run only with `pytest -m live`. The default suite must
     pass with no network, no GPU and no API key.
3. **Every acceptance check in `docs/REQUIREMENTS.md` becomes a test.**
4. **Before every push:** `pytest -q` (default suite) must pass. Run `pytest -m slow` before merging tool
   changes.

### NumPy-style docstrings, on every function

Every public and private function, method and class gets a NumPy-style docstring. Ruff enforces it
(`[tool.ruff.lint] select = ["D", ...]`, `[tool.ruff.lint.pydocstyle] convention = "numpy"`). Include
units in parameter descriptions and the requirement ID where one applies.

```python
def energy_above_hull(composition: str, energy_eV_per_atom: float,
                      competing_phases: list[dict]) -> tuple[float, list[str]]:
    """Energy above the convex hull of the candidate's chemical system (SE-5).

    Parameters
    ----------
    composition : str
        Formula in pymatgen notation, e.g. ``"SrTi0.875Sn0.125O3"``.
    energy_eV_per_atom : float
        Candidate energy from the same potential as the competing phases, in eV/atom.
    competing_phases : list of dict
        Phases with ``composition`` and ``energy_eV_per_atom``, relaxed with the same potential.

    Returns
    -------
    e_hull : float
        Energy above the hull in eV/atom; negative means below the current hull.
    decomposes_to : list of str
        Reduced formulae of the decomposition products.

    Raises
    ------
    ValueError
        If the chemical system has no elemental reference phases.
    """
```

### Other rules

- **Every number carries an uncertainty**, or says it has none (SYS-4).
- **The brief's numbers come only from stored results.** The LLM writes prose with placeholders, and a
  number check blocks anything else.
- **A failed optional step degrades the run but never stops it.** Errors are recorded as
  `{step, code, message}`.
- **No secrets in code, logs or the repo.** Tokens live in `~/.hephaestus/`, owner-only.
- **Read all files as UTF-8** (`read_text(encoding="utf-8")`). The Windows default codepage corrupts "≤" and "Å".
- **Commits reference requirement IDs.** Commit to your own branch and merge when your tests pass.
- **Commits and PRs are authored by the person making them, not by Claude.** Never add a
  `Co-Authored-By: Claude ...` trailer, a "Generated with Claude Code" line, or any other AI attribution
  to commit messages or PR descriptions. The author is the team member who commits.

## Environment gotchas

- **GPU on Windows:** with an NVIDIA driver at CUDA 12.2 (e.g. 537.x), cu126 torch wheels fail with
  "device busy or unavailable". Use
  `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121`.
- **One GPU job at a time:** MACE and MEGNet share one lock. Concurrent CUDA use from several threads can
  crash the process natively (access violation). Run the server with
  `faulthandler` on.
- **float32 vs float64:** float32 is about 3× faster on consumer GPUs and fine for screening. Use float64
  for the final re-check.
- **MCP Python SDK 2.x:** schemas are `Tool.input_schema` (snake_case). Valency's `get_keyword_trends` takes
  `query`, returns a list of `{period, paper_count}` by default, and omits empty years. See
  `docs/valency_tools.md`.
- **Valency OAuth:**
  - Restore the token's expiry on load, or the SDK never refreshes it and falls back to a browser sign-in.
  - Mark a rejected refresh token so the UI asks for a new sign-in.
- **Claude Code backend:**
  - The Agent SDK bundles the CLI, but it must be signed in (`claude auth login --claudeai`). Check
    `auth status` first: a signed-out CLI hangs instead of failing.
  - Start-up can exceed 60 s, so set `CLAUDE_CODE_STREAM_CLOSE_TIMEOUT=300000`.
  - Reuse one session per call type. A cold start costs about 100 s; a reused session answers in about 2 s.
- **Server restarts:**
  - Runs live inside the server process, so a restart kills them. Snapshot the workspace and support resume.
  - Load event history from disk, so a resumed run still shows its earlier events.
- **Windows file replace:** `os.replace` fails while another process reads the file. Retry briefly.

## Commands

```bash
pip install -e ".[sim,dev]"          # plus the torch line above for a GPU
pytest -q                            # default suite: offline, no GPU, no key
pytest -m slow                       # physics on cached cases
pytest -m live                       # real LLM / Valency / MP
python -m hephaestus serve                  # UI at http://127.0.0.1:8000, API docs at /docs
python -m hephaestus                        # interactive CLI
python -m hephaestus run "<goal>" --backend claude-code
python -m hephaestus login                  # Claude Code sign-in
python -m hephaestus signin                 # Valency sign-in
```

Environment: `MP_API_KEY` (required); `ANTHROPIC_API_KEY` (for the `api` backend); `HEPHAESTUS_LLM_MODEL`
(default `claude-opus-5-5`).
