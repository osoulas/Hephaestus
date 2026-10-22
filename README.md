<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hephaestus-wordmark-dark.svg">
  <img alt="Hephaestus" src="docs/assets/hephaestus-wordmark-light.svg" width="520">
</picture>

**A self-designing experiment agent for materials discovery.**

Describe the material you need in plain English. Hephaestus designs it, checks it by simulation, and writes a lab-ready brief that cites the literature.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-2a78d6)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-16130f)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-offline-0ca30c)](#testing)

</div>

---

> *a lead-free oxide perovskite with a smaller band gap than SrTiO3 that is thermodynamically stable
> (≤ 0.05 eV/atom above hull) and dynamically stable*

Hephaestus returns one recommended composition with its structure, its simulated properties and their
uncertainties, and a brief that cites every claim.

## Why it's different

Most LLM-driven discovery loops stop when the model says it is done. Hephaestus does not, and it can
build the tools a goal needs.

- **It builds the tools it lacks.** When no built-in tool computes a property, the agent researches the
  method in the literature, proposes a tool, and writes it once you accept the plan. You read the code and
  approve it before it runs. Approved tools are saved and reused in later runs, so the library grows with
  every goal. Each tool runs in a sandbox, and its results are labelled as agent-built.
- **Literature first.** The agent searches Valency Bond for the property, the chemistry and failed attempts,
  and every proposed composition cites the papers behind it.
- **Two independent scores must both pass.** Code scores each candidate on its checks, and an independent
  reviewer LLM scores it on its own confidence. The target is met only when the code's score and the
  reviewer's confidence both reach the threshold, the code checks pass, and the reviewer raises no critical
  finding. The model cannot override the code's checks, and the reviewer cannot override the code.
- **Checked twice.** The winner is re-checked at higher fidelity (a larger potential, full phonons, a
  recomputed reference) before the brief is written.
- **Honest numbers.** Every predicted value carries an uncertainty or says it has none, and the brief's
  numbers come only from stored results.

## How a run works

```
goal ─▶ parse ─▶ [build missing tools] ─▶ campaign ─▶ re-check ─▶ brief
         │              │                    │  ▲
         │              │                    │  └─ independent reviewer each round
         │              │                    └─ orchestrator calls tools; code gates every result
         │              └─ you approve the plan and the code
         └─ clarifying questions, if any
```

The orchestrator is an LLM (Claude) that works only through tools:

| Area | Tools |
| --- | --- |
| Literature | Valency Bond search, similar and citing papers, trends, evidence notes |
| Database | Materials Project entries and polymorphs |
| Structures | Prototype templates and Materials Project structures, site assignment, supercells |
| Simulation | MACE and CHGNet relaxations, energy above the hull, phonons with mode mapping, MD |
| Properties | MEGNet band gap (HSE, PBE, GLLB-SC or SCAN), elastic tensor and derived moduli |
| Agent-built | Tools the agent writes for properties no built-in tool computes |

## Quick start

Requires Python 3.11 or later. The simulation extras (torch, MACE, MEGNet) are large; skip `sim` to run only
the interface and the offline tests.

```bash
git clone https://github.com/osoulas/Hephaestus.git && cd Hephaestus
python -m venv .venv && source .venv/bin/activate
pip install -e ".[sim,dev]"
```

On Windows with an NVIDIA driver at CUDA 12.2, install torch first:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

You need a Materials Project key (`MP_API_KEY`) and one way to reach Claude:

- **Claude Code login** (uses your subscription): `python -m hephaestus login`
- **Anthropic API key**: set `ANTHROPIC_API_KEY`, or run `python -m hephaestus login --api-key`

Check that everything is ready:

```bash
python -m hephaestus backends
```

## Interfaces

Hephaestus has three ways in. All three call the same engine and show the same results.

### Web UI

```bash
python -m hephaestus serve      # http://127.0.0.1:8000
```

The UI has these parts:

- **New run:** write the goal, review the parsed specification and answer any questions, then run. Model
  access, the model, novelty, search depth and every other setting are in the setup panel.
- **Live activity:** the orchestrator's reasoning, every tool call with its result, reviews and decisions,
  filterable. The progress bar shows where the run is.
- **Tool decisions:** when the agent builds a tool, the plan and then the code appear as cards with
  **Accept**, **Request changes**, and cancel options. Package requests and questions from the agent appear
  the same way.
- **Overview, Candidates, Structure, Validation:** the recommended composition, the candidates ranked by the
  code's checks, a 3D view of the structure, and the final re-check.
- **Brief:** the lab brief and the computational brief (DFT settings and workflow), each downloadable as
  Markdown or PDF.
- **Replay:** any stored run plays back at demo speed with no network.
- **Tool library:** the tools you have approved, with their code, and files waiting for review.

The API documentation is served at http://127.0.0.1:8000/docs.

### Command line

```bash
python -m hephaestus login                 # sign in to Claude (Claude Code, or --api-key)
python -m hephaestus backends              # show which LLM backends and simulation parts are ready
python -m hephaestus serve --port 8000     # start the web UI and HTTP API
```

The commands `run`, `parse`, `replay`, `resume` and `signin` are not implemented yet; the CLI reports this
and exits with status 2.

### Python API

The same engine as a library. An `Experiment` is one run, stored in its own folder:

```python
from hephaestus.interfaces.python import Experiment

exp = Experiment("a lead-free oxide perovskite with a smaller band gap than SrTiO3",
                 llm_backend="claude-code", custom_tools="ask")

parsed = exp.parse()                 # the specification, clarifying questions and any capability gap
if parsed.questions:
    parsed = exp.parse({q: "..." for q in parsed.questions})

exp.start()                          # runs in the background
for event in exp.events():           # live progress: the history, then new events
    print(event.section, event.status)

state = exp.wait()                   # the final RunState
print(state.best, state.stop_reason)
print(exp.brief)                     # the lab brief, as Markdown
```

Other things you can do with an `Experiment`:

| Call | What it does |
| --- | --- |
| `Experiment.load(run_id)` | Open a stored run, with its goal, settings and state |
| `exp.resume()` / `exp.start(resume=True)` | Continue an interrupted or paused run from the same folder |
| `exp.stop("campaign")` / `exp.stop("all")` | Stop the search (and still get a brief), or stop everything |
| `exp.replay(speed=4)` | Replay a stored run's events with no network |
| `exp.tool("propose_candidate", ...)` | Call one tool through the same budgets, events and gates |
| `exp.tool_proposals()` / `exp.decide(id, "accept")` | Answer the agent's plan, code, package and question cards yourself |
| `hephaestus.interfaces.python.tool_library()` | List approved tools and files waiting for review |

Settings are keyword arguments, or a `RunConfig`:

```python
from hephaestus.core.config import RunConfig

cfg = RunConfig(max_iterations=2, band_gap_model="megnet-pbe", run_phonons=False)
exp = Experiment(goal, cfg)
```

## Settings that matter

Every run is configured by `RunConfig` (`hephaestus/core/config.py`). The most used options:

| Option | Default | Meaning |
| --- | --- | --- |
| `llm_backend` | `auto` | `api`, `claude-code`, or `auto` (API when a key exists) |
| `llm_model` | `claude-opus-5-5` | Any Claude model the account can use |
| `max_iterations` | `3` | Review rounds; `0` for no limit |
| `max_tool_calls` | `90` | Tool calls before the agent is told to submit |
| `time_budget_s` | `900` | Wall time for the search |
| `confidence_threshold` | `72` | Code rubric and reviewer must both reach this to stop on target |
| `potential` / `final_potential` | `mace-mp-0-small` / `mace-mp-0-medium` | Screening and re-check potentials |
| `band_gap_model` | `megnet-hse` | Band gap fidelity; all gaps in a run share one model |
| `run_phonons` / `run_hull` | `true` | Dynamic and thermodynamic stability checks |
| `require_novel` | `false` | Reject compositions already reported |
| `custom_tools` | `off` (Python), `ask` (UI) | Let the agent build a tool for a missing property |
| `offline` | `false` | Use cached results only; make no network requests |

## Agent-built tools

When a goal asks for a property no built-in tool computes, the agent can write one. It runs in four steps,
each waiting for you:

1. **Plan:** the method, its citations and what it needs. Accept, request changes, or cancel.
2. **Code:** the tool and its tests. Code checks it, runs the tests and a trial run on a reference material,
   and the trial value must fall in a range the agent took from the literature.
3. **Packages:** if the method needs a package outside the sandbox's allowlist, the agent asks first.
4. **Approval:** you read the source and approve it. Approved tools are saved to `./hephaestus_tools/` and
   reused by later runs.

Tools run in a sandboxed worker process: a fixed import allowlist, no network, time and memory limits, and
building blocks (relaxations, phonons, single points) served by the main process. The sandbox blocks
accidents, not a determined attacker, so reviewing the code before you approve it is the real safeguard.

## Project layout

```
hephaestus/
  core/          shared types, configuration, cache, event bus
  llm/           Claude backends (API, Claude Code), structured calls, sign-in
  tools/         tools the agent calls: literature, Materials Project, structures, simulation, agent-built
  harness/       orchestrator loop, reviewer, gates, capability phase, final re-check, briefs, runner
  interfaces/    Python API, HTTP API, CLI, web UI
docs/            requirements, tool contract, stability method, Valency tools
tests/           unit, integration, slow (real MACE and MEGNet) and live suites
```

## Testing

```bash
pytest -q                  # default suite: offline, no GPU, no API key
pytest -m slow             # real MACE and MEGNet on cached tiny cases
pytest -m live             # real LLM, Valency and Materials Project
ruff check . && ruff format --check .
```

The default suite blocks outbound network access. Mark a test `live` to let it use the network.

## Documentation

- [`CLAUDE.md`](CLAUDE.md): team guide, decisions, package layout and coding rules.
- [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md): the specification, with requirement IDs.
- [`docs/TOOLS.md`](docs/TOOLS.md): the tool contract, including agent-built tools and their events.
- [`docs/stability_check.md`](docs/stability_check.md): why dynamic stability uses mode mapping.
- [`docs/valency_tools.md`](docs/valency_tools.md): Valency Bond's live tool schemas.

## Acknowledgements

Built at the OSI Hackathon, 21–22 October 2026, by Oskar Soulas and the team. Builds on the prior work
LLMatDesign and PoreVoyant, and on
[MACE](https://github.com/ACEsuit/mace), [MEGNet and matgl](https://github.com/materialsvirtuallab/matgl),
[pymatgen](https://pymatgen.org), [Materials Project](https://materialsproject.org) and
[Valency Bond](https://valency.bond).

## License

[MIT](LICENSE).
