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
## Example Overview

#### You input:

> *A lead-free oxide perovskite with a smaller band gap than SrTiO3 that is thermodynamically stable
> (≤ 0.05 eV/atom above hull) and dynamically stable.*

#### Hephaestus:

* Returns a recommended composition,
* Determines a crystal structure,
* Simulates relevant properties,
* Performs a critical evaluation including uncertainties,
* Produces a brief that cites each claim

## Key Features

- **Literature first.** The agent searches Valency Bond for the property and the chemistry. Every proposed composition cites the papers behind it.

- **It can build the tools it lacks.** When no built-in tool computes a property, the agent researches the
  method in the literature, proposes a tool, and writes it once you accept the plan. You read the code and
  approve it before it runs. Approved tools are saved for reuse in later runs, so the library grows with
  every goal. Each tool runs in a sandbox, and its results are labelled as agent-built.

- **Two independent scores** Code scores each candidate on its checks, and an independent
  reviewer LLM scores it on its own confidence. The target is met only when the code's score and the
  reviewer's confidence both reach the threshold, the code checks pass, and the reviewer raises no critical
  finding. The model cannot override the code's checks, and the reviewer cannot override the code.
- **Checked twice.** The winner is re-checked at higher fidelity (e.g. a larger ML model, full phonons, a
  recomputed reference etc.) before the brief is written.
- **Numerical transparency.** Every predicted value carries an uncertainty or says it has none, and the brief's
  numbers come only from stored results.

## How a run works
<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hephaestus_workflow.svg">
  <img alt="Hephaestus" src="docs/assets/hephaestus_workflow.svg" width="">
</picture>
</div>

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

### Command Line Interface

`python -m hephaestus` (or `hephaestus`) opens an interactive session.
Type a goal, answer any clarifying questions, and watch the reasoning, the tool calls and each
review stream past, ending with the brief. Commands start with `/` and Tab completes them:

| Command | Does |
| --- | --- |
| `/model`, `/backend`, `/novel`, `/config`, `/profile` | Choose the model, the model access, novelty, every other run option (grouped), or a preset of budgets |
| `/runs`, `/show <id>`, `/rename [id] <title>` | List runs; show a run's rounds and candidates; retitle it |
| `/brief [id] [compute] [pdf]` | Print the lab brief, or the computational brief (methods, settings, every number) |
| `/tools [list\|review\|show\|remove] [name]` | The library of agent-built tools: review waiting files, read or remove approved ones |
| `/resume <id>`, `/replay <id> [speed]` | Continue an interrupted run; replay a stored one offline |
| `/backends`, `/login`, `/signin` | Readiness; Claude Code and Valency sign-in |
| `/help`, `/exit` | |

**Ctrl+C** opens a menu while a campaign runs: stop the search (the best candidate still gets
the re-check and the brief), stop everything (resume later), detach, or keep running. During a
replay it changes the speed, pauses, resumes or ends it. Without a terminal the first Ctrl+C stops
the search and the second detaches. Resume a run later with `hephaestus resume <id>`.

The subcommands do the same without the session:

```bash
hephaestus run "<goal>" [--backend claude-code] [--model claude-sonnet-5-5] [--novel] [--yes] [--json]
          [--profile quick|standard|thorough] [--band-gap-model megnet-hse] [--custom-tools ask|off]
          [--no-hull] [--no-phonons] [--e-hull-max 0.05] [--no-elastic] [--tool-library DIR]
hephaestus parse "<goal>" [--json]
hephaestus runs | show <id> | rename <id> <title> | brief <id> [--compute] [--pdf]
hephaestus tools [list | review [name] | show <name> | remove <name>] [--dir DIR]
hephaestus resume <id>
hephaestus replay <id or run folder> [--speed 4]
hephaestus backends | login | signin | serve
```

`run` leaves agent-built tools off unless `--custom-tools ask` is given, and refuses `ask` with
`--yes`, `--json` or no terminal, since nobody could review the tool. The exit codes are 0 (done), 1 (failed or refused), 2 (usage), 3 (needs input) and 130
(detached).


### Notebook (marimo)

```bash
marimo edit notebooks/run_goal.py      # or: marimo run notebooks/run_goal.py, to hide the code
```

A reactive notebook over the Python API: review the goal and answer the parser's questions, start the
run, watch the latest events, approve the agent's tool plans and code (or answer its questions and
package requests) on cards, then read the candidates and the lab brief. Every check and verdict is the
engine's own; the notebook only displays them and sends your decisions.

### Python API

This underlies both the web GUI and interactive CLI, and can be accessed directly:

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

## Settings

Every run is configured by `RunConfig` (`hephaestus/core/config.py`). The most useful options:

| Option | Default | Meaning |
| --- | --- | --- |
| `llm_backend` | `auto` | `api`, `claude-code`, or `auto` (API when a key exists) |
| `llm_model` | `claude-opus-5-5` | Any Claude model the account can use |
| `max_iterations` | `3` | Review rounds; `0` for no limit |
| `max_tool_calls` | `90` | Tool calls before the agent is told to submit |
| `time_budget_s` | `900` | Wall time for the search |
| `confidence_threshold` | `72` | Code rubric and reviewer must both reach this to stop on target (out of 100) |
| `potential` / `final_potential` | `mace-mp-0-small` / `mace-mp-0-medium` | Screening and re-check models |
| `band_gap_model` | `megnet-hse` | Band gap model/fidelity. All gaps in a run share one model |
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
building blocks (relaxations, phonons, single points) served by the main process. The sandbox only blocks
accidents, so reviewing the code before you approve it is a necessary safeguard.

## Acknowledgements

Built for the OSI Hackathon, 21–22 October 2026, by Oskar Soulas, Peter Russell, and Peter Graham.

## License

[MIT](LICENSE)
