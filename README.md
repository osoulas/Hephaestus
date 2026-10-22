# Hephaestus

A self-designing experiment agent for materials discovery. Type a goal in plain English:

> a lead-free oxide perovskite with a smaller band gap than SrTiO3 that is thermodynamically stable
> (≤ 0.05 eV/atom above hull) and dynamically stable

Hephaestus returns a recommended composition, checked by simulation, and a lab-ready brief that cites the
literature.

An LLM orchestrator runs the campaign through tools: Valency Bond literature search, Materials Project
lookups, structure building, MACE relaxations, convex-hull energies, phonons with mode mapping, MD, and
MEGNet band gaps. An independent reviewer LLM critiques each round. **Code, not the LLM, decides** the
stability verdicts, the target check, the submission gate, the budgets and the stop rules. The winner is
re-checked at higher fidelity before the brief is written.

Built at the OSI Hackathon, 21–22 October 2026.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[sim,dev]"     # drop "sim" to skip torch, MACE and MEGNet
pre-commit install
```

On Windows with an NVIDIA driver at CUDA 12.2, install torch first:
`pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121`.

Environment variables: `MP_API_KEY` (required), `ANTHROPIC_API_KEY` (for the `api` backend),
`HEPHAESTUS_LLM_MODEL` (default `claude-opus-5-5`).

## Use

```bash
python -m hephaestus serve                  # UI at http://127.0.0.1:8000, API docs at /docs
python -m hephaestus                        # interactive CLI
python -m hephaestus run "<goal>" --backend claude-code
python -m hephaestus login                  # Claude Code sign-in
python -m hephaestus signin                 # Valency sign-in
```

```python
from hephaestus.interfaces.python import Experiment

state = Experiment("a lead-free perovskite with a smaller band gap than SrTiO3").run()
```

### Command line

`python -m hephaestus` (or `hephaestus`) opens an interactive session: the logo, then a goal prompt.
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

**Agent-built tools.** When no tool computes a target property, the session (where
`custom_tools` defaults to `ask`) lets the agent research and write one. Each step waits for you
in the log: accept the plan, request changes or cancel; then review the code (paged and
highlighted, with its static check, tests and trial run) and approve it; allow or deny a package;
answer a question; or let the agent give up or point it at something else. The answers go
through `Experiment.decide`, so the engine decides what each one does. A run resumed while a card
waits asks again.

**Ctrl+C** opens a menu while a campaign runs: stop the search (the best candidate still gets
the re-check and the brief), stop everything (resume later), detach, or keep running. During a
replay it changes the speed, pauses, resumes or ends it. Without a terminal the first Ctrl+C stops
the search and the second detaches; resume the run later with `hephaestus resume <id>`.

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
`--yes`, `--json` or no terminal, since nobody could review the tool. `--yes` never asks: a goal that needs answers exits with code 3. `--json` prints events as JSON
lines. The exit codes are 0 (done), 1 (failed or refused), 2 (usage), 3 (needs input) and 130
(detached).

## Test

```bash
pytest -q          # default suite: offline, no GPU, no API key
pytest -m slow     # real MACE / MEGNet on cached tiny cases
pytest -m live     # real LLM, Valency and Materials Project
ruff check . && ruff format --check .
```

The default suite blocks outbound network access; mark a test `live` to use the network.

## Documentation

- [`CLAUDE.md`](CLAUDE.md): team guide, decisions, package layout and coding rules.
- [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md): the specification, with requirement IDs.
- [`docs/TOOLS.md`](docs/TOOLS.md): the tool contract between the harness and the tools.
- [`docs/stability_check.md`](docs/stability_check.md): why dynamic stability uses mode mapping.
- [`docs/valency_tools.md`](docs/valency_tools.md): Valency Bond's live tool schemas.

## License

[MIT](LICENSE).
