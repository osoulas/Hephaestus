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
