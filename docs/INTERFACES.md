# Building the CLI and the web UI

For B (`interfaces/cli/`) and C (`interfaces/web/`). This explains what to call and what to
render. The engine side is `interfaces/python.py` (Python API) and `interfaces/http.py` (HTTP API),
both owned by A.

## The one rule

**The CLI and the UI are renderers, not engines.** They start runs, show events and call stop,
resume and replay. They never decide a stability verdict, a target check, a budget, a gate or a
stop. Those come in the events, already decided by code. If you find yourself computing
"is this candidate stable?", read `thermo_stable` / `dyn_stable` / `verdict` from the event
instead.

You each write your own formatting (how a tool call, a review or a round looks). They don't need
to match.

```
web UI (JS) ──HTTP + SSE──► interfaces/http.py ──┐
                                                 ├──► interfaces/python.py (Experiment) ──► harness
CLI ───────────── in-process ────────────────────┘
```

- **CLI:** import `Experiment` and run in-process. You don't need a server.
- **UI:** use only `fetch` and `EventSource`. Never import Python.

## Start with the fixtures

You don't need a working engine, a GPU or an API key to build the display.

- `tests/fixtures/events/demo/events.jsonl`: a full recorded run with 67 events (parse, two
  rounds, review, submit, final check, brief). Use it as your test input.
- `tests/fixtures/events/demo/state.json` and `brief.md`: the matching final state and brief.
- `tests/fixtures/*.json`: one example per shared type (`run_state.json`, `critique.json`, …).

For a live but offline server, point the app at a runs folder that contains the demo run and
replay it:

```bash
mkdir -p runs && cp -r tests/fixtures/events/demo runs/r-demo
python -m hephaestus serve      # then POST /replay/r-demo {"speed": 4}
```

## The event stream

Everything the UI and CLI show comes from `ProgressEvent`s:

```json
{"run_id": "r-…", "seq": 39, "time": "2026-10-21T12:22:09Z",
 "section": "orchestrator", "status": "finished", "iteration": 1, "payload": {…}}
```

- `seq` counts up from 0 within a run. A subscription always sends the history first, then live
  events, so after a reload or a server restart you get the whole run again.
- `status` is `started`, `finished` or `failed`.
- `iteration` is the review round in which the step started.

| `section` | `status` | Payload | Show it as |
| --- | --- | --- | --- |
| `run` | started | `{resume, goal, config}` | Run header |
| `goal_parser` | started / finished | `{goal}` / `{spec}` | Parsed spec (UI-2) |
| `goal_parser` | failed | `{questions, errors, spec}` | Clarifying questions; the run stops with `needs_input` |
| `orchestrator` | started | `{resume, backend, model}` | "Campaign started on …" |
| `agent` | finished | `{kind: "text" \| "thinking", text}` | Orchestrator reasoning in the log (UI-14) |
| `tool` | started / finished / failed | see below | One log line per call (UI-14) |
| `orchestrator` | finished, per round | `{round, best, stop, rule, gate_note, scores, llm_scores}` | Rounds table, confidence chart (UI-5, UI-15) |
| `orchestrator` | finished, at the end | `{submitted, rule, usage}` | Submission |
| `final_calculation` | started / finished / failed | `{candidates}` / `{verdict, checks, notes}` | Validation panel |
| `brief_generator` | started / finished | `{}` / `{pdf, cited}` | "Brief ready": fetch `/runs/{id}/brief` |
| `run` | finished / failed | `{status, stop_reason, best, errors}` | End of run |

**Tool events.** These have one `started` and one `finished` (or `failed`) per call. Calls can run
in parallel, so pair them by `call_id`, not by order. The full details are under *Event payloads*
in `docs/TOOLS.md`.

- Always present: `tool`, `call_id`, `input`; also `result` on finish (shortened).
- Literature tools: `papers: [{paper_id, title, year, citations}]`, `mode` (UI-4).
- Candidate tools: flat fields `candidate_id`, `composition`, `cand_round`, `e_hull`,
  `thermo_stable`, `dyn_stable`, `min_freq`, `verdict`, `space_group`, `value`, `uncertainty`,
  `novelty`, `rationale`, `evidence_ids`. Keep a `candidate_id → latest fields` map; that is your
  candidates table.
- `request_review`: `review` holds the full critique (`score`, `score_breakdown`, `checks`,
  `findings`, `llm_confidence`, …). `result` is not shortened here.

**Confidence chart (UI-15).** It has two lines from the per-round `orchestrator` event: `scores`
(the code rubric, which drives the stop rules) and `llm_scores` (the reviewer's own confidence).
The stop threshold is `config.confidence_threshold` (default 72). `gate_note` explains why a stop was
blocked; show it.

**Every number carries an uncertainty** (SYS-4). Show `value ± uncertainty` whenever
`uncertainty` is present.

## Typical flows

1. **Goal → questions → run.** Parse first. If `questions` is non-empty, ask them, then parse
   again with `answers`. When `spec` is ready, start the run with the spec. The run then skips the
   parser.
2. **Watch.** Subscribe to events right after starting. Nothing is missed, because history comes
   first.
3. **Stop (UI-10).** The campaign submits its best candidate and the run still writes a brief.
   Keep streaming until `run finished`.
4. **Resume (UI-16).** If `GET /runs/{id}` shows a status other than `finished` and
   `running: false`, offer Resume.
5. **Replay (UI-9).** Re-emits a stored run's events under a new `replay_id`, with no network.
   Stream that ID.

Only one run executes at a time, because MACE and MEGNet share the GPU. A second start returns
409 while one is running. Replays don't count.

## HTTP API (UI)

Base URL `http://127.0.0.1:8000`; interactive docs at `/docs`. Bodies are JSON in the shared types.

| Endpoint | Body / query | Response |
| --- | --- | --- |
| `POST /parse` | `{raw_goal, answers?, llm_backend?, llm_model?, require_novel?, band_gap_fidelity?}` | `{spec, questions, errors, parser}` |
| `POST /runs` | `{spec, config}` or `{raw_goal, answers?, config}` | `201 {run_id}`; 409 busy; 503 backend not ready |
| `GET /runs` | | `[{run_id, raw_goal, status, best, stop_reason, created_at, running}]` newest first |
| `GET /runs/{id}` | | `RunState` + `running` |
| `GET /runs/{id}/events` | `?history=true` | SSE stream (below) |
| `GET /runs/{id}/structure/{candidate_id}` | | CIF text (3D viewer, UI-7) |
| `GET /runs/{id}/brief` | `?format=md\|pdf` | Markdown or PDF; 404 until written |
| `POST /runs/{id}/stop` | | `{stopping}` |
| `POST /runs/{id}/resume` | | `{run_id, resumed}`; 409 if finished or running |
| `POST /replay/{id}` | `{speed}` (0 = no pauses) | `{replay_id}` |
| `GET /health` | | `{ok, backends, device, valency, running}`; never blocks (`backends` is `null` until the first background check ends) |
| `GET /models` | | `{models, default}` |
| `GET /config/defaults` | | `RunConfig` defaults for the controls |
| `GET /auth/valency` | `?start=true` | `{signed_in, needs_signin, access_token_expired, pending, url, error}` (UI-11) |
| `GET /auth/claude` | | `{status: checking\|signed_in\|signed_out, signed_in, error, url}` (UI-12) |

Errors come back as `{"detail": "…"}` with the status code. Show `detail` to the user.

**SSE format.** Each event is

```
id: 39
event: progress
data: {…ProgressEvent JSON…}
```

and the stream closes with `event: end`. Close your `EventSource` on `end`, or it will reconnect
and replay the history again.

```js
const es = new EventSource(`/runs/${runId}/events`);
es.addEventListener("progress", (m) => render(JSON.parse(m.data)));
es.addEventListener("end", () => es.close());
```

**Run options (UI-12, UI-13).** Put the model-access selector, model selector and "Require novel
materials" checkbox into `config` (`llm_backend`, `llm_model`, `require_novel`) on `POST /runs`, and
the same fields on `POST /parse`. Use `GET /health` → `backends` to show readiness next to each
backend choice.

## Python API (CLI)

```python
from hephaestus.interfaces.python import Experiment, ExperimentError, list_runs, backend_status

exp = Experiment(goal, llm_backend="claude-code", llm_model="claude-sonnet-5-5", require_novel=False)

parsed = exp.parse()                    # ParseResult: spec, questions, errors, parser, .ready
while parsed.questions:
    answers = {q: input(q + " ") for q in parsed.questions}
    parsed = exp.parse(answers)

exp.start()                             # background thread; or exp.run() to block
try:
    for event in exp.events():          # history, then live; ends when the run ends
        print(render(event))            # your formatter in interfaces/cli/render.py
except KeyboardInterrupt:
    exp.stop()                          # campaign submits; keep reading events to get the brief
    for event in exp.events(include_history=False):
        print(render(event))

state = exp.wait()                      # RunState
print(exp.brief)                        # Markdown, or None
```

| Need | Call |
| --- | --- |
| Subcommand `run "<goal>"` | `Experiment(goal, **opts).start()` + `events()`, or `.run()` |
| `parse "<goal>"` | `Experiment(goal).parse()` → `ParseResult.to_dict()` |
| `resume <id>` | `Experiment.load(id).start(resume=True)` + `events()` |
| `replay <id> --speed 4` | `Experiment.load(id).replay(4)` → handle; iterate `handle.events()` |
| `backends` | `backend_status()` → `{api, claude-code, auto}` each with `ready`, `error` |
| `serve` | `from hephaestus.interfaces.http import serve; serve(host, port)` |
| List runs | `list_runs()` |
| Stored state / brief / CIF | `Experiment.load(id).state`, `.brief`, `.structure(candidate_id)` |

`ExperimentError` means the request can't be done (no such run, already running, already finished,
open parser questions). Print its message.

**Before starting a `claude-code` run, check `backend_status()["claude-code"]["ready"]`.** A
signed-out Claude Code CLI hangs instead of failing. If it's not ready, tell the user to run
`python -m hephaestus login`.

**Testing the CLI offline:** inject the scripted model as `tests/integration/test_python_api.py`
does: `Experiment(goal, root=tmp_path, deps=runner.Deps(llm=ScriptedLLM(...), handlers=stub_handlers(), …))`.
Or render `events.jsonl` from the fixture directly with `ProgressEvent.model_validate_json(line)`.

## Signing in to Claude

| Endpoint | Body | Response |
| --- | --- | --- |
| `GET /auth/claude` | | `{status, signed_in, url, code_prompt, error}`; `status` is `checking`, `signed_in`, `signed_out`, or a running sign-in's `starting`, `waiting`, `verifying`, `failed`, `cancelled` |
| `POST /auth/claude/start` | | Runs `claude auth login --claudeai`, which opens the Claude page and finishes by itself after authorising. `url` is the fallback page |
| `POST /auth/claude/code` | `{code}` | Passes the code the fallback page shows; 409 when no sign-in is waiting |
| `POST /auth/claude/cancel` | | Stops a running sign-in |
| `GET /auth/anthropic` | | `{present, source: saved\|environment\|null, hint: "sk-…WXYZ"}`; never the key |
| `POST /auth/anthropic` | `{api_key}` | Checks the key with one request, saves it owner-only in `~/.hephaestus/anthropic_api_key` and uses it at once; 422 if malformed or rejected |
| `DELETE /auth/anthropic` | | Deletes the saved key (a key from the environment stays) |

From a terminal: `python -m hephaestus login` (Claude subscription), `login --api-key` (prompts, hidden),
`login --forget-api-key`, and `python -m hephaestus backends` to check readiness. A saved key is loaded at
start-up by the server and the CLI unless `ANTHROPIC_API_KEY` is already set.

## The web UI as built

`interfaces/web/` is plain HTML, CSS and ES modules: no build step, no CDN, no web fonts, so it works
with Wi-Fi off. `python -m hephaestus serve` (or `uvicorn hephaestus.interfaces.http:app`) serves it at `/`.

| File | What it does |
| --- | --- |
| `js/app.js` | Hash router (`#/new`, `#/run/<id>/<tab>`), sidebar (runs, Claude/Valency/compute status, theme), goal composer |
| `js/run.js` | Run header (status, stop/resume/replay, phase stepper) and the tabs: Overview, Activity, Literature, Candidates, Structure, Validation, Brief, Spec |
| `js/model.js` | Folds `ProgressEvent`s and `RunState` into what the tabs show; decides nothing |
| `js/chart.js` | Confidence chart (code rubric and reviewer, threshold line, tooltip, table view) and the publications-per-year columns |
| `js/viewer.js` | Canvas 3D crystal viewer: CIF parsing with symmetry, shaded atoms, B–X bonds, unit cell |
| `js/markdown.js` | Escaping Markdown renderer for the brief |
| `css/app.css` | Design tokens (light and dark), layout, components |
