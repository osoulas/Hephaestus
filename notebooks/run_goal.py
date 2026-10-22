"""Interactive run notebook: review a goal, answer questions, run, decide on agent tools, read the brief.

Open with ``marimo edit notebooks/run_goal.py``. It drives the same Python API as the web UI and the CLI
(``hephaestus.interfaces.python.Experiment``), so every gate and verdict is the engine's own.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Hephaestus: run a goal")


@app.cell
def _():
    """Import the marimo API and the Python API."""
    import marimo as mo

    from hephaestus.interfaces.python import Experiment

    return Experiment, mo


@app.cell
def _(mo):
    """Goal, model access and the tool-builder setting."""
    get_session, set_session = mo.state(None)
    goal = mo.ui.text_area(
        label="Goal",
        value="a lead-free oxide perovskite with a smaller band gap than SrTiO3 that is thermodynamically stable "
        "(≤ 0.05 eV/atom above hull) and dynamically stable",
        full_width=True,
    )
    backend = mo.ui.dropdown(["auto", "api", "claude-code"], value="auto", label="Model access")
    custom = mo.ui.dropdown(["ask", "off"], value="ask", label="When no tool fits")

    def _review(_):
        """Create the run and parse the goal (a model call)."""
        from hephaestus.interfaces.python import Experiment as _Experiment

        exp = _Experiment(goal.value, llm_backend=backend.value, custom_tools=custom.value)
        set_session((exp, exp.parse()))

    review = mo.ui.button(label="Review goal", on_click=_review, kind="success")
    mo.vstack([goal, mo.hstack([backend, custom, review], justify="start")])
    return backend, custom, get_session, goal, review, set_session


@app.cell
def _(get_session, mo, set_session):
    """Show the parsed specification, the capability gap, and questions with answers."""
    mo.stop(get_session() is None, mo.md("_Review a goal to see its parsed specification._"))
    exp, parsed = get_session()
    spec = parsed.spec
    questions = list(parsed.questions)
    answers = mo.ui.array([mo.ui.text(label=q, full_width=True) for q in questions]) if questions else None

    def _recheck(_):
        """Parse again with the answers given."""
        given = {q: a.value for q, a in zip(questions, answers.value, strict=True) if a.value}
        set_session((exp, exp.parse(given or None)))

    recheck = mo.ui.button(label="Re-check with answers", on_click=_recheck)
    items = [
        mo.md(
            f"**Target:** {spec.target_property} · {spec.direction} · {spec.target_value} {spec.target_units}"
        )
    ]
    if parsed.gap:
        items.append(
            mo.callout(
                mo.md(
                    f"No built-in tool computes **{parsed.gap}**. The agent will propose one, and you approve it."
                ),
                kind="warn",
            )
        )
    if parsed.errors:
        items.append(mo.callout(mo.md("\n".join(f"- {e}" for e in parsed.errors)), kind="danger"))
    if questions:
        items += [mo.md("**The parser has questions:**"), answers, recheck]
    items.append(mo.md(f"_Parsed by {parsed.parser.get('model', 'the parser')}_"))
    mo.vstack(items)
    return exp, parsed


@app.cell
def _(exp, mo, parsed):
    """Start the run once the goal is ready."""
    ready = parsed.ready and parsed.spec is not None
    mo.stop(not ready, mo.md("_Answer the questions above, then start the run._"))

    def _start(_):
        """Start the run in the background."""
        exp.start()

    start = mo.ui.button(label="Start run", on_click=_start, kind="success")
    start
    return (start,)


@app.cell
def _(mo):
    """Refresh the live view every two seconds while a run is active."""
    tick = mo.ui.refresh(default_interval="2s", label="Live view")
    tick
    return (tick,)


@app.cell
def _(exp, mo, parsed, tick):
    """Status, the latest events, and the agent's tool cards."""
    tick.value  # re-run on every refresh
    mo.stop(exp is None or not parsed.ready, None)
    state = exp.state
    _rows = []
    for e in list(exp.events(include_history=True, follow=False))[-12:]:
        p = e.payload or {}
        label = p.get("tool") or p.get("kind") or p.get("text") or p.get("message") or ""
        _rows.append(
            {
                "time": e.time.strftime("%H:%M:%S"),
                "section": e.section,
                "status": e.status,
                "detail": str(label)[:120],
            }
        )
    header = mo.md(
        f"**Status:** `{state.status}` · round {len(state.iterations) or 1} · "
        f"{'running' if exp.running else 'idle'}"
    )
    events = mo.ui.table(_rows, selection=None, page_size=12) if _rows else mo.md("_No events yet._")
    mo.vstack([header, events])
    return (state,)


@app.cell
def _(exp, mo, tick):
    """Decisions the agent is waiting for: plans, code, package requests and questions."""
    tick.value
    mo.stop(exp is None, None)
    _waiting = [p for p in exp.tool_proposals() if p.get("waiting")]
    _cards = []
    for item in _waiting:
        pid = item.get("proposal_id") or item.get("request_id")
        kind = item.get("kind") or item.get("stage")
        title = item.get("name") or item.get("package") or "question"
        body = item.get("description") or item.get("question") or ""
        text = mo.ui.text(label="Change or answer", full_width=True)
        buttons = [
            mo.ui.button(
                label=a.replace("_", " "),
                on_click=lambda _, pid=pid, a=a, box=text: exp.decide(pid, a, box.value or None),
            )
            for a in item.get("allowed", [])
        ]
        _cards.append(
            mo.vstack([mo.md(f"**{title}** · {kind}\n\n{body}"), text, mo.hstack(buttons, justify="start")])
        )
    mo.vstack(_cards) if _cards else mo.md("_The agent is not waiting for a decision._")


@app.cell
def _(exp, mo, state, tick):
    """Results once the run has finished: the best candidate, the candidates and the brief."""
    tick.value
    mo.stop(
        exp is None or state.status not in ("finished", "stopped"),
        mo.md("_Results appear when the run finishes._"),
    )
    _spec = state.spec
    _rows = []
    for it in state.iterations:
        sims = {s.candidate_id: s for s in it.sims}
        preds = {p.candidate_id: p for p in it.predictions if p.property == _spec.target_property}
        for c in it.candidates:
            sim = sims.get(c.candidate_id)
            pred = preds.get(c.candidate_id)
            _rows.append(
                {
                    "candidate": c.candidate_id,
                    "composition": c.composition,
                    "best": "★" if c.candidate_id == state.best else "",
                    "energy above hull (eV/atom)": round(sim.energy_above_hull_eV_per_atom, 3)
                    if sim and sim.energy_above_hull_eV_per_atom is not None
                    else None,
                    _spec.target_property: f"{pred.value.value:.3g} ± {pred.value.uncertainty:.2g}"
                    if pred and pred.value.uncertainty is not None
                    else (f"{pred.value.value:.3g}" if pred else None),
                }
            )
    summary = mo.md(f"**Stopped by:** {state.stop_reason or 'n/a'} · **Best:** `{state.best or 'none'}`")
    table = mo.ui.table(_rows, selection=None) if _rows else mo.md("_No candidates._")
    brief = mo.md(exp.brief) if exp.brief else mo.md("_The brief is not written yet._")
    mo.vstack([summary, table, mo.accordion({"Lab brief": brief})])


if __name__ == "__main__":
    app.run()
