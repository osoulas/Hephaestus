"""Interactive run notebook: review a goal, answer questions, run, decide on agent tools, read the brief.

Open with ``marimo edit notebooks/run_goal.py`` (or ``marimo run`` to hide the code). It drives the same
Python API as the web UI and the CLI (``hephaestus.interfaces.python.Experiment``), so every gate and
verdict is the engine's own; this notebook only displays them and sends your decisions.

How the cells fit together
--------------------------
marimo re-runs a cell whenever something it reads changes. The run itself lives in a background thread,
so three pieces of ``mo.state`` connect it to the page:

- ``session``: the ``Experiment`` and its parsed goal, set when you review the goal;
- ``status``: the run's status, set by the live monitor only when it changes;
- ``waiting``: the decisions the agent is waiting for, set by the monitor only when they change.

The monitor polls every two seconds, but the decision cards and the results re-run only when ``waiting``
or ``status`` actually change, so what you type into a card is not wiped by the refresh.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hephaestus

    Describe the material you need. Review the goal, answer any questions, then start the run. If the
    goal needs a property no built-in tool computes, the agent proposes a tool and waits for you to
    approve the plan and then the code.
    """)
    return


@app.cell
def _(mo):
    """Create the state that links the background run to the page (see the module docstring)."""
    get_session, set_session = mo.state(None)  # {"exp": Experiment, "parsed": ParseResult} or {"error": str}
    get_status, set_status = mo.state(None)  # the run's status, e.g. "running", "finished"
    get_waiting, set_waiting = mo.state(())  # (signature, cards): the decisions the agent is waiting for
    return (
        get_session,
        get_status,
        get_waiting,
        set_session,
        set_status,
        set_waiting,
    )


@app.cell
def _(mo):
    """Show the goal box and the run settings."""
    goal = mo.ui.text_area(
        label="Goal",
        value="a lead-free oxide perovskite with a smaller band gap than SrTiO3 that is thermodynamically stable "
        "(≤ 0.05 eV/atom above hull) and dynamically stable",
        full_width=True,
        rows=3,
    )
    backend = mo.ui.dropdown(
        {"Auto": "auto", "Anthropic API key": "api", "Claude Code login": "claude-code"},
        value="Auto",
        label="Model access",
    )
    custom_tools = mo.ui.dropdown({"Ask me": "ask", "Off": "off"}, value="Ask me", label="When no tool fits")
    no_elastic = mo.ui.checkbox(label="Turn off the built-in elastic tools (demo)")
    review = mo.ui.run_button(label="Review goal", kind="success")
    mo.vstack([goal, mo.hstack([backend, custom_tools, no_elastic], justify="start"), review])
    return backend, custom_tools, goal, no_elastic, review


@app.cell
def _(
    Experiment,
    backend,
    custom_tools,
    goal,
    mo,
    no_elastic,
    review,
    set_session,
    set_status,
    set_waiting,
):
    """Parse the goal when Review goal is clicked: a new run each time (one model call)."""
    mo.stop(not review.value)
    with mo.status.spinner(title="Reading the goal…"):
        try:
            _exp = Experiment(
                goal.value,
                llm_backend=backend.value,
                custom_tools=custom_tools.value,
                disabled_capabilities=["elastic"] if no_elastic.value else [],
            )
            set_session({"exp": _exp, "parsed": _exp.parse()})
        except Exception as _e:  # a backend that is not ready, a network error: show it, do not crash
            set_session({"error": f"{type(_e).__name__}: {_e}"})
    set_status(None)
    set_waiting(())
    return


@app.cell
def _(get_session, mo):
    """Show the parsed specification, the capability gap, and the parser's questions."""
    session = get_session()
    mo.stop(session is None, mo.md("_Review a goal to see its parsed specification._"))
    mo.stop(
        "error" in session,
        mo.callout(mo.md(f"**Could not read the goal.** {session.get('error')}"), kind="danger"),
    )
    parsed = session["parsed"]
    spec = parsed.spec
    questions = list(parsed.questions)
    answers = mo.ui.array([mo.ui.text(label=q, full_width=True) for q in questions])
    recheck = mo.ui.run_button(label="Re-check with my answers")

    def _target_line(t):
        """Describe one target in words."""
        aim = {"minimise": "below", "maximise": "above", "hit": "near"}.get(t.direction, t.direction)
        than = t.reference_material or f"{t.value} {t.units}"
        return f"- **{t.property.replace('_', ' ')}** {aim} {than}"

    _items = []
    if spec is not None:
        _items.append(mo.md(f"### {spec.title or 'Parsed specification'}"))
        _lines = "\n".join(_target_line(t) for t in spec.targets())
        _items.append(mo.md("**Targets (every one must be met):**\n" + _lines))
        _items.append(mo.md("**Hard constraints:** " + ("; ".join(spec.hard_constraints) or "none")))
    if parsed.gap:
        _items.append(
            mo.callout(
                mo.md(
                    f"No built-in tool computes **{parsed.gap.replace('_', ' ')}**. When the run starts, the agent "
                    "will propose one, and you approve the plan and then the code."
                ),
                kind="warn",
            )
        )
    if parsed.errors:
        _items.append(mo.callout(mo.md("\n".join(f"- {e}" for e in parsed.errors)), kind="danger"))
    if questions:
        _items += [mo.md("**The parser has questions before it can start:**"), answers, recheck]
    mo.vstack(_items)
    return answers, parsed, questions, recheck


@app.cell
def _(answers, get_session, mo, questions, recheck, set_session):
    """Parse again with the answers when Re-check is clicked."""
    mo.stop(not recheck.value)
    _exp = get_session()["exp"]
    _given = {q: a for q, a in zip(questions, answers.value, strict=True) if a}
    with mo.status.spinner(title="Re-reading the goal…"):
        set_session({"exp": _exp, "parsed": _exp.parse(_given or None)})
    return


@app.cell
def _(get_session, get_status, mo, parsed):
    """Offer Start once the goal is ready, and Stop while the run is active."""
    mo.stop(not parsed.ready, mo.md("_Answer the questions above, then start the run._"))
    _exp = get_session()["exp"]

    def _stop(_):
        """Stop the search; the best candidate still gets the final re-check and the brief."""
        _exp.stop("campaign")

    start = mo.ui.run_button(label="Start run", kind="success", disabled=get_status() is not None)
    # Stop is only offered while the run is live (it means nothing before the start or after the end)
    _live = get_status() in ("running", "capability")
    stop = mo.ui.button(label="Stop the search", on_click=_stop, kind="danger", disabled=not _live)
    mo.hstack([start, stop], justify="start")
    return (start,)


@app.cell
def _(get_session, mo, set_status, start):
    """Start the run in the background when Start is clicked."""
    mo.stop(not start.value)
    get_session()["exp"].start()
    set_status("running")
    return


@app.cell
def _(mo):
    """Tick every two seconds, so the monitor below can read the run's progress."""
    tick = mo.ui.refresh(default_interval="2s", label="Live view")
    tick
    return (tick,)


@app.cell
def _(get_session, get_status, get_waiting, mo, set_status, set_waiting, tick):
    """Monitor the run: show the latest events, and update status and waiting only when they change."""
    tick.value  # re-run on every tick
    _session = get_session()
    mo.stop(not _session or "exp" not in _session or not _session["exp"].exists, None)
    _exp = _session["exp"]
    _state = _exp.state
    _status = "running" if _exp.running else _state.status
    if _status != get_status():
        set_status(_status)
    # Only push the waiting cards when the set changes, so typing in a card is not wiped
    _cards = [p for p in _exp.tool_proposals() if p.get("waiting")]
    _sig = tuple(
        (c.get("proposal_id") or c.get("request_id"), c.get("stage") or c.get("kind"), c.get("revision"))
        for c in _cards
    )
    if _sig != (get_waiting() or ((), ()))[0]:
        set_waiting((_sig, _cards))
    _rows = []
    for _e in list(_exp.events(include_history=True, follow=False))[-12:]:
        _p = _e.payload or {}
        _detail = (
            _p.get("tool") or _p.get("kind") or _p.get("text") or _p.get("message") or _p.get("status") or ""
        )
        _rows.append(
            {
                "time": _e.time.strftime("%H:%M:%S"),
                "step": _e.section,
                "status": _e.status,
                "detail": str(_detail)[:120],
            }
        )
    _round = max(1, len(_state.iterations))
    mo.vstack(
        [
            mo.md(
                f"**Status:** `{_status}` · round {_round} · {len(_state.errors)} recorded issue{'' if len(_state.errors) == 1 else 's'}"
            ),
            mo.ui.table(list(reversed(_rows)), selection=None, page_size=12, label="Latest events")
            if _rows
            else mo.md("_No events yet._"),
        ]
    )
    return


@app.cell
def _(get_session, get_waiting, mo):
    """Show a card for each decision the agent is waiting for, with only the actions it offers.

    marimo only delivers clicks and typing for widgets held in a named variable of the cell, so every
    message box and button lives in the two dictionaries ``card_messages`` and ``card_buttons``; the cards
    then place those widgets. Building them inside a helper function would leave them dead.
    """
    _waiting = get_waiting()
    mo.stop(not _waiting or not _waiting[1], mo.md("_The agent is not waiting for a decision._"))
    _exp = get_session()["exp"]
    _items = _waiting[1]
    _labels = {
        "accept": "Accept",
        "revise": "Request changes",
        "cancel_tool": "Cancel the tool",
        "cancel_run": "Cancel the run",
        "allow": "Allow",
        "deny": "Don't allow",
        "answer": "Send answer",
    }

    def _id(item):
        """Return a card's proposal or request ID."""
        return item.get("proposal_id") or item.get("request_id")

    def _send(pid, action, message):
        """Send one decision; one no longer offered (already decided) is reported, not raised."""
        try:
            _exp.decide(pid, action, message or None)
            return f"Sent: {action}"
        except Exception as e:
            return f"Not sent: {e}"

    # One message box per card that can carry a message (a change request or an answer)
    card_messages = mo.ui.dictionary(
        {
            _id(it): mo.ui.text_area(label="Your message (for changes or answers)", full_width=True, rows=2)
            for it in _items
            if {"revise", "answer"} & set(it.get("allowed", []))
        }
    )

    def _message(pid):
        """Return what the user typed in a card's box, read when the button is clicked."""
        return (card_messages.value or {}).get(pid) or None

    # One button per offered action, plus one per suggested answer to a question
    _buttons = {}
    for it in _items:
        _pid = _id(it)
        for _a in it.get("allowed", []):
            _buttons[f"{_pid}|{_a}"] = mo.ui.button(
                label=_labels.get(_a, _a),
                on_click=lambda _, pid=_pid, a=_a: _send(pid, a, _message(pid)),
                kind="danger" if _a == "cancel_run" else "neutral",
            )
        for _k, _o in enumerate(it.get("options") or []):
            _buttons[f"{_pid}|option{_k}"] = mo.ui.button(
                label=_o, on_click=lambda _, pid=_pid, o=_o: _send(pid, "answer", o)
            )
    card_buttons = mo.ui.dictionary(_buttons)

    def _describe(item, kind):
        """Return the markdown and widgets that describe one card."""
        if kind == "plan":
            return [
                mo.md(
                    f"### Tool plan: `{item['name']}` · revision {item['revision']}\n\n{item['description']}\n\n"
                    f"**Method:** {item['method']}\n\n**Citations:** {', '.join(item['citations']) or 'none'}\n\n"
                    f"**If cancelled:** {item['if_cancelled']}"
                )
            ]
        if kind == "code":
            checks = "\n".join(
                f"- {'✅' if c['passed'] else '❌'} **{k}**: {c['report']}" for k, c in item["checks"].items()
            )
            parts = [mo.md(f"### Code review: `{item['name']}` · revision {item['revision']}\n\n{checks}")]
            trial = item.get("trial") or {}
            if trial:
                expected = (
                    f" (expected {trial['range'][0]} to {trial['range'][1]})" if trial.get("range") else ""
                )
                parts.append(
                    mo.md(
                        f"**Trial run on {trial.get('material')}:** {trial.get('value'):.4g} {trial.get('units')}{expected}"
                    )
                )
            parts.append(
                mo.accordion(
                    {
                        "Tool source": mo.md(f"```python\n{item.get('source') or ''}\n```"),
                        "Tests": mo.md(f"```python\n{item.get('tests') or ''}\n```"),
                    }
                )
            )
            return parts
        if kind == "package":
            install = (
                "Already installed on the server."
                if item.get("installed")
                else f"Not installed: allowing it runs `pip install {item.get('pip_name')}`."
            )
            return [
                mo.md(f"### Package request: `{item['package']}`\n\n**Why:** {item['question']}\n\n{install}")
            ]
        if kind == "give_up":
            return [
                mo.md(
                    f"### The agent proposes to stop without a tool\n\n**Why:** {item['question']}\n\n"
                    "**Allow** lets it stop; **Send answer** with a message tells it what to build instead."
                )
            ]
        return [mo.md(f"### Question from the tool builder\n\n{item['question']}")]

    def _card(item):
        """Lay out one card from its description and its widgets in the two dictionaries."""
        pid = _id(item)
        kind = item.get("kind") or item.get("stage")
        parts = _describe(item, kind)
        options = [card_buttons[k] for k in _buttons if k.startswith(f"{pid}|option")]
        if options:  # suggested answers send at once
            parts.append(mo.hstack(options, justify="start", wrap=True))
        if pid in card_messages:
            parts.append(card_messages[pid])
        actions = [card_buttons[f"{pid}|{a}"] for a in item.get("allowed", [])]
        parts.append(mo.hstack(actions, justify="start"))
        return mo.callout(mo.vstack(parts), kind="warn")

    mo.vstack([_card(c) for c in _items])
    return


@app.cell
def _(get_session, get_status, mo):
    """Show the results once the run has ended: the best candidate, the candidates and the brief."""
    _status = get_status()
    mo.stop(
        _status not in ("finished", "stopped", "failed", "paused"),
        mo.md("_Results appear when the run ends._"),
    )
    _exp = get_session()["exp"]
    _state = _exp.state
    _out = []
    if _status in ("failed", "paused"):
        _why = (
            _state.pause.get("message")
            if _state.pause
            else "; ".join(f"{e.step}: {e.message}" for e in _state.errors[-3:])
        )
        _out.append(
            mo.callout(mo.md(f"**Run {_status}.** {_why}"), kind="danger" if _status == "failed" else "warn")
        )
    _spec = _state.spec
    _rows = []
    for _it in _state.iterations:
        _sims = {s.candidate_id: s for s in _it.sims}
        for _c in _it.candidates:
            _row = {
                "candidate": _c.candidate_id,
                "composition": _c.composition,
                "best": "★" if _c.candidate_id == _state.best else "",
            }
            _sim = _sims.get(_c.candidate_id)
            _ehull = _sim.energy_above_hull_eV_per_atom if _sim else None
            _row["energy above hull (eV/atom)"] = None if _ehull is None else round(_ehull, 3)
            for _t in _spec.targets() if _spec else []:
                _p = next(
                    (
                        p
                        for p in _it.predictions
                        if p.candidate_id == _c.candidate_id and p.property == _t.property
                    ),
                    None,
                )
                _col = (
                    f"{_t.property.replace('_', ' ')} ({_t.units})"
                    if _t.units
                    else _t.property.replace("_", " ")
                )
                if _p is None:
                    _row[_col] = None
                elif _p.value.uncertainty is None:  # e.g. an agent-built tool: no error bar
                    _row[_col] = f"{_p.value.value:.3g}"
                else:
                    _row[_col] = f"{_p.value.value:.3g} ± {_p.value.uncertainty:.2g}"
            _rows.append(_row)
    _verdict = _state.validation[0].verdict if _state.validation else "not run"
    _out += [
        mo.md(
            f"**Best:** `{_state.best or 'none'}` · **stopped by:** {_state.stop_reason or 'n/a'} · "
            f"**final re-check:** {_verdict}"
        ),
        mo.ui.table(_rows, selection=None, label="Candidates") if _rows else mo.md("_No candidates._"),
        mo.accordion({"Lab brief": mo.md(_exp.brief or "_The brief was not written._")}),
    ]
    mo.vstack(_out)
    return


if __name__ == "__main__":
    app.run()
