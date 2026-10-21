// The run view: header (status, actions, phase stepper), tabs, and each tab's content.
// Live data comes from the event stream; the stored RunState fills in the rest.

import { api, streamEvents } from "./api.js";
import { columnChart, confidenceChart, confidenceLegend, scoreRing } from "./chart.js";
import { ago, check, clamp, download, duration, empty, formula, h, icon, label, num, pill, pm, store, stored, toast } from "./dom.js";
import { renderMarkdown } from "./markdown.js";
import { PHASES, RunModel, toolKind } from "./model.js";
import { CONFIG_GROUPS, configName, modelName, specGrid, targetText } from "./spec.js";
import { CrystalViewer, elementStyle, parseCif } from "./viewer.js";

const TABS = [
  { key: "overview", name: "Overview", icon: "overview" },
  { key: "activity", name: "Activity", icon: "log" },
  { key: "literature", name: "Literature", icon: "book" },
  { key: "candidates", name: "Candidates", icon: "flask" },
  { key: "structure", name: "Structure", icon: "cube" },
  { key: "validation", name: "Validation", icon: "shield" },
  { key: "brief", name: "Brief", icon: "doc" },
  { key: "spec", name: "Spec", icon: "sliders" },
];

const SCORE_MAX = { property: 30, thermo: 20, dynamic: 20, reliability: 15, novelty_synthesis: 15 };
const SEVERITY = { critical: "critical", major: "serious", minor: "warning", info: "info" };

/**
 * Mount the run view.
 * @param {HTMLElement} root The scrolling main element.
 * @param {string} id Run or replay ID.
 * @param {string} tab Initial tab.
 * @param {{onChange: () => void}} hooks Called when the run starts or ends.
 * @returns {{unmount: () => void, setTab: (t: string) => void}}
 */
export function mountRun(root, id, tab, hooks) {
  const model = new RunModel(id);
  const ui = {
    tab: TABS.some((t) => t.key === tab) ? tab : "overview",
    filter: "all",
    open: new Set(),
    follow: true,
    selected: null,
    paperQuery: "",
    openAbstracts: new Set(),
    collapsedRounds: new Set(),
    brief: null,
    briefLoading: false,
    notFound: false,
    replaySpeed: stored("heph-replay-speed", 4),
    replayInfo: null, // {source, total_events, speed} for a replay
  };
  const head = h("header.run-head");
  const body = h("div.tab-body", { role: "tabpanel" });
  root.append(head, body);

  let viewer = null;
  let lastTabSig = null;
  let lastHeadSig = null;
  let scheduled = false;
  let closeStream = null;
  let stateTimer = null;
  let alive = true;

  const schedule = () => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      if (alive) render();
    });
  };

  async function loadState() {
    try {
      model.setState(await api(`/runs/${encodeURIComponent(id)}`));
    } catch (e) {
      if (e.status === 404) ui.notFound = true;
    }
    schedule();
  }

  async function loadReplayInfo() {
    try {
      const st = await api(`/runs/${encodeURIComponent(id)}`);
      ui.replayInfo = st.replay || null;
      if (ui.replayInfo) ui.replaySpeed = ui.replayInfo.speed;
      if (!model.goal) model.goal = st.raw_goal;
    } catch (e) {
      if (e.status === 404) ui.notFound = true;
    }
    lastHeadSig = null;
    schedule();
  }

  async function setReplaySpeed(sp) {
    ui.replaySpeed = sp;
    store("heph-replay-speed", sp);
    // Update the pickers in place: re-rendering the header would close an open popover.
    for (const b of head.querySelectorAll(".speed-picker button")) b.setAttribute("aria-pressed", String(Number(b.dataset.speed) === sp));
    if (model.isReplay && model.running) {
      try {
        await api(`/replay/${encodeURIComponent(id)}/speed`, { body: { speed: sp } });
      } catch (e) {
        toast(e.message, "critical");
      }
    }
  }

  async function stopReplay() {
    try {
      await api(`/runs/${encodeURIComponent(id)}/stop`, { method: "POST" });
    } catch (e) {
      toast(e.message, "critical");
    }
  }

  const loadStateSoon = () => {
    clearTimeout(stateTimer);
    stateTimer = setTimeout(loadState, 350);
  };

  function connect() {
    if (closeStream) closeStream();
    closeStream = streamEvents(
      id,
      (e) => {
        if (!model.apply(e)) return;
        const done = e.section === "run" && e.status !== "started";
        if (!model.isReplay && (["orchestrator", "final_calculation", "brief_generator"].includes(e.section) || done)) loadStateSoon();
        if (e.section === "run" && e.status === "started") hooks.onChange();
        schedule();
      },
      () => {
        model.ended = true;
        model.running = false;
        lastHeadSig = null;
        if (model.isReplay) loadState();
        else loadStateSoon();
        hooks.onChange();
        schedule();
      },
    );
  }

  if (!model.isReplay) loadState();
  else loadReplayInfo();
  connect();
  const ticker = setInterval(() => model.running && tick(), 1000);
  const onTheme = () => viewer && viewer.draw();
  window.addEventListener("themechange", onTheme);

  // ------------------------------------------------------------------ actions

  async function stop() {
    try {
      const r = await api(`/runs/${encodeURIComponent(id)}/stop`, { method: "POST" });
      toast(r.stopping ? "Stopping: the campaign will submit its best candidate and write a brief." : "The run is not executing in this server.");
    } catch (e) {
      toast(e.message, "critical");
    }
  }

  async function resume() {
    try {
      await api(`/runs/${encodeURIComponent(id)}/resume`, { method: "POST" });
      toast("Resumed. A fresh orchestrator picks up where the run stopped.");
      model.ended = false;
      model.running = true;
      connect();
      hooks.onChange();
      schedule();
    } catch (e) {
      toast(e.message, "critical");
    }
  }

  async function replay() {
    try {
      const src = model.isReplay ? (ui.replayInfo && ui.replayInfo.source) || sourceId() : id;
      const { replay_id } = await api(`/replay/${encodeURIComponent(src)}`, { body: { speed: ui.replaySpeed } });
      location.hash = `#/run/${encodeURIComponent(replay_id)}/activity`;
    } catch (e) {
      toast(e.message, "critical");
    }
  }

  const sourceId = () => (model.state && model.state.run_id) || id;

  const selectCandidate = (cid) => {
    ui.selected = cid;
    location.hash = `#/run/${encodeURIComponent(id)}/structure`;
  };

  // ------------------------------------------------------------------ render

  function render() {
    renderHead();
    const sig = tabSignature();
    if (sig === lastTabSig) return;
    lastTabSig = sig;
    const keepScroll = root.scrollTop;
    // Follow the newest entry only while live and already at the bottom, like a chat.
    const atBottom = root.scrollHeight - root.scrollTop - root.clientHeight < 160;
    if (viewer && ui.tab !== "structure") {
      viewer.destroy();
      viewer = null;
    }
    const content = ui.notFound && !model.events.length ? notFound() : TAB_RENDER[ui.tab]();
    body.replaceChildren(content);
    syncAnimations(body);
    tick();
    if (ui.tab === "literature") markLongAbstracts();
    root.scrollTop = keepScroll;
    if (ui.tab === "activity" && ui.follow && model.running && model.ended === false && atBottom && keepScroll > 0) root.scrollTop = root.scrollHeight;
  }

  function tabSignature() {
    const base = `${ui.tab}|${ui.notFound}|${model.state ? model.state.status + model.state.iterations.length + (model.state.brief ? 1 : 0) + model.state.validation.length : "-"}|${model.ended}`;
    switch (ui.tab) {
      case "structure":
        return `${base}|${model.candidateIds.join(",")}|${ui.selected}|${model.useState}`;
      case "brief":
        return `${base}|${model.phase.brief}|${ui.brief ? ui.brief.length : 0}|${ui.briefLoading}`;
      case "spec":
        return `${base}|${model.spec ? 1 : 0}`;
      case "literature":
        return `${base}|${model.papers.size}|${ui.paperQuery}|${model.useState}`;
      default:
        return `${base}|${model.lastSeq}|${ui.filter}|${ui.open.size}|${[...ui.open].join()}|${ui.follow}|${model.running}`;
    }
  }

  function renderHead() {
    const fs = model.finalStep;
    const sig = `${model.status}|${JSON.stringify(model.phase)}|${model.round}|${model.toolCount}|${model.best}|${ui.tab}|${model.running}|${model.resumable}|${model.paperList.length}|${model.candidateIds.length}|${model.log.length}|${fs ? fs.seq + fs.status : ""}|${model.isReplay ? model.events.length + JSON.stringify(ui.replayInfo) + model.ended : ""}`;
    if (sig === lastHeadSig) return tick();
    lastHeadSig = sig;
    head.replaceChildren(headContent());
    syncAnimations(head);
    tick();
  }

  // Update the live counters in place, so nothing is re-rendered (and no animation restarts) each second.
  function tick() {
    const cfg = model.config || (model.state && model.state.config) || {};
    for (const el of root.querySelectorAll("[data-tick]")) {
      const kind = el.dataset.tick;
      if (kind === "elapsed") {
        el.textContent = model.elapsed == null ? "" : model.running ? `${duration(model.elapsed)} elapsed` : `took ${duration(model.elapsed)}`;
      } else if (kind === "campaign" || kind === "final") {
        const used = kind === "campaign" ? model.span(model.campaignStart, model.campaignEnd) : model.span(model.finalStart, model.finalEnd);
        const max = kind === "campaign" ? cfg.time_budget_s || 900 : cfg.final_time_budget_s || 600;
        setMeter(el, used || 0, max, `${duration(used || 0)} / ${duration(max)}`);
      } else if (kind === "finalstep" && model.finalStep && model.finalStep.status === "started") {
        el.textContent = duration((Date.now() - new Date(model.finalStep.started)) / 1000);
      }
    }
  }

  function headContent() {
    const cfg = model.config || (model.state && model.state.config) || {};
    const status = model.status;
    const statusPill = model.running
      ? h("span.pill.live", null, h("span.dot.live"), model.isReplay ? "Replaying" : "Running")
      : status === "finished"
        ? pill("Finished", "good", "check")
        : status === "needs_input"
          ? pill("Needs input", "warning", "message")
          : status === "failed"
            ? pill("Failed", "critical", "x")
            : status === "created"
              ? pill("Created")
              : pill("Interrupted", "warning", "alert");
    const actions = [];
    if (model.running && !model.isReplay) actions.push(h("button.btn.danger", { onclick: stop }, icon("stop"), "Stop"));
    if (model.resumable) actions.push(h("button.btn.primary", { onclick: resume }, icon("resume"), "Resume"));
    if (!model.running && !model.isReplay && (model.events.length || model.state)) {
      actions.push(
        h(
          "details.menu.replay-menu",
          null,
          h("summary.btn", { title: "Replay this run" }, icon("replay"), "Replay"),
          h(
            "div.menu-list.replay-pop",
            null,
            h("div.replay-pop-title", null, "Replay this run"),
            h("div.muted.replay-pop-sub", null, "Plays the recorded events back with no network, as they happened."),
            speedPicker(),
            h(
              "button.btn.primary",
              {
                onclick: (e) => {
                  e.currentTarget.closest("details").open = false;
                  replay();
                },
              },
              icon("play"),
              "Start replay",
            ),
          ),
        ),
      );
    }
    const meta = [
      cfg.llm_model ? h("span", null, modelName(cfg.llm_model)) : null,
      model.startTime ? h("span", null, `${model.isReplay ? "Recorded" : "Started"} ${ago(model.startTime)}`) : model.state ? h("span", null, `Created ${ago(model.state.created_at)}`) : null,
      model.startTime ? h("span.num", { dataset: { tick: "elapsed" } }) : null,
      cfg.require_novel ? pill("Novel only", "info", "star") : null,
    ].filter(Boolean);
    const metaWithSeps = [];
    meta.forEach((m, i) => {
      if (i) metaWithSeps.push(h("span.sep"));
      metaWithSeps.push(m);
    });
    const counts = {
      activity: model.log.filter((e) => e.type === "tool").length || null,
      literature: model.paperList.length || null,
      candidates: model.candidateIds.length || null,
    };
    return h(
      "div.run-head-inner",
      null,
      h(
        "div.run-title-row",
        null,
        h("div", null, h("div", { style: { display: "flex", gap: "10px", alignItems: "center", marginBottom: "6px" } }, statusPill), h("h1.run-goal", { title: id }, model.goal || (model.state && model.state.raw_goal) || id), h("div.run-meta", null, metaWithSeps)),
        h("div.run-actions", null, actions),
      ),
      model.isReplay ? replayBar() : null,
      stepper(),
      h(
        "nav.tabs",
        { role: "tablist" },
        TABS.map((t) =>
          h(
            "a.tab",
            { href: `#/run/${encodeURIComponent(id)}/${t.key}`, role: "tab", "aria-selected": String(ui.tab === t.key) },
            icon(t.icon),
            t.name,
            counts[t.key] ? h("span.count", null, counts[t.key]) : null,
          ),
        ),
      ),
    );
  }

  function speedPicker() {
    const speeds = [
      [1, "1×"],
      [2, "2×"],
      [4, "4×"],
      [10, "10×"],
      [0, "Instant"],
    ];
    return h(
      "div.segmented.speed-picker",
      { role: "group", "aria-label": "Replay speed" },
      speeds.map(([sp, text]) => h("button", { type: "button", dataset: { speed: String(sp) }, "aria-pressed": String(ui.replaySpeed === sp), onclick: (e) => (e.preventDefault(), setReplaySpeed(sp)) }, text)),
    );
  }

  function replayBar() {
    const info = ui.replayInfo || {};
    const total = info.total_events || 0;
    const seen = model.events.length;
    const frac = total ? Math.min(1, seen / total) : model.ended ? 1 : 0;
    const source = info.source || sourceId();
    const live = model.running || (!model.ended && seen < total);
    return h(
      "div.replay-bar",
      null,
      h("div.replay-icon", null, icon("replay")),
      h(
        "div.replay-main",
        null,
        h(
          "div.replay-line",
          null,
          h("strong", null, live ? "Replaying" : model.ended && seen < total ? "Replay stopped" : "Replay finished"),
          h("span.muted", null, " · recorded run "),
          h("a.mono", { href: `#/run/${encodeURIComponent(source)}/${ui.tab}` }, source),
          h("span.muted.num", { style: { marginLeft: "auto" } }, total ? `${seen} / ${total} events` : `${seen} events`),
        ),
        h("div.meter-track.replay-track", null, h("div.meter-fill", { style: { width: `${frac * 100}%` } })),
      ),
      live
        ? h("div.replay-actions", null, speedPicker(), h("button.btn.sm", { onclick: stopReplay }, icon("stop"), "Stop"))
        : h(
            "div.replay-actions",
            null,
            h("button.btn.sm", { onclick: replay }, icon("replay"), "Replay again"),
            h("a.btn.sm", { href: `#/run/${encodeURIComponent(source)}/overview` }, "Open recorded run"),
          ),
    );
  }

  function stepper() {
    const cfg = model.config || (model.state && model.state.config) || {};
    const detail = {
      parse: { pending: "Waiting", active: "Reading the goal…", done: model.spec ? h("span", null, targetText(model.spec)) : "Done", input: "Needs answers", failed: "Failed" },
      campaign: {
        failed: "Failed: see Overview",
        pending: "Not started",
        active: `Round ${Math.max(model.round, 1)} of ${cfg.max_iterations ?? "?"} · ${model.toolCount} tool calls`,
        done: `${model.roundRows.length} round${model.roundRows.length === 1 ? "" : "s"}${model.submitted ? ` · submitted ${model.submitted.candidate_id}` : ""}`,
      },
      final: {
        pending: cfg.run_final === false ? "Off" : "Not started",
        active: model.finalStep
          ? h("span", null, finalStepName(model.finalStep), model.finalStep.status === "started" ? h("span", null, " · ", h("span.num", { dataset: { tick: "finalstep" } })) : null)
          : `Re-checking with ${cfg.final_potential || "the final potential"}…`,
        done: model.final && model.final.verdict ? label(model.final.verdict) : finalFromState() || "Done",
        failed: "Failed (run continues)",
        skipped: "Skipped",
      },
      brief: { pending: "Not started", active: "Writing…", done: "Ready", failed: "Failed" },
    };
    return h(
      "div.stepper",
      null,
      PHASES.map((p) => {
        const st = model.phase[p.key];
        const cls = st === "done" ? "done" : st === "active" && model.running ? "active" : st === "failed" ? "failed" : st === "input" ? "failed" : "";
        const n = PHASES.indexOf(p) + 1;
        const mark =
          cls === "done"
            ? h("span.step-mark.done", null, icon("check"))
            : cls === "active"
              ? h("span.step-mark.active", null, h("span.spinner"))
              : cls === "failed"
                ? h("span.step-mark.failed", null, icon("x"))
                : h("span.step-mark", null, n);
        return h(
          `div.step${cls ? "." + cls : ""}`,
          null,
          h("div.step-name", null, mark, p.name),
          h("div.step-detail", null, (detail[p.key] || {})[st] || ""),
        );
      }),
    );
  }

  const finalFromState = () => model.useState && model.state.validation[0] && label(model.state.validation[0].verdict);

  // ------------------------------------------------------------------ tabs

  const TAB_RENDER = {
    overview: overviewTab,
    activity: activityTab,
    literature: literatureTab,
    candidates: candidatesTab,
    structure: structureTab,
    validation: validationTab,
    brief: briefTab,
    spec: specTab,
  };

  function notFound() {
    return empty("search", "Run not found", "It may have been a replay from an earlier server session. Pick a run on the left.");
  }

  // ---------------------------------------------------------------- overview

  function overviewTab() {
    const cfg = model.config || (model.state && model.state.config) || {};
    const rows = model.roundRows;
    return h(
      "div.grid",
      null,
      failureCallout(),
      h(
      "div.grid.overview",
      null,
      h("div.grid", null, heroCard(), confidenceCard(rows, cfg.confidence_threshold ?? 72, cfg.max_iterations || 3)),
      h("div.grid", { style: { alignContent: "start" } }, budgetCard(cfg), latestReviewCard()),
      ),
    );
  }

  function runErrors() {
    const fromEvent = (model.runResult && Array.isArray(model.runResult.errors) && model.runResult.errors) || [];
    const fromState = (model.useState && model.state.errors) || [];
    const seen = new Set();
    return [...fromEvent, ...fromState].filter((e) => {
      const k = `${e.step}|${e.code}|${e.message}`;
      return !seen.has(k) && seen.add(k);
    });
  }

  function failureCallout() {
    if (model.status !== "failed") return null;
    const errs = runErrors();
    return h(
      "div.callout.critical",
      { style: { alignItems: "flex-start" } },
      icon("alert"),
      h(
        "div",
        { style: { display: "grid", gap: "6px" } },
        h("strong", null, "The run failed before it could finish"),
        errs.length
          ? h("ul.list-plain", null, errs.map((e) => h("li", null, h("span"), h("span", null, h("strong", null, `${label(e.step)}: `), e.message))))
          : h("span", null, "No error was recorded. See the Activity tab."),
        errs.some((e) => /No module named 'mace'|MP_API_KEY/.test(e.message))
          ? h("span", null, "Fix the setup (see the sidebar), then start a new run.")
          : null,
      ),
    );
  }

  function heroCard() {
    const best = model.best;
    const spec = model.spec || {};
    if (!best) {
      const lastTool = [...model.calls.values()].pop();
      return h(
        "section.card.hero",
        null,
        h("div.hero-label", null, icon("star"), "Recommended candidate"),
        h("div.hero-formula.muted", { style: { fontSize: "26px" } }, model.running ? "Searching…" : "No candidate yet"),
        h(
          "p.hero-why",
          null,
          model.running
            ? lastTool
              ? h("span", null, "Latest step: ", h("span.mono", null, lastTool.tool), " ", summary(lastTool))
              : "The orchestrator is reading the goal and the literature."
            : model.status === "needs_input"
              ? "The goal parser asked clarifying questions. Start a new experiment and answer them."
              : "This run ended before a review picked a best candidate.",
        ),
      );
    }
    const c = model.candidate(best);
    const thr = spec.stability_thresholds || {};
    const review = [...model.reviews].reverse().find((r) => r.best_of_round === best);
    const met = review && review.checks ? review.checks.target_met_central : null;
    const metCons = review && review.checks ? review.checks.target_met_conservative : null;
    const verdict = (model.final && model.final.verdict) || (model.useState && model.state.validation[0] && model.state.validation[0].verdict);
    const finished = model.submitted || (model.useState && model.state.submission);
    return h(
      "section.card.hero",
      null,
      h("div.hero-label", null, icon("star"), finished ? "Recommended candidate" : "Best so far", review ? h("span.muted", { style: { textTransform: "none", letterSpacing: 0, fontWeight: 500 } }, `· round ${review.round}`) : null),
      h("div.hero-formula", null, formula(c.composition || best)),
      h(
        "div.hero-sub",
        null,
        h("span.mono.muted", null, best),
        c.space_group ? pill(c.space_group, "outline") : null,
        c.novelty ? pill(label(c.novelty), c.novelty === "novel" ? "info" : null) : null,
        verdict ? verdictPill(verdict) : null,
      ),
      c.rationale || (c.proposal && c.proposal.rationale) ? h("div.hero-why", null, clamp(c.rationale || c.proposal.rationale, 2)) : null,
      h(
        "div.stats",
        null,
        stat(
          label(c.property || spec.target_property || "target"),
          c.value != null ? h("span", null, num(c.value), h("small", null, c.uncertainty != null ? `± ${num(c.uncertainty)} ${c.units || ""}` : c.units || "")) : "–",
          met == null ? h("span.muted", null, spec.reference_material ? h("span", null, "vs ", formula(spec.reference_material)) : "") : h("span", { title: metCons ? "Met even at the conservative end of the error bar" : "Met at the central value" }, check(met, metCons ? "Met, within error" : "Met (central)", "Not met")),
        ),
        stat("Energy above hull", c.e_hull != null ? h("span", null, num(c.e_hull, 3), h("small", null, "eV/atom")) : "–", c.thermo_stable == null ? null : check(c.thermo_stable, `≤ ${thr.e_hull_max_eV_per_atom ?? 0.05}`, `> ${thr.e_hull_max_eV_per_atom ?? 0.05}`)),
        stat("Lowest phonon", c.min_freq != null ? h("span", null, num(c.min_freq), h("small", null, "THz")) : "–", c.dyn_stable == null ? null : check(c.dyn_stable, "Dynamically stable", "Unstable")),
        stat("Final re-check", verdict ? label(verdict) : model.phase.final === "active" ? "Running…" : "–", verdict ? h("span.muted", null, model.config ? model.config.final_potential : "") : null),
      ),
      outcomeLine(),
    );
  }

  function outcomeLine() {
    const st = model.useState ? model.state : null;
    const sub = model.submitted || (st && st.submission);
    if (!sub) return null;
    const rule = (model.runResult && model.runResult.stop_reason) || (st && st.stop_reason) || (model.log.filter((e) => e.type === "submit").pop() || {}).rule;
    const parts = [
      rule ? `Stopped: ${label(rule).toLowerCase()}` : null,
      `submit gate ${sub.gate_passed ? "passed" : sub.accept_not_met ? "not met (accepted)" : "failed"}`,
      `picked by ${sub.picked_by}`,
      sub.runner_up_id ? `runner-up ${sub.runner_up_id}` : null,
    ].filter(Boolean);
    return h("div.hero-foot", null, parts.join(" · "));
  }

  function confidenceCard(rows, threshold, maxRounds) {
    const tableMode = ui.chartTable;
    const toggle = h(
      "div.segmented",
      null,
      h("button", { "aria-pressed": String(!tableMode), onclick: () => ((ui.chartTable = false), (lastTabSig = null), render()) }, "Chart"),
      h("button", { "aria-pressed": String(Boolean(tableMode)), onclick: () => ((ui.chartTable = true), (lastTabSig = null), render()) }, "Table"),
    );
    let bodyEl;
    if (!rows.length) bodyEl = empty("trend", "No review yet", "Each review round adds a point: the code rubric and the reviewer's own confidence.");
    else if (tableMode) {
      bodyEl = h(
        "div.table-wrap",
        null,
        h(
          "table.data",
          null,
          h("thead", null, h("tr", null, h("th", null, "Round"), h("th", null, "Best"), h("th.num", null, "Code rubric"), h("th.num", null, "Reviewer"), h("th", null, "Stop"))),
          h(
            "tbody",
            null,
            rows.map((r) => h("tr", null, h("td", null, r.round), h("td.mono", null, r.best || "–"), h("td.num", null, r.score ?? "–"), h("td.num", null, r.llm ?? "–"), h("td", null, r.stop ? label(r.rule || "yes") : r.gateNote ? "Blocked" : "No"))),
          ),
        ),
      );
    } else bodyEl = h("div", { style: { display: "grid", gap: "12px" } }, confidenceLegend(), confidenceChart(rows, threshold, maxRounds));
    return h(
      "section.card",
      null,
      h("div.card-head", null, h("div", null, h("div.card-title", null, "Confidence by round"), h("div.card-sub", null, `Stops on target when both reach ${threshold}.`)), toggle),
      h("div.card-body", null, bodyEl),
    );
  }

  function budgetCard(cfg) {
    const maxCalls = cfg.max_tool_calls || 90;
    const used = model.toolCount || (model.state && model.state.usage ? 0 : 0);
    const rounds = model.roundRows.length;
    const meter = (lab, v, max, text, tickKind) => {
      const el = h(
        "div.meter",
        tickKind ? { dataset: { tick: tickKind } } : null,
        h("div.meter-top", null, h("span.secondary", null, lab), h("span.num.meter-text", null, text)),
        h("div.meter-track", null, h("div.meter-fill")),
      );
      setMeter(el, v, max, text);
      return el;
    };
    const usage = (model.state && model.state.usage) || {};
    return h(
      "section.card",
      null,
      h("div.card-head", null, h("div.card-title", null, "Budgets")),
      h(
        "div.card-body",
        { style: { display: "grid", gap: "14px" } },
        meter("Tool calls", used, maxCalls, `${used} / ${maxCalls}`),
        meter("Review rounds", rounds, cfg.max_iterations || 3, `${rounds} / ${cfg.max_iterations || 3}`),
        meter("Campaign time", 0, cfg.time_budget_s || 900, "", "campaign"),
        model.finalStart || model.phase.final === "active"
          ? meter("Final re-check time", 0, cfg.final_time_budget_s || 600, "", "final")
          : null,
        usage.input_tokens || usage.output_tokens
          ? h("div.muted", { style: { fontSize: "12px" } }, `${(usage.input_tokens || 0).toLocaleString()} input · ${(usage.output_tokens || 0).toLocaleString()} output tokens`)
          : null,
      ),
    );
  }

  function latestReviewCard() {
    const r = model.reviews[model.reviews.length - 1];
    if (!r) return null;
    const round = model.roundRows.find((x) => x.round === r.round);
    const gate = (round && round.gateNote) || r.gate_note;
    return h(
      "section.card",
      null,
      h("div.card-head", null, h("div.card-title", null, `Latest review · round ${r.round}`), h("a.card-sub", { href: `#/run/${encodeURIComponent(id)}/candidates` }, "All rounds")),
      h(
        "div.card-body",
        { style: { display: "grid", gap: "12px" } },
        gate ? h("div.callout.warning", null, icon("gate"), h("div", null, gate)) : null,
        findingsList((r.findings || []).slice(0, 2), 2),
      ),
    );
  }

  // ---------------------------------------------------------------- activity

  function activityTab() {
    const groups = { all: () => true, reasoning: (e) => e.type === "thinking" || e.type === "text", tools: (e) => e.type === "tool" && e.call.tool !== "request_review", reviews: (e) => ["round", "submit", "final", "brief"].includes(e.type) || (e.type === "tool" && e.call.tool === "request_review"), issues: (e) => (e.type === "tool" && e.call.error) || e.status === "failed" };
    const count = (k) => model.log.filter(groups[k]).length;
    const chips = [
      ["all", "All"],
      ["reasoning", "Reasoning"],
      ["tools", "Tool calls"],
      ["reviews", "Reviews & decisions"],
      ["issues", "Issues"],
    ].map(([k, n]) =>
      h(
        "button.chip",
        {
          "aria-pressed": String(ui.filter === k),
          onclick: () => {
            ui.filter = k;
            render();
          },
        },
        n,
        h("span.count", null, count(k)),
      ),
    );
    const entries = model.log.filter(groups[ui.filter]);
    const list = h("div.log", { role: "log", "aria-live": model.running ? "polite" : "off" });
    let round = null;
    for (const e of entries) {
      const r = e.type === "tool" ? e.call.round : e.round && typeof e.round === "number" ? e.round : e.type === "round" ? e.round.round : null;
      if (r && r !== round && e.type !== "round") {
        round = r;
        list.append(h("div.log-round", null, `Round ${r}`));
      }
      list.append(logEntry(e));
    }
    if (!entries.length) list.append(empty("log", model.events.length ? "Nothing in this filter" : "Waiting for the first event", model.events.length ? "" : "Events stream in as the orchestrator works."));
    if (model.running) list.append(h("div.log-entry", null, h("div.log-icon", null, h("span.spinner")), h("div.log-main.muted", null, model.isReplay ? "Replaying…" : "Working…")));
    return h(
      "div",
      null,
      h(
        "div.log-toolbar",
        null,
        h("div.chips", null, chips),
        h(
          "label.switch",
          { title: "Keep the newest entry in view" },
          h("input", {
            type: "checkbox",
            checked: ui.follow,
            onchange: (e) => {
              ui.follow = e.target.checked;
              render();
            },
          }),
          "Follow",
        ),
      ),
      list,
    );
  }

  function toggleOpen(key) {
    if (ui.open.has(key)) ui.open.delete(key);
    else ui.open.add(key);
    ui.follow = false;
    render();
  }

  function logEntry(e) {
    const key = String(e.seq);
    const open = ui.open.has(key);
    const time = h("span.nowrap.log-time", null, new Date(e.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
    switch (e.type) {
      case "thinking":
      case "text": {
        const thinking = e.type === "thinking";
        const long = e.text.length > 280;
        return h(
          `div.log-entry.reasoning${thinking ? ".is-thinking" : ""}${long ? ".expandable" : ""}${open ? ".open" : ""}`,
          {
            onclick: long ? () => toggleOpen(key) : null,
            role: long ? "button" : null,
            tabindex: long ? 0 : null,
            "aria-expanded": long ? String(open) : null,
            onkeydown: long ? (ev) => (ev.key === "Enter" || ev.key === " ") && (ev.preventDefault(), toggleOpen(key)) : null,
          },
          h("div.log-icon", null, icon(thinking ? "sparkles" : "message")),
          h(
            "div.log-main",
            null,
            h("div.reasoning-kind", null, thinking ? "Thinking" : "Orchestrator"),
            h(`div.${thinking ? "log-thinking" : "log-text"}`, null, long && !open ? e.text.slice(0, 280) + "…" : e.text),
          ),
          h("div.log-side", null, time, long ? h("span.expand-hint", null, open ? "Show less" : "Show more", icon("chevron")) : null),
        );
      }
      case "tool":
        return toolEntry(e, key, open, time);
      case "round": {
        const r = e.round;
        return h(
          "div.log-entry",
          null,
          h("div.log-icon.review", null, icon("gate")),
          h(
            "div.log-main",
            null,
            h("div.log-line", null, h("strong", null, `Round ${r.round} decided`), h("span.log-summary", null, r.best ? `best ${r.best}` : "")),
            h(
              "div.log-result",
              null,
              `Code rubric ${r.score ?? "–"} · reviewer ${r.llm ?? "–"} · `,
              r.stop ? h("strong", null, `stop: ${label(r.rule || "yes")}`) : r.gateNote ? r.gateNote : "continue",
            ),
          ),
          h("div.log-side", null, r.stop ? pill("Stop", "good", "check") : r.gateNote ? pill("Blocked", "warning", "gate") : pill("Continue"), time),
        );
      }
      case "submit": {
        const s = e.submitted;
        return h(
          "div.log-entry",
          null,
          h("div.log-icon.review", null, icon("star")),
          h(
            "div.log-main",
            null,
            h("div.log-line", null, h("strong", null, "Submitted"), s ? h("span.log-summary", null, formula(model.candidate(s.candidate_id).composition || s.candidate_id), ` (${s.candidate_id})`) : h("span.log-summary", null, "nothing")),
            s ? h("div.log-result", null, `Gate ${s.gate_passed ? "passed" : "failed"} · picked by ${s.picked_by}${s.runner_up_id ? ` · runner-up ${s.runner_up_id}` : ""}`) : null,
          ),
          h("div.log-side", null, s ? check(s.gate_passed, "Gate passed", "Gate failed") : null, time),
        );
      }
      case "final": {
        const p = e.payload;
        const text = e.status === "started" ? `Re-checking ${(p.candidates || []).filter(Boolean).join(" and ")} at higher fidelity` : e.status === "finished" ? `Verdict: ${label(p.verdict)}` : `Failed: ${p.error || ""}`;
        return h(
          "div.log-entry",
          null,
          h(`div.log-icon.${e.status === "failed" ? "error" : "sim"}`, null, icon("shield")),
          h("div.log-main", null, h("div.log-line", null, h("strong", null, "Final re-check"), h("span.log-summary", null, text))),
          h("div.log-side", null, e.status === "finished" ? verdictPill(p.verdict) : null, time),
        );
      }
      case "finalstep": {
        const i = e.info || {};
        const dur = e.finished ? (new Date(e.finished) - new Date(e.started)) / 1000 : null;
        return h(
          "div.log-entry",
          null,
          h(`div.log-icon.${e.status === "failed" ? "error" : "sim"}`, null, icon(e.tool === "phonons" ? "wave" : e.tool === "predict_property" ? "gauge" : "atom")),
          h(
            "div.log-main",
            null,
            h("div.log-line", null, h("strong", null, "Final re-check"), h("span.log-summary", null, finalStepName(e), i.potential ? ` · ${i.potential}` : "")),
            e.status === "failed" ? h("div.log-result.error", null, "Failed") : null,
          ),
          h("div.log-side", null, dur != null ? h("span", null, dur < 60 ? `${dur.toFixed(1)} s` : duration(dur)) : h("span.spinner"), time),
        );
      }
      case "brief":
        return h(
          "div.log-entry",
          null,
          h("div.log-icon", null, icon("doc")),
          h("div.log-main", null, h("div.log-line", null, h("strong", null, "Lab brief"), h("span.log-summary", null, e.status === "started" ? "Writing the brief" : e.status === "finished" ? `Ready · ${(e.payload.cited || []).length} papers cited` : "Failed"))),
          h("div.log-side", null, e.status === "finished" ? h("a.btn.sm", { href: `#/run/${encodeURIComponent(id)}/brief` }, "Open") : null, time),
        );
      case "parser": {
        const p = e.payload;
        const text = e.status === "started" ? "Parsing the goal" : e.status === "finished" ? "Specification ready" : (p.questions || []).length ? `Asked: ${p.questions.join(" ")}` : `Failed: ${(p.errors || []).join(" ")}`;
        return h(
          "div.log-entry",
          null,
          h(`div.log-icon${e.status === "failed" ? ".error" : ""}`, null, icon("sliders")),
          h("div.log-main", null, h("div.log-line", null, h("strong", null, "Goal parser"), h("span.log-summary", null, text))),
          h("div.log-side", null, time),
        );
      }
      case "marker":
        return h(
          "div.log-entry",
          null,
          h(`div.log-icon${e.status === "failed" ? ".error" : ""}`, null, icon(e.text.includes("finished") ? "check" : e.text.includes("failed") ? "x" : "play")),
          h(
            "div.log-main",
            null,
            h("strong", null, e.text),
            e.detail && e.detail.stop_reason ? h("span.muted", null, ` · ${label(e.detail.stop_reason)}`) : null,
            e.detail && Array.isArray(e.detail.errors)
              ? e.detail.errors.map((x) => h("div.log-result.error", null, `${label(x.step)}: ${x.message}`))
              : null,
          ),
          h("div.log-side", null, time),
        );
      default:
        return h("div");
    }
  }

  function toolEntry(e, key, open, time) {
    const c = e.call;
    const kind = c.error ? "error" : { lit: "lit", db: "lit", sim: "sim", review: "review" }[toolKind(c.tool)] || "";
    const ic = { lit: "book", db: "database", sim: c.tool === "phonons" ? "wave" : c.tool === "predict_property" ? "gauge" : c.tool === "build_structure" ? "cube" : "atom", review: "gate", candidate: "flask", control: c.tool === "submit_final" ? "star" : "info" }[toolKind(c.tool)];
    const dur = c.finished ? (new Date(c.finished) - new Date(c.started)) / 1000 : null;
    const statusEl = c.status === "started" ? h("span.spinner") : c.error ? h("span.check.bad", null, icon("x")) : null;
    const res = resultLine(c);
    const detail = open
      ? h(
          "div.log-detail",
          { onclick: (ev) => ev.stopPropagation() },
          c.review ? reviewBlock(c.review) : null,
          h("div", null, h("div.log-detail-label", null, "Input"), h("pre", null, JSON.stringify(c.input, null, 2))),
          c.result !== undefined && !c.review ? h("div", null, h("div.log-detail-label", null, "Result (shortened for the stream)"), h("pre", null, JSON.stringify(c.result, null, 2))) : null,
        )
      : null;
    return h(
      `div.log-entry.expandable${open ? ".open" : ""}`,
      { onclick: () => toggleOpen(key), title: "Show input and result" },
      h(`div.log-icon.${kind || "x"}`, null, icon(c.error ? "alert" : ic)),
      h(
        "div.log-main",
        null,
        h("div.log-line", null, h("span.log-tool", null, c.tool), h("span.log-summary", null, summary(c))),
        res ? h(`div.log-result${c.error ? ".error" : ""}`, null, res) : null,
      ),
      h("div.log-side", null, dur != null && dur >= 0.05 ? h("span", null, dur < 60 ? `${dur.toFixed(1)} s` : duration(dur)) : null, statusEl, time),
      detail,
    );
  }

  // ---------------------------------------------------------------- literature

  function literatureTab() {
    const lit = model.useState ? model.state.literature : null;
    const papers = model.paperList.slice().sort((a, b) => (b.relevance || 0) - (a.relevance || 0) || (b.citation_count || 0) - (a.citation_count || 0));
    const insights = lit ? lit.insights : [...model.calls.values()].filter((c) => c.tool === "note_evidence" && !c.error).map((c) => ({ ...c.input, value: c.input.value != null ? { value: c.input.value, units: c.input.units } : null }));
    const cited = new Set((model.useState && model.state.brief && model.state.brief.cited_paper_ids) || (model.brief && model.brief.cited) || []);
    const q = ui.paperQuery.toLowerCase();
    const shown = q ? papers.filter((p) => `${p.title} ${p.abstract || ""} ${(p.authors || []).join(" ")}`.toLowerCase().includes(q)) : papers;
    const trend = lit && Object.keys(lit.field_trend || {}).length ? lit.field_trend : trendFromCalls();

    const search = h("input.input", {
      type: "search",
      placeholder: "Filter papers",
      value: ui.paperQuery,
      style: { maxWidth: "260px" },
      oninput: (e) => {
        ui.paperQuery = e.target.value;
        const pos = e.target.selectionStart;
        render();
        const again = body.querySelector("input[type=search]");
        if (again) {
          again.focus();
          again.setSelectionRange(pos, pos);
        }
      },
    });

    const paperCards = shown.length
      ? shown.map((p) =>
          h(
            "article.card.paper",
            null,
            h(
              "div",
              null,
              h("div.paper-title", null, p.url || p.doi ? h("a", { href: p.url || `https://doi.org/${p.doi}`, target: "_blank", rel: "noopener" }, p.title) : p.title),
              h(
                "div.paper-meta",
                { style: { marginTop: "4px" } },
                p.authors && p.authors.length ? h("span", null, p.authors.slice(0, 3).join(", ") + (p.authors.length > 3 ? " et al." : "")) : null,
                p.year ? h("span.num", null, p.year) : null,
                p.citation_count != null ? h("span.num", null, `${p.citation_count} citations`) : null,
                p.found_via ? pill(label(p.found_via), "outline") : null,
                cited.has(p.paper_id) ? pill("Cited in brief", "good", "doc") : null,
                h("span.mono", null, p.paper_id),
              ),
            ),
            p.relevance ? h("div.relevance", null, `Relevance ${Math.round(p.relevance * 100)}%`, h("div.meter-track", null, h("div.meter-fill", { style: { width: `${p.relevance * 100}%` } }))) : h("div"),
            p.abstract ? abstractBlock(p) : null,
          ),
        )
      : [empty("book", papers.length ? "No paper matches" : "No papers yet", papers.length ? "" : "Papers appear as the orchestrator searches Valency Bond.")];

    const listCard = (title, items, ic, kind) =>
      items && items.length
        ? h(
            "section.card",
            null,
            h("div.card-head", null, h("div.card-title", null, title), h("span.card-sub.num", null, items.length)),
            h("div.card-body", null, h("ul.list-plain", null, items.map((t) => h("li", null, h(`span.check.${kind || "unknown"}`, null, icon(ic)), h("span", null, t))))),
          )
        : null;

    return h(
      "div.grid",
      { style: { gridTemplateColumns: "minmax(0, 1.6fr) minmax(0, 1fr)", alignItems: "start" } },
      h(
        "div.grid",
        null,
        h(
          "div.stats",
          { style: { marginTop: 0 } },
          stat("Papers", papers.length),
          stat("Evidence notes", insights.length),
          stat("Failed approaches", lit ? lit.failed_approaches.length : "–"),
          stat("Cited in brief", cited.size || "–"),
        ),
        h("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center" } }, h("h3", { style: { fontSize: "14px" } }, "Papers"), search),
        h("div.papers", null, paperCards),
      ),
      h(
        "div.grid",
        null,
        trend && Object.keys(trend).length
          ? h(
              "section.card",
              null,
              h("div.card-head", null, h("div", null, h("div.card-title", null, "Publications per year"), h("div.card-sub", null, "Valency Bond, for the campaign's topic"))),
              h("div.card-body", null, columnChart(Object.entries(trend).sort((a, b) => a[0] - b[0]), "papers")),
            )
          : null,
        insights.length
          ? h(
              "section.card",
              null,
              h("div.card-head", null, h("div.card-title", null, "Evidence notes"), h("span.card-sub.num", null, insights.length)),
              h(
                "div.card-body",
                { style: { display: "grid", gap: "12px" } },
                insights.map((n) =>
                  h(
                    "div",
                    { style: { display: "grid", gap: "3px" } },
                    h("div", { style: { display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" } }, pill(label(n.finding_type || n.property), n.finding_type === "negative" ? "critical" : n.finding_type === "positive" ? "good" : null), n.composition ? h("strong", null, formula(n.composition)) : null, n.value ? h("span.num.secondary", null, pm(n.value.value, n.value.uncertainty, n.value.units)) : null),
                    h("div", null, n.summary),
                    h("div.muted.mono", { style: { fontSize: "11.5px" } }, n.paper_id, n.conditions ? ` · ${n.conditions}` : ""),
                  ),
                ),
              ),
            )
          : null,
        lit ? listCard("Failed approaches", lit.failed_approaches, "x", "bad") : null,
        lit ? listCard("Mechanisms", lit.mechanisms, "atom") : null,
        lit ? listCard("Synthesis routes", lit.synthesis_routes, "flask", "good") : null,
        lit ? listCard("Open questions", lit.open_questions, "message") : null,
        lit && lit.coverage_note ? h("div.callout.info", null, icon("info"), h("div", null, lit.coverage_note)) : null,
      ),
    );
  }

  function abstractBlock(p) {
    const open = ui.openAbstracts.has(p.paper_id);
    const text = h(`p.paper-abstract${open ? "" : ".clamp"}`, null, p.abstract);
    const btn = h(
      "button.link.abstract-toggle",
      {
        hidden: true,
        onclick: () => {
          const now = !ui.openAbstracts.has(p.paper_id);
          if (now) ui.openAbstracts.add(p.paper_id);
          else ui.openAbstracts.delete(p.paper_id);
          text.classList.toggle("clamp", !now);
          btn.textContent = now ? "Show less" : "Show more";
        },
      },
      open ? "Show less" : "Show more",
    );
    return h("div.paper-abstract-wrap", null, text, btn);
  }

  // Show "Show more" only where the abstract is longer than its three lines.
  function markLongAbstracts() {
    requestAnimationFrame(() => {
      for (const wrap of body.querySelectorAll(".paper-abstract-wrap")) {
        const text = wrap.querySelector(".paper-abstract");
        const btn = wrap.querySelector(".abstract-toggle");
        const long = !text.classList.contains("clamp") || text.scrollHeight > text.clientHeight + 1;
        btn.hidden = !long;
      }
    });
  }

  function trendFromCalls() {
    const call = [...model.calls.values()].reverse().find((c) => c.tool === "publication_trend" && c.result && c.result.papers_per_year);
    return call ? call.result.papers_per_year : null;
  }

  // ---------------------------------------------------------------- candidates

  function candidatesTab() {
    const ids = model.candidateIds;
    if (!ids.length) return empty("flask", "No candidates yet", "Candidates appear when the orchestrator proposes them.");
    const best = model.best;
    const spec = model.spec || {};
    const byRound = new Map();
    for (const cid of ids) {
      const c = model.candidate(cid);
      const r = c.cand_round || Number((cid.match(/^c(\d+)-/) || [])[1]) || 1;
      if (!byRound.has(r)) byRound.set(r, []);
      byRound.get(r).push(c);
    }
    const reviews = model.reviews;
    const rounds = model.roundRows;
    return h(
      "div",
      null,
      [...byRound.entries()].map(([r, cands]) => {
        const review = reviews.find((x) => x.round === r);
        const row = rounds.find((x) => x.round === r);
        const collapsed = ui.collapsedRounds.has(r);
        const content = h("div.round-content", { hidden: collapsed });
        const chevron = icon("chevron", "round-chevron");
        const toggle = () => {
          const now = !ui.collapsedRounds.has(r);
          if (now) ui.collapsedRounds.add(r);
          else ui.collapsedRounds.delete(r);
          content.hidden = now;
          headBtn.setAttribute("aria-expanded", String(!now));
        };
        const headBtn = h(
          "button.round-head",
          { "aria-expanded": String(!collapsed), onclick: toggle, title: "Collapse or expand this round" },
          h("span.round-title", null, chevron, h("h3", null, `Round ${r}`)),
          h(
            "span.muted",
            null,
            `${cands.length} candidate${cands.length === 1 ? "" : "s"}`,
            row && row.best ? ` · best ${row.best}` : "",
            row && row.score != null ? ` · rubric ${row.score}` : "",
            row && row.llm != null ? ` · reviewer ${row.llm}` : "",
          ),
        );
        content.append(
          h(
            "div.card",
            null,
            h(
              "div.table-wrap",
              null,
              h(
                "table.data",
                null,
                h(
                  "thead",
                  null,
                  h(
                    "tr",
                    null,
                    h("th", null, "Candidate"),
                    h("th", null, "Structure"),
                    h("th.num", null, label(spec.target_property || "Target")),
                    h("th.num", null, "E above hull"),
                    h("th", null, "Thermo"),
                    h("th.num", null, "Min phonon"),
                    h("th", null, "Dynamic"),
                    h("th", null, "Novelty"),
                    h("th", null, ""),
                  ),
                ),
                h(
                  "tbody",
                  null,
                  cands.map((c) =>
                    h(
                      `tr.clickable${c.candidate_id === best ? ".best" : ""}`,
                      { onclick: () => selectCandidate(c.candidate_id), title: c.rationale || "" },
                      h(
                        "td",
                        null,
                        h("div", { style: { display: "flex", alignItems: "center", gap: "6px" } }, c.candidate_id === best ? h("span.star", { title: "Best candidate" }, icon("star")) : null, h("span.cand-formula", null, formula(c.composition || c.candidate_id))),
                        h("div.cand-id", null, c.candidate_id),
                      ),
                      h("td", null, c.space_group || (c.proposal && label(c.proposal.structure_type)) || "–"),
                      h("td.num", null, c.value != null ? pm(c.value, c.uncertainty, c.units) : "–"),
                      h("td.num", null, c.e_hull != null ? `${num(c.e_hull, 3)} eV/atom` : "–"),
                      h("td", null, c.e_hull == null ? h("span.muted", null, "–") : check(c.thermo_stable, "Stable", "Off hull")),
                      h("td.num", null, c.min_freq != null ? `${num(c.min_freq)} THz` : "–"),
                      h("td", null, c.dyn_stable == null ? h("span.muted", null, "–") : check(c.dyn_stable, c.verdict ? label(c.verdict) : "Stable", c.verdict ? label(c.verdict) : "Unstable")),
                      h("td", null, c.novelty ? pill(label(c.novelty), c.novelty === "novel" ? "info" : null) : "–"),
                      h("td", null, h("span.muted", { title: "View structure" }, icon("cube"))),
                    ),
                  ),
                ),
              ),
            ),
          ),
        );
        if (review) content.append(h("section.card", { style: { marginTop: "12px" } }, h("div.card-body", null, reviewBlock(review, row && row.gateNote))));
        return h("section.round-block", null, headBtn, content);
      }),
    );
  }

  // ---------------------------------------------------------------- structure

  function structureTab() {
    const ids = model.candidateIds;
    if (!ids.length) return empty("cube", "No structures yet", "Structures appear once candidates are built.");
    if (!ui.selected || !ids.includes(ui.selected)) ui.selected = model.best && ids.includes(model.best) ? model.best : ids[ids.length - 1];
    const cid = ui.selected;
    const c = model.candidate(cid);
    const picker = h(
      "div.card.cand-picker",
      null,
      ids.map((x) => {
        const cc = model.candidate(x);
        return h(
          "button.cand-pick",
          {
            "aria-pressed": String(x === cid),
            onclick: () => {
              ui.selected = x;
              render();
            },
          },
          h("span", { style: { display: "flex", alignItems: "center", gap: "6px" } }, x === model.best ? h("span.star", null, icon("star")) : null, h("span.cand-formula", null, formula(cc.composition || x))),
          h("span.cand-id", null, x),
        );
      }),
    );
    const canvas = h("canvas", { "aria-label": `3D structure of ${c.composition || cid}`, role: "img" });
    const legend = h("div.viewer-legend");
    const status = h("div.empty", { style: { position: "absolute", inset: 0 } }, h("span.spinner"), "Loading structure…");
    const tools = h(
      "div.viewer-tools",
      null,
      h("button.btn.sm", { title: "Show or hide bonds", onclick: () => viewer && ((viewer.showBonds = !viewer.showBonds), viewer.draw()) }, icon("link"), "Bonds"),
      h("button.btn.sm", { title: "Show or hide the unit cell", onclick: () => viewer && ((viewer.showCell = !viewer.showCell), viewer.draw()) }, icon("box"), "Cell"),
      h("button.btn.sm.icon", { title: "Reset view", onclick: () => viewer && viewer.reset() }, icon("reset")),
    );
    const stage = h("div.card.viewer", null, canvas, legend, tools, status, h("div.viewer-hint", null, "Drag to rotate · scroll to zoom · double-click to reset"));
    const info = h("div.card", null, h("div.card-head", null, h("div.card-title", null, "Structure")), h("div.card-body", null, structureFacts(c)));

    if (viewer) viewer.destroy();
    viewer = new CrystalViewer(canvas);
    const v = viewer;
    api(`/runs/${encodeURIComponent(id)}/structure/${encodeURIComponent(cid)}`, { text: true })
      .then((cif) => {
        if (v !== viewer) return;
        const crystal = parseCif(cif);
        if (!crystal) throw new Error("unreadable");
        status.remove();
        v.load(crystal);
        v.autoSpin(true);
        const counts = {};
        for (const s0 of crystal.sites) counts[s0.el] = (counts[s0.el] || 0) + 1;
        legend.replaceChildren(...Object.entries(counts).map(([el, n]) => h("span.el-chip", null, h("span.el-ball", { style: { background: elementStyle(el)[0] } }), el, h("span.muted.num", null, n))));
        info.querySelector(".card-head").append(h("button.btn.sm", { onclick: () => download(`${cid}.cif`, cif, "chemical/x-cif") }, icon("download"), "CIF"));
      })
      .catch(() => {
        if (v !== viewer) return;
        status.replaceChildren(icon("cube"), h("div.empty-title", null, "No structure to show"), h("div", null, "This candidate has no stored atomic structure yet."));
      });
    return h("div.viewer-layout", null, picker, stage, info);
  }

  function structureFacts(c) {
    const sim = c.sim || {};
    const lat = sim.lattice;
    const rec = (sim.relaxed && sim.relaxed.n_atoms ? sim.relaxed : c.structure) || {};
    const rows = [
      ["Composition", formula(c.composition || c.candidate_id)],
      ["Space group", c.space_group || "–"],
      ["Prototype", c.proposal ? label(c.proposal.structure_type) : "–"],
      ["Source", rec.source || "–"],
      ["Atoms", rec.n_atoms ?? "–"],
      ["Supercell", rec.supercell ? rec.supercell.join(" × ") : "–"],
      ["Lattice (Å)", lat ? `${num(lat.a, 3)}, ${num(lat.b, 3)}, ${num(lat.c, 3)}` : "–"],
      ["Angles (°)", lat ? `${num(lat.alpha, 1)}, ${num(lat.beta, 1)}, ${num(lat.gamma, 1)}` : "–"],
      ["Density", sim.density_g_cm3 ? `${num(sim.density_g_cm3, 3)} g/cm³` : "–"],
      ["Potential", sim.potential || "–"],
      ["Energy", sim.energy_eV_per_atom != null ? `${num(sim.energy_eV_per_atom, 3)} eV/atom` : "–"],
      ["Relaxation", sim.potential ? (sim.converged ? "Converged" : "Not converged") : "–"],
    ];
    return h(
      "div",
      { style: { display: "grid", gap: "14px" } },
      h("dl.kv", null, rows.map(([k, v]) => [h("dt", null, k), h("dd", null, v)])),
      sim.domain_note ? h("div.callout.info", null, icon("info"), h("div", null, sim.domain_note)) : null,
    );
  }

  // ---------------------------------------------------------------- validation

  function validationTab() {
    const st = model.useState ? model.state : null;
    const reports = st && st.validation.length ? st.validation : model.final && model.final.verdict ? [{ candidate_id: model.submitted ? model.submitted.candidate_id : model.best, ...model.final }] : [];
    const sub = model.submitted || (st && st.submission);
    const errors = (st && st.errors) || [];
    const parts = [];
    if (!reports.length) {
      parts.push(
        h(
          "section.card",
          null,
          model.phase.final === "active"
            ? empty("shield", "Re-checking…", "The finalists are re-relaxed with the larger potential, with full phonons and a fresh hull.")
            : model.final && model.final.error
              ? h("div.card-body", null, h("div.callout.critical", null, icon("alert"), h("div", null, `The final re-check failed: ${model.final.error}. The run continued and the brief says so.`)))
              : empty("shield", "No final re-check yet", "After submission the winner and runner-up are re-checked at higher fidelity."),
        ),
      );
    }
    for (const r of reports) {
      const kind = { confirmed: "good", weakened: "warning", overturned: "critical" }[r.verdict] || "warning";
      const text = {
        confirmed: "The higher-fidelity re-check agrees with screening.",
        weakened: "The re-check moved some results the wrong way; read the changes below.",
        overturned: "The re-check contradicts screening. Do not take this candidate to the lab as is.",
      }[r.verdict];
      parts.push(
        h(
          "section.card",
          null,
          h(
            `div.verdict.${kind}`,
            null,
            h("div.verdict-icon", null, icon(kind === "good" ? "check" : kind === "critical" ? "x" : "alert")),
            h("div", null, h("div.verdict-title", null, `${label(r.verdict)}: `, formula(model.candidate(r.candidate_id).composition || r.candidate_id), h("span.muted.mono", { style: { fontSize: "13px", marginLeft: "8px" } }, r.candidate_id)), h("div.secondary", null, text)),
          ),
          r.checks && r.checks.length
            ? h(
                "div.table-wrap",
                { style: { borderTop: "1px solid var(--hairline)" } },
                h(
                  "table.data",
                  null,
                  h("thead", null, h("tr", null, h("th", null, "Check"), h("th.num", null, "Screening"), h("th.num", null, "Final"), h("th.num", null, "Change"))),
                  h(
                    "tbody",
                    null,
                    r.checks.map((ch) =>
                      h(
                        "tr",
                        null,
                        h("td", null, checkName(ch.name)),
                        h("td.num", null, fmtCheck(ch.screening_value)),
                        h("td.num", null, fmtCheck(ch.final_value)),
                        h("td.num", null, ch.delta != null ? `${ch.delta > 0 ? "+" : ""}${num(ch.delta, Math.abs(ch.delta) >= 10 ? 1 : 3)}` : "–"),
                      ),
                    ),
                  ),
                ),
              )
            : null,
          r.notes && r.notes.length ? h("div.card-body", null, h("ul.list-plain", null, r.notes.map((n) => h("li", null, icon("info"), h("span.secondary", null, n))))) : null,
        ),
      );
    }
    const side = [];
    if (sub) {
      side.push(
        h(
          "section.card",
          null,
          h("div.card-head", null, h("div.card-title", null, "Submission")),
          h(
            "div.card-body",
            null,
            h(
              "dl.kv",
              null,
              [
                ["Winner", h("span", null, formula(model.candidate(sub.candidate_id).composition || ""), " ", h("span.mono.muted", null, sub.candidate_id))],
                ["Runner-up", sub.runner_up_id ? h("span", null, formula(model.candidate(sub.runner_up_id).composition || ""), " ", h("span.mono.muted", null, sub.runner_up_id)) : "–"],
                ["Submit gate", check(sub.gate_passed, "Passed", "Failed")],
                sub.accept_not_met ? ["Accepted not met", "Yes"] : null,
                ["Picked by", label(sub.picked_by)],
                ["Justification", sub.justification ? clamp(sub.justification, 5) : "–"],
              ]
                .filter(Boolean)
                .map(([k, v]) => [h("dt", null, k), h("dd", null, v)]),
            ),
          ),
        ),
      );
    }
    side.push(
      errors.length
        ? h(
            "details.card.issues",
            null,
            h("summary.card-head", null, h("div.card-title", null, icon("chevron", "round-chevron"), `${errors.length} recorded issue${errors.length === 1 ? "" : "s"}`), h("span.card-sub", null, "Degraded steps; the run continued")),
            h(
              "div.card-body",
              null,
              h("ul.list-plain", null, errors.map((e) => h("li", null, h("span.check.bad", null, icon("alert")), h("span", null, h("strong", null, `${label(e.step)}: `), h("span.secondary", null, e.message))))),
            ),
          )
        : null,
    );
    return h("div.grid", { style: { gridTemplateColumns: "minmax(0, 1.6fr) minmax(0, 1fr)", alignItems: "start" } }, h("div.grid", null, parts), h("div.grid", null, side));
  }

  // ---------------------------------------------------------------- brief

  function briefTab() {
    const ready = model.phase.brief === "done" || (model.useState && model.state.brief);
    if (!ready) {
      return h("section.card", null, model.phase.brief === "active" ? empty("doc", "Writing the brief…", "Numbers come only from stored results; a number check blocks anything else.") : empty("doc", "No brief yet", "The lab brief is written after the final re-check."));
    }
    if (ui.brief == null) {
      if (!ui.briefLoading) {
        ui.briefLoading = true;
        api(`/runs/${encodeURIComponent(model.isReplay ? sourceId() : id)}/brief`, { text: true })
          .then((md) => (ui.brief = md))
          .catch(() => (ui.brief = (model.state && model.state.brief && model.state.brief.markdown) || ""))
          .finally(() => {
            ui.briefLoading = false;
            schedule();
          });
      }
      return h("section.card", null, empty("doc", "Loading the brief…"));
    }
    const { html, headings } = renderMarkdown(ui.brief);
    const doc = h("article.card.doc");
    doc.innerHTML = html; // escaped by renderMarkdown
    const pdf = model.state && model.state.brief && model.state.brief.pdf_path;
    const rid = model.isReplay ? sourceId() : id;
    const cited = (model.state && model.state.brief && model.state.brief.cited_paper_ids) || [];
    return h(
      "div.brief-layout",
      null,
      doc,
      h(
        "aside.toc",
        null,
        h(
          "div",
          { style: { display: "grid", gap: "8px" } },
          h("button.btn", { onclick: () => download(`${rid}_brief.md`, ui.brief, "text/markdown") }, icon("download"), "Markdown"),
          pdf ? h("a.btn", { href: `/runs/${encodeURIComponent(rid)}/brief?format=pdf`, download: `${rid}_brief.pdf` }, icon("download"), "PDF") : null,
          h("button.btn", { onclick: () => window.print() }, icon("printer"), "Print"),
          h(
            "button.btn",
            {
              onclick: async () => {
                try {
                  await navigator.clipboard.writeText(ui.brief);
                  toast("Brief copied as Markdown.");
                } catch (e) {
                  toast("Copy failed: your browser blocked clipboard access.", "critical");
                }
              },
            },
            icon("copy"),
            "Copy",
          ),
        ),
        headings.filter((x) => x.level === 2).length
          ? h(
              "div",
              null,
              h("div.field-label", { style: { marginBottom: "6px" } }, "Sections"),
              h(
                "nav.toc-links",
                null,
                headings
                  .filter((x) => x.level === 2)
                  .map((x) =>
                    h(
                      "a",
                      {
                        href: `#${x.id}`,
                        onclick: (e) => {
                          e.preventDefault();
                          const t = doc.querySelector(`#${CSS.escape(x.id)}`);
                          if (t) root.scrollTo({ top: t.offsetTop - 210, behavior: "smooth" });
                        },
                      },
                      x.text,
                    ),
                  ),
              ),
            )
          : null,
        cited.length ? h("div.muted", { style: { fontSize: "12px" } }, `${cited.length} paper${cited.length === 1 ? "" : "s"} cited`) : null,
      ),
    );
  }

  // ---------------------------------------------------------------- spec

  function specTab() {
    const cfg = model.config || (model.state && model.state.config);
    const spec = model.spec;
    const groups = cfg
      ? [
          ["Model access", ["llm_backend", "llm_model", "require_novel"]],
          ...CONFIG_GROUPS.map((g) => [g.name, Object.keys(g.fields)]),
        ]
      : [];
    return h(
      "div.grid",
      { style: { gridTemplateColumns: "minmax(0, 1.5fr) minmax(0, 1fr)", alignItems: "start" } },
      h(
        "section.card",
        { style: { overflow: "hidden" } },
        h("div.card-head", { style: { paddingBottom: "14px" } }, h("div", null, h("div.card-title", null, "Search specification"), h("div.card-sub", null, "What the goal parser produced, after the code expanded it"))),
        spec ? specGrid(spec, cfg) : h("div.card-body", null, empty("sliders", "No specification yet")),
        spec && spec.reference_value
          ? h("div.card-body", null, h("div.callout.info", null, icon("info"), h("div", null, "Reference value ", h("strong", null, pm(spec.reference_value.value, spec.reference_value.uncertainty, spec.reference_value.units)), ` from ${spec.reference_value.source}: computed with the same model as every candidate, not quoted.`)))
          : null,
      ),
      h(
        "section.card",
        null,
        h("div.card-head", null, h("div.card-title", null, "Run configuration")),
        h(
          "div.card-body",
          { style: { display: "grid", gap: "18px" } },
          groups.map(([title, keys]) =>
            h(
              "div",
              null,
              h("div.field-label", { style: { marginBottom: "8px" } }, title),
              h("dl.kv", null, keys.filter((k) => k in cfg).map((k) => [h("dt", null, k === "llm_backend" ? "Model access" : k === "llm_model" ? "Model" : k === "require_novel" ? "Novel materials only" : configName(k)), h("dd.mono", null, fmtConfig(k, cfg[k]))])),
            ),
          ),
        ),
      ),
    );
  }

  // ------------------------------------------------------------------ shared pieces

  function reviewBlock(r, gateNote) {
    const breakdown = Object.entries(r.score_breakdown || {});
    const checks = r.checks || {};
    const scores = h(
      "div.review-scores",
      null,
      h(
        "div.score-pair",
        null,
        h("div.score-item", null, scoreRing(r.score, "var(--series-1)"), h("div", null, h("div.score-name", null, "Code rubric"), h("div.muted.score-sub", null, `best ${r.best_of_round || "–"}`))),
        h("div.score-item", null, scoreRing(r.llm_confidence, "var(--series-2)"), h("div", null, h("div.score-name", null, "Reviewer"), h("div.muted.score-sub", null, "own confidence"))),
      ),
      h(
        "div.check-list",
        null,
        check(checks.target_met_central, "Target met", "Target not met"),
        checks.target_met_conservative != null ? check(checks.target_met_conservative, "Within error bar", "Not within error bar") : null,
        check(checks.thermo_stable, "On the hull", "Off hull"),
        check(checks.dyn_stable, "Dynamically stable", "Dynamically unstable"),
      ),
      breakdown.length
        ? h(
            "div.breakdown",
            null,
            breakdown.map(([k, v]) => {
              const max = SCORE_MAX[k] || 100;
              return h("div.breakdown-row", null, h("span.secondary", null, label(k)), h("div.meter-track", null, h("div.meter-fill", { style: { width: `${Math.min(100, (v / max) * 100)}%`, background: "var(--series-1)" } })), h("span.num.muted", { style: { textAlign: "right" } }, `${v}/${max}`));
            }),
          )
        : null,
    );
    const words = h(
      "div.review-words",
      null,
      gateNote || r.gate_note ? h("div.callout.warning", null, icon("gate"), h("div", null, gateNote || r.gate_note)) : null,
      r.llm_confidence_reason ? h("div", null, h("div.field-label", null, "Reviewer's view"), clamp(r.llm_confidence_reason, 3)) : null,
      (r.findings || []).length ? h("div", null, h("div.field-label", null, "Findings"), findingsList(r.findings, 3)) : null,
      r.recommended_next ? h("div.secondary", { style: { fontSize: "13px" } }, "Recommended next: ", h("strong", null, formula(r.recommended_next)), r.why ? h("span.muted", null, ` — ${r.why}`) : null) : null,
    );
    return h("div.review", null, scores, words);
  }

  function findingsList(findings, lines = 3) {
    if (!findings.length) return null;
    return h(
      "ul.findings",
      null,
      findings.map((f) =>
        h(
          "li.finding",
          null,
          pill(label(f.severity), SEVERITY[f.severity], f.severity === "critical" ? "alert" : null),
          clamp([h("span.muted", null, `${label(f.category)} · `), f.text, f.suggestion ? h("div.suggest", null, f.suggestion) : null], lines),
        ),
      ),
    );
  }

  // ------------------------------------------------------------------ lifecycle

  render();
  return {
    unmount() {
      alive = false;
      clearInterval(ticker);
      clearTimeout(stateTimer);
      window.removeEventListener("themechange", onTheme);
      if (closeStream) closeStream();
      if (viewer) viewer.destroy();
    },
    setTab(t) {
      ui.tab = TABS.some((x) => x.key === t) ? t : "overview";
      root.scrollTop = 0;
      lastTabSig = null;
      render();
    },
  };
}

// ------------------------------------------------------------------ helpers

/**
 * One-line summary of a tool call's input.
 * @param {object} c Call.
 * @returns {string}
 */
function summary(c) {
  const i = c.input || {};
  const p = c.payload || {};
  const comp = p.composition || i.composition;
  switch (c.tool) {
    case "search_literature":
      return `“${i.query}”${i.mode ? ` · ${i.mode}` : ""}`;
    case "related_papers":
      return `${i.paper_id || ""}${i.mode ? ` · ${i.mode}` : ""}`;
    case "get_paper":
      return i.paper_id || "";
    case "publication_trend":
      return `“${i.query}”`;
    case "note_evidence":
      return i.summary || "";
    case "lookup_materials_project":
      return i.composition || i.mp_id || "";
    case "propose_candidate":
      return `${i.composition} · ${label(i.structure_type || "")}`;
    case "request_review":
      return i.summary || "";
    case "submit_final":
      return `${i.candidate_id}${i.runner_up_id ? ` (runner-up ${i.runner_up_id})` : ""}`;
    case "predict_property":
      return `${i.candidate_id}${comp ? ` ${comp}` : ""} · ${label(i.property)}`;
    default:
      return i.candidate_id ? `${i.candidate_id}${comp ? ` ${comp}` : ""}` : Object.values(i).slice(0, 2).join(" · ");
  }
}

/**
 * One-line summary of a tool call's result.
 * @param {object} c Call.
 * @returns {string|null}
 */
function resultLine(c) {
  if (c.status === "started") return null;
  if (c.error) return String(c.error);
  const r = c.result || {};
  switch (c.tool) {
    case "search_literature":
    case "related_papers":
      return `${r.n ?? (r.papers || []).length} papers${(r.papers || [])[0] ? ` · top: ${r.papers[0].title}` : ""}`;
    case "get_paper":
      return r.title || null;
    case "publication_trend":
      return r.papers_per_year ? `${Object.keys(r.papers_per_year).length} years` : null;
    case "note_evidence":
      return r.recorded ? `Recorded${r.value_dropped_not_in_abstract ? " · value dropped (not in the abstract)" : ""}` : "Not recorded";
    case "lookup_materials_project":
      return `${r.n_entries ?? 0} entries${r.default_polymorph ? ` · default ${r.default_polymorph}` : ""}${r.note ? ` · ${r.note}` : ""}`;
    case "propose_candidate":
      return r.accepted ? `Accepted as ${r.candidate_id}${r.tolerance_factor ? ` · tolerance factor ${num(r.tolerance_factor)}` : ""}` : `Rejected: ${(r.reasons || []).join("; ")}`;
    case "build_structure":
      return `${r.n_atoms} atoms from ${r.source}${r.supercell ? ` · ${r.supercell.join("×")} supercell` : ""}`;
    case "relax":
      return `${r.converged ? "Converged" : "Not converged"} · ${num(r.energy_eV_per_atom, 3)} eV/atom · ${r.space_group || "?"}`;
    case "energy_above_hull":
      return `${num(r.e_hull_eV_per_atom, 3)} eV/atom above hull · ${r.thermo_stable ? "stable" : "unstable"} (≤ ${r.threshold})`;
    case "phonons":
      return `Min ${num(r.min_frequency_THz)} THz · ${label(r.verdict || (r.dyn_stable ? "stable" : "unstable"))}${(r.mode_mapping_rounds || []).length ? ` · ${r.mode_mapping_rounds.length} mode-mapping rounds` : ""}`;
    case "run_md":
      return `${r.stable ? "Stable" : "Unstable"} at ${r.temperature_K} K · max displacement ${num(r.max_displacement_A)} Å`;
    case "predict_property":
      return `${pm(r.value, r.uncertainty, r.units)} · ${r.model}${r.in_domain === false ? " · out of domain" : ""}`;
    case "candidate_status":
      return r.candidates ? `${r.candidates.length} candidates · ${r.budget ? r.budget.tool_calls_used : "?"} calls used` : null;
    case "request_review": {
      const v = c.review || {};
      return `Rubric ${v.score ?? "–"} · reviewer ${v.llm_confidence ?? "–"}${v.gate_note ? " · stop blocked" : v.stop_rule ? ` · ${label(v.stop_rule)}` : ""}`;
    }
    case "submit_final":
      return r.accepted ? `Accepted · gate ${r.gate_passed ? "passed" : "not met"}` : `Refused: ${(r.reasons || [r.note]).join("; ")}`;
    default:
      return null;
  }
}

const FINAL_STEP_NAMES = {
  relax: "Re-relax",
  phonons: "Full phonons",
  energy_above_hull: "Hull",
  predict_property: "Target property",
  build_structure: "Build structure",
  run_md: "MD",
};

/**
 * Name of a final-check step, e.g. "Full phonons · c3-1".
 * @param {{tool: string, info?: object}} e Final step entry.
 * @returns {string}
 */
function finalStepName(e) {
  const cid = e.info && e.info.candidate_id;
  const who = cid === "ref" ? "reference" : cid;
  return `${FINAL_STEP_NAMES[e.tool] || label(e.tool)}${who ? ` · ${who}` : ""}`;
}

/**
 * Set a meter's fill and text; over budget shows full and red.
 * @param {HTMLElement} el A `.meter`.
 * @param {number} used Used amount.
 * @param {number} max Budget.
 * @param {string} text Label.
 */
function setMeter(el, used, max, text) {
  const frac = max > 0 ? used / max : 0;
  const fill = el.querySelector(".meter-fill");
  fill.style.width = `${Math.min(1, frac) * 100}%`;
  fill.className = `meter-fill${frac > 1.001 ? " critical" : frac >= 0.9 ? " warning" : ""}`; // red only when over
  const t = el.querySelector(".meter-text");
  if (t) t.textContent = text;
}

const ANIMATION_MS = { spinner: 800, step: 1800, dot: 1600 };

/**
 * Keep looping animations in phase across re-renders, so a re-render does not restart them.
 * @param {HTMLElement} el Freshly rendered subtree.
 */
function syncAnimations(el) {
  const now = performance.now();
  for (const n of el.querySelectorAll(".spinner")) n.style.animationDelay = `-${now % ANIMATION_MS.spinner}ms`;
  for (const n of el.querySelectorAll(".dot.live")) n.style.animationDelay = `-${now % ANIMATION_MS.dot}ms`;
  for (const n of el.querySelectorAll(".step.active")) n.style.setProperty("--sync", `-${now % ANIMATION_MS.step}ms`);
}

/**
 * A stat tile.
 * @param {string} lab Label.
 * @param {any} value Value.
 * @param {any} [foot] Footer.
 * @returns {HTMLElement}
 */
function stat(lab, value, foot) {
  return h("div.stat", null, h("div.stat-label", null, lab), h("div.stat-value", null, value), foot ? h("div.stat-foot", null, foot) : null);
}

/**
 * A verdict pill with icon.
 * @param {string} v confirmed | weakened | overturned.
 * @returns {HTMLElement}
 */
function verdictPill(v) {
  const k = { confirmed: "good", weakened: "warning", overturned: "critical" }[v] || "warning";
  return pill(label(v), k, k === "good" ? "shield" : "alert");
}


const CHECK_NAMES = {
  e_hull_eV_per_atom: "Energy above hull (eV/atom)",
  thermo_stable: "On the hull",
  dyn_stable: "Dynamically stable",
  min_freq_THz: "Lowest phonon (THz)",
  min_frequency_THz: "Lowest phonon (THz)",
  md_stable: "Stable in MD",
  target_met: "Target met",
  target_met_conservative: "Target met within error",
  band_gap: "Band gap (eV)",
  bulk_modulus: "Bulk modulus (GPa)",
  disorder_spread_eV_per_atom: "Disorder spread (eV/atom)",
};

/**
 * Readable name of a validation check.
 * @param {string} n Check name.
 * @returns {string}
 */
function checkName(n) {
  return CHECK_NAMES[n] || label(n);
}

/**
 * Format a check value (number or boolean).
 * @param {any} v Value.
 * @returns {Node|string}
 */
function fmtCheck(v) {
  if (typeof v === "boolean") return check(v, "Yes", "No");
  return num(v, 3);
}

/**
 * Format a config value.
 * @param {string} k Key.
 * @param {any} v Value.
 * @returns {string}
 */
function fmtConfig(k, v) {
  if (v == null) return "–";
  if (typeof v === "boolean") return v ? "on" : "off";
  if (k.endsWith("_s")) return duration(v);
  return String(v);
}
