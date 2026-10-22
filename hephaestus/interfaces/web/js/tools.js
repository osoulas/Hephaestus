// Agent-built tools: plan and code-review cards for the activity log, and the tool library page.
// The server decides what is offered (the "allowed" actions); these only render and send the choice.

import { api } from "./api.js";
import { append, h, icon, label, pill, toast } from "./dom.js";

const KEYWORDS = new Set(
  "and as assert break class continue def del elif else except False finally for from global if import in is lambda None nonlocal not or pass raise return True try while with yield".split(" "),
);
const TOKEN = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|\b(\d+(?:\.\d+)?(?:e[-+]?\d+)?)\b|\b([A-Za-z_]\w*)\b/g;

/**
 * Python source with syntax colouring and line numbers.
 * @param {string} src Source code.
 * @returns {HTMLElement}
 */
export function codeView(src) {
  // Built with textContent, not h(): code is shown verbatim, never filtered like prose
  const tok = (cls, text) => {
    const el = document.createElement("span");
    el.className = cls;
    el.textContent = text;
    return el;
  };
  const code = document.createElement("code");
  code.className = "py";
  let last = 0;
  let prev = "";
  for (const m of (src || "").matchAll(TOKEN)) {
    if (m.index > last) code.append(src.slice(last, m.index));
    const [text, comment, str, num, word] = m;
    if (comment) code.append(tok("tok-com", text));
    else if (str) code.append(tok("tok-str", text));
    else if (num) code.append(tok("tok-num", text));
    else if (word && KEYWORDS.has(word)) code.append(tok("tok-kw", text));
    else if (word && (prev === "def" || prev === "class")) code.append(tok("tok-fn", text));
    else code.append(text);
    if (word) prev = word;
    last = m.index + text.length;
  }
  if (last < (src || "").length) code.append(src.slice(last));
  const n = (src || "").split("\n").length;
  const gutter = h("div.code-gutter", { "aria-hidden": "true" }, Array.from({ length: n }, (_, i) => h("span", null, String(i + 1))));
  const pre = document.createElement("pre");
  pre.append(code);
  return h("div.code-view", null, gutter, pre);
}

const ACTION_TEXT = {
  accept: "Accepted",
  revise: "Changes requested",
  cancel_tool: "Tool cancelled",
  cancel_run: "Run cancelled",
};

/**
 * Buttons for a proposal waiting for the user: accept, request changes, cancel the tool or the run.
 * @param {object} prop ToolProposal.
 * @param {string[]} allowed Actions the server offers.
 * @param {object} ctx {runId, drafts, onSent}.
 * @returns {HTMLElement}
 */
function decisionBar(prop, allowed, ctx) {
  const key = `${prop.proposal_id}:${prop.stage}`;
  const draft = ctx.drafts[key] || { open: false, text: "" };
  ctx.drafts[key] = draft;
  const send = async (action, message = null, btn = null) => {
    if (btn) btn.disabled = true;
    try {
      await api(`/runs/${encodeURIComponent(ctx.runId)}/tool-decisions`, { body: { proposal_id: prop.proposal_id, action, message } });
      draft.sent = action;
      ctx.onSent();
    } catch (e) {
      toast(e.message, "critical");
      if (btn) btn.disabled = false;
    }
  };
  const stageWord = prop.stage === "plan" ? "plan" : "code";
  const bar = h("div.tool-actions");
  if (draft.sent) {
    bar.append(h("span.muted", null, h("span.spinner"), ` ${ACTION_TEXT[draft.sent]}: waiting for the agent…`));
    return bar;
  }
  const accept = h(
    "button.btn.primary",
    { onclick: (ev) => send("accept", null, ev.currentTarget) },
    icon("check"),
    prop.stage === "plan" ? "Accept plan" : "Approve code",
  );
  const revise = allowed.includes("revise")
    ? h(
        "button.btn",
        {
          onclick: () => {
            draft.open = !draft.open;
            ctx.onSent();
          },
          "aria-expanded": String(draft.open),
        },
        icon("edit"),
        "Request changes",
      )
    : null;
  const cancelTool = h(
    "button.btn.ghost",
    {
      onclick: (ev) => {
        const consequence = prop.supplies_target ? `\n\n${prop.if_cancelled}` : "";
        if (confirm(`Cancel ${prop.name}? The run continues without it.${consequence}`)) send("cancel_tool", null, ev.currentTarget);
      },
    },
    icon("x"),
    "Cancel the tool",
  );
  const cancelRun = h(
    "button.btn.ghost.danger",
    { onclick: (ev) => confirm("Cancel the whole run? Nothing else runs.") && send("cancel_run", null, ev.currentTarget) },
    icon("stop"),
    "Cancel the run",
  );
  append(bar, [accept, revise, cancelTool, cancelRun]);
  const wrap = h("div.tool-decide", null, bar);
  if (!allowed.includes("revise")) wrap.append(h("div.field-hint", null, "No more change requests for this tool: accept it or cancel."));
  if (draft.open) {
    const box = h("textarea.input", {
      rows: 3,
      placeholder: `What should change in the ${stageWord}? e.g. use an equation-of-state fit instead of strains`,
      value: draft.text,
      oninput: (e) => {
        draft.text = e.target.value;
      },
    });
    wrap.append(
      h(
        "div.tool-revise",
        null,
        box,
        h(
          "button.btn.primary",
          {
            onclick: (ev) => {
              if (!draft.text.trim()) return toast("Say what should change.", "warning");
              send("revise", draft.text.trim(), ev.currentTarget);
            },
          },
          icon("message"),
          "Send to the agent",
        ),
      ),
    );
    setTimeout(() => box.focus(), 0);
  }
  return wrap;
}

function checkRow(name, c) {
  const names = { static: "Static check", tests: "Tests", trial: "Trial run" };
  return h(
    "li.tool-check",
    null,
    h(`span.check.${c.passed ? "good" : "critical"}`, null, icon(c.passed ? "check" : "x")),
    h("strong", null, names[name] || label(name)),
    h("span.secondary", null, c.report),
  );
}

/**
 * The card for a plan or the code of an agent-built tool.
 * @param {object} entry Log entry {kind, proposal, allowed, decision}.
 * @param {object} ctx {runId, waiting (bool), drafts, onSent, open (bool), toggle}.
 * @returns {HTMLElement}
 */
export function proposalCard(entry, ctx) {
  const p = entry.proposal;
  const isPlan = entry.kind === "plan";
  const who = p.author === "user" ? "Your library file" : "The agent";
  const status = entry.decision
    ? pill(ACTION_TEXT[entry.decision.action] || label(entry.decision.action), entry.decision.action === "accept" ? "good" : entry.decision.action === "revise" ? "info" : "warning", entry.decision.action === "accept" ? "check" : "edit")
    : ctx.waiting
      ? pill("Needs your decision", "warning", "alert")
      : pill("Not decided", "", "info");
  const head = h(
    "div.tool-card-head",
    null,
    h("div.log-icon.decision", null, icon(isPlan ? "wrench" : "code")),
    h(
      "div",
      null,
      h("div.tool-card-kicker", null, isPlan ? `Tool plan · revision ${p.revision}` : `Code review · revision ${p.revision}`),
      h("div.tool-card-title", null, h("span.mono", null, p.name), ` computes ${label(p.property).toLowerCase()}`, p.units ? h("span.muted", null, ` (${p.units})`) : null),
      h("div.muted", null, `${who} ${isPlan ? "proposes this tool" : "wrote this code"}. ${isPlan ? "Nothing is written until you accept the plan." : "It only runs in the campaign once you approve it."}`),
    ),
    status,
  );
  const body = h("div.tool-card-body");
  if (isPlan) {
    const kv = (k, v) => (v ? [h("dt", null, k), h("dd", null, v)] : []);
    body.append(
      h("p", null, p.description),
      h(
        "dl.kv",
        null,
        ...kv("Method", p.method),
        ...kv("Citations", (p.citations || []).join(", ") || "none"),
        ...kv("Building blocks", (p.building_blocks || []).join(", ") || "none: its own calculation"),
        ...kv("Expected time", p.expected_runtime_s != null ? `about ${Math.round(p.expected_runtime_s)} s per candidate` : null),
        ...kv("Inputs", Object.keys((p.inputs && p.inputs.properties) || {}).join(", ") || "only the candidate"),
        ...kv("If you cancel it", p.if_cancelled),
      ),
    );
  } else {
    const checks = Object.entries(p.checks || {});
    body.append(h("ul.tool-checks", null, checks.map(([k, c]) => checkRow(k, c))));
    if (p.trial) body.append(h("div.callout.info", null, icon("flask"), h("div", null, `Trial run on ${p.trial.material}: `, h("strong", null, `${Number(p.trial.value).toPrecision(4)} ${p.units}`))));
    if ((p.toolkit_calls || []).length) body.append(h("div.muted", null, `Building blocks used: ${p.toolkit_calls.join(", ")}`));
    body.append(h("div.field-label", null, "Tool"), codeView(p.source));
    if (p.tests) {
      body.append(
        h(
          "details.tool-tests",
          { open: ctx.open || null },
          h("summary", null, "Tests"),
          codeView(p.tests),
        ),
      );
    }
  }
  if (entry.decision && entry.decision.message) body.append(h("div.callout", null, icon("message"), h("div", null, h("strong", null, "You asked: "), entry.decision.message)));
  if (ctx.waiting && !entry.decision) body.append(decisionBar(p, entry.allowed || [], ctx));
  return h(`div.log-entry.tool-card${ctx.waiting && !entry.decision ? ".waiting" : ""}`, { id: `tool-${p.proposal_id}-${entry.kind}` }, head, body);
}

/**
 * One-line log entries for the other capability events.
 * @param {object} e Log entry {kind, payload}.
 * @param {HTMLElement} time Time element.
 * @returns {HTMLElement}
 */
export function capabilityLine(e, time) {
  const p = e.payload;
  let ic = "wrench";
  let title = "Tool builder";
  let text = "";
  let side = null;
  switch (e.kind) {
    case "gap":
      title = "Capability gap";
      text = `No tool computes ${label(p.property).toLowerCase()}${p.units ? ` (${p.units})` : ""}. The agent will propose one.`;
      ic = "alert";
      break;
    case "library":
      title = "Tool library";
      text = `Reusing ${p.reused.join(", ")} for ${label(p.property).toLowerCase()}: approved earlier.`;
      side = pill("Reused", "good", "check");
      break;
    case "checks": {
      const failed = Object.entries(p.checks || {}).filter(([, c]) => !c.passed);
      title = "Code checks";
      text = failed.length ? `${failed.map(([k]) => label(k)).join(", ")} failed; the agent is fixing it` : "All passed";
      ic = failed.length ? "x" : "check";
      side = failed.length ? pill("Failed", "warning", "x") : pill("Passed", "good", "check");
      break;
    }
    case "registered":
      title = "Tool approved";
      text = `${p.tool.name} is saved to the tool library and available to the campaign.`;
      ic = "check";
      side = pill("Registered", "good", "check");
      break;
    case "cancelled":
      title = p.scope === "run" ? "Run cancelled" : "No tool";
      text = p.scope === "run" ? "You cancelled the run." : p.reason === "budget" ? "The tool builder ran out of time. The run screens stability only." : "The run continues without the tool and screens stability only.";
      ic = "x";
      break;
    case "log":
      title = "Tool";
      text = p.message;
      ic = "terminal";
      break;
    case "decision":
      title = "Your decision";
      text = `${ACTION_TEXT[p.decision.action] || p.decision.action} (${p.name}, ${p.decision.stage})`;
      ic = "message";
      break;
    default:
      text = e.kind;
  }
  return h(
    "div.log-entry",
    null,
    h(`div.log-icon${ic === "x" ? ".error" : ".decision"}`, null, icon(ic)),
    h("div.log-main", null, h("div.log-line", null, h("strong", null, title), h("span.log-summary", null, text))),
    h("div.log-side", null, side, time),
  );
}

/**
 * The tool library page: approved tools with their code, files waiting for review, remove.
 * @param {HTMLElement} root Mount point.
 * @returns {{unmount: () => void}}
 */
export function mountTools(root) {
  const view = h("div.tools-page");
  root.append(view);
  let alive = true;
  const st = { lib: null, error: null, open: new Set() };

  async function load() {
    try {
      st.lib = await api("/tools");
      st.error = null;
    } catch (e) {
      st.error = e.message;
    }
    if (alive) draw();
  }

  async function approve(f) {
    try {
      await api(`/tools/${encodeURIComponent(f.name)}/approve`, { body: { sha256: f.sha256 } });
      toast(`Approved ${f.name}.`);
      load();
    } catch (e) {
      toast(e.message, "critical");
    }
  }

  async function remove(name) {
    if (!confirm(`Delete ${name} and its tests from the library?`)) return;
    try {
      await api(`/tools/${encodeURIComponent(name)}`, { method: "DELETE" });
      toast(`Removed ${name}.`);
      load();
    } catch (e) {
      toast(e.message, "critical");
    }
  }

  const toggle = (k) => {
    if (st.open.has(k)) st.open.delete(k);
    else st.open.add(k);
    draw();
  };

  function toolRow(t, approved) {
    const header = approved ? t : { ...t.header, name: t.name };
    const key = `${approved ? "a" : "u"}:${t.name}`;
    const open = st.open.has(key);
    const check = !approved && t.check;
    return h(
      "div.card.tool-lib-card",
      null,
      h(
        "div.card-head",
        null,
        h(
          "div",
          null,
          h("div.card-title.mono", null, t.name),
          h("div.card-sub", null, `${label(header.property || "?")} · ${header.units || "no units"} · written by ${header.author === "user" ? "you" : "the agent"}${approved && t.run_id ? ` in run ${t.run_id}` : ""}`),
        ),
        approved ? pill("Approved", "good", "check") : t.changed ? pill("Edited: review again", "warning", "alert") : pill("Needs review", "warning", "alert"),
      ),
      h(
        "div.card-body",
        null,
        header.description ? h("p", null, header.description) : null,
        t.error ? h("div.callout.critical", null, icon("alert"), t.error) : null,
        check ? h(`div.callout.${check.passed ? "info" : "critical"}`, null, icon(check.passed ? "check" : "alert"), h("div", null, h("strong", null, "Static check: "), check.report)) : null,
        h(
          "div.tool-lib-actions",
          null,
          h("button.btn.sm", { onclick: () => toggle(key), "aria-expanded": String(open) }, icon("code"), open ? "Hide code" : "Read the code"),
          !approved && check && check.passed ? h("button.btn.sm.primary", { onclick: () => approve(t) }, icon("check"), "Approve as shown") : null,
          h("button.btn.sm.ghost.danger", { onclick: () => remove(t.name) }, icon("x"), "Remove"),
        ),
        open ? h("div", null, codeView(t.source), t.tests ? h("details.tool-tests", { open: true }, h("summary", null, "Tests"), codeView(t.tests)) : null) : null,
      ),
    );
  }

  function draw() {
    const lib = st.lib;
    view.replaceChildren();
    append(view, [
      h(
        "div.composer-hero",
        null,
        h("div.eyebrow", null, icon("wrench"), "Tool library"),
        h("h1", null, "Agent-built tools"),
        h(
          "p",
          null,
          "Tools the agent wrote and you approved. Runs reuse them when a goal needs the same property. A file you add or edit here loads only after you approve it.",
        ),
      ),
      st.error ? h("div.callout.critical", null, icon("alert"), st.error) : null,
      lib ? h("div.muted.mono", { style: { margin: "0 0 14px" } }, lib.dir) : h("div.muted", null, h("span.spinner"), " Loading…"),
      lib && lib.unapproved.length ? h("h2.section-title", null, "Waiting for review") : null,
      lib ? lib.unapproved.map((f) => toolRow(f, false)) : null,
      lib ? h("h2.section-title", null, "Approved") : null,
      lib && !lib.approved.length ? h("div.muted", null, "No approved tools yet. They appear here when you approve a tool in a run.") : null,
      lib ? lib.approved.map((t) => toolRow(t, true)) : null,
    ]);
  }

  draw();
  load();
  return {
    unmount() {
      alive = false;
      view.remove();
    },
  };
}
