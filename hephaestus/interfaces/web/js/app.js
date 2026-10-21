// Shell: hash router, sidebar (runs, sign-in and backend status, theme) and the goal composer.
// Routes: #/new, #/run/<id>/<tab>.

import { api, ApiError } from "./api.js";
import { ago, append, cid, cidLong, h, icon, label, pill, store, stored, toast, wordmark } from "./dom.js";
import { openConnectDialog } from "./auth.js";
import { mountRun } from "./run.js";
import { CONFIG_GROUPS, TOP_LEVEL_SETTINGS, modelName, specGrid } from "./spec.js";

const main = document.getElementById("main");
const sidebar = document.getElementById("sidebar");

const app = {
  runs: [],
  health: null,
  defaults: null,
  models: null,
  current: null, // unmount function of the current view
  route: null,
};

const EXAMPLES = [
  {
    short: "Smaller band gap than SrTiO3, stable",
    goal: "a lead-free oxide perovskite with a smaller band gap than SrTiO3 that is thermodynamically stable (≤ 0.05 eV/atom above hull) and dynamically stable",
  },
  {
    short: "Stiffer than SrTiO3",
    goal: "a lead-free oxide perovskite with a higher bulk modulus than SrTiO3 that is thermodynamically stable (≤ 0.05 eV/atom above hull) and dynamically stable",
  },
  {
    short: "Visible-light absorber, no Pb or Cd",
    goal: "an oxide perovskite photocatalyst with a band gap between 2.0 and 2.6 eV, without Pb or Cd, stable on the hull",
  },
];

// ------------------------------------------------------------------ router

/** Render the view for the current hash. */
function route() {
  const parts = (location.hash.replace(/^#\/?/, "") || "new").split("/");
  const next = { view: parts[0], id: parts[1] ? decodeURIComponent(parts[1]) : null, tab: parts[2] || "overview" };
  const same = app.route && app.route.view === next.view && app.route.id === next.id;
  app.route = next;
  if (same && app.current && app.current.setTab) {
    app.current.setTab(next.tab);
  } else {
    if (app.current) app.current.unmount();
    main.replaceChildren();
    main.scrollTop = 0;
    if (next.view === "run" && next.id) app.current = mountRun(main, next.id, next.tab, { onChange: refreshRuns });
    else app.current = mountComposer(main);
  }
  renderSidebar();
}

// ------------------------------------------------------------------ sidebar

/** Fetch the run list and redraw the sidebar. */
async function refreshRuns() {
  try {
    app.runs = await api("/runs");
  } catch (e) {
    /* server down: keep the old list */
  }
  renderSidebar();
}

/** Fetch backend, Valency and device status. */
async function refreshHealth() {
  try {
    app.health = await api("/health");
  } catch (e) {
    app.health = { ok: false };
  }
  renderSidebar();
  if (app.current && app.current.onHealth) app.current.onHealth(app.health);
}

/** Draw the sidebar. */
function renderSidebar() {
  const active = app.route && app.route.view === "run" ? app.route.id : null;
  const list = h("nav.run-list", { "aria-label": "Runs" });
  if (!app.runs.length) {
    list.append(h("div.muted", { style: { padding: "8px 10px", fontSize: "12.5px" } }, "No runs yet. Start one above."));
  }
  for (const r of app.runs) {
    const kind = r.running ? "live" : r.status === "finished" ? "good" : r.status === "failed" ? "critical" : "warning";
    list.append(
      h(
        `a.run-item${r.run_id === active ? ".active" : ""}`,
        { href: `#/run/${encodeURIComponent(r.run_id)}/overview`, title: r.raw_goal },
        h("div.run-item-goal", null, r.raw_goal || r.run_id),
        h(
          "div.run-item-meta",
          null,
          h(`span.dot.${kind}`),
          r.running ? "Running" : label(r.status),
          r.best ? h("span", { title: cidLong(r.best) }, "· ", cid(r.best)) : null,
          h("span", { style: { marginLeft: "auto" } }, ago(r.created_at)),
        ),
      ),
    );
  }
  sidebar.replaceChildren(
    h(
      "a.brand",
      { href: "#/new", title: "Hephaestus: new experiment" },
      wordmark(),
      h("div.brand-tag", { title: "forged in simulation · checked by code · grounded in the literature" }, "self-designing experiments for materials discovery"),
    ),
    h("div.sidebar-actions", null, h("a.btn.primary", { href: "#/new" }, icon("plus"), "New experiment")),
    h("div.sidebar-section", null, "Runs", h("span.num", null, app.runs.length || "")),
    list,
    statusPanel(),
  );
}

/**
 * Backend, Valency and device status, with sign-in actions (UI-11, UI-12).
 * @returns {HTMLElement}
 */
function statusPanel() {
  const hl = app.health;
  const rows = [];
  const row = (state, name, value, onclick, title) =>
    h(
      `div.status-row${onclick ? ".clickable" : ""}`,
      { role: onclick ? "button" : null, tabindex: onclick ? 0 : null, title: title || null, onclick: onclick || null, onkeydown: onclick ? (e) => e.key === "Enter" && onclick() : null },
      state === "pending" ? h("span.spinner") : h(`span.dot${state ? "." + state : ""}`),
      h("span.label", null, name),
      h(`span.value${state === "critical" ? ".action" : ""}`, null, value),
    );
  if (!hl) {
    rows.push(row("pending", "Checking…", ""));
  } else if (!hl.ok) {
    rows.push(row("critical", "Server unreachable", ""));
  } else {
    const b = hl.backends;
    const sim = hl.simulation || {};
    const v = hl.valency || {};
    const d = hl.device || {};
    if (!b) rows.push(row("pending", "Claude", "Checking…"));
    else {
      const auto = b.auto || {};
      const via = auto.resolves_to === "api" ? "API key" : "Claude Code";
      rows.push(row(auto.ready ? "good" : "critical", "Claude", auto.ready ? via : "Sign in", () => connect("claude")));
    }
    rows.push(
      row(
        v.signed_in ? "good" : v.pending ? "warning" : "critical",
        "Valency literature",
        v.signed_in ? "Signed in" : v.pending ? "Waiting…" : v.needs_signin && v.access_token_expired ? "Sign in again" : "Sign in",
        v.signed_in ? null : signInValency,
        v.error || null,
      ),
    );
    rows.push(row(sim.mp_key ? "good" : "critical", "Materials Project", sim.mp_key ? "Key set" : "Add key", () => connect("mp")));
    rows.push(
      sim.mace === false
        ? row("critical", "Simulation", "MACE missing", null, 'pip install -e ".[sim]"')
        : row(d.cuda ? "good" : "", "Compute", d.cuda ? "GPU (CUDA)" : "CPU"),
    );
  }
  return h(
    "div.sidebar-foot",
    null,
    h("div.sidebar-section.foot-title", null, "Connections"),
    ...rows,
    h("div.foot-tools", null, themeButton(), h("a.btn.ghost.sm", { href: "/docs", target: "_blank", rel: "noopener" }, icon("terminal"), "API")),
  );
}

/**
 * Open the "Connections" dialog.
 * @param {"claude"|"api"} [focus] Option to highlight.
 */
export function connect(focus) {
  openConnectDialog({ focus, onChange: refreshHealth });
}

/** Start the Valency browser sign-in and open its page when the URL is ready (UI-11). */
async function signInValency() {
  try {
    let st = await api("/auth/valency?start=true");
    toast("Opening the Valency sign-in page…");
    for (let i = 0; i < 20 && !st.url && !st.signed_in && !st.error; i++) {
      await new Promise((r) => setTimeout(r, 500));
      st = await api("/auth/valency");
    }
    if (st.url) window.open(st.url, "_blank", "noopener");
    if (st.error) toast(st.error, "critical");
    const poll = setInterval(async () => {
      const s2 = await api("/auth/valency").catch(() => null);
      if (s2 && (s2.signed_in || !s2.pending)) {
        clearInterval(poll);
        if (s2.signed_in) toast("Signed in to Valency.");
        refreshHealth();
      }
    }, 2000);
  } catch (e) {
    toast(e.message, "critical");
  }
  refreshHealth();
}

/**
 * Theme toggle: system → light → dark.
 * @returns {HTMLElement}
 */
function themeButton() {
  const cur = document.documentElement.dataset.theme || "system";
  const next = { system: "light", light: "dark", dark: "system" }[cur];
  const names = { system: "System theme", light: "Light theme", dark: "Dark theme" };
  return h(
    "button.btn.ghost.sm",
    {
      title: `${names[cur]} (switch to ${names[next].toLowerCase()})`,
      onclick: () => {
        if (next === "system") delete document.documentElement.dataset.theme;
        else document.documentElement.dataset.theme = next;
        try {
          if (next === "system") localStorage.removeItem("heph-theme");
          else localStorage.setItem("heph-theme", next);
        } catch (e) {
          /* not stored */
        }
        renderSidebar();
        window.dispatchEvent(new Event("themechange"));
      },
    },
    icon(cur === "dark" ? "moon" : cur === "light" ? "sun" : "monitor"),
    names[cur].replace(" theme", ""),
  );
}

// ------------------------------------------------------------------ composer

/**
 * The goal composer: goal, model access, model, novelty and every other run setting; parse, answer, run.
 * @param {HTMLElement} root Mount point.
 * @returns {{unmount: () => void, onHealth: (h: object) => void}}
 */
function mountComposer(root) {
  const saved = stored("heph-options", {});
  const st = {
    goal: stored("heph-goal-draft", ""),
    backend: saved.backend || "auto",
    model: saved.model || null,
    novel: Boolean(saved.novel),
    overrides: saved.overrides || {}, // RunConfig fields changed from the defaults
    busy: false,
    parsed: null,
    answers: {},
    error: null,
  };
  const persist = () =>
    store("heph-options", {
      backend: st.backend,
      model: st.model,
      novel: st.novel,
      overrides: st.overrides,
      advancedOpen: stored("heph-options", {}).advancedOpen,
    });

  const view = h("div.composer");
  root.append(view);

  const config = () => {
    const d = app.defaults || {};
    const known = Object.fromEntries(Object.entries(st.overrides).filter(([k]) => k in d));
    return {
      ...d,
      ...known,
      llm_backend: st.backend,
      llm_model: st.model || (app.models && app.models.default) || d.llm_model,
      require_novel: st.novel,
    };
  };

  async function parse() {
    if (!st.goal.trim() || st.busy) return;
    st.busy = "parse";
    st.error = null;
    draw();
    try {
      const cfg = config();
      st.parsed = await api("/parse", {
        body: {
          raw_goal: st.goal.trim(),
          answers: Object.keys(st.answers).length ? st.answers : null,
          llm_backend: cfg.llm_backend,
          llm_model: cfg.llm_model,
          require_novel: cfg.require_novel,
          band_gap_fidelity: cfg.band_gap_fidelity,
        },
      });
    } catch (e) {
      st.error = e instanceof ApiError ? e.message : String(e);
    }
    st.busy = false;
    draw();
  }

  async function start() {
    if (st.busy) return;
    st.busy = "run";
    st.error = null;
    draw();
    try {
      const body = st.parsed && st.parsed.spec && !st.parsed.questions.length
        ? { spec: st.parsed.spec, config: config() }
        : { raw_goal: st.goal.trim(), answers: Object.keys(st.answers).length ? st.answers : null, config: config() };
      const { run_id } = await api("/runs", { body });
      store("heph-goal-draft", "");
      toast("Run started.");
      await refreshRuns();
      location.hash = `#/run/${encodeURIComponent(run_id)}/activity`;
    } catch (e) {
      st.error = e instanceof ApiError ? e.message : String(e);
      st.busy = false;
      draw();
    }
  }

  function draw() {
    const d = app.defaults || {};
    const models = (app.models && app.models.models) || ["claude-opus-5-5"];
    const hl = app.health || {};
    const b = hl.backends;
    const ready = (k) => (b ? Boolean(b[k] && b[k].ready) : null);
    const readyDot = (k) => h(`span.dot.${ready(k) == null ? "" : ready(k) ? "good" : "critical"}`);

    const textarea = h("textarea", {
      placeholder: "Describe the material you need, e.g. a lead-free oxide perovskite with a smaller band gap than SrTiO3 that is stable on the hull…",
      rows: 4,
      value: st.goal,
      "aria-label": "Goal",
      oninput: (e) => {
        st.goal = e.target.value;
        st.parsed = null;
        st.answers = {};
        store("heph-goal-draft", st.goal);
        syncButtons();
      },
      onkeydown: (e) => {
        if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
          e.preventDefault();
          parse();
        }
      },
    });
    const parseBtn = h(
      "button.btn.primary.lg",
      { onclick: parse, disabled: !st.goal.trim() || Boolean(st.busy) },
      st.busy === "parse" ? h("span.spinner") : icon("sparkles"),
      st.parsed ? "Re-check goal" : "Review goal",
    );
    const syncButtons = () => {
      parseBtn.disabled = !st.goal.trim() || Boolean(st.busy);
    };

    const backendBtn = (key, text, ic) =>
      h(
        "button",
        {
          type: "button",
          "aria-pressed": String(st.backend === key),
          onclick: () => {
            st.backend = key;
            persist();
            draw();
          },
        },
        key === "auto" ? icon(ic) : readyDot(key),
        text,
      );

    const selectedReady = () => (st.backend === "auto" ? Boolean(b.auto && b.auto.ready) : ready(st.backend));
    const backendNote = (() => {
      if (!b) return "Checking which access is available…";
      if (st.backend === "auto") return `Uses ${b.auto.resolves_to === "api" ? "the Anthropic API key" : "your Claude Code login"}${b.auto.ready ? "." : " (not ready)."}`;
      const s = b[st.backend];
      if (s && s.ready) return st.backend === "api" ? "API key found." : "Signed in to Claude Code.";
      return st.backend === "api" ? "No API key yet." : "Claude Code is signed out.";
    })();

    view.replaceChildren();
    append(view, [
      h(
        "div.composer-hero",
        null,
        h("div.eyebrow", null, icon("flame"), "New experiment"),
        h("h1", null, "What material should we design?"),
        h(
          "p",
          null,
          "Describe the goal in plain English. You get back a candidate that passed code-enforced stability checks, and a cited lab brief.",
        ),
      ),
      h(
        "div.goal-box",
        null,
        textarea,
        h(
          "div.goal-foot",
          null,
          h("span.muted", { style: { fontSize: "12px" } }, h("span.kbd", null, navigator.platform.includes("Mac") ? "⌘" : "Ctrl"), " ", h("span.kbd", null, "Enter"), " to review"),
          h("div", { style: { display: "flex", gap: "8px" } }, parseBtn),
        ),
      ),
      h(
        "div.examples",
        null,
        EXAMPLES.map((ex) =>
          h(
            "button.example",
            {
              type: "button",
              title: ex.goal,
              onclick: () => {
                st.goal = ex.goal;
                st.parsed = null;
                st.answers = {};
                store("heph-goal-draft", st.goal);
                draw();
              },
            },
            icon("sparkles"),
            ex.short,
          ),
        ),
      ),
      st.error ? h("div.callout.critical", { style: { marginTop: "18px" } }, icon("alert"), h("div", null, st.error)) : null,
      blockersCallout(),
      st.parsed ? specReview() : null,
      h(
        "div.card.options",
        null,
        h(
          "div.field.span-2",
          null,
          h("span.field-label", null, "Model access"),
          h(
            "div.option-row",
            null,
            h("div.segmented", { role: "group", "aria-label": "Model access" }, backendBtn("auto", "Auto", "sparkles"), backendBtn("api", "Anthropic API key"), backendBtn("claude-code", "Claude Code login")),
          ),
          h(
            "div",
            { style: { display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" } },
            h("span.field-hint", null, backendNote),
            b && !selectedReady() ? h("button.btn.sm", { onclick: () => connect(st.backend === "api" ? "api" : "claude") }, icon("key"), st.backend === "api" ? "Add API key" : "Sign in to Claude") : null,
          ),
        ),
        h(
          "label.field",
          null,
          h("span.field-label", null, "Model"),
          h(
            "select.select",
            {
              onchange: (e) => {
                st.model = e.target.value;
                persist();
              },
            },
            models.map((m) => h("option", { value: m, selected: (st.model || (app.models && app.models.default)) === m }, modelName(m))),
          ),
        ),
        h(
          "div.field",
          null,
          h("span.field-label", null, "Novelty"),
          h(
            "label.switch",
            null,
            h("input", {
              type: "checkbox",
              checked: st.novel,
              onchange: (e) => {
                st.novel = e.target.checked;
                st.parsed = null;
                persist();
                draw();
              },
            }),
            "Require novel materials",
          ),
          h("span.field-hint", null, "Rejects compositions with an ICSD-backed Materials Project entry or a prior report."),
        ),
        settingsPanel(),
      ),
    ]);
    if (!st.goal) requestAnimationFrame(() => textarea.focus());
  }

  function setOverride(key, value) {
    const d = app.defaults || {};
    if (value === d[key] || (value === "" && d[key] == null)) delete st.overrides[key];
    else st.overrides[key] = value;
    if (["band_gap_fidelity"].includes(key)) st.parsed = null; // changes the parsed target
    persist();
  }

  function settingControl(key, prop, meta) {
    const d = app.defaults || {};
    const [name, hint, unit] = meta;
    const value = key in st.overrides ? st.overrides[key] : d[key];
    const changed = key in st.overrides;
    const enumValues = prop.enum || (prop.anyOf || []).flatMap((x) => x.enum || []);
    const type = prop.type || (prop.anyOf || []).map((x) => x.type).find((t) => t && t !== "null");
    const nullable = (prop.anyOf || []).some((x) => x.type === "null");
    let control;
    if (type === "boolean") {
      control = h(
        "label.switch",
        null,
        h("input", { type: "checkbox", checked: Boolean(value), onchange: (e) => (setOverride(key, e.target.checked), draw()) }),
        value ? "On" : "Off",
      );
    } else if (enumValues.length) {
      control = h(
        "select.select",
        { onchange: (e) => (setOverride(key, e.target.value), draw()) },
        enumValues.map((v) => h("option", { value: v, selected: v === value }, String(v))),
      );
    } else if (type === "integer" || type === "number") {
      const scale = unit === "min" ? 60 : 1;
      control = h("input.input.num", {
        type: "number",
        min: 0,
        step: type === "integer" ? 1 : "any",
        value: value == null ? "" : +(value / scale).toFixed(2),
        onchange: (e) => {
          const raw = e.target.value;
          if (raw === "") setOverride(key, d[key]);
          else setOverride(key, type === "integer" ? Math.round(Number(raw)) : Number(raw) * scale);
          draw();
        },
      });
    } else {
      control = h("input.input.mono", {
        value: value ?? "",
        placeholder: nullable ? "none" : "",
        list: key.includes("potential") ? "potential-options" : null,
        onchange: (e) => (setOverride(key, e.target.value.trim() || (nullable ? null : d[key])), draw()),
      });
    }
    const def = d[key];
    const defText = unit === "min" && typeof def === "number" ? `${+(def / 60).toFixed(2)} min` : def == null ? "none" : String(def);
    return h(
      "label.field.setting",
      { class: changed ? "changed" : "" },
      h("span.field-label", null, name, unit === "min" ? h("span.muted", null, " (min)") : null, changed ? h("span.changed-dot", { title: `Default: ${defText}` }) : null),
      control,
      h("span.field-hint", null, hint, changed ? h("span", null, ` Default: ${defText}.`) : null),
    );
  }

  function settingsPanel() {
    const schema = app.schema && app.schema.properties;
    if (!schema || !app.defaults) return null;
    const used = new Set();
    const groups = CONFIG_GROUPS.map((g) => {
      const fields = Object.entries(g.fields).filter(([k]) => schema[k]);
      fields.forEach(([k]) => used.add(k));
      return { name: g.name, fields };
    });
    const other = Object.keys(schema).filter((k) => !used.has(k) && !TOP_LEVEL_SETTINGS.has(k));
    if (other.length) groups.push({ name: "Other", fields: other.map((k) => [k, [label(k), schema[k].title || ""]]) });
    const nChanged = Object.keys(st.overrides).filter((k) => k in app.defaults).length;
    const open = Boolean(stored("heph-options", {}).advancedOpen);
    return h(
      "details.advanced",
      {
        open,
        ontoggle: (e) => store("heph-options", { ...stored("heph-options", {}), advancedOpen: e.currentTarget.open }),
      },
      h(
        "summary",
        null,
        icon("chevron"),
        "All settings",
        nChanged ? h("span.pill.info", { style: { marginLeft: "8px" } }, `${nChanged} changed`) : h("span.muted", { style: { fontWeight: 500, marginLeft: "6px" } }, "· defaults"),
      ),
      h(
        "datalist",
        { id: "potential-options" },
        ["mace-mp-0-small", "mace-mp-0-medium", "mace-mp-0-large", "chgnet"].map((v) => h("option", { value: v })),
      ),
      groups.map((g) =>
        h(
          "div.settings-group",
          null,
          h("div.settings-group-name", null, g.name),
          h("div.advanced-grid", null, g.fields.map(([k, meta]) => settingControl(k, schema[k], meta))),
        ),
      ),
      nChanged
        ? h(
            "div",
            { style: { marginTop: "14px" } },
            h(
              "button.btn.sm",
              {
                onclick: (e) => {
                  e.preventDefault();
                  st.overrides = {};
                  st.parsed = null;
                  persist();
                  draw();
                },
              },
              icon("reset"),
              "Reset to defaults",
            ),
          )
        : null,
    );
  }

  function blockers() {
    const hl = app.health || {};
    const out = [...((hl.simulation && hl.simulation.blockers) || [])];
    const b = hl.backends;
    if (b) {
      const key = st.backend === "auto" ? b.auto.resolves_to : st.backend;
      if (!(b[key] && b[key].ready)) out.unshift(st.backend === "api" || key === "api" ? "No Anthropic API key." : "Claude Code is signed out.");
    }
    return out;
  }

  function blockersCallout() {
    const list = blockers();
    if (!list.length) return null;
    return h(
      "div.callout.warning",
      { style: { marginTop: "18px", alignItems: "flex-start" } },
      icon("alert"),
      h(
        "div",
        { style: { display: "grid", gap: "8px", flex: 1 } },
        h("strong", null, "Runs cannot start yet"),
        h("ul.list-plain", null, list.map((t) => h("li", null, h("span"), h("span", null, t)))),
        h("div", null, h("button.btn.sm", { onclick: () => connect(list.some((t) => t.includes("Materials Project")) && !list.some((t) => t.includes("Claude") || t.includes("Anthropic")) ? "mp" : "claude") }, icon("key"), "Open connections")),
      ),
    );
  }

  function specReview() {
    const p = st.parsed;
    const spec = p.spec;
    const open = (p.questions || []).length > 0;
    const errs = (p.errors || []).filter(Boolean);
    const card = h("div.card.spec-review");
    card.append(
      h(
        "div.card-head",
        { style: { paddingBottom: "14px" } },
        h("div", null, h("div.card-title", null, "Parsed specification"), h("div.card-sub", null, `Checked by ${p.parser.model || "the parser"} · edit the goal and review again to change it`)),
        open ? pill("Needs answers", "warning", "alert") : errs.length ? pill("Check the goal", "critical", "alert") : pill("Ready to run", "good", "check"),
      ),
    );
    if (spec) card.append(specGrid(spec, config()));
    if (open) {
      card.append(
        h(
          "div.questions",
          null,
          h("div.questions-title", null, icon("message"), "The parser has questions before it can start"),
          p.questions.map((q) =>
            h(
              "label.field",
              null,
              h("span", null, q),
              h("input.input", {
                value: st.answers[q] || "",
                placeholder: "Your answer",
                oninput: (e) => {
                  st.answers[q] = e.target.value;
                },
                onkeydown: (e) => e.key === "Enter" && parse(),
              }),
            ),
          ),
          h("div", null, h("button.btn", { onclick: parse }, st.busy === "parse" ? h("span.spinner") : icon("check"), "Update specification")),
        ),
      );
    }
    if (errs.length) card.append(h("div.card-body", null, h("div.callout.critical", null, icon("alert"), h("div", null, errs.join(" ")))));
    card.append(
      h(
        "div.card-body",
        { style: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: "12px" } },
        h("span.muted", { style: { fontSize: "12.5px" } }, "Stability verdicts, the target check, budgets and stop rules are decided by code, never by the model."),
        h("button.btn.primary.lg", { onclick: start, disabled: open || Boolean(st.busy) || !spec || blockers().length > 0, title: blockers().join(" ") }, st.busy === "run" ? h("span.spinner") : icon("play"), "Start run"),
      ),
    );
    return card;
  }

  draw();
  return {
    unmount() {},
    onHealth() {
      if (!st.busy) draw();
    },
  };
}

// ------------------------------------------------------------------ boot

async function boot() {
  renderSidebar();
  const [defaults, models, schema] = await Promise.all([
    api("/config/defaults").catch(() => null),
    api("/models").catch(() => null),
    api("/config/schema").catch(() => null),
  ]);
  app.defaults = defaults;
  app.models = models;
  app.schema = schema;
  window.addEventListener("hashchange", route);
  route();
  refreshRuns();
  refreshHealth();
  setInterval(refreshHealth, 15000);
  setInterval(refreshRuns, 10000);
}

boot();
