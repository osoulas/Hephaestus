// "Connections" dialog (UI-12): sign in with a Claude subscription through Claude Code, or save
// an Anthropic API key; and save the Materials Project key every run needs. The server runs
// `claude auth login`, which opens the Claude page itself and
// finishes on its own after authorising; a code box covers the case where the page shows a code.

import { api } from "./api.js";
import { append, h, icon, pill, toast } from "./dom.js";

const ACTIVE = new Set(["starting", "waiting", "verifying"]);

const KEYS = [
  {
    id: "api",
    endpoint: "/auth/anthropic",
    env: "ANTHROPIC_API_KEY",
    title: "Anthropic API key",
    sub: "Alternative to a subscription: billed to your Anthropic Console account.",
    placeholder: "sk-ant-…",
    icon: "key",
    note: "Checked with one request.",
  },
  {
    id: "mp",
    endpoint: "/auth/materials-project",
    env: "MP_API_KEY",
    title: "Materials Project API key",
    sub: "Required: free from next-gen.materialsproject.org/api.",
    placeholder: "32 letters and digits",
    icon: "database",
    note: "Checked with one request.",
    required: true,
  },
];

/**
 * Open the dialog.
 * @param {{focus?: "claude"|"api"|"mp", onChange: () => void}} opts Which option to highlight; called after a change.
 */
export function openConnectDialog(opts) {
  const dlg = h("dialog.modal", { "aria-labelledby": "connect-title" });
  const st = { claude: null, keys: {}, inputs: {}, errors: {}, code: "", busy: null };
  let timer = null;
  let closed = false;

  const close = () => {
    closed = true;
    clearTimeout(timer);
    if (st.claude && ACTIVE.has(st.claude.status)) api("/auth/claude/cancel", { method: "POST" }).catch(() => {});
    dlg.close();
    dlg.remove();
  };
  dlg.addEventListener("cancel", (e) => {
    e.preventDefault();
    close();
  });
  dlg.addEventListener("click", (e) => {
    if (e.target === dlg) close(); // click on the backdrop
  });

  async function refresh() {
    const [claude, ...keys] = await Promise.all([api("/auth/claude").catch(() => null), ...KEYS.map((k) => api(k.endpoint).catch(() => null))]);
    const was = st.claude && st.claude.status;
    st.claude = claude;
    KEYS.forEach((k, i) => (st.keys[k.id] = keys[i]));
    if (claude && claude.status === "signed_in" && was && ACTIVE.has(was)) {
      toast("Signed in to Claude.");
      opts.onChange();
      setTimeout(opts.onChange, 3000); // readiness is re-checked in the background
    }
    draw();
    clearTimeout(timer);
    if (!closed && claude && (ACTIVE.has(claude.status) || claude.status === "checking")) timer = setTimeout(refresh, 1200);
  }

  async function startSignIn() {
    st.busy = "claude";
    draw();
    try {
      st.claude = await api("/auth/claude/start", { method: "POST" });
    } catch (e) {
      toast(e.message, "critical");
    }
    st.busy = null;
    refresh();
  }

  async function submitCode() {
    if (!st.code.trim()) return;
    st.busy = "code";
    draw();
    try {
      st.claude = await api("/auth/claude/code", { body: { code: st.code.trim() } });
      st.code = "";
    } catch (e) {
      toast(e.message, "critical");
    }
    st.busy = null;
    refresh();
  }

  async function cancelSignIn() {
    await api("/auth/claude/cancel", { method: "POST" }).catch(() => {});
    refresh();
  }

  async function saveKey(k) {
    const value = (st.inputs[k.id] || "").trim();
    if (!value) return;
    st.busy = k.id;
    st.errors[k.id] = null;
    draw();
    try {
      st.keys[k.id] = await api(k.endpoint, { body: { api_key: value } });
      st.inputs[k.id] = "";
      toast(`${k.title} saved.`);
      opts.onChange();
      setTimeout(opts.onChange, 3000);
    } catch (e) {
      st.errors[k.id] = e.message;
    }
    st.busy = null;
    draw();
  }

  async function forgetKey(k) {
    try {
      st.keys[k.id] = await api(k.endpoint, { method: "DELETE" });
      toast(`Saved ${k.title.toLowerCase()} removed.`);
      opts.onChange();
    } catch (e) {
      toast(e.message, "critical");
    }
    draw();
  }

  function claudePanel() {
    const c = st.claude;
    const status = c ? c.status : "checking";
    const head = h(
      "div.connect-head",
      null,
      h("div.connect-icon", null, icon("sparkles")),
      h("div", null, h("div.connect-title", null, "Claude subscription"), h("div.muted", null, "Uses your Claude Code login (Pro, Max, Team or Enterprise).")),
      c && c.signed_in ? pill("Signed in", "good", "check") : status === "checking" ? pill("Checking…") : null,
    );
    let body;
    if (status === "starting" || status === "waiting" || status === "verifying") {
      body = h(
        "div.connect-body",
        null,
        h("div.connect-wait", null, h("span.spinner"), h("div", null, h("strong", null, status === "verifying" ? "Finishing sign-in…" : "Waiting for you to authorise"), h("div.muted", null, "A Claude sign-in page opened in your browser. Approve access there; this finishes by itself."))),
        c.url ? h("div.muted", { style: { fontSize: "12.5px" } }, "No page? ", h("a", { href: c.url, target: "_blank", rel: "noopener" }, "Open the sign-in page"), ". If it shows a code, paste it here:") : null,
        h(
          "div.connect-row",
          null,
          h("input.input.mono", {
            placeholder: "Code from the sign-in page",
            value: st.code,
            autocomplete: "off",
            spellcheck: false,
            oninput: (e) => (st.code = e.target.value),
            onkeydown: (e) => e.key === "Enter" && submitCode(),
          }),
          h("button.btn", { onclick: submitCode, disabled: st.busy === "code" }, st.busy === "code" ? h("span.spinner") : null, "Submit code"),
        ),
        h("div", null, h("button.btn.ghost.sm", { onclick: cancelSignIn }, "Cancel sign-in")),
      );
    } else if (c && c.signed_in) {
      body = h(
        "div.connect-body",
        null,
        h("div.callout.good", null, icon("check"), h("div", null, "Claude Code is signed in. Choose “Claude Code login” or “Auto” as model access.")),
        h("button.btn.ghost.sm", { onclick: startSignIn, disabled: Boolean(st.busy) }, "Sign in with another account"),
      );
    } else {
      body = h(
        "div.connect-body",
        null,
        status === "failed" && c.error ? h("div.callout.critical", null, icon("alert"), h("div", null, c.error)) : null,
        status === "cancelled" ? h("div.muted", null, "Sign-in cancelled.") : null,
        h("button.btn.primary.lg", { onclick: startSignIn, disabled: Boolean(st.busy) || status === "checking" }, st.busy === "claude" ? h("span.spinner") : icon("key"), status === "failed" ? "Try again" : "Sign in with Claude"),
        h("div.muted", { style: { fontSize: "12px" } }, "Runs ", h("code", null, "claude auth login"), " on this machine. The token stays in Claude Code's own store."),
      );
    }
    return h(`section.connect-panel${opts.focus === "claude" ? ".focus" : ""}`, null, head, body);
  }

  function keyPanel(k) {
    const cur = st.keys[k.id];
    const head = h(
      "div.connect-head",
      null,
      h("div.connect-icon", null, icon(k.icon)),
      h("div", null, h("div.connect-title", null, k.title), h("div.muted", null, k.sub)),
      cur && cur.present ? pill(cur.source === "environment" ? "From environment" : "Saved", "good", "check") : cur ? (k.required ? pill("Missing", "critical", "alert") : pill("Not set")) : null,
    );
    const body = h("div.connect-body");
    if (cur && cur.present) {
      body.append(
        h(
          "div.connect-row",
          { style: { justifyContent: "space-between" } },
          h("span", null, "Using key ", h("code", null, cur.hint)),
          cur.source === "saved" ? h("button.btn.ghost.sm.danger", { onclick: () => forgetKey(k) }, "Remove") : h("span.muted", { style: { fontSize: "12px" } }, `Set by ${k.env}`),
        ),
      );
    }
    append(body, [
      h(
        "div.connect-row",
        null,
        h("input.input.mono", {
          type: "password",
          placeholder: cur && cur.present ? "Replace with a new key" : k.placeholder,
          value: st.inputs[k.id] || "",
          autocomplete: "off",
          spellcheck: false,
          oninput: (e) => (st.inputs[k.id] = e.target.value),
          onkeydown: (e) => e.key === "Enter" && saveKey(k),
          "aria-label": k.title,
        }),
        h("button.btn", { onclick: () => saveKey(k), disabled: st.busy === k.id }, st.busy === k.id ? h("span.spinner") : null, "Save key"),
      ),
      st.errors[k.id] ? h("div.callout.critical", null, icon("alert"), h("div", null, st.errors[k.id])) : null,
      h("div.muted", { style: { fontSize: "12px" } }, k.note, " Stored owner-only in ", h("code", null, "~/.hephaestus/"), "; never shown again."),
    ]);
    return h(`section.connect-panel${opts.focus === k.id ? ".focus" : ""}`, null, head, body);
  }

  function draw() {
    const active = document.activeElement;
    const restore = active && active.closest && dlg.contains(active) && active.tagName === "INPUT" ? active.getAttribute("placeholder") : null;
    dlg.replaceChildren(
      h(
        "div.modal-head",
        null,
        h("div", null, h("h2", { id: "connect-title" }, "Connections"), h("div.muted", null, "Claude runs the orchestrator, reviewer and brief. Materials Project supplies structures and the competing phases for the hull.")),
        h("button.btn.ghost.icon", { onclick: close, "aria-label": "Close" }, icon("x")),
      ),
      h(
        "div.modal-body",
        null,
        h("div.modal-section", null, "Claude: pick one"),
        claudePanel(),
        keyPanel(KEYS[0]),
        h("div.modal-section", null, "Materials data"),
        keyPanel(KEYS[1]),
      ),
    );
    if (restore) {
      const again = [...dlg.querySelectorAll("input")].find((i) => i.getAttribute("placeholder") === restore);
      if (again) {
        again.focus();
        again.setSelectionRange(again.value.length, again.value.length);
      }
    }
  }

  document.body.append(dlg);
  draw();
  dlg.showModal();
  refresh();
}
