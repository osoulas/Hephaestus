// DOM helpers, icons and formatters shared by every view.
// Text always goes in through text nodes: run data (paper titles, LLM text) is untrusted.

/**
 * Create an element.
 * @param {string} tag Tag name, optionally with classes: "div.card.hero".
 * @param {object|null} attrs Attributes; `on*` keys add listeners, `class` appends classes.
 * @param {...any} children Nodes, strings, numbers, arrays, or null/false (skipped).
 * @returns {HTMLElement}
 */
export function h(tag, attrs, ...children) {
  const [name, ...classes] = tag.split(".");
  const el = document.createElement(name || "div");
  if (classes.length) el.className = classes.join(" ");
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === "class") el.className = [el.className, v].filter(Boolean).join(" ");
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k === "value") el.value = v; // a property: the attribute does not fill a textarea
    else if (k in el && typeof v !== "string") el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  append(el, children);
  return el;
}

/**
 * Append children to a node, flattening arrays and turning strings into text nodes.
 * @param {Node} el Parent.
 * @param {any[]} children Children.
 */
export function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false || c === true) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * Create an SVG element.
 * @param {string} tag SVG tag.
 * @param {object} attrs Attributes.
 * @param {...any} children Child nodes.
 * @returns {SVGElement}
 */
export function s(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else el.setAttribute(k, v);
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

// Stroke icons, 24-unit grid.
const ICONS = {
  plus: "M12 5v14M5 12h14",
  flame:
    "M12 3c1 3 4.5 4.8 4.5 9.2A4.5 4.5 0 0 1 12 16.7a4.5 4.5 0 0 1-4.5-4.5c0-1.9.9-3.2 2-4.2.2 1.5 1 2.5 1.9 2.8-.6-3.3.2-6.1.6-7.8Z M5 20h14",
  play: "M7 5v14l11-7z",
  stop: "M7 7h10v10H7z",
  resume: "M4 12a8 8 0 1 0 2.3-5.7M4 4v4h4",
  replay: "M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5M10 9l5 3-5 3z",
  overview: "M4 4h7v7H4zM13 4h7v4h-7zM13 10h7v10h-7zM4 13h7v7H4z",
  log: "M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01",
  book: "M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2zM4 19V5M8 7h7",
  flask: "M9 3h6M10 3v6L4.5 18.5A1.7 1.7 0 0 0 6 21h12a1.7 1.7 0 0 0 1.5-2.5L14 9V3M7.5 14h9",
  cube: "M12 2 3 7v10l9 5 9-5V7zM3 7l9 5 9-5M12 12v10",
  shield: "M12 3 5 6v6c0 4.4 3 7.8 7 9 4-1.2 7-4.6 7-9V6zM9 12l2 2 4-4",
  doc: "M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h6",
  sliders: "M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0M14 4v4M8 10v4M16 16v4",
  search: "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM20 20l-4-4",
  database: "M12 3c4.4 0 8 1.3 8 3s-3.6 3-8 3-8-1.3-8-3 3.6-3 8-3zM4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
  atom: "M12 12h.01M12 3c4 0 7 4 7 9s-3 9-7 9-7-4-7-9 3-9 7-9zM3.5 7.5c2-3.5 7-3 11 0s7 8 5 11.5-7 3-11 0-7-8-5-11.5z",
  wave: "M2 12c2-5 4-5 6 0s4 5 6 0 4-5 6 0 2 3 2 3",
  gauge: "M12 14l4-4M4 18a9 9 0 1 1 16 0",
  sparkles: "M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8zM19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z",
  message: "M4 5h16v11H8l-4 4z",
  check: "M5 12.5l4.5 4.5L19 7.5",
  x: "M6 6l12 12M18 6 6 18",
  alert: "M12 4 2.5 20h19zM12 10v4M12 17h.01",
  info: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 11v5M12 8h.01",
  chevron: "M9 6l6 6-6 6",
  download: "M12 4v11M7 10l5 5 5-5M5 20h14",
  printer: "M7 9V3h10v6M7 17H4v-7h16v7h-3M7 14h10v7H7z",
  copy: "M9 9h11v11H9zM5 15H4V4h11v1",
  sun: "M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10zM12 1v2M12 21v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M1 12h2M21 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4",
  moon: "M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z",
  monitor: "M3 4h18v12H3zM8 20h8M12 16v4",
  star: "M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1-4.4-4.3 6.1-.9z",
  key: "M15 7a4 4 0 1 0-3.8 5.3L4 19.5V21h3v-2h2v-2h2l1.2-1.2A4 4 0 0 0 15 7zM16 8h.01",
  terminal: "M4 5h16v14H4zM7 9l3 3-3 3M12 15h5",
  link: "M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1",
  external: "M14 4h6v6M20 4l-9 9M18 14v6H4V6h6",
  reset: "M3 12a9 9 0 1 0 9-9 9.7 9.7 0 0 0-6.7 2.7L3 8M3 3v5h5",
  rotate: "M21 12a9 9 0 1 1-3-6.7L21 8M21 3v5h-5",
  box: "M4 4h16v16H4z",
  cpu: "M7 7h10v10H7zM9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3",
  arrowRight: "M5 12h14M13 6l6 6-6 6",
  trend: "M3 17l6-6 4 4 8-8M15 7h6v6",
  gate: "M5 21V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16M3 21h18M9 11h.01",
};

/**
 * An inline stroke icon.
 * @param {string} name Key of ICONS.
 * @param {string} [cls] Extra class.
 * @returns {SVGElement}
 */
export function icon(name, cls) {
  const d = ICONS[name] || ICONS.info;
  return s(
    "svg",
    { viewBox: "0 0 24 24", class: `icon ${cls || ""}`.trim(), "aria-hidden": "true" },
    s("path", { d }),
  );
}

// The CLI's logo (interfaces/cli/theme.py): HEPHAESTUS in ANSI Shadow block letters graded from
// molten red to gold, with an anvil in iron to the right. Drawn as SVG, one terminal cell per unit
// (cells are twice as tall as wide): full blocks become filled cells, box-drawing shadows thin lines.
const LETTERS = {
  H: ["██╗  ██╗", "██║  ██║", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"],
  E: ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"],
  P: ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔═══╝ ", "██║     ", "╚═╝     "],
  A: [" █████╗ ", "██╔══██╗", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"],
  S: ["███████╗", "██╔════╝", "███████╗", "╚════██║", "███████║", "╚══════╝"],
  T: ["████████╗", "╚══██╔══╝", "   ██║   ", "   ██║   ", "   ██║   ", "   ╚═╝   "],
  U: ["██╗   ██╗", "██║   ██║", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
};
const ANVIL = [
  "████████████████████╗",
  "╚════█████████████╔═╝",
  "     ╚══███████╔══╝  ",
  "        ███████║     ",
  "    ███████████████╗ ",
  "    ╚══════════════╝ ",
];
const ANVIL_GAP = 3;

// Half-cell strokes for each box-drawing character: [x1, y1, x2, y2] in cell units (cell 1 × 2).
const STROKES = {
  "═": [[0, 1, 1, 1]],
  "║": [[0.5, 0, 0.5, 2]],
  "╗": [[0, 1, 0.5, 1], [0.5, 1, 0.5, 2]],
  "╔": [[1, 1, 0.5, 1], [0.5, 1, 0.5, 2]],
  "╝": [[0, 1, 0.5, 1], [0.5, 1, 0.5, 0]],
  "╚": [[1, 1, 0.5, 1], [0.5, 1, 0.5, 0]],
};

/**
 * Draw rows of ANSI Shadow art into an SVG group.
 * @param {string[]} rows Six rows of block and box-drawing characters.
 * @param {number} x0 Left column.
 * @param {string} fillVar CSS variable prefix for block rows, e.g. "--molten" (rows 1-6).
 * @param {string} shadowVar CSS variable for the shadow strokes.
 * @returns {SVGElement}
 */
function shadowArt(rows, x0, fillVar, shadowVar) {
  const g = s("g");
  const lines = [];
  rows.forEach((row, r) => {
    const y = r * 2;
    let run = null; // merge each run of blocks into one rect, so no seams show between cells
    [...row, " "].forEach((ch, c) => {
      const x = x0 + c;
      if (ch === "█") {
        if (run === null) run = x;
        return;
      }
      if (run !== null) {
        g.append(s("rect", { x: run, y: y - 0.02, width: x - run, height: 2.06, fill: `var(${fillVar}-${r + 1})` }));
        run = null;
      }
      if (STROKES[ch]) for (const [a, b, e, f] of STROKES[ch]) lines.push(`M${x + a} ${y + b}L${x + e} ${y + f}`);
    });
  });
  g.prepend(s("path", { d: lines.join(""), stroke: `var(${shadowVar})`, "stroke-width": 0.42, "stroke-linecap": "square", fill: "none" }));
  return g;
}

/**
 * The HEPHAESTUS wordmark with its anvil, as in the CLI banner.
 * @param {{anvil?: boolean}} [opts] Draw the anvil (default true).
 * @returns {SVGElement}
 */
export function wordmark(opts = {}) {
  const anvil = opts.anvil !== false;
  const word = "HEPHAESTUS";
  const letterRows = [0, 1, 2, 3, 4, 5].map((r) => [...word].map((ch) => LETTERS[ch][r]).join(""));
  const textW = letterRows[0].length;
  const width = textW + (anvil ? ANVIL_GAP + ANVIL[0].length : 0);
  return s(
    "svg",
    { viewBox: `-0.3 -0.3 ${width + 0.6} 12.6`, class: "wordmark", role: "img", "aria-label": "Hephaestus" },
    shadowArt(letterRows, 0, "--molten", "--scorch"),
    anvil ? shadowArt(ANVIL, textW + ANVIL_GAP, "--iron", "--iron-shadow") : null,
  );
}

// ------------------------------------------------------------------ formatters

/**
 * Render a chemical formula with subscripted amounts: "SrTi0.75Fe0.25O3".
 * @param {string} f Formula.
 * @returns {HTMLElement}
 */
export function formula(f) {
  const el = h("span.formula");
  if (!f) return el;
  for (const part of String(f).match(/[A-Z][a-z]?|[\d.]+|[^A-Za-z\d.]+/g) || []) {
    if (/^[\d.]+$/.test(part)) el.append(h("sub", null, part));
    else el.append(document.createTextNode(part));
  }
  return el;
}

/**
 * Display form of a candidate ID: "c1-2" becomes "1-2" (round 1, candidate 2).
 * The engine keeps "c1-2"; only what the UI shows changes.
 * @param {string|null|undefined} id Candidate ID.
 * @returns {string}
 */
export function cid(id) {
  if (!id) return "–";
  if (id === "ref") return "reference";
  const m = String(id).match(/^c(\d+)-(\d+)(.*)$/);
  return m ? `${m[1]}-${m[2]}${m[3]}` : String(id);
}

/**
 * Long form of a candidate ID for tooltips: "Round 1 · Candidate 2".
 * @param {string} id Candidate ID.
 * @returns {string}
 */
export function cidLong(id) {
  const m = String(id || "").match(/^c(\d+)-(\d+)/);
  return m ? `Round ${m[1]} · Candidate ${m[2]}` : String(id || "");
}

/**
 * A candidate ID as shown in the UI, with its long form on hover.
 * @param {string} id Candidate ID.
 * @param {string} [cls] Element and classes, e.g. "span.cand-id".
 * @returns {HTMLElement}
 */
export function cidTag(id, cls = "span.cid") {
  return h(cls, { title: cidLong(id) }, cid(id));
}

/**
 * Format a number with sensible precision.
 * @param {number|null|undefined} v Value.
 * @param {number} [digits] Significant decimals.
 * @returns {string}
 */
export function num(v, digits = 2) {
  if (v == null || Number.isNaN(v)) return "–";
  if (typeof v !== "number") return String(v);
  const abs = Math.abs(v);
  if (abs !== 0 && (abs < 0.001 || abs >= 1e5)) return v.toExponential(1);
  if (abs !== 0 && abs < 0.01) return v.toFixed(Math.max(digits, 3));
  return v.toFixed(abs >= 100 ? 0 : digits).replace(/\.?0+$/, (m) => (m.startsWith(".") ? "" : m));
}

/**
 * Value with uncertainty, e.g. "2.2 ± 0.65 eV". Says so when no uncertainty is known (SYS-4).
 * @param {number} v Value.
 * @param {number|null} u Uncertainty.
 * @param {string} [units] Units.
 * @returns {string}
 */
export function pm(v, u, units) {
  if (v == null) return "–";
  const unit = units ? ` ${units}` : "";
  return u == null ? `${num(v)}${unit}` : `${num(v)} ± ${num(u)}${unit}`;
}

/**
 * Human duration: 5 s, 3 min 12 s, 1 h 4 min.
 * @param {number} sec Seconds.
 * @returns {string}
 */
export function duration(sec) {
  if (sec == null || !isFinite(sec)) return "–";
  sec = Math.max(0, Math.round(sec));
  if (sec < 60) return `${sec} s`;
  const m = Math.floor(sec / 60);
  if (m < 60) return `${m} min ${sec % 60} s`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

/**
 * Relative time: "just now", "5 min ago", "3 days ago", else a date.
 * @param {string|Date} t Time.
 * @returns {string}
 */
export function ago(t) {
  const d = new Date(t);
  const sec = (Date.now() - d.getTime()) / 1000;
  if (sec < 45) return "just now";
  if (sec < 3600) return `${Math.round(sec / 60)} min ago`;
  if (sec < 86400) return `${Math.round(sec / 3600)} h ago`;
  if (sec < 86400 * 7) return `${Math.round(sec / 86400)} d ago`;
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

/**
 * Sentence-case a snake_case name: "band_gap" -> "Band gap".
 * @param {string} k Name.
 * @returns {string}
 */
export function label(k) {
  if (!k) return "";
  const t = String(k).replace(/_/g, " ");
  return t.charAt(0).toUpperCase() + t.slice(1);
}

/**
 * Text clamped to a few lines, with "Show more" when it is longer.
 * @param {any} content Text or nodes.
 * @param {number} [lines] Lines shown while collapsed.
 * @returns {HTMLElement}
 */
export function clamp(content, lines = 3) {
  const text = h("div.clamp-text.clamped", null, content);
  text.style.setProperty("--lines", String(lines));
  const btn = h("button.link.clamp-toggle", { type: "button", hidden: true }, "Show more");
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const open = !text.classList.toggle("clamped");
    btn.textContent = open ? "Show less" : "Show more";
  });
  requestAnimationFrame(() => {
    if (text.scrollHeight > text.clientHeight + 1) btn.hidden = false;
  });
  return h("div.clamp", null, text, btn);
}

/**
 * A status pill.
 * @param {string} text Label.
 * @param {string} [kind] good | warning | serious | critical | info | live.
 * @param {string} [iconName] Optional icon.
 * @returns {HTMLElement}
 */
export function pill(text, kind, iconName) {
  return h(`span.pill${kind ? "." + kind : ""}`, null, iconName ? icon(iconName) : null, text);
}

/**
 * A pass/fail mark with icon and label, so status is never colour alone.
 * @param {boolean|null|undefined} ok Result; null means not computed.
 * @param {string} [yes] Label when true.
 * @param {string} [no] Label when false.
 * @returns {HTMLElement}
 */
export function check(ok, yes = "Pass", no = "Fail") {
  if (ok == null) return h("span.check.unknown", null, icon("info"), "Not run");
  return h(`span.check.${ok ? "good" : "bad"}`, null, icon(ok ? "check" : "x"), ok ? yes : no);
}

/**
 * Show a short-lived message.
 * @param {string} text Message.
 * @param {string} [kind] "critical" for errors.
 */
export function toast(text, kind) {
  // A modal dialog sits in the top layer, above the page's toasts: show them inside it instead.
  const dlg = document.querySelector("dialog[open]");
  let root = document.getElementById("toasts");
  if (dlg) {
    root = dlg.querySelector(".toasts") || dlg.appendChild(h("div.toasts"));
  }
  const el = h(`div.toast${kind ? "." + kind : ""}`, { role: "status" }, icon(kind ? "alert" : "check"), text);
  root.append(el);
  setTimeout(() => el.remove(), kind ? 7000 : 3500);
}

/**
 * An empty state.
 * @param {string} iconName Icon.
 * @param {string} title Title.
 * @param {string} [text] Body.
 * @returns {HTMLElement}
 */
export function empty(iconName, title, text) {
  return h("div.empty", null, icon(iconName), h("div.empty-title", null, title), text ? h("div", null, text) : null);
}

/**
 * Download text as a file.
 * @param {string} name File name.
 * @param {string} text Content.
 * @param {string} [type] MIME type.
 */
export function download(name, text, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h("a", { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Read a value from localStorage, which may be unavailable.
 * @param {string} key Key.
 * @param {any} fallback Default.
 * @returns {any}
 */
export function stored(key, fallback) {
  try {
    const v = localStorage.getItem(key);
    return v == null ? fallback : JSON.parse(v);
  } catch (e) {
    return fallback;
  }
}

/**
 * Write a value to localStorage when possible.
 * @param {string} key Key.
 * @param {any} value Value.
 */
export function store(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (e) {
    /* private mode: per-viewer conveniences only */
  }
}
