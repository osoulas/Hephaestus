// Turns the rendered brief (plain Markdown HTML) into a designed document: a summary of tiles,
// one card per section with its own icon and accent, a timeline for the synthesis steps, coloured
// pass/stop criteria, severity and yes/no pills, "inferred" tags and reference cards.
// It only rearranges and decorates what the Markdown already says; no text is added or changed.

import { h, icon } from "./dom.js";
import { slug } from "./markdown.js";

const SECTIONS = {
  "recommended-composition-and-structure": { icon: "star", tone: "ember" },
  "predicted-properties-against-target": { icon: "gauge", tone: "blue" },
  "physical-rationale": { icon: "atom", tone: "violet" },
  "validation-verdict": { icon: "shield", tone: "green" },
  "synthesis-route": { icon: "flask", tone: "ember" },
  "characterisation-plan": { icon: "search", tone: "blue" },
  "risks-and-fallbacks": { icon: "alert", tone: "red" },
  "runner-up-candidates": { icon: "flask", tone: "slate" },
  "iteration-history": { icon: "log", tone: "slate" },
  "literature-support": { icon: "book", tone: "blue" },
  "scope-and-assumptions": { icon: "info", tone: "slate" },
};

const GLANCE_ICONS = {
  "what to make": "flask",
  why: "trend",
  "stable?": "shield",
  "final re-check": "gate",
  "first experiment": "play",
};

/**
 * Decorate a rendered brief in place.
 * @param {HTMLElement} doc The article holding the rendered Markdown.
 */
export function decorateBrief(doc) {
  doc.classList.add("brief");
  titleBlock(doc);
  glance(doc);
  const blockquote = doc.querySelector(":scope > blockquote");
  if (blockquote) blockquote.classList.add("brief-alert");
  wrapSections(doc);
  for (const sec of doc.querySelectorAll(".bsec")) decorateSection(sec);
  tagWords(doc);
}

function titleBlock(doc) {
  const h1 = doc.querySelector(":scope > h1");
  if (!h1) return;
  const name = h1.innerHTML.replace(/^Lab brief:\s*/, "");
  const head = h("header.brief-head", null, h("div.brief-eyebrow", null, icon("doc"), "Lab brief"));
  const title = h("h1.brief-title");
  title.innerHTML = name; // already escaped Markdown output
  head.append(title);
  h1.replaceWith(head);
}

function glance(doc) {
  const head = doc.querySelector(".brief-head");
  let list = head ? head.nextElementSibling : null;
  while (list && list.tagName === "BLOCKQUOTE") list = list.nextElementSibling;
  if (!list || list.tagName !== "UL") return;
  const grid = h("div.glance");
  for (const li of list.querySelectorAll(":scope > li")) {
    const strong = li.querySelector(":scope > strong");
    if (!strong) continue;
    const key = strong.textContent.replace(/:$/, "").trim();
    strong.remove();
    const value = h("div.glance-value");
    value.innerHTML = li.innerHTML.trim();
    const lk = key.toLowerCase();
    let tone = "";
    if (lk === "final re-check") {
      const t = value.textContent.toLowerCase();
      tone = t.startsWith("confirmed") ? "good" : t.startsWith("overturned") ? "critical" : "warning";
    }
    if (lk === "stable?") tone = /unstable/.test(value.textContent) ? "warning" : "good";
    grid.append(
      h(
        `div.glance-tile${lk === "first experiment" ? ".wide.next" : ""}${tone ? "." + tone : ""}`,
        null,
        h("div.glance-key", null, icon(GLANCE_ICONS[lk] || "info"), key),
        value,
      ),
    );
  }
  list.replaceWith(grid);
}

function wrapSections(doc) {
  const nodes = [...doc.childNodes];
  let current = null;
  let n = 0;
  for (const node of nodes) {
    if (node.nodeType === 1 && node.tagName === "H2") {
      n += 1;
      const id = node.id || slug(node.textContent);
      const meta = SECTIONS[id] || { icon: "doc", tone: "slate" };
      current = h(`section.bsec.tone-${meta.tone}`, { dataset: { kind: id } });
      const head = h("div.bsec-head", null, h("span.bsec-icon", null, icon(meta.icon)), h("h2", { id }, node.textContent));
      current.append(head);
      node.replaceWith(current);
      continue;
    }
    if (current) current.append(node);
  }
}

function decorateSection(sec) {
  const kind = sec.dataset.kind;
  for (const td of sec.querySelectorAll("td")) {
    const t = td.textContent.trim();
    if (/^yes([,.;]|$)/i.test(t)) td.classList.add("cell-yes");
    else if (/^no([,.;]|$)/i.test(t)) td.classList.add("cell-no");
    else if (/^(critical|major|minor)$/i.test(t)) td.replaceChildren(h(`span.sev.sev-${t.toLowerCase()}`, null, t));
  }
  for (const p of sec.querySelectorAll(":scope > p")) {
    const lead = p.querySelector(":scope > strong:first-child");
    const key = lead ? lead.textContent.trim().toLowerCase() : "";
    if (key.startsWith("approach")) p.classList.add("callout-approach");
    else if (key.startsWith("watch for")) p.classList.add("callout-watch");
    else if (/^(confirmed|weakened|overturned)\./.test(key)) p.classList.add("verdict-line", `v-${key.split(".")[0]}`);
  }
  if (kind === "synthesis-route" || kind === "characterisation-plan") {
    for (const ol of sec.querySelectorAll(":scope > ol")) ol.classList.add("steps");
  }
  if (kind === "literature-support") {
    for (const ol of sec.querySelectorAll(":scope > ol")) ol.classList.add("refs");
  }
  if (kind === "scope-and-assumptions") sec.classList.add("quiet");
}

// Coloured words: "inferred" as a tag; "pass if" / "stop if" in green / red.
function tagWords(doc) {
  const walker = document.createTreeWalker(doc, NodeFilter.SHOW_TEXT);
  const targets = [];
  while (walker.nextNode()) {
    const t = walker.currentNode;
    if (/\binferred\b|\b(pass|stop) if\b/i.test(t.nodeValue) && !t.parentElement.closest("a, code, .glance-key")) targets.push(t);
  }
  for (const t of targets) {
    const frag = document.createDocumentFragment();
    let last = 0;
    const re = /\(inferred\)|\binferred\b|\b(pass|stop) if\b/gi; // a bare "(inferred)" becomes just the tag
    let m;
    while ((m = re.exec(t.nodeValue))) {
      frag.append(t.nodeValue.slice(last, m.index));
      const word = m[0].toLowerCase();
      if (word === "inferred" || word === "(inferred)") frag.append(h("span.tag-inferred", { title: "Not from a cited source: an inference" }, "inferred"));
      else frag.append(h(`span.${word.startsWith("pass") ? "crit-pass" : "crit-stop"}`, null, m[0]));
      last = m.index + m[0].length;
    }
    frag.append(t.nodeValue.slice(last));
    t.replaceWith(frag);
  }
}
