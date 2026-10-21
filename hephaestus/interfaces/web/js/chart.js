// SVG charts (no library: the UI must work offline). Specs follow the dataviz method:
// 2px lines, >= 8px markers with a 2px surface ring, hairline solid grid, legend for >= 2 series,
// crosshair + tooltip, and a table view so no value hides behind hover.

import { h, s } from "./dom.js";

/**
 * Confidence across rounds: the code rubric (drives the stop rules) and the reviewer's own
 * confidence, with the stop threshold (UI-6, UI-15).
 * @param {object[]} rows `{round, score, llm}` per round.
 * @param {number} threshold Stop threshold (0-100).
 * @param {number} [maxRounds] Rounds allowed, so the x axis shows the whole budget.
 * @returns {HTMLElement}
 */
export function confidenceChart(rows, threshold, maxRounds = 3) {
  const series = [
    { key: "score", name: "Code rubric", color: "var(--series-1)" },
    { key: "llm", name: "Reviewer confidence", color: "var(--series-2)" },
  ];
  const W = 560;
  const H = 220;
  const m = { top: 14, right: 46, bottom: 28, left: 34 };
  const iw = W - m.left - m.right;
  const ih = H - m.top - m.bottom;
  const n = Math.max(rows.length, 1);
  const maxRound = Math.max(maxRounds, n, 2);
  const x = (r) => m.left + (maxRound === 1 ? iw / 2 : ((r - 1) / (maxRound - 1)) * iw);
  const y = (v) => m.top + ih - (v / 100) * ih;

  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Confidence by round" });
  for (const v of [0, 25, 50, 75, 100]) {
    svg.append(
      s("line", { class: v === 0 ? "baseline" : "grid-line", x1: m.left, x2: m.left + iw, y1: y(v), y2: y(v) }),
      s("text", { class: "tick", x: m.left - 8, y: y(v) + 4, "text-anchor": "end" }, v),
    );
  }
  for (let r = 1; r <= maxRound; r++) {
    svg.append(s("text", { class: "tick", x: x(r), y: H - 8, "text-anchor": "middle" }, `Round ${r}`));
  }
  svg.append(
    s("line", { class: "threshold", x1: m.left, x2: m.left + iw, y1: y(threshold), y2: y(threshold) }),
    s("text", { class: "threshold-label", x: m.left + iw + 6, y: y(threshold) + 4 }, `Stop ${threshold}`),
  );

  const ends = [];
  for (const se of series) {
    const pts = rows.filter((r) => r[se.key] != null).map((r) => [x(r.round), y(r[se.key]), r[se.key]]);
    if (!pts.length) continue;
    if (pts.length > 1) {
      svg.append(
        s("polyline", {
          points: pts.map((p) => `${p[0]},${p[1]}`).join(" "),
          fill: "none",
          stroke: se.color,
          "stroke-width": 2,
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
        }),
      );
    }
    for (const p of pts) {
      svg.append(s("circle", { cx: p[0], cy: p[1], r: 4, fill: se.color, stroke: "var(--chart-surface)", "stroke-width": 2 }));
    }
    const last = pts[pts.length - 1];
    ends.push({ x: last[0], y: last[1], v: last[2] });
  }
  // Direct end labels only when they do not collide; the legend and tooltip carry the rest.
  if (ends.length === 2 && Math.abs(ends[0].y - ends[1].y) < 14) ends.length = 0;
  for (const e of ends) svg.append(s("text", { class: "end-label", x: e.x + 9, y: e.y + 4 }, e.v));

  const cross = s("line", { class: "crosshair", y1: m.top, y2: m.top + ih, opacity: 0 });
  svg.append(cross);
  const tip = h("div.tooltip");
  const wrap = h("div.chart", null, svg, tip);
  const show = (row, px) => {
    cross.setAttribute("x1", x(row.round));
    cross.setAttribute("x2", x(row.round));
    cross.setAttribute("opacity", 1);
    tip.replaceChildren(
      h("div.tooltip-title", null, `Round ${row.round}${row.best ? ` · best ${row.best}` : ""}`),
      ...series.map((se) =>
        h("div.tooltip-row", null, h("span.legend-key", { style: { background: se.color } }), h("strong", null, row[se.key] ?? "–"), h("span", null, se.name)),
      ),
    );
    const rect = svg.getBoundingClientRect();
    const scale = rect.width / W;
    const left = Math.min(px * scale + 12, rect.width - 170);
    tip.style.left = `${Math.max(0, left)}px`;
    tip.style.top = `${m.top * scale}px`;
    tip.classList.add("show");
  };
  rows.forEach((row, i) => {
    const cx = x(row.round);
    const half = maxRound === 1 ? iw / 2 : iw / (maxRound - 1) / 2;
    const hit = s("rect", {
      class: "hit",
      x: cx - half,
      y: m.top,
      width: half * 2,
      height: ih,
      tabindex: 0,
      "aria-label": `Round ${row.round}: rubric ${row.score ?? "none"}, reviewer ${row.llm ?? "none"}`,
    });
    hit.addEventListener("pointerenter", () => show(row, cx));
    hit.addEventListener("focus", () => show(row, cx));
    svg.append(hit);
  });
  const hide = () => {
    cross.setAttribute("opacity", 0);
    tip.classList.remove("show");
  };
  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("focusout", hide);
  return wrap;
}

/**
 * Legend for the confidence chart (always shown: two series).
 * @returns {HTMLElement}
 */
export function confidenceLegend() {
  return h(
    "div.legend",
    null,
    h("span.legend-item", null, h("span.legend-key", { style: { background: "var(--series-1)" } }), "Code rubric (drives stop rules)"),
    h("span.legend-item", null, h("span.legend-key", { style: { background: "var(--series-2)" } }), "Reviewer confidence"),
    h("span.legend-item", null, h("span.legend-key.dashed"), "Stop threshold"),
  );
}

/**
 * Single-series column chart, e.g. papers per year. One colour (slot 1): the categories are
 * ordered years, the height carries the value.
 * @param {[string, number][]} data `[label, value]` pairs.
 * @param {string} unit Tooltip unit, e.g. "papers".
 * @returns {HTMLElement}
 */
export function columnChart(data, unit) {
  const W = 560;
  const H = 180;
  const m = { top: 18, right: 8, bottom: 24, left: 30 };
  const iw = W - m.left - m.right;
  const ih = H - m.top - m.bottom;
  const max = Math.max(...data.map((d) => d[1]), 1);
  const step = niceStep(max);
  const top = Math.ceil(max / step) * step;
  const band = iw / data.length;
  const bw = Math.min(24, band * 0.62);
  const y = (v) => m.top + ih - (v / top) * ih;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${unit} per year` });
  for (let v = 0; v <= top; v += step) {
    svg.append(
      s("line", { class: v === 0 ? "baseline" : "grid-line", x1: m.left, x2: m.left + iw, y1: y(v), y2: y(v) }),
      s("text", { class: "tick", x: m.left - 8, y: y(v) + 4, "text-anchor": "end" }, v),
    );
  }
  const tip = h("div.tooltip");
  const wrap = h("div.chart", null, svg, tip);
  const peak = data.reduce((a, d, i) => (d[1] > data[a][1] ? i : a), 0);
  data.forEach(([lab, v], i) => {
    const cx = m.left + band * i + band / 2;
    const bh = Math.max(0, y(0) - y(v));
    const r = Math.min(4, bh / 2, bw / 2);
    const x0 = cx - bw / 2;
    const y0 = y(v);
    // 4px rounded data end, square at the baseline.
    const d = `M${x0},${y(0)}V${y0 + r}Q${x0},${y0} ${x0 + r},${y0}H${x0 + bw - r}Q${x0 + bw},${y0} ${x0 + bw},${y0 + r}V${y(0)}Z`;
    const bar = s("path", { class: "bar", d, fill: "var(--series-1)" });
    svg.append(bar);
    if (data.length <= 14 || i % 2 === 0) svg.append(s("text", { class: "tick", x: cx, y: H - 7, "text-anchor": "middle" }, lab));
    if (i === peak) svg.append(s("text", { class: "end-label", x: cx, y: y0 - 6, "text-anchor": "middle" }, v));
    const hit = s("rect", { class: "hit", x: cx - band / 2, y: m.top, width: band, height: ih, tabindex: 0, "aria-label": `${lab}: ${v} ${unit}` });
    const show = () => {
      svg.querySelectorAll(".bar").forEach((b) => b.classList.toggle("dim", b !== bar));
      tip.replaceChildren(h("div.tooltip-title", null, lab), h("div.tooltip-row", null, h("strong", null, v), h("span", null, unit)));
      const rect = svg.getBoundingClientRect();
      const scale = rect.width / W;
      tip.style.left = `${Math.min(cx * scale + 10, rect.width - 160)}px`;
      tip.style.top = `${Math.max(0, y0 * scale - 50)}px`;
      tip.classList.add("show");
    };
    hit.addEventListener("pointerenter", show);
    hit.addEventListener("focus", show);
    svg.append(hit);
  });
  svg.addEventListener("pointerleave", () => {
    svg.querySelectorAll(".bar").forEach((b) => b.classList.remove("dim"));
    tip.classList.remove("show");
  });
  return wrap;
}

/**
 * A round tick step for an axis maximum.
 * @param {number} max Largest value.
 * @returns {number}
 */
function niceStep(max) {
  const raw = max / 4;
  const mag = 10 ** Math.floor(Math.log10(raw || 1));
  for (const f of [1, 2, 2.5, 5, 10]) if (raw <= f * mag) return f * mag;
  return 10 * mag;
}

/**
 * Circular score (0-100) for a review.
 * @param {number|null} v Score.
 * @param {string} color Stroke colour.
 * @returns {HTMLElement}
 */
export function scoreRing(v, color) {
  const r = 24;
  const c = 2 * Math.PI * r;
  const frac = v == null ? 0 : Math.max(0, Math.min(1, v / 100));
  return h(
    "div.score-ring",
    null,
    s(
      "svg",
      { viewBox: "0 0 56 56", "aria-hidden": "true" },
      s("circle", { class: "track", cx: 28, cy: 28, r, fill: "none", "stroke-width": 5 }),
      s("circle", { class: "fill", cx: 28, cy: 28, r, fill: "none", stroke: color, "stroke-width": 5, "stroke-dasharray": c, "stroke-dashoffset": c * (1 - frac) }),
    ),
    h("div.val", null, v ?? "–"),
  );
}
