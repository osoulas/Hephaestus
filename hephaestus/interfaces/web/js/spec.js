// The parsed spec and model names, shared by the composer and the run view.

import { formula, h, icon, label } from "./dom.js";

/**
 * The parsed spec as a grid of facts.
 * @param {object} spec SearchSpec.
 * @param {object} [cfg] RunConfig.
 * @returns {HTMLElement}
 */
export function specGrid(spec, cfg) {
  const cell = (k, v) => h("div.spec-cell", null, h("div.k", null, k), h("div.v", null, v));
  const target = targetText(spec);
  const thr = spec.stability_thresholds || {};
  const grid = h(
    "div.spec-grid",
    null,
    cell("Material family", spec.material_family || "–"),
    cell("Target", target),
    cell("Thermodynamic stability", `≤ ${thr.e_hull_max_eV_per_atom ?? 0.05} eV/atom above hull`),
    cell("Dynamic stability", "Phonons with mode mapping"),
    cell("Excluded elements", spec.excluded_elements && spec.excluded_elements.length ? spec.excluded_elements.join(", ") : "None"),
    cell("Conditions", `${spec.conditions?.temperature_K ?? 300} K, ${spec.conditions?.pressure_GPa ?? 0.0001} GPa`),
  );
  const lists = h("div.card-body", { style: { display: "grid", gap: "12px" } });
  if (spec.hard_constraints && spec.hard_constraints.length) {
    lists.append(h("div", null, h("div.field-label", null, "Hard constraints"), h("ul.list-plain", { style: { marginTop: "6px" } }, spec.hard_constraints.map((c) => h("li", null, h("span.check.good", null, icon("check")), h("span", null, c))))));
  }
  if (spec.assumptions && spec.assumptions.length) {
    lists.append(h("div", null, h("div.field-label", null, "Assumptions"), h("ul.list-plain", { style: { marginTop: "6px" } }, spec.assumptions.map((a) => h("li", null, icon("info"), h("span.secondary", null, a))))));
  }
  if (cfg && cfg.require_novel) lists.append(h("div.callout.info", null, icon("star"), "Novel materials only: known compositions are rejected."));
  return h("div", null, grid, lists.childNodes.length ? lists : null);
}

/**
 * The target in words, e.g. "Band gap below SrTiO3 (same model)".
 * @param {object} spec SearchSpec.
 * @returns {Node}
 */
export function targetText(spec) {
  const prop = label(spec.target_property);
  const dir = { minimise: "below", maximise: "above", hit: "near" }[spec.direction] || spec.direction;
  if (spec.reference_material) {
    const ref = spec.reference_value ? ` (${spec.reference_value.value.toFixed(2)} ${spec.reference_value.units})` : "";
    return h("span", null, `${prop} ${dir} `, formula(spec.reference_material), ref);
  }
  if (spec.target_value != null) {
    const tol = spec.direction === "hit" && spec.target_tolerance ? ` ± ${spec.target_tolerance}` : "";
    return h("span", null, `${prop} ${dir} ${spec.target_value}${tol} ${spec.target_units}`);
  }
  return h("span", null, `${prop} (${spec.direction})`);
}

/**
 * Display name of a model ID.
 * @param {string} m Model ID.
 * @returns {string}
 */
export function modelName(m) {
  const t = String(m || "").match(/^claude-(\w+)-(\d+)-(\d+)/);
  return t ? `${t[1][0].toUpperCase()}${t[1].slice(1)} ${t[2]}.${t[3]}` : m;
}
