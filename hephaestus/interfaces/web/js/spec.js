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

/** Run settings by group, with display names and hints; keys follow RunConfig (OR-4). */
export const CONFIG_GROUPS = [
  {
    name: "Campaign",
    fields: {
      max_iterations: ["Review rounds", "Rounds before it must submit."],
      max_tool_calls: ["Tool-call budget", "Calls before it is told to submit."],
      time_budget_s: ["Campaign time budget", "Wall clock for the search. When it runs out, the best candidate goes to the final re-check.", "min"],
      max_parallel_tools: ["Parallel tool calls", "Calls on different candidates that run at once."],
      confidence_threshold: ["Stop threshold", "Code rubric and reviewer must both reach this to stop on target."],
    },
  },
  {
    name: "Science",
    fields: {
      band_gap_fidelity: ["Band gap fidelity", "HSE (±0.65 eV) or PBE (runs about 1.6 eV low)."],
      potential: ["Screening potential", "Machine-learned potential for every candidate."],
      final_potential: ["Final potential", "Larger potential for the final re-check."],
      cross_check_potential: ["Cross-check potential", "Optional second potential in the final re-check. Empty for none."],
      phonon_mode: ["Phonon mode", "Fast or full phonons during screening."],
      phonon_gate: ["Phonon gate", "Mode mapping freezes in soft modes; strict fails any imaginary mode."],
      distortion_threshold_meV: ["Mode-mapping threshold (meV/atom)", "Energy drop that counts as a real instability."],
      run_phonons: ["Run phonons", "Dynamic stability check."],
      max_atoms: ["Max atoms", "Largest supercell built."],
    },
  },
  {
    name: "Final re-check",
    fields: {
      run_final: ["Final re-check", "Re-check the winner and runner-up at higher fidelity."],
      final_time_budget_s: ["Final time budget", "Optional extras are skipped when they would not fit; the core re-check always runs.", "min"],
      final_dtype: ["Final precision", "Floating-point precision for the re-check."],
    },
  },
  {
    name: "Compute",
    fields: {
      device: ["Device", "auto uses a GPU when one is available."],
      dtype: ["Screening precision", "float32 is about 3× faster on consumer GPUs."],
      seed: ["Seed", "For disorder configurations and anything random."],
      offline: ["Offline", "Use cached results only; make no network requests."],
    },
  },
];

/** Settings chosen above the advanced section, not inside it. */
export const TOP_LEVEL_SETTINGS = new Set(["llm_backend", "llm_model", "require_novel"]);

/**
 * Display name of a setting.
 * @param {string} key RunConfig field.
 * @returns {string}
 */
export function configName(key) {
  for (const g of CONFIG_GROUPS) if (g.fields[key]) return g.fields[key][0];
  return label(key);
}
