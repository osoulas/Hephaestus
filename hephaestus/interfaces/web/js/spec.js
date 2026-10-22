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

/**
 * Run settings by group; keys follow RunConfig. Each field is
 * [name, hint, unit, expert] where unit "min" shows seconds as minutes and expert fields hide
 * behind "Show expert settings".
 */
export const CONFIG_GROUPS = [
  {
    name: "Search budget",
    id: "budget",
    icon: "gauge",
    fields: {
      max_iterations: ["Review rounds", "Rounds of proposing, simulating and reviewing before it must submit. 0 for no limit.", "rounds"],
      max_tool_calls: ["Tool calls", "Calls before it is told to submit. 0 for no limit.", "calls"],
      time_budget_s: ["Time for the search", "When it runs out, the best candidate goes to the final re-check. 0 for no limit.", "min"],
      confidence_threshold: ["Stop threshold", "Code rubric and reviewer must both reach this to stop on target.", "/ 100", true],
      max_parallel_tools: ["Parallel tool calls", "Calls on different candidates that run at once.", "", true],
    },
    summary: (c) =>
      [
        c.max_iterations ? `${c.max_iterations} rounds` : "unlimited rounds",
        c.max_tool_calls ? `${c.max_tool_calls} tool calls` : "unlimited tool calls",
        c.time_budget_s ? `${minutes(c.time_budget_s)} min` : "no time limit",
      ].join(" · "),
  },
  {
    name: "Models",
    id: "models",
    icon: "atom",
    fields: {
      band_gap_model: ["Band gap model", "Used for the reference and every candidate, so all gaps share one scale."],
      potential: ["Screening potential", "Machine-learned potential for relaxations, hull and phonons."],
      final_potential: ["Final re-check potential", "A larger potential for the higher-fidelity re-check."],
      cross_check_potential: ["Second opinion", "Optional second potential in the final re-check."],
    },
    summary: (c, bandGapLabel) => `${bandGapLabel(c.band_gap_model)} · ${potentialName(c.potential)} → ${potentialName(c.final_potential)}`,
  },
  {
    name: "Stability checks",
    id: "checks",
    icon: "shield",
    fields: {
      run_hull: ["Energy above hull", "Check thermodynamic stability against competing phases relaxed with the same potential."],
      e_hull_max_eV_per_atom: ["Hull threshold", "Highest energy above hull that counts as stable. Leave empty to use the goal's (0.05 if it names none).", "eV/atom"],
      run_phonons: ["Phonons", "Check dynamic stability."],
      phonon_mode: ["Phonon detail", "Fast during screening; full is slower but samples more of the Brillouin zone."],
      run_final: ["Final re-check", "Re-check the winner and runner-up at higher fidelity."],
      final_time_budget_s: ["Time for the re-check", "Optional extras are skipped if they would not fit. 0 for no limit.", "min"],
      phonon_gate: ["Instability rule", "Mode mapping freezes soft modes in and re-relaxes; strict fails any imaginary mode.", "", true],
      distortion_threshold_meV: ["Mode-mapping threshold", "Energy drop that counts as a real instability.", "meV/atom", true],
      max_atoms: ["Largest cell", "Biggest supercell built.", "atoms", true],
    },
    summary: (c) =>
      [
        c.run_hull ? `hull ≤ ${c.e_hull_max_eV_per_atom ?? "goal's"} eV/atom`.replace("goal's eV/atom", "goal's threshold") : "no hull",
        c.run_phonons ? `${c.phonon_mode === "full" ? "full" : "fast"} phonons` : "no phonons",
        c.run_final ? `final re-check (${c.final_time_budget_s ? `${minutes(c.final_time_budget_s)} min` : "no time limit"})` : "no final re-check",
      ]
        .join(" · ")
        .replace(/^./, (x) => x.toUpperCase()),
  },
  {
    name: "Agent-built tools",
    id: "tools",
    icon: "wrench",
    fields: {
      custom_tools: ["When no tool fits", "If the goal needs a property no tool computes, the agent can propose one. You review the plan and the code before it runs."],
      disabled_capabilities: ["Turn off elastic", "Hide the built-in elastic calculations, so the agent has to build its own (for demos)."],
      max_custom_tools: ["Tools per run", "Most tools the agent may propose in one run.", "tools", true],
      max_tool_revisions: ["Change requests", "How many times you can ask for changes to one tool.", "per tool", true],
      capability_time_budget_s: ["Time to build a tool", "The agent's own time; time waiting for you does not count. 0 for no limit.", "min", true],
      custom_tool_timeout_s: ["Time per tool call", "A tool call that runs longer is stopped.", "min", true],
      tool_library_dir: ["Tool library folder", "Approved tools are saved here, relative to the server's working directory.", "", true],
    },
    summary: (c) =>
      [
        c.custom_tools === "ask" ? "Proposes a tool when none fits" : "Off",
        (c.disabled_capabilities || []).includes("elastic") ? "elastic turned off" : null,
      ]
        .filter(Boolean)
        .join(" · "),
  },
  {
    name: "Compute",
    id: "compute",
    icon: "cpu",
    fields: {
      device: ["Device", "Auto uses a GPU when one is available."],
      offline: ["Offline", "Use cached results only; make no network requests."],
      dtype: ["Screening precision", "float32 is about 3× faster on consumer GPUs.", "", true],
      final_dtype: ["Re-check precision", "Precision for the final re-check.", "", true],
      seed: ["Seed", "For disorder configurations and anything random.", "", true],
    },
    summary: (c) => `${c.device === "auto" ? "Auto device" : c.device === "cuda" ? "GPU" : "CPU"}${c.offline ? " · offline" : ""}`,
  },
];

/** Fields shown only while the switch they depend on is on. */
export const DEPENDS_ON = {
  e_hull_max_eV_per_atom: "run_hull",
  phonon_mode: "run_phonons",
  phonon_gate: "run_phonons",
  distortion_threshold_meV: "run_phonons",
  final_time_budget_s: "run_final",
  max_custom_tools: "custom_tools",
  max_tool_revisions: "custom_tools",
  capability_time_budget_s: "custom_tools",
  custom_tool_timeout_s: "custom_tools",
};

/** Budgets where 0 means no limit. */
export const LIMITS = new Set(["max_iterations", "max_tool_calls", "time_budget_s", "final_time_budget_s", "capability_time_budget_s"]);

function minutes(s) {
  return +(s / 60).toFixed(1);
}

/** Profiles: one click sets the budget and depth of checking. */
export const PROFILES = [
  {
    id: "quick",
    name: "Quick look",
    hint: "Two short rounds; useful to check a goal makes sense.",
    values: { max_iterations: 2, max_tool_calls: 50, time_budget_s: 480, final_time_budget_s: 300, phonon_mode: "fast" },
  },
  { id: "standard", name: "Standard", hint: "The defaults: three rounds and a 10-minute re-check.", values: {} },
  {
    id: "thorough",
    name: "Thorough",
    hint: "Five rounds, full phonons and more time for the re-check.",
    values: { max_iterations: 5, max_tool_calls: 150, time_budget_s: 1800, final_time_budget_s: 1200, phonon_mode: "full" },
  },
];

/** Readable names for setting values. */
export const VALUE_NAMES = {
  fast: "Fast",
  full: "Full",
  mode_map: "Mode mapping",
  strict: "Strict",
  auto: "Auto",
  cpu: "CPU",
  cuda: "GPU (CUDA)",
  ask: "Ask me",
  off: "Off",
};

/** Machine-learned potentials offered in the model menus. */
export const POTENTIALS = ["mace-mp-0-small", "mace-mp-0-medium", "mace-mp-0-large", "chgnet"];

/**
 * Display name of a potential: "mace-mp-0-small" -> "MACE-MP-0 small".
 * @param {string|null} p Potential ID.
 * @returns {string}
 */
export function potentialName(p) {
  if (!p) return "None";
  const m = String(p).match(/^mace-mp-0-(\w+)$/);
  if (m) return `MACE-MP-0 ${m[1]}`;
  return p === "chgnet" ? "CHGNet" : p;
}

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
