# Hephaestus tools reference

The orchestrator LLM acts **only** through the tools below. This file is the contract between the harness
(person A) and the tool owners (B and C). It covers the name, purpose, owner, inputs, outputs and which
verdicts code computes, for every tool. Changing a tool's name, inputs or result keys needs all three people
to agree, because the harness prompts, the UI renderers and the tests depend on them.

Schemas live in `hephaestus/tools/registry.py` as `TOOLS`: a list of `{name, description, input_schema}`
with `strict: true` and `additionalProperties: false`. That list is what both LLM backends see. Handlers
live in the owning module and are registered by name.

## How tools work

**Signature.** Every handler is `handler(ws: Workspace, **args) -> dict`.

- `ws` is the per-run `Workspace` (see below). Handlers read and update it; they keep no state of their own.
- **Results** are JSON-serialisable dicts. Keep them compact: the LLM reads every byte, on every later turn.
- **Errors** raise `ToolError("message")`. The dispatcher (`registry.execute`) turns that into
  `{"error": "..."}` with `is_error: true` for the LLM. Any other exception is caught and reported the same
  way: a tool never crashes the run.
- **Verdicts are code.** Results include booleans such as `thermo_stable` and `dyn_stable`, computed by
  code from the thresholds in `SearchSpec`. The LLM reads them and cannot set them (SC-1, SE-13).
- **Caching.** Every external call and every relaxation is cached by its exact inputs (`core/store.py`), so
  repeating a tool call is cheap and the demo works offline.
- **Budget.** Each call counts towards `RunConfig.max_tool_calls`. Compute tools also add to `ws.compute_s`.
  A budget of 0 (`max_iterations`, `max_tool_calls`, `time_budget_s`, `final_time_budget_s`) means no limit.
- **Events.** The harness, not the tool, emits a `ProgressEvent` before and after each call. The payload
  carries `tool`, `input`, `result`, plus flat fields for the UI (see *Event payloads*).

### The Workspace (per-run state)

`Workspace` (in `tools/registry.py`) holds everything the tools share in one run. It is snapshotted to
`runs/<id>/workspace.json` after every call, so a run can be resumed (OR-6).

| Field | Contents |
| --- | --- |
| `spec`, `cfg` | `SearchSpec` and `RunConfig` for the run |
| `round` | current review round, starting at 1; `request_review` advances it |
| `candidates` | `candidate_id -> CandidateState` (below) |
| `papers` | every `Paper` retrieved this run, by `paper_id` (LM-9: never fetched twice) |
| `insights` | `Insight`s recorded with `note_evidence` |
| `trend`, `queries` | field-activity trend; every literature query made |
| `tool_calls`, `compute_s` | budget counters |
| `submitted` | the accepted submission, or `None` |

`CandidateState` holds:
- `candidate` (`Candidate`) and the starting `record` (`StructureRecord`)
- the current `structure`, which is the relaxed one, or the distorted one after mode mapping
- `energy_eV_per_atom`, `converged` and the `potential` used
- `e_hull`, `e_form`, `decomposes_to` and `hull_stale` (set true when mode mapping changed the structure
  after the hull was computed)
- `phonon` (`PhononResult`) and `phonon_details`
- `predictions` (property name → `PropertyPrediction`)
- `md`, `errors` and `round`

`ws.sim_result(cid)` assembles a `SimulationResult` from these. `ws.facts(cid)` returns the code-computed
checks: the target check and both stability verdicts, as used by the gate, the reviewer and the UI.

## Tool catalogue

Each section below gives the tool's module; owners are listed in `CLAUDE.md`.

### Literature: owner C, `tools/literature.py`

Valency Bond is the only literature source (LM-1); see `docs/valency_tools.md` for its live schemas.

#### `search_literature`
Search Valency.
- `mode=semantic` searches by concept.
- `mode=abstract` is keyword search in abstracts; use it to find failed attempts such as secondary phases
  or solubility limits.
- `mode=citations` finds highly cited foundational work.

| Input | Type | Notes |
| --- | --- | --- |
| `query` | string | |
| `mode` | `semantic` \| `abstract` \| `citations` | |
| `limit` | integer | capped at 15 |

**Returns:** `{n, papers: [{paper_id, title, year, citations, source, abstract (≤ 700 chars)}]}`.
Papers are added to `ws.papers`.

#### `related_papers`
Papers similar to, or citing, a given paper. Citing papers catch later revisions of headline numbers.

| Input | Type | Notes |
| --- | --- | --- |
| `paper_id` | string | |
| `relation` | `similar` \| `citing` | |
| `limit` | integer | capped at 10 |

**Returns:** `{n, papers: [...]}`, same shape as above, with abstracts ≤ 400 chars.

#### `get_paper`
Full record for one paper. **Input:** `paper_id`.
**Returns:** `{paper_id, title, year, citations, source, abstract (≤ 3000), doi, url, authors}`.

#### `publication_trend`
Papers per year for a keyword query (LM-8). **Input:** `query`.
**Returns:** `{query, papers_per_year: {year: n}}`. Missing years are filled with 0.

#### `note_evidence`
Record a finding for the reviewer and the brief.

| Input | Type | Notes |
| --- | --- | --- |
| `paper_id` | string | must already be in `ws.papers` |
| `composition` | string \| null | normalised to pymatgen notation |
| `property` | string | |
| `value` | number \| null | **dropped by code unless the number appears in the abstract** (LM-14) |
| `units` | string \| null | |
| `conditions` | string | |
| `finding_type` | `positive` \| `negative` \| `mechanism` \| `synthesis` | |
| `summary` | string | |

**Returns:** `{recorded, value_dropped_not_in_abstract, n_insights}`.

### Materials Project: owner C, `tools/materials_project.py`

#### `lookup_materials_project`
All MP polymorphs of an exact composition. **Input:** `composition`.

**Returns:** `{formula, n_entries, default_polymorph, note, entries: [...]}` (at most 8 entries, lowest
E_hull first). Each entry has `mp_id`, `space_group`, `e_hull_eV_per_atom` (PBE),
`formation_energy_eV_per_atom`, `experimental` (ICSD-backed), `n_icsd` (how often it has been reported),
`band_gap_pbe_eV` and `nsites`. The default polymorph and the others within 0.05 eV/atom (up to 3) also
carry `phonon: {available, min_frequency_THz, where, method}`: MP's own band-path phonons, **for information
only**. They flag soft modes on known-stable phases (cubic SrTiO3: −2 THz at R), so `dyn_stable` always
comes from the `phonons` tool.

`default_polymorph` is the most-reported experimentally observed entry within 0.05 eV/atom of the lowest:
a proxy for the ambient phase. The 0 K DFT ground state can be a high-temperature phase (LaFeO3: R-3c in
DFT, Pnma at room temperature).

- **IDs.** MP now issues IDs such as `mp-aaaaahtd`. Legacy IDs (`mp-5229`) still resolve when passed to
  `build_structure`.
- **Solid solutions.** Fractional formulas return `n_entries: 0` with a note to look up the end-members,
  and no request is made.
- **Caching and offline.** Results are cached (`cache/mp/`). With `RunConfig.offline`, an uncached
  composition is a tool error.

### Candidates and structures: owner B, `tools/structures.py`

#### `propose_candidate`
Register a candidate before any calculation. **Code checks**, returning `accepted: false` with `reasons`
when any fail:
- **Hard constraints:** no excluded elements (CG-2).
- **Charge balance** under common oxidation states (CG-3). These are pymatgen's common states, plus
  well-established oxide states it omits: Mn3+, Ni3+, Fe4+, Co4+, Cu+, Ti3+, V3+/4+ and others; see
  `EXTRA_OXIDATION_STATES` in `tools/structures.py`. An intentional exception is declared with an
  `off-stoichiometric: <reason>` risk. Up to 5 % missing anions then become vacancies.
- **Structure:** the composition fits the prototype's sites. Elements are placed by ionic radius, not by
  formula order (CG-9, SG-5).
- **Ordered compounds only (for now).** Every site must hold a single element. Doped solid solutions and
  partly vacant sites are rejected with the shared sites named. They need a choice of dopant site and an
  ensemble of configurations, which is parked until the ordered workflow is complete
  (`ALLOW_PARTIAL_OCCUPANCY` in `tools/structures.py`). The builders still support supercells for
  partial occupancy (SG-4, SG-8); only candidates are refused.
- **No repeats** of a composition already proposed in this run (CG-6).
- **Evidence:** at least one `evidence_id`, and each is a retrieved `paper_id` or an `mp:<id>` (CG-4).
- **Novelty** when `RunConfig.require_novel`: no ICSD-backed MP entry for the exact composition.

| Input | Type |
| --- | --- |
| `composition` | string (pymatgen notation; site fractions in multiples of 0.125 for supercells ≤ 80 atoms) |
| `structure_type` | a prototype key from `hephaestus/resources/prototypes.json` (list below), or `mp` |
| `rationale` | string |
| `evidence_ids` | string[] |
| `prior_estimate` | number \| null |
| `novelty` | `known` \| `variant` \| `novel` |
| `synthesis_feasibility` | `easy` \| `moderate` \| `difficult` \| `unknown` |
| `risks` | string[] |

**Returns:** `{accepted: true, candidate_id: "c<round>-<n>", sites, tolerance_factor, descriptors,
warnings}`, or `{accepted: false, reasons: [...]}`.
- `sites` maps each site label to `{element: fraction}`, e.g. `{"A": {"Sr": 1.0}, "B": {"Ti": 0.875,
  "Fe": 0.125}, "X": {"O": 1.0}}`.
- `tolerance_factor` is set for perovskites, else null.
- `descriptors` adds the octahedral factor for perovskites, or rA/rB for pyrochlores.
- `warnings` include known inverse spinels (NiFe2O4, CoFe2O4, Fe3O4 ...) proposed on the `spinel`
  template, which always builds the normal arrangement.

**`structure_type: "mp"`** takes the structure from Materials Project as-is, for compounds that no
prototype fits (a garnet, a scheelite ...).
- The prototype-fit, disorder and supercell checks are skipped. Every other check still applies,
  charge balance included.
- The exact composition must be in MP. Otherwise the candidate is refused at proposal, since there is no
  prototype to fall back on.
- The result has `sites: null` and adds `mp_default_polymorph: {mp_id, space_group, n_entries}`.

**Prototypes** (`structure_type`). Each is a reference compound written as labelled Wyckoff orbits in
`hephaestus/resources/prototypes.json`. Adding one is a single JSON entry, validated on import by
`tools/prototypes.py`.

| Family | Keys |
| --- | --- |
| Perovskite | `perovskite_cubic`, `perovskite_tetragonal`, `perovskite_orthorhombic`, `perovskite_rhombohedral`, `double_perovskite`, `ruddlesden_popper`, `antiperovskite` |
| Fluorite | `fluorite`, `antifluorite`, `pyrochlore` |
| Spinel, corundum and others | `spinel`, `ilmenite`, `corundum`, `olivine`, `layered_rocksalt` |
| Binary | `rocksalt`, `cesium_chloride`, `zincblende`, `wurtzite`, `rutile` |
| Intermetallic | `b2`, `l12`, `half_heusler`, `full_heusler` |

#### `build_structure`
Build the starting structure.
- `source=mp` with an `mp_id` builds that exact polymorph.
- `source=auto` uses `default_polymorph` when the exact composition exists in MP, else the template.
- `source=template` forces the prototype template, e.g. for solid solutions. The supercell is the smallest that
  matches every site fraction within 5 % (SG-4), capped at `RunConfig.max_atoms` (SG-8). Sites are decorated
  by a fixed `seed` (SG-6), and geometry is checked (SG-9).

| Input | Type |
| --- | --- |
| `candidate_id` | string |
| `source` | `auto` \| `mp` \| `template` |
| `mp_id` | string \| null |
| `seed` | integer |

**Returns:** `{candidate_id, source ("mp:<id>" | "template:<name>"), n_atoms, supercell, composition_error,
disorder_seed, space_group, notes}`. `composition_error` is the largest relative error of any element's
fraction on any site.
- **MP builds** also return `polymorph: {space_group, experimental, n_icsd, matches_structure_type,
  other_polymorphs}` and a note to confirm the phase.
  - `matches_structure_type` is false when the MP polymorph's space group differs from the proposed
    prototype's. LaFeO3 proposed as `perovskite_cubic` builds as MP's Pnma phase. A note then says the
    candidate is evaluated in the MP phase and how to get the prototype instead.
  - It is null for `structure_type: "mp"`.
  - **Cell size (SG-8).** An MP cell above `RunConfig.max_atoms` is first reduced to its primitive cell.
    If it is still too big, `auto` falls back to the template and says so in `notes`. An explicit
    `mp_id`, or a `structure_type: "mp"` candidate, fails with `supercell`.
- **Template builds** also return `sites` and `lattice_abc_A`. The lattice is scaled from the reference
  compound by bond lengths (Shannon radii); cubic perovskites use a fitted formula, within 1 % of known
  cells (SG-13).
- **`auto` degrades.** If MP is unreachable it falls back to the template and says so in `notes`.
  A `structure_type: "mp"` candidate has no template: there `source=template` fails with `no_template`, and
  an unreachable MP fails with `mp_unavailable`.

Building resets that candidate's later results. A failure is a `ToolError` with a named code
(`no_template`, `disordered`, `supercell`, `geometry`, `mp_missing`, `mp_unavailable`), also recorded in the
candidate's `errors` (SG-10).

### Simulation: owner B, `tools/simulation.py`

#### `relax`
Relax positions and cell together with an ML potential (SE-1, SE-3), using FIRE with `FrechetCellFilter`.

| Input | Type | Notes |
| --- | --- | --- |
| `candidate_id` | string | needs `build_structure` first |
| `potential` | `mace-mp-0-small` \| `mace-mp-0-medium` \| `mace-mpa-0` \| `mace-omat-0` \| `chgnet` | |
| `fmax` | number (eV/Å) | 0.02 for screening, 0.005 for a careful check; clamped to [0.002, 0.1] |

**Returns:** `{candidate_id, potential, converged, energy_eV_per_atom, space_group, lattice_abc_A,
volume_per_atom_A3, wall_time_s, domain_note}`. Relaxing resets the hull, phonon and prediction results.

#### `energy_above_hull`
E_hull against MP stable phases of every subsystem, re-relaxed with the **same** potential (SE-4, SE-5).
The first use of a new chemical system relaxes its competing phases, which can take minutes on CPU.
**Input:** `candidate_id`.

**Returns:** `{candidate_id, e_hull_eV_per_atom, formation_energy_eV_per_atom, decomposes_to,
n_competing_phases, thermo_stable, threshold, wall_time_s}`. `thermo_stable` (code) means
E_hull ≤ `spec.stability_thresholds.e_hull_max_eV_per_atom`.

`RunConfig.run_hull = false` turns the tool off (it returns an error) and the gate stops requiring it.
`RunConfig.e_hull_max_eV_per_atom`, when set, replaces the goal's threshold; `null` keeps the goal's
(0.05 eV/atom if the goal names none). `run_phonons = false` does the same for `phonons`.

#### `phonons`
Finite-displacement phonons with phonopy, at commensurate q-points (SE-6 to SE-8).
- `mode=fast`: supercell ≥ 7 Å.
- `mode=full`: supercell ≥ 10 Å, plus the high-symmetry band path.

**Mode mapping** (when `RunConfig.phonon_gate="mode_map"`, the default):
1. Freeze the softest imaginary mode into its commensurate supercell, scanning amplitudes 0.5 to 3 Å·√amu.
2. Re-relax the lowest-energy structure from the scan.
3. If the energy drops by more than `RunConfig.distortion_threshold_meV` (2 meV/atom), the instability is
   **real**: the candidate's structure becomes the distorted one, and steps 1–3 repeat (up to 2 rounds).
   Otherwise it is **spurious**.

With `phonon_gate="strict"`, any frequency below −0.1 THz fails.

**Input:** `candidate_id`, `mode` (`fast` | `full`).

**Returns:** `{candidate_id, mode, min_frequency_THz, where, dyn_stable, verdict, mode_mapping_rounds,
supercell, wall_time_s}`.
- `verdict` is one of `stable`, `spurious_soft_mode`, `stable_after_distortion`, `unstable`.
- Each `mode_mapping_rounds` entry is `{q, dE_meV_per_atom, space_group, real}`.
- When the structure changed, a `note` tells the agent to rerun `energy_above_hull` and
  `predict_property`, and `hull_stale` is set.

`dyn_stable` is computed by code.

#### `run_md`
Short NVT (Langevin) MD at `spec.conditions.temperature_K` (SE-9). It counts as stable if no atom drifts
more than 1 Å after removing the centre-of-mass motion.
**Input:** `candidate_id`, `duration_ps` (clamped to [0.5, 10]).
**Returns:** `{candidate_id, stable, max_displacement_A, temperature_K}`.

### Properties: `tools/simulation.py`

#### `predict_property`
Predict a property on the current (relaxed or distorted) structure, with its uncertainty (PP-4) and a
domain flag (PP-5). The registry (`simulation.property_registry`) lists every key with its units and error.

| Property | Model | Units | Uncertainty |
| --- | --- | --- | --- |
| `band_gap` | MEGNet-BandGap-mfi-MP-2019.4.1; `RunConfig.band_gap_model` picks the output: `megnet-hse` (default), `megnet-pbe`, `megnet-gllb-sc`, `megnet-scan` | eV | HSE ±0.65 (MAE vs 15 measured perovskite gaps); PBE ±0.57 vs MP PBE gaps, about 1.6 eV below experiment; GLLB-SC and SCAN not benchmarked, so no uncertainty |
| `bulk_modulus`, `shear_modulus` | elastic tensor from strain-stress with the run's potential (±0.25 and ±0.5 % strains, ions relaxed at each strain), Voigt-Reuss-Hill | GPa | 15 / 20 (typical MLIP MAE) |
| `youngs_modulus`, `poisson_ratio`, `pugh_ratio` | from K and G | GPa, –, – | propagated from K and G |
| `sound_velocity_longitudinal`, `sound_velocity_transverse`, `sound_velocity_mean`, `debye_temperature` | from K, G and the density | km/s, K | propagated from K and G |
| `cauchy_pressure`, `anisotropy_index`, `zener_ratio` | from individual elastic constants (C12 − C44; universal A^U; 2C44/(C11 − C12)) | GPa, –, – | none available (stated) |
| `formation_energy` | from `energy_above_hull`'s references | eV/atom | 0.05 |
| `density` | relaxed structure | g/cm³ | 0.1 |

Out-of-domain chemistry widens the uncertainty: ×2 outside the potential's training set, ×1.5 for mixed
anions. **Input:** `candidate_id`, `property`.

**Returns:** `{candidate_id, property, value, uncertainty, units, model, in_domain, scale_note}`. For the
target property it also returns `target: {direction, value, reference}`. Every mechanical property comes from
one elastic-tensor calculation (cached), so any of them also returns `related` (all the others, with
uncertainties), `mechanically_stable` (Born criteria: the tensor is positive definite) and
`elastic_tensor_GPa`. All mechanical predictions are stored on the candidate at once.

### Run control: owner A, `harness/`

These three are handled by the harness itself, not a tool module, because they touch the run, not one
candidate.

#### `candidate_status`
No input. **Returns:** `{round, candidates: [facts...], budget: {tool_calls_used, compute_s}}`, where
`facts` is `ws.facts(cid)` for every candidate.

#### `request_review`
Ends the current round. An independent reviewer LLM receives:
- the orchestrator's `summary`
- the code facts for every candidate, with phonon interpretations
- the code-fixed best of the round (`select_iteration_best`) and its rubric score (`score_rubric`)
- a novelty search for that best candidate
- the evidence notes, the paper list and earlier reviews

The reviewer returns:
- findings, each `{severity, category, text, suggestion, paper_ids}`
- one recommended next composition, validated by code (SC-6, SC-7)
- a stop recommendation
- **its own confidence** (0–100) that the best candidate would meet the goal in the lab

**Input:** `summary`.

**Returns:** `{round, best_of_round, score, score_breakdown, checks, findings, recommended_next, why,
reviewer_recommends_stop, stop_reason, llm_confidence, llm_confidence_reason}`. Then either:
- `{stop_rule, instruction: "call submit_final"}` when a stop rule fired, or
- `{next_round, instruction}` otherwise.

When the code checks pass but the stop is blocked, `gate_note` gives the reason: a critical finding, or
reviewer confidence below the threshold.

#### `submit_final`
Submit the recommended candidate. **Code gate:**
- **Required calculations:** `relax`, `energy_above_hull` (not stale), `phonons` and the target
  `predict_property` must have run. A stability check turned off for the run is not required.
- **Checks:** the target check and both stability verdicts must pass. A check turned off for the run
  reports `null` and is left out; the code rubric is then rescaled to 0–100 without it.
- **Novelty** (when `require_novel`): the literature search must find no prior report of the composition.
- **A failing gate** returns `accepted: false` with the reasons. The agent can improve the candidate, or
  resubmit with `accept_not_met: true` to report the best attempt as not meeting the target.
- **Budget exhausted or a stop rule fired:** the submission is accepted with `gate_passed: false`.

| Input | Type |
| --- | --- |
| `candidate_id` | string |
| `runner_up_id` | string \| null |
| `justification` | string |
| `accept_not_met` | boolean |

**Returns:** `{accepted, gate_passed, note}`. Acceptance ends the campaign. The winner and runner-up go to
the final re-check: the final potential, fmax 0.005 and full phonons (Section 9). Then the brief is written.

## Parallel calls

A model may request several tool calls in one turn, and the Claude Code backend may issue them
concurrently. `Campaign.call_tool` enforces, for every backend:
- calls on the **same candidate** run one at a time, in the order given within a turn;
- calls on **different candidates**, and literature and database lookups, run in parallel (up to
  `RunConfig.max_parallel_tools`, default 4);
- `propose_candidate`, `candidate_status`, `request_review` and `submit_final` run with nothing else in
  flight; so do workspace snapshots;
- identical expensive work (the same structure relaxed by two calls) runs once: `cached` holds a lock per
  key, and the second caller reads the first one's result.

What handlers must do:
- touch only their own candidate's `CandidateState`;
- for shared collections (`ws.papers`, `ws.insights`, `ws.queries`), only add (`setdefault`, `append`);
- add compute time with `ws.add_compute(seconds)`, never `ws.compute_s +=`;
- get calculators from `load_potential` (one per thread, sharing the model weights) and wrap model calls
  in `model_lock(calc)`: on a GPU this is the shared `GPU_LOCK` (SE-15), on CPU it is a no-op.

## Event payloads (what the UI and CLI render)

The harness emits one `ProgressEvent` before (`started`) and after (`finished` | `failed`) each tool call.
Its `iteration` is the round in which the call **started**. Calls can run in parallel (see *Parallel calls*),
so events of different calls can interleave: pair them by `call_id`. The payload contains:
- `tool`, `call_id` (unique per call in a run) and `input`.
- `result`, the tool's return value. It is shortened for the stream, except `review`, which carries
  `request_review`'s result **in full**.
- `papers: [{paper_id, title, year, citations}]` and `mode`, for literature tools.
- Flat candidate fields when the call concerns a candidate: `candidate_id`, `composition`, `cand_round`
  (the candidate's own round), `e_hull`, `thermo_stable`, `dyn_stable`, `min_freq`, `verdict`,
  `space_group`, `value`, `uncertainty`, `novelty`, `rationale`, `evidence_ids`.

Other event sections:
- `agent` — the orchestrator's text and thinking: `{kind: "text" | "thinking", text}`.
- `capability` — agent-built tools; see *Agent-built tools* below.
- `orchestrator` — round results `{best, stop, rule, scores, llm_scores}`, and the end of the campaign
  `{submitted, usage}`.
- `final_calculation` — `{verdict, checks, notes}`.
- `brief_generator` — `{pdf, cited}`.

## Adding a tool

1. **Agree the name, inputs and result keys** with the team, and add them to this file.
2. **Write the failing tests first.** They go in `tests/unit/tools/test_<module>.py` and cover:
   - the happy path
   - each `ToolError` case
   - every code verdict
3. **Write the handler** with a NumPy docstring, then add its schema to `TOOLS` in `registry.py`.
4. **Add a scripted-model case** in `tests/integration/test_agent_loop.py` if the harness must react to the
   new tool's result.
5. **Tell the UI and CLI owners** if the event payload gains fields.

## Agent-built tools

When the goal's property has no tool, a capability phase runs before the campaign: the agent proposes a tool,
the user accepts the plan, the agent writes it, and the user reviews the code (`docs/REQUIREMENTS.md`,
Section 14). This section is the contract for the phase's tools, the tool file, the toolkit and the events.

### Capability-phase tools

These exist only in the capability phase's conversation, never in the campaign (AT-2). The literature tools
are also available in the phase.

#### `read_tool_source`
Read our own tools for inspiration and scope (AT-5). **Input:** `function` (string or null). With null it
returns `{functions: [{module, name, signature, docstring}]}` for the public functions of
`tools/simulation.py` and `tools/structures.py`; with a name it returns `{module, name, source}`. Functions of
`RunConfig.disabled_capabilities` are hidden; asking for one returns an error.

#### `propose_tool`
Propose a plan, or a revised plan, for the user's decision (AT-6, AT-7). **Input:** the plan fields of
`ToolProposal`: `name`, `property`, `units`, `description`, `method`, `citations`, `inputs` (JSON schema),
`building_blocks`, `expected_runtime_s`, `supplies_target`, `if_cancelled`. **Returns**, once the user has
decided: `{proposal_id, decision: {action, message}}`. The call waits for the user; the wait counts against no
budget (AT-25).

#### `write_tool`
Submit the source for an accepted plan (AT-10 to AT-13). **Input:** `proposal_id`, `source` (the tool file),
`tests` (the test file), `reason` (null, or why it replaces code the user is reviewing). Code runs the
static check, the tests and the trial run. **Returns** either
`{checks}` with the failing check's report (the agent fixes it and calls again, up to 3 times), or, once the
checks pass and the user has decided, `{proposal_id, checks, decision: {action, message}}`.

#### `request_package`
Ask the user to allow a package outside the allowlist, e.g. phonopy (and to install it with pip when the
server lacks it). **Input:** `module` (import name), `pip_name` (or null), `reason`. Returns at once; the
answer (`allow`, `deny` or `cancel_run`) is the next message. An allowed package goes in `TOOL["packages"]`;
the static check, the sandbox and later runs then allow it for that tool only (`CustomTool.packages`).
Names are checked: no URLs, paths or pip flags.

#### `ask_user`
Ask the user a question (`question`, `options`). The answer is the next message. The user never reads the
agent's plain text as a question.

#### `give_up`
End the phase without a tool (`reason`); the run screens stability only and the brief says why.

**The agent's second look.** Every tool declares `TOOL["trial_range"]`, the plausible value for the trial
material from the literature; a trial value outside it fails the checks before the user sees the code. If
the agent finds a bug after its code is shown, `write_tool` with a `reason` withdraws the open card (event
`withdrawn`, `{proposal_id, revision, reason}`) and resubmits; it uses none of the user's change requests but
counts against the three fix attempts, and is refused once the user has decided.

**No dead ends.** A change request at the code stage may be answered with a new plan (`propose_tool`
again) or new code. Two turns in a row without a tool call, with nothing waiting for the user, end the
phase with "the agent made no progress", instead of nudging forever.

### The tool file

One tool per file, `<tool_library_dir>/<name>.py`, with its tests in `test_<name>.py`. A user can write one by
hand; it is reviewed like an agent's (AT-21). The `TOOL` header must be a literal dict, because code reads it
with `ast.literal_eval` before the file is approved and never imports an unapproved file.

```python
"""Bulk modulus from a Birch-Murnaghan fit of energy against volume."""

TOOL = {
    "name": "bulk_modulus_eos",
    "property": "bulk_modulus",
    "units": "GPa",
    "description": "Isotropic strain of ±4 % in 7 steps, ions relaxed at each volume, third-order "
                   "Birch-Murnaghan fit.",
    "method": "Equation-of-state fit (Birch 1947).",
    "citations": ["<paper_id>"],
    "inputs": {"type": "object", "properties": {}, "additionalProperties": False},
    "building_blocks": ["relax", "single_point"],
    "author": "agent",
}


def run(structure: dict, inputs: dict, hp) -> dict:
    """Compute the bulk modulus of one structure.

    Parameters
    ----------
    structure : dict
        The candidate's current structure, as a pymatgen ``Structure.as_dict()``.
    inputs : dict
        The declared inputs, validated against ``TOOL["inputs"]``.
    hp : Toolkit
        Building blocks, run by the main process (see *The toolkit*).

    Returns
    -------
    dict
        ``{"value": float, "units": str, "details": dict}``; ``units`` must equal ``TOOL["units"]``.
    """
```

The test file holds plain `test_*` functions. Each may take any of `run` (the tool's function), `hp`,
`structure` (the relaxed trial structure of the reference material, or SrTiO3) and `tool` (the header); they
run in the sandbox like the tool. Results must be JSON. Allowed imports: `math`, `cmath`, `statistics`,
`itertools`, `functools`, `collections`, `typing`, `dataclasses`, `numpy`, `scipy`, `pymatgen.core`,
`pymatgen.analysis`, `pymatgen.symmetry`, `ase`, `ase.atoms`, `ase.build`, `ase.units`, `ase.eos`
(`ALLOWED_IMPORTS` in `tools/custom.py`). The static check also refuses names and attributes that reach
files or processes (`open`, `getattr`, `.save`, `.to`, `.write`, anything starting with `_`).

### The toolkit (`hp`)

The worker process sends each call over a pipe; the main process runs it with the run's potential (the final
potential and dtype in the final re-check), under `model_lock` and the cache (AT-15). Structures are pymatgen
dicts; units are eV, Å, GPa and THz.

| Call | Returns |
| --- | --- |
| `hp.single_point(structure)` | `{energy_eV, forces_eV_per_A, stress_GPa}` (stress 3×3, ASE sign: negative under compression) |
| `hp.relax(structure, fix_cell=False, fmax=0.01)` | `{structure, energy_eV_per_atom, converged}` |
| `hp.phonons(structure, supercell=None)` | `{frequencies_THz, qpoints, min_frequency_THz, supercell}` |
| `hp.md(structure, temperature_K, duration_ps)` | `{energies_eV, temperatures_K, final_structure, max_displacement_A, stable}` (every 10 steps; at most 10 ps) |
| `hp.log(message)` | none; shown in the orchestrator log |

Each toolkit call adds to the run's compute time and is cached by its exact inputs. A tool may use none of
them. The worker compiles the tool from source on every call (no bytecode cache), gets no environment
credentials, and its time limit excludes the time building blocks spend in the main process.

### In the campaign

An approved tool appears as `custom_<name>` with input `candidate_id` plus its declared inputs (AT-16). The
harness runs it on the candidate's current structure. **Returns:** `{candidate_id, property, value, units,
model: "agent-built:<name>@<sha8>", uncertainty: null, note: "agent-built tool"}`, plus `target` when it
supplies the target property (AT-17). It never sets `thermo_stable`, `dyn_stable` or `mechanically_stable`
(AT-18). Results are cached under the tool's sha256 and the structure (AT-23).

### Capability events

Section `capability`. Each event's payload has `kind` plus:

| `kind` | Payload |
| --- | --- |
| `gap` | `{property, units, custom_tools}` |
| `library` | `{reused: [name], unapproved: [file]}` |
| `plan` | `{proposal, allowed}`: the `ToolProposal` at the plan stage and the actions offered |
| `checks` | `{proposal_id, revision, checks}` after each `write_tool` |
| `code` | `{proposal, allowed}`: the `ToolProposal` with `source`, `tests`, `checks`, `trial` and `toolkit_calls` |
| `decision` | the `ToolDecision` |
| `registered` | the `CustomTool` |
| `cancelled` | `{proposal_id, scope: "tool" or "run", supplies_target, reason}` |
| `log` | `{message}` from `hp.log` |
| `package`, `question` | `{request, allowed}`: a `UserRequest` waiting for the user |

Tool calls of the phase are ordinary `tool` events with `phase: "capability"` and string call IDs
(`cap1`, `cap2`, ...), so they never collide with the campaign's.

A `plan` or `code` event with no later `decision` for the same `proposal_id` means the run is waiting for the
user; the UI and CLI show the card with its actions (AT-24, AT-27, AT-28).
