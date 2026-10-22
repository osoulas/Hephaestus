# Hephaestus (Self-Designing Experiment): Requirements

## Changes since the 16 October version

This version reflects the design decisions made before the hackathon. Requirement IDs are kept where the intent is
unchanged; new ones are marked *(new)*. Tool-level detail lives in `docs/TOOLS.md`.

- **Architecture.** The fixed pipeline is replaced. An LLM orchestrator drives the search with tools, and an
  independent reviewer LLM critiques each round. Sections 2, 3, 7 and 8 are rewritten. OR-1 no longer
  fixes the order of steps inside the loop.
- **Code-only verdicts are kept and strengthened.** The target check, stability verdicts, submission gate,
  budgets and stop rules are code (SC-1, SE-13).
- **Band gap** uses MEGNet's HSE fidelity, ±0.65 eV against measured gaps. The PBE fidelity is about
  1.6 eV low (PP-1, Section 6 decision).
- **Dynamic stability** uses mode mapping: an imaginary mode only fails if freezing it in lowers the energy
  by more than 2 meV/atom (SE-6, SE-13, SE-14).
- **Polymorphs.** The ambient-temperature polymorph is preferred over the 0 K DFT ground state (SG-1, SG-12).
- **Stop rule "target met"** also needs the reviewer's own confidence above the threshold, and no critical
  reviewer finding (OR-2, SC-12).
- **New requirements:**
  - optional novelty requirement (GP-13, CG-12)
  - LLM backend and model chosen per run (OR-11)
  - tool-call budget (OR-10)
  - resume after a crash or restart (OR-6, DL-10)
  - crash diagnostics (DL-11)
  - new UI controls (UI-12 to UI-16)
- **Shared types:** `Critique` gains `llm_confidence` and `llm_confidence_reason`; `RunConfig` is now listed.
- **Several property targets** *(22 October)*: `SearchSpec.extra_targets`; every target must be met. References are computed per target, each target without a tool gets its own capability phase, and the rubric's property points are averaged over the targets.
- **Agent-built tools** *(22 October, new Section 14)*. When no tool can compute the goal's property, the agent
  proposes a tool, writes it once the user accepts the plan, and the user reviews the code before it runs.
  Approved tools are saved as files in the working directory and reused by later runs. A capability phase is
  added between goal parsing and the campaign (OR-1, GP-2); new types `ToolProposal`, `ToolDecision` and
  `CustomTool`; new `RunConfig` fields.

Oct 16, 2026 · @Oskar Soulas

## Purpose and scope

The package takes a plain-English materials research goal and returns a simulation-checked candidate composition with an experimental brief. It parses the goal, then hands the campaign to an LLM orchestrator. The orchestrator works only through tools: it reads the literature, consults the database, proposes candidates, builds structures, runs simulations with ML interatomic potentials and predicts properties. An independent reviewer LLM critiques each round. Code decides whether results pass, and when to stop. The winner is re-checked at higher fidelity, then the brief is written.

This document specifies what each section of the package must do, not how. No code or library choice is binding unless a requirement names it.

**In scope:** inorganic crystalline materials, composition and structure screening, ground-state energetics, dynamic stability, ML-predicted functional properties, synthesis and characterisation suggestions.

**Out of scope:** running physical lab equipment, full DFT as the default path, molecules and polymers, kinetics of synthesis beyond literature-derived advice.

**Conventions used below**

- Each requirement has an ID (prefix per section), a priority, and an acceptance check.
- Priority: **Must** = needed for the hackathon demo; **Should** = needed for a credible tool; **Could** = stretch.
- "Candidate" = one proposed composition plus structure type. "Iteration" = one pass of propose, simulate, critique.
- Units: energies in eV/atom, frequencies in THz, lengths in Å, temperatures in K unless stated.

## Prior work and what's new

No past hackathon project has built this, but the core loop (LLM proposes, a tool evaluates, the LLM reflects and iterates) already exists in published work. The pitch must name that work and lead with what this package adds. Checked against all 120 submissions from 2025 and the 34 from 2024, as of 16 October 2026.

| Project | Event | What it does | How it differs from ours |
| --- | --- | --- | --- |
| [LLMatDesign](https://arxiv.org/abs/2406.13163) | Published paper, 2024 | LLM agent modifies materials toward a target property, evaluates with tools, self-reflects on past decisions, repeats until the target is met. | Same core loop. Evaluation relies mainly on ML property predictors; no live literature grounding, no phonon or hull checks, no lab brief. |
| PoreVoyant ([code](https://github.com/mehradans92/PoreVoyant)) | 2024, 3rd place | ReAct agent iteratively proposes MOFs with lower band gap, guided by rules from 7 papers, scored by a surrogate ensemble; reverts when a step gets worse. | MOFs as SMILES, fixed 7-paper corpus, no physical stability checks. |
| LangSim | 2024, 1st place | Natural-language interface that runs ASE workflows with MACE to compute bulk properties of elemental crystals. | Same simulation stack, but a tool interface, not a design loop. |
| [MAGE](https://github.com/truptimohanty/MAGE) | 2025 | Fine-tuned LLM for property prediction and inverse design; MEGNet formation energy as a stability hint. | One-shot generation; no critique loop, phonons or literature. |
| [SKY](https://github.com/hspark1212/synthesis-agent) | 2025, 1st place | Similarity search over Materials Project synthesis recipes; LLM adapts them into a protocol for a target material. | Answers how to make a chosen material; ours decides what to make. Overlaps only with our brief's synthesis section. |
| [PhaseDiag Agents](https://github.com/Jverduzc/maps) | 2025 | Agent runs MACE molecular dynamics on Materials Project structures to build phase diagrams. | MLIP use only; no design loop. |

**What is new here, in pitch order:**

1. A stability gate the LLM cannot override: every answer must pass energy above hull and phonon checks (SC-1, SE-13).
2. Live literature grounding through Valency, including dedicated searches for failed attempts (LM-5).
3. A higher-fidelity re-check of the winner before anyone spends lab time (Section 9).
4. A cited, lab-ready brief rather than a ranked list.

Pitch line: "LLMatDesign showed an LLM can iterate toward a target. We built the version you would trust before spending a week in the lab."

Sources: [2025 submissions data](https://llmhackathon.github.io/submissions/), [2024 hackathon report](https://arxiv.org/abs/2411.15221), [hackathon site](https://llmhackathon.github.io/).

## System-level requirements

These apply across every section. A section-level requirement never overrides them.

| ID | Requirement | Priority | Acceptance check |
| --- | --- | --- | --- |
| SYS-1 | One run goes from a free-text goal to an experimental brief with no manual step in between. | Must | A demo goal completes end to end unattended. |
| SYS-2 | A full run of 3 iterations finishes in under 15 minutes on a laptop CPU. | Must | Timed run on a team laptop, phonons on the small supercell. |
| SYS-3 | Every claim in the final brief is traceable to a paper, a database entry or a named simulation result. | Must | Spot-check 10 claims; each has a source or run ID. |
| SYS-4 | Every predicted number carries an uncertainty or an explicit "no uncertainty available". | Must | No bare property values in any output. |
| SYS-5 | Sections talk only through defined data contracts, so any one can be replaced or stubbed. | Must | Each section runs in isolation against a fixture input. |
| SYS-6 | Failure of an optional step (phonons, property model, a search angle) degrades the run but does not stop it. | Must | Kill each optional step in turn; the run still produces a brief that names what is missing. |
| SYS-7 | The run is reproducible: same goal, same seeds, same cached data gives the same candidates and numbers. | Should | Two runs from cache give identical simulation outputs. |
| SYS-8 | All external calls (LLM, Valency, Materials Project) are cached by input, so re-runs cost nothing and work offline for the demo. | Should | Second run makes zero network calls. |
| SYS-9 | The system states its own coverage limits in every brief (corpus covered, structure types supported, model training domain). | Should | Brief contains a scope section. |
| SYS-10 | No secrets are stored in code or logs; credentials come from the environment or the OAuth token store. | Must | Grep of repo and logs finds no keys or tokens. |
| SYS-11 | Runs on Python 3.11+, Linux and macOS, CPU only; a GPU is used when present. | Should | Runs on two team laptops without a GPU. |

## Shared data types

Every function in the package passes one of the records below. Define them once, in one shared module, so sections can be built in parallel against the same contracts (SYS-5).

**Format rules for all records**

- Each record is a typed model (e.g. Pydantic) that serialises to JSON with no loss.
- IDs are strings: `run_id`, `candidate_id` (`c{iteration}-{n}`, e.g. `c2-3`), `paper_id` (Valency's ID as given).
- Compositions are formula strings in pymatgen notation, e.g. `Ba0.7Sr0.3TiO3`; element lists are symbol strings.
- Structures are stored twice: a CIF string for people and tools, and the pymatgen structure dictionary for exact reload.
- Units are fixed per field and written into the field name where ambiguous (`energy_above_hull_eV_per_atom`, `min_frequency_THz`, `band_gap_eV`).
- Times are ISO 8601 in UTC. Errors are `{step, code, message}` objects collected in a list, never raised past a section boundary.

| Type | Key fields (type, units) | Produced by | Stored as |
| --- | --- | --- | --- |
| `Quantity` | value (float), uncertainty (float or null), units (str), source (str: model name, paper ID or database ID) | Many | Inside other records |
| `SearchSpec` | raw\_goal (str); material\_family (str); elements (list\[str\]); excluded\_elements (list\[str\]); target\_property (str, registry key); target\_value (float); target\_units (str); direction (maximise, minimise or hit); reference\_material (str or null); reference\_value (Quantity or null); conditions (temperature\_K, pressure\_GPa); hard\_constraints (list\[str\]); soft\_preferences (list\[str\]); stability\_thresholds (e\_hull\_max\_eV\_per\_atom, phonon\_min\_THz); assumptions (list\[str\]); target\_tool, target\_evaluable (agent-built tools); **extra\_targets (list\[Target\]: property, value, units, direction, tolerance, reference\_material, reference\_value, tool, evaluable): further property targets, every one of which must be met** | Goal Parser | `spec.json` |
| `Paper` | paper\_id; title; authors (list\[str\]); year (int or null); doi; url; source (arxiv, pubmed, biorxiv, medrxiv); abstract; citation\_count (int or null); relevance (float 0 to 1); found\_via (semantic, cited, negative, similar, citing) | Literature Miner | Inside `literature.json` |
| `Insight` | composition (str or null); property (str); value (Quantity or null); conditions (str); finding\_type (positive, negative, mechanism, synthesis, method); summary (str); paper\_id (str) | Literature Miner | Inside `literature.json` |
| `LiteratureContext` | papers (list\[Paper\]); insights (list\[Insight\]); property\_map (dict: composition to Quantity); failed\_approaches (list\[str\]); mechanisms (list\[str\]); synthesis\_routes (list\[str\]); open\_questions (list\[str\]); field\_trend (dict: year to count); coverage\_note (str) | Literature Miner | `literature.json` |
| `Candidate` | candidate\_id; iteration (int); composition; structure\_type (str, registry key); rationale (str); evidence\_ids (list\[str\]); prior\_estimate (Quantity or null); novelty (known, variant, novel); synthesis\_feasibility (easy, moderate, difficult, unknown); risks (list\[str\]) | Candidate Generator | `iter_N/candidates.json` |
| `StructureRecord` | candidate\_id; cif (str); pmg\_dict (dict); source (mp:\<id>, template:\<name>, substitution); supercell (list\[int\], 3 values); n\_atoms (int); disorder\_seed (int or null); composition\_error (float, max fractional error per site) | Structure Generator | `iter_N/<candidate_id>/start.cif` + JSON |
| `PhononResult` | mode (fast or full); stable (bool); min\_frequency\_THz (float); instability\_at (str or null); supercell (list\[int\]) | Simulation Engine | Inside SimulationResult |
| `SimulationResult` | candidate\_id; potential (name and version); converged (bool); relaxed (StructureRecord); space\_group (str); lattice (a, b, c in Å; alpha, beta, gamma in degrees); density\_g\_cm3; energy\_eV\_per\_atom; formation\_energy\_eV\_per\_atom; energy\_above\_hull\_eV\_per\_atom; thermo\_stable (bool); phonons (PhononResult or null); md\_stable (bool or null); domain\_note (str); wall\_time\_s (float); errors (list) | Simulation Engine | `iter_N/<candidate_id>/sim.json` + `relaxed.cif` |
| `PropertyPrediction` | candidate\_id; property (str); value (Quantity); model (name and version); in\_domain (bool); literature\_value (Quantity or null) | Property Predictor | `iter_N/<candidate_id>/prop.json` |
| `Critique` | iteration (int, = review round); candidate\_id (best of round, chosen by code); target\_met\_central (bool); target\_met\_conservative (bool); thermo\_stable (bool); dyn\_stable (bool); score (int 0 to 100, code rubric); score\_breakdown (dict: rubric component to points); findings (list of {severity, category, text, suggestion, paper\_ids}); next\_composition (str); next\_rationale (str); stop\_recommended (bool); stop\_reason (str or null); **llm\_confidence (int 0 to 100 or null): the reviewer's own confidence that the best candidate would meet the goal in the lab; llm\_confidence\_reason (str or null)** | Reviewer (Section 7) | `iter_N/critique.json` |
| `ValidationReport` | candidate\_id; checks (list of {name, screening\_value, final\_value, delta}); verdict (confirmed, weakened, overturned); notes (list\[str\]) | Final Calculation Runner | `final/validation.json` |
| `Brief` | run\_id; markdown (str); pdf\_path (str or null); cited\_paper\_ids (list\[str\]) | Brief Generator | `brief.md`, `brief.pdf` |
| `ProgressEvent` | run\_id; time; section (str); status (started, finished, failed); iteration (int or null); payload (small JSON summary) | Orchestrator | Streamed to UI; appended to `events.jsonl` |
| `RunState` | run\_id; spec; literature; iterations (one per review round: {candidates, structures, sims, predictions, critique}); best (candidate\_id); stop\_reason; validation; brief; errors | Orchestrator | `state.json`, rewritten after every tool call |
| `RunConfig` *(new)* | max\_iterations (review rounds, default 3); confidence\_threshold (72); max\_tool\_calls (90); llm\_backend (auto, api, claude-code); llm\_model (default claude-opus-5-5); band\_gap\_fidelity (hse, pbe); require\_novel (bool); potential; final\_potential; cross\_check\_potential; phonon\_mode; phonon\_gate (mode\_map, strict); distortion\_threshold\_meV (2.0); run\_phonons; device; dtype; max\_atoms (80); time\_budget\_s; final\_time\_budget\_s; run\_final; seed; offline; custom\_tools (off, ask); capability\_time\_budget\_s (600); max\_custom\_tools (3); max\_tool\_revisions (3); custom\_tool\_timeout\_s (300); tool\_library\_dir (`./hephaestus_tools`); disabled\_capabilities (list\[str\], empty) | Caller (UI, CLI, Python API) | `progress.json` |
| `ToolProposal` *(new)* | proposal\_id; name (snake\_case); property (str, the gap it fills); units; description; method (str); citations (list\[str\], paper IDs); inputs (JSON schema, beyond `candidate_id`); building\_blocks (list\[str\], may be empty); expected\_runtime\_s (float); supplies\_target (bool); if\_cancelled (str, what the run loses); revision (int); stage (plan, code); source (str or null, set at the code stage); tests (str or null); checks (static, tests, trial: each {passed, report}) | Capability phase (Section 14) | `capability/proposals.json` |
| `ToolDecision` *(new)* | proposal\_id; stage (plan, code); action (accept, revise, cancel\_tool, cancel\_run); message (str or null); time | User (UI, CLI, Python API) | `capability/decisions.json`, and as an event |
| `CustomTool` *(new)* | name; property; units; file (path); sha256; author (agent, user); approved\_at; run\_id (str or null, the run that built it) | Capability phase, tool library | `<tool_library_dir>/manifest.json` |
| Workspace snapshot *(new)* | everything the agent's tools share in one run: candidates with their structures and results, papers, evidence, round, budget counters, submission (see `docs/TOOLS.md`) | Harness | `workspace.json`, after every tool call |

## 1. Goal Parser

The Goal Parser turns a free-text research goal into a structured search specification that every later section reads. If the goal is ambiguous, it asks rather than guesses.

**Input:** free-text goal, e.g. "a lead-free perovskite with dielectric constant above 2000 at room temperature, made by solid-state synthesis".

**Output:** a search specification containing material family, element set, target property (name, value, units, direction: maximise, minimise or hit), operating conditions, hard constraints, soft preferences, and a list of assumptions made.

| ID | Requirement | Priority | Acceptance check |
| --- | --- | --- | --- |
| GP-1 | Extract material family, elements, target property, target value, units and direction from the goal. A goal with several property targets keeps the main one in the target fields and the rest in `extra_targets`; every target must be met by the target check, the submission gate and the stop rule. | Must | 10 test goals parsed correctly by hand review; "smaller gap and higher bulk modulus than SrTiO3" gives two targets. |
| GP-2 | Map the target property to one the Property Predictor supports, or flag it as unsupported before the run starts. An unsupported property keeps its plain name and units, and becomes a capability gap (AT-1). | Must | An unsupported property (e.g. creep rate) is flagged, not silently swapped. |
| GP-3 | Separate hard constraints (must hold: "no lead") from soft preferences ("cheap if possible"). | Must | Constraints are labelled hard or soft in the output. |
| GP-4 | Expand implied constraints into explicit ones, e.g. "lead-free" becomes an excluded element. | Must | Excluded-element list is populated from wording. |
| GP-5 | Ask the user at most 3 clarifying questions when a required field is missing or ambiguous. | Should | Goal without a target value triggers a question, not a default. |
| GP-6 | Record every assumption it made, in plain words, for the final brief. | Must | Assumptions list is non-empty whenever a default was used. |
| GP-7 | Default operating conditions to 300 K and ambient pressure when unstated, and record that. | Must | Default appears in assumptions. |
| GP-8 | Validate element symbols and reject impossible combinations (unknown elements, empty element set). | Must | "Xx" or an empty set is rejected with a message. |
| GP-10 | Let the user edit the parsed specification before the run starts. | Could | UI shows editable fields. |
| GP-11 | Accept targets phrased relative to a known material ("smaller band gap than SrTiO3"). Compute the reference value with the same Property Predictor used for candidates, so both sides sit on the same scale; record the literature value alongside for context. | Must | "Better than BaTiO3" yields a numeric target taken from a model run on BaTiO3. |
| GP-12 | Add thermodynamic and dynamic stability to every specification as hard constraints, with the thresholds from SE-13, unless the user explicitly relaxes them. | Must | Both constraints appear in every parsed specification. |
| GP-13 *(new)* | When the user asks for novel materials (UI checkbox, `RunConfig.require_novel`), add "novel: not an experimentally known compound and not previously reported" to the hard constraints. | Should | The constraint appears when the option is set. |

### Functions

| Function | Inputs | Output | Notes |
| --- | --- | --- | --- |
| `parse_goal` | raw\_goal (str) | draft SearchSpec + list of clarifying questions (list\[str\], max 3) | One structured LLM call. Never fills a missing target value with a guess (GP-5). |
| `apply_answers` | draft SearchSpec, answers (dict: question to answer) | SearchSpec | Merges user answers; records any remaining defaults in `assumptions`. |
| `expand_constraints` | SearchSpec | SearchSpec | Turns wording into explicit fields, e.g. "lead-free" adds Pb to `excluded_elements` (GP-4). Adds stability thresholds (GP-12). |
| `resolve_reference` | reference\_material (str), target\_property (str) | Quantity | For relative goals (GP-11): gets the reference structure from Materials Project (ambient polymorph, SG-1), relaxes it with the screening potential, runs the Property Predictor on it with the run's band gap fidelity. |
| `validate_spec` | SearchSpec | list of errors (empty when valid) | Checks element symbols, property is in the registry, direction matches target, units match the registry. |

## 2. Literature Miner (Valency Bond)

Literature is gathered from Valency Bond, Valency's hosted MCP server for scientific literature. **In the agent design the orchestrator searches whenever it needs evidence, not in one batch at the start.** It uses the literature tools in `docs/TOOLS.md` (`search_literature`, `related_papers`, `get_paper`, `publication_trend`, `note_evidence`). The papers retrieved and the evidence recorded are aggregated into the `LiteratureContext` for the reviewer and the brief. The retrieval requirements below are met by tool capabilities plus the orchestrator's instructions; the processing requirements by code.

**Source:** Valency Bond at `https://mcp.valency.io/` over MCP (streamable HTTP). Corpus covers arXiv, bioRxiv, medRxiv and PubMed, 500M+ records, updated daily. Authentication is a one-time browser OAuth sign-in; there is no API key.

**Input:** tool calls from the orchestrator during the run.

**Output:** a literature context aggregated from the run: the papers retrieved (with Valency paper IDs and links), evidence recorded with `note_evidence` (compositions, reported values, failed approaches, mechanisms, synthesis routes), open questions, a field-activity trend and a coverage note.

### Retrieval

| ID | Requirement | Priority | Acceptance check |
| --- | --- | --- | --- |
| LM-1 | Use Valency Bond as the only literature source. | Must | No other literature API is called. |
| LM-2 | Authenticate once via browser sign-in and reuse cached tokens; refresh silently when they expire. | Must | Second run opens no browser. |
| LM-3 | Search from several angles (property, elements, structure-property, synthesis). The orchestrator is instructed to research before proposing. | Must | A run's log shows ≥ 4 distinct queries. |
| LM-4 | Run one citation-sorted pass to capture foundational papers. | Should | Top-cited papers appear in results. |
| LM-5 | Run dedicated searches for negative results (failed synthesis, secondary phases, no improvement) using abstract-level search. | Must | At least one paper tagged as a negative-result source when such papers exist. |
| LM-6 | Expand around the top seeds with similar-paper search to catch work using different vocabulary. | Should | Expanded papers are tagged by how they were found. |
| LM-7 | Fetch papers that cite the most-cited seeds, to catch later revisions of headline numbers. | Should | Citing papers appear when the seed has citations. |
| LM-8 | Get a yearly publication trend for the material family, with missing years filled as zero. | Should | Trend covers every year from first to last. |
| LM-9 | Reuse Valency paper IDs across calls instead of repeating searches. | Must | No paper is fetched twice in one run. |
| LM-10 | Retry a failed Valency call once, then continue without it and note the gap. | Must | Simulated tool failure does not stop the run. |

### Processing

| ID | Requirement | Priority | Acceptance check |
| --- | --- | --- | --- |
| LM-11 | Merge duplicate records (arXiv and PubMed copies of one paper) by DOI, or by normalised title plus first author. | Must | A known duplicate pair yields one record. |
| LM-12 | Derive each paper's year from its journal reference when present, else its first-available date. | Should | Years match a hand-checked sample. |
| LM-13 | Record structured insights through `note_evidence`: composition, property value, units, conditions, finding type (positive, negative, mechanism, synthesis), one-line summary, source ID. | Must | 20 random insights hand-checked; ≥ 85% correct. |
| LM-14 | Never invent a value: code drops any recorded number that does not appear in the paper's abstract. | Must | Zero fabricated values in the hand check; a test value absent from the abstract is dropped. |
| LM-15 | Normalise compositions to one notation so the same material from two papers matches. | Must | "SrTiO3" and "Sr1Ti1O3" map to one entry. |
| LM-16 | Keep the best reported value per composition, respecting the property's direction. | Must | Property map matches hand-picked best values. |
| LM-17 | Produce 3 to 5 specific open questions the literature leaves unanswered. | Should | Questions name compositions or mechanisms, not generic gaps. |
| LM-18 | Cap the context passed downstream (e.g. 30 papers, 20 insights per type) and rank by relevance, citations and a bonus for negative results. | Must | Context fits the LLM budget with headroom. |
| LM-19 | State coverage in the output: Valency indexes preprint and PubMed records, so journal-only papers with no preprint are missing. | Must | Coverage note appears in the brief. |

**Resolved (16 Oct 2026):** tool and argument names confirmed against the live server's schema listing; see `docs/valency_tools.md`. All planned tools exist under these names; `get_keyword_trends` takes `query`, and `get_citing_papers.sort_by` is citations/year/year\_asc.

### Functions

The `valency_*` functions wrap single Valency Bond tool calls. Argument names are confirmed against the live schema; see `docs/valency_tools.md`.

| Function | Inputs | Output | Notes |
| --- | --- | --- | --- |
| `valency_connect` | none (reads cached tokens) | open MCP session | Restores the token expiry so expired tokens refresh silently (LM-2); a rejected refresh marks the token for a new sign-in. |
| `valency_semantic_search` | session, query (str), limit (int, default 15), sort\_by (relevance or citations) | list\[Paper\] | Wraps `semantic_search_papers` \[query, limit, sort\_by, include\_abstract, max\_authors\]. |
| `valency_abstract_search` | session, query (str), limit (int) | list\[Paper\] | Wraps `search_by_abstract`. Used for negative-result phrasing. |
| `valency_similar` | session, paper\_id (str), limit (int, default 8) | list\[Paper\] | Wraps `find_similar_papers`. Similarity is to the seed paper, so scale relevance by the seed's. |
| `valency_citing` | session, paper\_id (str), limit (int, default 8) | list\[Paper\] | Wraps `get_citing_papers` \[paper\_id, limit, sort\_by (citations, year, year\_asc), max\_authors\]. |
| `valency_get_paper` | session, paper\_id (str) | Paper or null | Wraps `get_paper_by_id`. |
| `valency_trend` | session, term (str) | dict: year (int) to count (int) | Wraps `get_keyword_trends` \[query, granularity="year"\]; counts arrive under `periods`; fills missing years with 0 (LM-8). |
| `merge_duplicates` | list\[Paper\] | list\[Paper\] | By DOI, else normalised title plus first author's surname (LM-11). |
| `rank_papers` | list\[Paper\], cap (int) | list\[Paper\] | 0.7 × relevance + 0.3 × capped citations + bonus for negative results (LM-18). |
| `normalise_composition` | formula (str) | str or null | pymatgen reduced notation; rejects sentence words that parse as formulae (LM-15). |
| `aggregate_context` | papers, insights, field\_trend, SearchSpec | LiteratureContext | Property map (best value per composition, LM-16), failed approaches, mechanisms, synthesis routes, coverage note. |
| `find_open_questions` | list\[Insight\], SearchSpec | list\[str\], 3 to 5 | One LLM call (LM-17). |
| `check_novelty` | composition, property key | {hits (int), paper\_ids} | Targeted search for prior reports of the exact composition (SC-11, CG-12). |
| Literature tools | see `docs/TOOLS.md` | | `search_literature`, `related_papers`, `get_paper`, `publication_trend`, `note_evidence`. |

## 3. Candidate proposal

The orchestrator proposes candidates itself, grounded in the literature and database evidence it has gathered. It registers each one with the `propose_candidate` tool, which applies the code checks below before any calculation. Requirements marked *(prompt)* are instructions to the orchestrator, checked by the reviewer; the rest are enforced by code.

**Input:** the orchestrator's composition, structure type, rationale, evidence IDs, prior estimate, novelty class, synthesis feasibility and risks.

**Output:** a `Candidate` with ID `c<round>-<n>`, or the rejection reasons.

| ID | Requirement | Priority | Acceptance check |
| --- | --- | --- | --- |
| CG-1 | Propose 3 to 5 candidates per round *(prompt)*. | Should | Reviewer flags rounds outside the range. |
| CG-2 | Every candidate obeys all hard constraints from the specification. | Must | `propose_candidate` rejects violators (e.g. an excluded element). |
| CG-3 | Every candidate is charge-balanced under common oxidation states, or flagged as intentionally off-stoichiometric with a reason. Works for fractional compositions. | Must | `Sr0.75La0.25Ti0.75Al0.25O3` is accepted; `SrTi0.5Nb0.5O2` is rejected. |
| CG-4 | Every rationale cites at least one retrieved paper ID or an `mp:<id>`. | Must | Uncited or unknown IDs are rejected. |
| CG-5 | Include a spread: at least one conservative and one exploratory candidate *(prompt)*. | Should | Reviewer comments on the spread. |
| CG-6 | Never re-propose a composition already proposed in this run. | Must | A repeat is rejected. |
| CG-7 | Avoid compositions the literature reports as failed, unless the rationale explains why this attempt differs *(prompt)*. | Should | Reviewer flags unexplained overlap. |
| CG-8 | From round 2, address the reviewer's recommended composition: try it or say why not *(prompt)*. | Must | The log links each review to the next round. |
| CG-9 | Only supported structure types; perovskites must be ABX3. | Must | Unsupported types never reach simulation. |
| CG-10 | Record the orchestrator's prior property estimate. | Should | Prior stored when given. |
| CG-12 *(new)* | When novelty is required, reject compositions with an experimentally observed (ICSD-backed) Materials Project entry. | Should | A known compound is rejected with the MP IDs. |

### Functions

| Function | Inputs | Output | Notes |
| --- | --- | --- | --- |
| `propose_candidate` (tool) | see `docs/TOOLS.md` | {accepted, candidate\_id, sites, tolerance\_factor} or {accepted: false, reasons} | Runs the checks below. |
| `check_charge_balance` | composition (str) | oxidation states or null | Enumerates common oxidation states; works for fractional compositions (CG-3). |
| `assign_sites` | composition, structure\_type | {A, B, X} occupancies | By ionic radius, not list order (SG-5); also validates ABX3 (CG-9). |
| `is_known_compound` | composition | {known, mp\_ids} | ICSD-backed MP entries (CG-12). |

## 4. Structure Generator

The Structure Generator turns a candidate's composition and structure type into an atomic structure the Simulation Engine can relax. It must handle solid solutions, which most interesting candidates are.

**Input:** one candidate.

**Output:** a starting structure (lattice, species, fractional coordinates), its source (database entry, prototype template, or substitution), supercell size, and how disorder was represented.

| ID | Requirement | Priority | Acceptance check |
| --- | --- | --- | --- |
| SG-1 | If the exact composition exists in Materials Project, start from the polymorph stable at ambient conditions, not just the 0 K DFT ground state. The default is the most-reported experimentally observed (ICSD-backed) entry within 0.05 eV/atom of the lowest; with none observed, the lowest-energy entry. | Must | LaFeO3 builds as Pnma (not R-3c), SrTiO3 as cubic Pm-3m; the MP ID is recorded. |
| SG-2 | Otherwise build from a prototype template; cubic ABO3 perovskite is the minimum supported type. | Must | Perovskite candidates always produce a structure. |
| SG-3 | Support further prototypes: tetragonal and orthorhombic perovskite, spinel, rocksalt, fluorite, L1₂. | Should | Each listed type builds a valid structure. |
| SG-4 | Represent fractional site occupancy with a supercell sized so the composition is matched within 5% per site. | Must | Built composition matches the target within tolerance. |
| SG-5 | Assign species to sites by crystal-chemistry rules (ionic radius and oxidation state), not by list order. | Must | A-site and B-site cations land on correct sites in test cases. |
| SG-6 | Generate disorder with a fixed random seed, recorded with the structure. | Must | Same seed gives an identical structure. |
| SG-7 | Optionally generate 2 to 3 disorder configurations and pass the lowest-energy one forward. | Could | Multiple configurations recorded with energies. |
| SG-8 | Keep supercells at or below 80 atoms by default so CPU runs stay inside the SYS-2 time budget. | Must | Default cells never exceed the cap. |
| SG-9 | Reject structures with atoms closer than a minimum distance (e.g. 0.7 × sum of covalent radii) before relaxation. | Must | Overlapping atoms are caught before simulation. |
| SG-10 | Report a clear failure (not a guess) when no supported template fits the composition. | Must | Unsupported composition returns a named failure the Critique Engine can read. |
| SG-11 | Support anion substitution on the oxygen sites (e.g. partial N for O, as in oxynitride perovskites), with anions ordered to keep the cell charge-balanced. | Should | An ABO2N candidate builds a valid, charge-balanced structure. |
| SG-12 *(new)* | Let the orchestrator build an exact MP polymorph by `mp_id`, and report the alternatives it did not pick, so it can confirm the ambient phase from the literature. | Must | `build_structure(source="mp", mp_id=...)` builds that entry; MP builds list the other polymorphs. |
| SG-13 *(new)* | Estimate template lattice parameters from the B–X bond length with a tolerance-factor correction, within ~2 % of known cubic cells. | Should | SrTiO3, BaTiO3, SrZrO3 templates within 2 % of measured cubic a. |

### Functions

| Function | Inputs | Output | Notes |
| --- | --- | --- | --- |
| `mp_polymorphs` | composition (str) | list of {mp\_id, space\_group, e\_hull, experimental, n\_icsd, band\_gap\_pbe, structure} | All MP entries for the exact composition, cached. |
| `choose_polymorph` | polymorph list | entry | Default pick (SG-1). |
| `lookup_mp_structure` | composition (str), mp\_id (str or null) | {structure, mp\_id, space\_group, experimental, alternatives} or null | The given `mp_id`, else `choose_polymorph` (SG-1, SG-12). |
| `template_registry` | none | dict: structure\_type to template builder | Minimum key: `perovskite_cubic`. Later: tetragonal and orthorhombic perovskite, spinel, rocksalt, fluorite, L12 (SG-3). |
| `assign_sites` | composition (str), structure\_type (str) | dict: site label (A, B, X) to {element: fraction} | Uses oxidation states and ionic radii, not list order (SG-5). |
| `choose_supercell` | site occupancies, max\_atoms (int, default 80) | list\[int\], 3 values | Smallest cell matching every fraction within 5% (SG-4, SG-8). |
| `build_from_template` | structure\_type, site occupancies, supercell, seed (int) | pymatgen structure | Random site decoration with a fixed seed (SG-6). Handles anion substitution when enabled (SG-11). |
| `check_geometry` | pymatgen structure | list of problems (empty when valid) | Rejects atoms closer than 0.7 × sum of covalent radii (SG-9). |
| `generate_structure` | Candidate | StructureRecord, or a named failure {code: no\_template or geometry, message} | Public entry point: MP lookup, else template build, then geometry check (SG-10). |
| `generate_disorder_variants` | Candidate, k (int, 2 to 3) | list\[StructureRecord\] | Different seeds; used by the Final Calculation Runner (SG-7, FC-4). |

## 5. Simulation Engine

The Simulation Engine relaxes each structure with a universal ML interatomic potential and runs the stability checks. It answers two questions per candidate: is it thermodynamically stable, and is it dynamically stable.

**Input:** a starting structure and the candidate it came from.

**Output:** a simulation result: relaxed structure, space group, lattice parameters, density, total energy, formation energy per atom, energy above the convex hull, phonon stability verdict with the lowest frequency and where it occurs, optional MD verdict, which potential was used, convergence status, and wall time.

| ID | Requirement | Priority | Acceptance check |
| --- | --- | --- | --- |
| SE-1 | Relax atomic positions and cell shape and volume together. | Must | Relaxed cell differs from the template when it should. |
| SE-2 | Use MACE-MP-0 as the default potential; support CHGNet as an alternative (preferred for magnetic systems). | Must | Potential is selectable and recorded per result. |
| SE-3 | Converge to a force threshold of 0.02 eV/Å or report non-convergence; never pass a non-converged result as converged. | Must | Convergence flag matches the optimiser log. |
| SE-4 | Compute formation energy per atom against elemental references taken from one consistent source. | Must | Formation energies of 3 known compounds within 0.1 eV/atom of Materials Project values. |
| SE-5 | Compute energy above hull against the competing phases in that chemical system. | Must | Known stable compound gives ≤ 0.02 eV/atom above hull. |
| SE-6 | Compute phonons at commensurate q-points (and the band path in full mode). Follow imaginary modes by mode mapping (SE-14). | Must | A known unstable cubic phase (CaTiO3) is shown to distort; DFT-stable controls (BaTiO3 R3m, SrZrO3 Pnma, SrSnO3 Pnma) pass. |
| SE-7 | Report where an instability sits (Γ, X, M, R or other high-symmetry point). | Should | Location is reported for flagged cases. |
| SE-8 | Offer a fast phonon mode (smaller supercell) for screening and a full mode for the final candidate. | Must | Both modes run; mode is recorded. |
| SE-9 | Optionally run a short NVT MD (e.g. 10 ps) at operating temperature and report whether the structure stays intact. | Could | MD verdict recorded for the final candidate. |
| SE-10 | Run candidates in one iteration in parallel when cores allow. | Should | Iteration time is less than the sum of single runs. |
| SE-11 | Catch and record failures per step (relax, phonon, MD) without losing the steps that succeeded. | Must | A phonon crash still returns the relaxation result. |
| SE-12 | Report how close the composition is to the potential's training domain (e.g. element coverage, nearest known compound). | Should | Result carries a domain-distance note used by the Critique Engine. |
| SE-13 | Give every candidate two stability verdicts, set only by code. Thermodynamically stable: energy above hull at or below 0.05 eV/atom. Dynamically stable: no imaginary mode whose frozen-in distortion lowers the energy by more than the distortion threshold (mode mapping, default), or no frequency below −0.1 THz (`phonon_gate="strict"`). Thresholds configurable. | Must | Both verdicts present on every result; the DFT-stable controls pass both. |
| SE-14 *(new)* | Mode mapping: freeze the softest imaginary mode into its commensurate supercell, scan amplitudes, re-relax, and compare energies. A drop beyond 2 meV/atom is a real instability: carry the distorted structure forward (and mark the hull stale) and repeat, up to 2 rounds. Otherwise the mode is spurious. Report each round's q-point, energy change and resulting space group. | Must | Cubic BaTiO3 distorts to a polar phase (about −13 meV/atom); Pnma CaTiO3's Γ soft mode is spurious (under 1 meV/atom). |
| SE-15 *(new)* | Run only one GPU job at a time across the process (MACE and MEGNet share one lock). | Must | No concurrent CUDA calls from different threads. |

### Functions

| Function | Inputs | Output | Notes |
| --- | --- | --- | --- |
| `load_potential` | name (mace-mp-0-small, mace-mp-0-medium, chgnet), device (cpu or cuda) | calculator object | Loaded once per process and reused; name and version recorded on every result (SE-2). |
| `relax` | StructureRecord, calculator, fmax\_eV\_per\_A (float, default 0.02), max\_steps (int, default 500) | {relaxed: StructureRecord, energy\_eV\_per\_atom (float), converged (bool), steps (int)} | Positions and cell together (SE-1, SE-3). |
| `get_reference_data` | elements (list\[str\]) | {elemental\_refs: dict element to eV/atom, competing\_phases: list of {formula, energy\_eV\_per\_atom}} | From Materials Project, cached. Competing phases are relaxed once with the same potential so the hull is self-consistent. |
| `formation_energy` | relaxed structure, energy\_eV\_per\_atom, elemental\_refs | float, eV/atom | SE-4. |
| `energy_above_hull` | composition, energy\_eV\_per\_atom, competing\_phases | float, eV/atom | Convex hull built in the candidate's chemical system (SE-5). |
| `phonons` | relaxed structure, potential, config, mode (fast or full), fmax | PhononResult + mode-mapping details (rounds, verdict, distorted structure) | Fast: supercell ≥ 7 Å, commensurate q-points. Full: supercell ≥ 10 Å plus band path. Mode mapping (SE-14). |
| `run_md` | relaxed StructureRecord, calculator, temperature\_K (float), duration\_ps (float, default 10) | {stable (bool), max\_displacement\_A (float)} | Optional (SE-9). |
| `domain_check` | composition, potential name | str note | Element coverage and nearest known compound (SE-12). |
| `stability_verdicts` | energy\_above\_hull, PhononResult, SearchSpec thresholds | {thermo\_stable (bool), dyn\_stable (bool)} | Deterministic rule from SE-13; the only place these booleans are set. |
| `simulate` | StructureRecord, SearchSpec, options {potential, phonon\_mode, run\_md} | SimulationResult | Public entry point; a failed step is recorded in `errors` and later steps still run where possible (SE-11). |
| `simulate_batch` | list\[StructureRecord\], SearchSpec, options, workers (int) | list\[SimulationResult\] | Parallel across candidates (SE-10). |

## 6. Property Predictor

The Property Predictor estimates the target functional property from the relaxed structure, because the interatomic potential alone gives energies and forces, not band gaps or dielectric constants. Its uncertainty is as important as its value.

**Input:** relaxed structure and the target property key.

**Output:** predicted value, uncertainty, the model used, and whether the structure lies inside that model's training domain.

| ID | Requirement | Priority | Acceptance check |
| --- | --- | --- | --- |
| PP-1 | Keep a registry of supported properties, each mapped to a model, its units and its published error. | Must | Registry lists every property the Goal Parser can accept. |
| PP-2 | Support at least: band gap, bulk modulus, shear modulus, formation energy, density. | Must | Each returns a value on a test structure. |
| PP-3 | Derive mechanical properties directly from the potential (strain-stress) where possible, as a cross-check on ML models. | Should | Bulk modulus from both routes agrees within the stated error. |
| PP-4 | Return an uncertainty for every prediction: an ensemble spread where available, else the model's published error for that property. | Must | No prediction without an uncertainty field. |
| PP-5 | Flag predictions for chemistries far from the model's training data and widen the uncertainty accordingly. | Should | Out-of-domain flag is set on a deliberately exotic test case. |
| PP-6 | Return "not available" rather than a substitute property when no model covers the target. | Must | Dielectric constant without a model returns not available. |
| PP-7 | Compare the prediction with the best literature value for the same composition when one exists, and report the gap. | Should | Gap appears for known compositions. |
| PP-8 | Allow a new property model to be added through the registry alone. | Could | Adding an entry exposes a new property end to end. |

**Decision:** the demo property is band gap, with thermodynamic and dynamic stability part of the target. The model is the pretrained multi-fidelity MEGNet (`matgl`, `MEGNet-BandGap-mfi-MP-2019.4.1`) at its **HSE fidelity**. Against 15 measured perovskite gaps, HSE has MAE 0.65 eV (mean signed error −0.33). PBE has MAE 1.58 eV and always runs low. Neither orders all pairs correctly (about 75–80 %): BaSnO3 comes out below SrTiO3 on both scales, though experiment says otherwise. So the reviewer must judge the lab-scale plausibility of a model-scale win (SC-12). Relative targets still use the same model and fidelity for the reference and the candidates (GP-11). Do not train a new model in-session.

### Functions

| Function | Inputs | Output | Notes |
| --- | --- | --- | --- |
| `property_registry` | fidelity (hse or pbe) | dict: property key to {model, units, published\_error, error\_note, scale\_note} | Must include `band_gap` (eV, MEGNet at the chosen fidelity), `bulk_modulus` (GPa), `shear_modulus` (GPa), `formation_energy` (eV/atom), `density` (g/cm³) (PP-1, PP-2). |
| `is_supported` | property key (str) | bool | Used by the Goal Parser before a run starts (GP-2). |
| `predict` | relaxed StructureRecord, property key | PropertyPrediction, or {status: not\_available} | Ensemble spread as uncertainty when available, else the registry's published error (PP-4, PP-6). |
| `elastic_from_potential` | relaxed StructureRecord, calculator | {bulk\_modulus\_GPa, shear\_modulus\_GPa} | Strain-stress route; cross-check for the ML value (PP-3). |
| `domain_flag` | composition, model name | {in\_domain (bool), widen\_factor (float)} | Widens the uncertainty when out of domain (PP-5). |
| `compare_to_literature` | PropertyPrediction, LiteratureContext | PropertyPrediction with `literature_value` set | PP-7. |

## 7. Reviewer (self-critique)

An independent reviewer LLM, separate from the orchestrator, reviews each round when the orchestrator calls `request_review`. It critiques like a harsh expert referee, and gives its own confidence that the best candidate would work in the lab. Code computes the facts it reviews: the best of the round, the target check, the stability verdicts and the rubric score. The orchestrator must respond to each review.

**Input:** the orchestrator's summary of the round, code facts for every candidate (with phonon interpretations), the code-fixed best of the round and its rubric score, a novelty search, the recorded evidence and papers, and earlier reviews.

**Output:** a `Critique`: code checks and rubric score, findings, one recommended next composition, a stop recommendation, and the reviewer's own confidence with a reason.

| ID | Requirement | Priority | Acceptance check |
| --- | --- | --- | --- |
| SC-1 | Decide "target met" by a deterministic check covering the property target and both stability verdicts; no LLM can override it. | Must | Target-met flag always equals the numeric rule. |
| SC-2 | Judge target met in two ways: on the central value, and on the value minus its uncertainty. | Must | Both verdicts appear in the critique. |
| SC-3 | Interpret every phonon instability physically (tilt, polar mode, phase transition, wrong structure type). | Must | Flagged instabilities always have an interpretation. |
| SC-4 | Assess reliability of the potential and the property model for this chemistry. | Must | A reliability finding is present every round. |
| SC-5 | Include at least one sceptic challenge: novelty versus the literature, or synthesis feasibility. | Must | A finding of that category is present. |
| SC-6 | Recommend exactly one specific next composition (an exact formula). | Must | Recommendation parses as a valid formula. |
| SC-7 | Not repeat a recommendation, and not recommend a composition already tried. | Must | Validation rejects repeats; the reviewer is asked again once. |
| SC-8 | Compute the confidence score from the rubric below, in code. | Must | Score equals the sum of the breakdown. |
| SC-9 | Cite literature by Valency paper ID when a finding relies on a paper. | Must | Literature-based findings carry an ID. |
| SC-10 | When the reviewer LLM fails, record a minimal critique with the code checks and score, and say the review was unavailable. | Must | Simulated LLM failure still produces a critique. |
| SC-11 | Check novelty with a targeted Valency search for the best candidate's exact composition. | Should | Novelty finding reports hits or none. |
| SC-12 *(new)* | Report the reviewer's own confidence (0–100) with a one-sentence reason, separate from the code rubric. It may differ from the rubric, e.g. when a model-scale win is unlikely to hold in the lab. | Must | `llm_confidence` and its reason are present in every LLM review. |
| SC-13 *(new)* | Check each candidate was evaluated in the polymorph stable at the operating temperature. | Should | A wrong-polymorph evaluation draws a finding. |

**Confidence rubric** (code, unchanged): property target 30 (30 if met on value minus uncertainty, 15 if met on the central value only); thermodynamic stability 20 (20 if E\_hull ≤ 0.025, 10 if ≤ 0.05); dynamic stability 20; model reliability 15 (5 each: converged, inside the potential's domain, inside the property model's domain); novelty and synthesis 15 (8 if no prior report of the composition, 7 if the literature gives a plausible synthesis route).

### Functions

| Function | Inputs | Output | Notes |
| --- | --- | --- | --- |
| `select_iteration_best` | list of (SimulationResult, PropertyPrediction), SearchSpec | candidate\_id | Most checks passed first, then closest to target. |
| `check_target` | PropertyPrediction, SimulationResult, SearchSpec | {target\_met\_central, target\_met\_conservative, thermo\_stable, dyn\_stable, all\_passed} | Pure rule (SC-1, SC-2). |
| `score_rubric` | check\_target output, SimulationResult, PropertyPrediction, novelty\_hits, has\_synthesis\_route | {score, breakdown} | SC-8. |
| `phonon_interpretation` | SimulationResult, mode-mapping details | str | Physical reading of the phonon result (SC-3). |
| `review_round` | as in the input above | Critique | One structured LLM call; validated by `validate_critique` and asked again once on failure. |
| `validate_critique` | Critique, previous critiques, tried compositions | list of problems | SC-6, SC-7, SC-9, SC-4, SC-5. |
| `request_review` (tool) | summary | review result + stop instruction | See `docs/TOOLS.md`. |

## 8. Orchestrator and convergence

The orchestrator is an LLM agent. After goal parsing, it runs the campaign through the tools in `docs/TOOLS.md`:
- reading the literature and database
- proposing candidates
- building, relaxing and checking them
- predicting properties
- asking for reviews
- finally submitting one candidate

It decides what to compute and in what order. Code holds the budget, the gates and the stop rules. The conversation is append-only (thinking blocks are passed back unchanged).

**State it holds:** the `Workspace` (candidates, structures and results, papers, evidence, round, budget counters, submission), mirrored into `RunState` after every tool call; review critiques; usage.

**Stop rules, checked in this order after every review.** When one fires, the orchestrator is told to submit:

1. User stop.
2. Maximum review rounds reached (default 3).
3. Target met. All four must hold:
   - the code checks pass (SC-1)
   - the code rubric is ≥ threshold (default 72)
   - the reviewer's own confidence is ≥ the same threshold
   - the reviewer raised no critical finding
4. Reviewer recommends stopping, with a reason.
5. Confidence plateau: less than 5 points of change in the rubric across the last 3 rounds.
6. Persistent instability: the best candidate is dynamically unstable for 2 rounds running.
7. Wall-time budget exhausted.

The tool-call budget (OR-10) and a user stop also prompt submission. A hard limit (budget + 8 calls, or 125 % of the wall time) ends the campaign and code picks the best candidate.

| ID | Requirement | Priority | Acceptance check |
| --- | --- | --- | --- |
| OR-1 | Fixed outer order: parse the goal, the capability phase (only when there is a gap, Section 14), run the agent campaign, final re-check, brief. Inside the campaign the orchestrator chooses the order of tool calls, but every result passes through the code gates. | Must | Log order of the outer steps matches on every run. |
| OR-2 | Apply the stop rules above, in order, and record which one fired. | Must | Stop reason is always one of the rules, or "submitted". |
| OR-3 | The submitted candidate is the answer; without a submission, code picks the best by checks, then distance to target. | Must | A late worse round does not replace an earlier better submission. |
| OR-4 | Make thresholds, budgets and limits configurable per run (`RunConfig`). | Must | Changing the threshold changes stop behaviour. |
| OR-5 | Append to history; never overwrite earlier rounds. | Must | All rounds are present in the final state. |
| OR-6 | Save state and a workspace snapshot after every tool call. Resume an interrupted run (crash or restart): restore the workspace, start a fresh orchestrator conversation told what was already done, and continue in the same run folder. | Must | A killed run resumes without repeating finished calculations; a run with only `state.json` also resumes. |
| OR-7 | Emit progress events for every tool call (started, finished or failed), the orchestrator's text, reviews, round results and the final steps. | Must | UI and CLI update live during a run. |
| OR-8 | Let a user stop the run early; the orchestrator submits and a brief is still produced. | Should | Manual stop produces a brief. |
| OR-9 | Enforce a total wall-clock budget. | Should | Budget stop is recorded as a stop reason. |
| OR-10 *(new)* | Enforce a tool-call budget (default 90). | Must | The orchestrator is told to submit when it runs out. |
| OR-11 *(new)* | Reach the LLM by a backend chosen per run: Anthropic API (`api`); Claude Code login through the Claude Agent SDK (`claude-code`), with built-in Claude Code tools disabled and no user settings loaded; or `auto` (`api` when API credentials exist). The Claude model is chosen per run (default `claude-opus-5-5`). The orchestrator, reviewer, goal parser and brief all use the run's choice. | Must | The same campaign runs on both backends; a signed-out Claude Code is reported, not left hanging. |
| OR-12 *(new)* | Tell the orchestrator why a stop was blocked (critical finding, or reviewer confidence below the threshold). | Should | The review result carries a gate note. |

### Functions

| Function | Inputs | Output | Notes |
| --- | --- | --- | --- |
| `new_run` | raw\_goal, RunConfig, spec (optional) | RunState | Creates the run folder (DL-1). |
| `run` | raw\_goal, RunConfig, spec, answers, state, client, resume | RunState (finished) | Public entry point: parse, campaign, final re-check, brief. `client` injects a scripted model for tests. |
| `resume` | run\_id, background | run\_id or RunState | OR-6. Refuses finished runs and runs already running. |
| `should_stop` | RunState, RunConfig, start time | {stop, rule} | OR-2, OR-9. |
| `sync_state` | RunState, Workspace, critiques | RunState | One IterationRecord per review round. |
| `request_stop` | run\_id | none | OR-8. |
| `replay` | run\_id, speed | replay\_id | Re-emits stored events (UI-9). |
| Agent loop | Workspace, config, backend | submission or none | API backend: Messages API with tools, adaptive thinking, refusal fallback, prompt caching. Claude Code backend: Agent SDK with the tools on an in-process MCP server. |

## 9. Final Calculation Runner

The Final Calculation Runner re-checks the winning candidate (and the runner-up) at higher accuracy than screening allowed. Its job is to confirm or overturn the loop's answer before anyone spends lab time on it.

**Input:** the submitted candidate and the runner-up named in the submission.

**Output:** a validation report: the same quantities as the Simulation Engine at higher fidelity, the change from the screening values, and a verdict (confirmed, weakened, overturned).

| ID | Requirement | Priority | Acceptance check |
| --- | --- | --- | --- |
| FC-1 | Re-relax the winner with the larger, more accurate potential model and tighter force convergence (e.g. 0.005 eV/Å). | Must | Final run records the higher-fidelity settings. |
| FC-2 | Run full phonons (larger supercell, full band path) on the winner. | Must | Band-path phonons recorded. |
| FC-3 | Repeat with a second potential (e.g. CHGNet) and report agreement between the two. | Could | Two-potential comparison table present. |
| FC-4 | Test 2 to 3 disorder configurations of the winner and report the energy spread. | Should | Spread reported in eV/atom. |
| FC-5 | Run MD at operating temperature on the winner. | Could | MD verdict present. |
| FC-6 | Validate the runner-up at the same fidelity so the brief can offer a tested fallback. | Should | Runner-up has a validation report. |
| FC-7 | Mark the verdict "overturned" when the final run loses stability or misses the target the screening run met. | Must | Verdict rule is applied automatically. |
| FC-8 | Optionally export DFT-ready input files (structure plus suggested settings) for users with DFT access. | Could | Export files written for the winner. |
| FC-9 | Stay within a separate time budget (e.g. 10 minutes on CPU) and skip optional checks first when short. | Should | Budget overrun drops Could items before Should items. |

### Functions

| Function | Inputs | Output | Notes |
| --- | --- | --- | --- |
| `refine` | best StructureRecord, SearchSpec | SimulationResult | Larger potential, fmax 0.005 eV/Å, full phonons (FC-1, FC-2). Reuses Simulation Engine functions with different options. |
| `cross_check_potential` | best StructureRecord, second potential name | SimulationResult | Stretch item (FC-3). |
| `disorder_spread` | Candidate | {energies\_eV\_per\_atom (list\[float\]), spread\_eV\_per\_atom (float)} | Uses `generate_disorder_variants` (FC-4). |
| `verdict` | screening SimulationResult and PropertyPrediction, final ones, SearchSpec | ValidationReport | Overturned if the final run fails any check the screening run passed (FC-7). |
| `export_dft_inputs` | relaxed StructureRecord | folder path | Optional (FC-8). |
| `validate_finalists` | RunState, time\_budget\_s | list\[ValidationReport\] (winner, runner-up) | Public entry point; drops Could items first when short of time (FC-6, FC-9). |

## 10. Experimental Brief Generator

The Experimental Brief Generator writes the document a researcher takes to the lab. It may only state what earlier sections produced; it adds explanation and structure, never new numbers.

**Input:** the full run state plus the validation report.

**Output:** a brief with these parts, in order: recommended composition and structure; predicted properties against target, each with uncertainty; physical rationale; validation verdict; synthesis route; characterisation plan; risks and fallbacks; runner-up candidates; iteration history; literature support with links; scope and assumptions.

| ID | Requirement | Priority | Acceptance check |
| --- | --- | --- | --- |
| EB-1 | Contain every part listed above, in that order. | Must | Template check passes. |
| EB-2 | Take every number from the run state; the LLM writes prose only. | Must | Each number in the brief matches a stored value. |
| EB-3 | Show predicted properties as a table: property, predicted value ± uncertainty, target, met or not. | Must | Table present with all four columns. |
| EB-4 | Base the synthesis route on literature synthesis insights for the same or closest composition, citing them; for a compound never made, follow the closest analogues and say which. Lab conditions (temperatures, times, atmospheres) come from cited evidence or abstracts, or are marked "inferred"; computed results still come only from the run state (EB-2). | Must | Each synthesis step names its source paper or says "inferred". |
| EB-5 | Order the characterisation plan by what most quickly confirms or kills the candidate (e.g. XRD for phase first). | Must | First step tests phase identity. |
| EB-6 | List risks with severity and a concrete fallback for each critical one. | Must | Every critical risk has a fallback. |
| EB-7 | Link each cited paper to its Valency record. | Must | All citations resolve. |
| EB-8 | Give one line per iteration showing composition, key result and confidence. | Should | History length equals iteration count. |
| EB-9 | Include the scope note: corpus coverage, potential used, property model domain, assumptions from the Goal Parser. | Must | Scope section present. |
| EB-10 | Export as Markdown and PDF. | Should | Both formats open correctly. |
| EB-11 | Say plainly when the target was not met, and present the best attempt as such. | Must | A failed run's brief opens with that fact. |

### Functions

| Function | Inputs | Output | Notes |
| --- | --- | --- | --- |
| `collect_facts` | RunState | dict of every number and label the brief may show, each with its source | The only source of numbers for the brief (EB-2). |
| `property_table` | facts | Markdown table: property, predicted ± uncertainty, target, met | Built in code, not by the LLM (EB-3). |
| `write_sections` | facts, LiteratureContext, ValidationReport, submission | dict: section name to Markdown prose | One LLM call per section (rationale, synthesis route, characterisation plan, risks, literature support), given the orchestrator's justification. Numbers appear only as placeholders filled from `facts`. Template sections replace any LLM section that fails the number check. |
| `check_numbers` | draft Markdown, facts | list of numbers not found in facts | Any hit blocks the brief until fixed or removed (EB-2). |
| `link_citations` | draft Markdown, list\[Paper\] | Markdown with Valency links | EB-7. |
| `assemble_brief` | sections, tables, scope note, iteration history | Brief (Markdown) | Fixed section order (EB-1). Opens with "target not met" when that is true (EB-11). |
| `export_pdf` | Brief | file path | EB-10. |

## 11. User interface

The interface lets a researcher enter a goal, watch the agent reason and simulate live, and read the brief. For the hackathon it is also the demo, so watching the loop matters as much as the result.

| ID | Requirement | Priority | Acceptance check |
| --- | --- | --- | --- |
| UI-1 | One text box for the goal and a Run button. | Must | A goal can be submitted in one step. |
| UI-2 | Show the parsed specification and any clarifying questions before the run starts. | Must | Questions appear and answers feed the run. |
| UI-3 | Show live progress per section, driven by Orchestrator events. | Must | Each section lights up as it runs. |
| UI-4 | Show literature found: count, top papers with links, failed approaches. | Must | Panel populates after mining. |
| UI-5 | Show each iteration as a card: candidates, key simulation numbers, critique summary, confidence. | Must | One card per iteration. |
| UI-6 | Plot confidence across iterations. | Should | Line updates after each critique. |
| UI-7 | Render the relaxed structure of the current best candidate in 3D. | Should | Structure viewer rotates. |
| UI-8 | Show the final brief with export buttons. | Must | Brief renders; exports download. |
| UI-9 | Offer a replay mode that plays a cached run at demo speed with no network. | Must | Demo runs with Wi-Fi off. |
| UI-10 | Let the user stop a run and still get a brief. | Should | Stop button produces a brief. |
| UI-11 | Prompt the Valency sign-in on first use with a one-line explanation; show "sign in again" when the stored token was rejected. | Must | First run shows the sign-in prompt. |
| UI-12 *(new)* | Model access selector (Auto, Anthropic API key, Claude Code login) with readiness shown, a "Sign in to Claude" button that opens the Claude page and finishes by itself after authorising, and a model selector. | Must | Choosing Claude Code while signed out offers the sign-in; after authorising, the backend shows ready. |
| UI-13 *(new)* | "Require novel materials" checkbox next to the goal. | Should | The run's config and spec reflect it. |
| UI-14 *(new)* | Live orchestrator log: reasoning, each tool call with a one-line result, rationales, reviews and the submission; scrolls without squashing older entries. | Must | Every tool call appears. |
| UI-15 *(new)* | Confidence chart with two lines: the code rubric (drives the stop rules) and the reviewer's own confidence. | Should | Both lines update after each review. |
| UI-16 *(new)* | Resume button on interrupted runs; the page shows a run's full history after a server restart. | Must | Resume continues the run; history before the restart is visible. |

### Backend endpoints

The UI talks to the backend over HTTP plus one event stream. All bodies are JSON in the shared data types.

| Endpoint | Request | Response | Notes |
| --- | --- | --- | --- |
| `POST /parse` | {raw\_goal, answers, llm\_backend, llm\_model} | {spec: SearchSpec, questions: list\[str\], errors, parser} | UI-2. |
| `POST /runs` | {spec: SearchSpec, config: RunConfig} | {run\_id} | Starts a run in the background; 503 when the chosen LLM backend is not ready. |
| `GET /runs/{run_id}/events` | none | stream of ProgressEvent (server-sent events) | Drives the live view (UI-3, UI-5, UI-6). |
| `GET /runs/{run_id}` | none | RunState | Full state for refresh or replay. |
| `GET /runs/{run_id}/structure/{candidate_id}` | none | CIF text | For the 3D viewer (UI-7). |
| `GET /runs/{run_id}/brief` | query: format (md or pdf), kind (lab or compute) | file | UI-8. |
| `POST /runs/{run_id}/stop` | none | {stopping: true} | UI-10. |
| `POST /replay/{run_id}` | {speed (float, default 1)} | {replay\_id} | Re-emits a stored run's events with no network (UI-9). |
| `GET /auth/valency` | query: start (bool) | {signed\_in, needs\_signin, access\_token\_expired, pending, url, error} | `?start=true` opens the browser sign-in (UI-11). |
| `POST /runs/{run_id}/resume` *(new)* | none | {run\_id, resumed} | OR-6, UI-16. 409 if finished or already running. |
| `GET /auth/claude`, `POST /auth/claude/start`, `POST /auth/claude/code`, `POST /auth/claude/cancel` *(new)* | {code} for `/code` | sign-in state {status, url, signed\_in, error} | Runs `claude auth login --claudeai`; the code endpoint is a fallback when the page shows a code (UI-12). |
| `GET /runs/{run_id}/tool-proposals` *(new)* | none | list\[ToolProposal\] | Pending and decided proposals, with source at the code stage (AT-13). |
| `POST /runs/{run_id}/tool-decisions` *(new)* | ToolDecision | {accepted: true} | AT-7, AT-13. 409 if the proposal is not waiting for that stage. |
| `GET /tools`, `POST /tools/{name}/approve`, `DELETE /tools/{name}` *(new)* | none | list\[CustomTool\] and unapproved files; the approved CustomTool; {removed: true} | The tool library (AT-20, AT-21). Approve takes the file's current sha256 and refuses if it changed since it was shown. |
| `GET /models`, `GET /health`, `GET /config/defaults` *(new)* | none | models; backend, Valency and device status; RunConfig defaults | `/health` never blocks: slow checks run in the background. |

## 12. Data, logging and persistence

Every run leaves a complete, inspectable record, and every expensive external call is cached. This is what makes the demo safe and the results auditable.

| ID | Requirement | Priority | Acceptance check |
| --- | --- | --- | --- |
| DL-1 | Store each run in its own folder named by run ID: specification, literature context, every candidate, structure, simulation result, critique, validation report and brief. | Must | Folder holds all artefacts after a run. |
| DL-2 | Save structures in a standard format (CIF) alongside the result records. | Must | CIFs open in a structure viewer. |
| DL-3 | Cache Valency responses, LLM responses and Materials Project responses keyed by their exact inputs. | Must | Repeat call is served from cache. |
| DL-4 | Log every LLM prompt and response with the section and iteration it belongs to. | Must | Any critique can be traced to its prompt. |
| DL-5 | Log every external call with time, duration and success or failure. | Should | Log shows timings per call. |
| DL-6 | Record versions of potentials, property models and the LLM used. | Must | Versions appear in the run record and the brief. |
| DL-7 | Keep OAuth tokens in a user-only file outside the repository. | Must | Token file permissions are owner-only. |
| DL-8 | Provide a script that replays a stored run into the UI. | Must | Replay works from a copied run folder. |
| DL-9 | Track token use and cost per run. | Could | Cost shown at end of run. |
| DL-10 *(new)* | Save `workspace.json` (the agent's workspace) after every tool call, and load a run's event history from disk when a new process first publishes to it. | Must | Resume works; history survives restarts. |
| DL-11 *(new)* | Enable `faulthandler` in the server, writing all threads' native stacks to `runs/_crash/`. | Should | A native crash leaves a stack dump. |
| DL-12 *(new)* | Read and write every file as UTF-8, and retry atomic replaces that Windows refuses while another process reads the file. | Must | "≤" and "Å" round-trip; concurrent reads do not crash a save. |

### Functions

| Function | Inputs | Output | Notes |
| --- | --- | --- | --- |
| `cached` | namespace (llm, valency, mp), key inputs (JSON-serialisable), compute function | the compute function's result | Key = hash of namespace + canonical JSON of inputs + model or tool version. Stored on disk (DL-3). |
| `save_record` | run\_id, relative path, record | none | Writes JSON (and CIF for structures) atomically (DL-1, DL-2). |
| `load_record` | run\_id, relative path, type | record | Validates against the shared type on load. |
| `log_llm_call` | section, iteration, prompt, response, model, tokens | none | Appends to `llm_log.jsonl` (DL-4, DL-9). |
| `log_external_call` | service, operation, duration\_s, ok (bool), error | none | Appends to `calls.jsonl` (DL-5). |
| `versions` | none | dict of package, potential, property model and LLM versions | Stored in `state.json` and the brief (DL-6). |

**Run folder layout**

```
runs/<run_id>/
  state.json
  spec.json
  literature.json
  events.jsonl
  llm_log.jsonl
  calls.jsonl
  workspace.json
  progress.json
  submission.json
  capability/          # only when there was a gap (Section 14)
    proposals.json  decisions.json
    drafts/<name>-r<revision>.py  drafts/test_<name>-r<revision>.py
    checks/<name>-r<revision>.json
  iter_1/
    candidates.json
    critique.json
    c1-1/  start.cif  relaxed.cif  sim.json  prop.json
    c1-2/  ...
  iter_2/ ...
  final/
    validation.json
    relaxed.cif
  brief.md
  brief.pdf
cache/
  llm/  valency/  mp/  custom/

hephaestus_tools/      # in the working directory (tool_library_dir), shared by runs
  manifest.json
  <name>.py  test_<name>.py
```

## 13. Hackathon MVP cut

For the 24-hour build, ship every **Must**, restricted to oxide perovskites. The team split, package layout and coding rules are in `CLAUDE.md`; the tool contracts are in `docs/TOOLS.md`.

**MVP restrictions**

- Material family: ABO3 oxide perovskites and their solid solutions.
- Structure types: cubic perovskite template plus Materials Project polymorphs (SG-1, SG-2, SG-12).
- Target property: band gap (MEGNet HSE fidelity), plus thermodynamic and dynamic stability; bulk modulus as the backup demo.
- Potentials: MACE-MP-0 small for screening (float32 on GPU is fine), MACE-MP-0 medium (float64) for the final re-check.
- Phonons: fast mode with mode mapping in the loop, full mode on the winner only.
- Review rounds: 3 maximum; confidence threshold 72 for both the rubric and the reviewer.
- LLM: Claude Opus 5.5 by default, via API key or Claude Code login.
- Demo: a pre-cached run in replay mode (UI-9), plus one live run if time allows.
- Agent-built tools (Section 14), demonstrated by rebuilding the elastic tool with `elastic` disabled.

## 14. Agent-built tools *(new)*

When a goal asks for a property that no tool can compute, the agent can build the tool itself. It proposes a
plan, writes the tool only once the user accepts that plan, and the user reads the code before it runs. Approved
tools are saved as files in the working directory and reused by later runs; nothing is added to the package.

Code decides when a tool may be proposed, runs every tool in a sandbox, and keeps agent-built results out of the
stability verdicts. The tool file format, the toolkit and the event payloads are in `docs/TOOLS.md`
(*Agent-built tools*).

**Flow.** The capability phase runs between goal parsing and the campaign, and only when there is a gap:

1. **Gap (code).** The target property is not in the property registry. The registry here means the built-in
   properties, minus `RunConfig.disabled_capabilities`, plus approved library tools.
2. **Library.** An approved library tool for the same property is reused without asking. Unapproved files in
   the library are shown for code review first (AT-21).
3. **Scoping.** The scoping literature search also looks for how papers compute the property. The agent reads
   our own tools for inspiration and scope.
4. **Plan.** The agent proposes the tool. The user decides.
5. **Code.** The agent writes the tool and its tests, and code checks them.
6. **Code review.** The user reads the code and decides.
7. **Registration.** The tool is saved to the library and added to the campaign's tool list.

**User decisions**, at both the plan and the code stage:

| Action | Effect |
| --- | --- |
| Accept | Plan stage: the agent writes the tool. Code stage: the tool is saved and registered. |
| Request changes | The user's message goes to the agent, which revises and proposes again (up to `max_tool_revisions`). |
| Cancel the tool | The run continues without it. If it supplied the target property, the run screens stability only and cannot meet the target. |
| Cancel the run | The run ends before the campaign, with stop reason "cancelled by user". |

**Decisions made.**
- Agent-built results can meet the target. The target verdict is labelled "agent-built tool" and no
  uncertainty is shown. That label is the value's explicit no-uncertainty statement (SYS-4). Relative targets
  keep the comparison fair: the reference is computed with the same tool. "Target met" still needs the
  reviewer's confidence above the threshold and no critical finding (OR-2).
- No calibration step. The user's review of the plan and the code is the check.
- The agent may write the tool from scratch or use our building blocks. Atomistic-model evaluations (energies,
  forces, stresses, relaxations, phonons, MD) go through the toolkit, so they run under the GPU lock and the
  cache (AT-15).
- Tools live in the working directory, not in the package. A user can also write a tool by hand and import it.

| ID | Requirement | Priority | Acceptance check |
| --- | --- | --- | --- |
| AT-1 | Code, not the LLM, detects a capability gap: the target property is not in the registry (built-in, minus `disabled_capabilities`, plus approved library tools). Only a gap opens the capability phase. | Must | A band gap goal never enters the phase; a thermal expansion goal does. |
| AT-2 | The agent can propose and write tools only in the capability phase. The capability tools are not in the campaign's tool list. | Must | Calling `propose_tool` during the campaign returns an error. |
| AT-3 | `RunConfig.custom_tools` is `off` or `ask`. Defaults: `off` for `hephaestus run` and the Python API, `ask` for the interactive CLI and the UI. With `off`, a gap ends the run before the campaign with a message naming the missing property. | Must | A gap with `off` stops with that message and no campaign. |
| AT-4 | The scoping literature search records how papers compute the missing property as `Insight`s with finding\_type `method`, and the plan cites them. | Should | A plan's citations are papers found in this run. |
| AT-5 | `read_tool_source` returns the signatures and docstrings of the public functions in `tools/simulation.py` and `tools/structures.py`, and the full source of one named function on request. It hides the code of disabled capabilities. | Must | With `elastic` disabled, `elastic_tensor` is neither listed nor returned. |
| AT-6 | A plan states: name, property, units, description, method with citations, inputs, building blocks used (may be none), expected runtime, whether it supplies the target, and what the run loses if it is cancelled. | Must | A plan missing a field is refused by the tool schema. |
| AT-7 | At the plan stage the user can accept, request changes with a message, cancel the tool or cancel the run. Revisions are capped by `max_tool_revisions` (default 3); after that, only accept or cancel are offered. | Must | Each of the four actions has its effect; a fourth revision request is refused. |
| AT-8 | When the cancelled tool supplied the target property, the card says so before the user chooses. The run then screens stability only, "target met" cannot fire, and the brief says the target property was not evaluated. | Must | After cancelling, the stop rule is never "target met" and the brief names the missing property. |
| AT-9 | Cancel the run ends the run before the campaign, with stop reason "cancelled by user" and no brief. | Must | The state records the stop reason; no `iter_1/` folder exists. |
| AT-10 | The agent writes the tool only after the plan is accepted. It writes a tool file and a test file in the format of `docs/TOOLS.md`. | Must | No source exists before a plan `accept`. |
| AT-11 | Static checks (code): imports only from the allowlist; no `open`, `exec`, `eval`, `compile`, `__import__`, `getattr`, `setattr`, `globals`, `vars`, `input` or `breakpoint`; no access to attributes starting with `_`; at most 500 lines; a valid `TOOL` header and `run` function. A failure goes back to the agent, which gets up to 3 fix attempts that do not count as user revisions. | Must | `import os`, `open(...)` and `x.__class__` each fail with a named reason. |
| AT-12 | The agent's tests and one trial run, on the reference material (or SrTiO3 when the goal has none), pass in the sandbox. The trial result is a finite number in the declared units. | Must | A tool returning NaN or the wrong units fails the trial and goes back to the agent. |
| AT-13 | The code review shows the full source and test file with syntax highlighting, the check results, the trial output and the toolkit calls used. The user has the same four actions as at the plan stage. | Must | The card shows every line of the file that will be saved. |
| AT-14 | Each tool call runs in a worker process: Python's isolated mode (`python -I`), allowlisted imports only, sockets disabled, CPU and memory limits (POSIX; timeout only on Windows), and `custom_tool_timeout_s` (default 300). Inputs and outputs pass as JSON. | Must | A tool that loops forever is killed at the timeout and the call returns an error. |
| AT-15 | The worker asks for building blocks over a pipe. The main process runs them with the run's potential, under the GPU lock and the cache. The worker never imports torch, MACE or matgl. | Must | Custom tool calls in parallel with relaxations never use the GPU at the same time; `import torch` fails the static check. |
| AT-16 | Approved tools are added to the campaign's tool list as `custom_<name>` before the campaign's LLM session starts. Inputs are `candidate_id` plus the declared inputs. Each result is stored as a `PropertyPrediction` with model `agent-built:<name>@<sha8>` and uncertainty null, with the note "agent-built tool". | Must | The campaign's first turn lists the tool; its predictions carry the model tag. |
| AT-17 | If the tool supplies the target property, the target check uses it for the reference material and every candidate. The verdict is labelled "agent-built tool" in the UI, CLI and brief, and no uncertainty is shown. The conservative target check equals the central one. The reviewer receives the tool's source. | Must | The reference value comes from the tool; the label appears in all three outputs. |
| AT-18 | Agent-built results never feed the thermodynamic, dynamic or mechanical stability verdicts, budgets or stop rules other than the target check. | Must | A tool returning any value leaves `thermo_stable` and `dyn_stable` unchanged. |
| AT-19 | The final re-check runs the tool again on the winner and the reference, with the final potential and dtype. | Should | `validation.json` has a check row for the tool's property. |
| AT-20 | On approval, the tool is saved as `<tool_library_dir>/<name>.py` and `test_<name>.py`, and its sha256 is added to `manifest.json` as a `CustomTool`. Runs import only files whose hash matches the manifest. | Must | Editing one byte of an approved file stops it loading until it is approved again. |
| AT-21 | A new or edited file in the library, including one written by hand, gets the same code review before it is imported: at the start of the capability phase, in the UI's tool library, or with `hephaestus tools review`. Before approval, a file is only parsed (for its header), never imported. | Must | A hand-written tool appears for review and loads after approval. |
| AT-22 | An approved library tool for the gap's property is reused without a new proposal, with an event saying so. | Should | A second run with the same gap proposes nothing. |
| AT-23 | Cache keys for custom tool results include the tool's sha256, so a revised or edited tool never returns an earlier version's result. | Must | Changing one line of the tool recomputes its result. |
| AT-24 | Every proposal and decision is an event and is saved in the run folder. Replay and resume never ask again. A run killed while waiting for a decision shows the same card again on resume. | Must | Resume after a kill at the code review shows the card; replay shows the decision without asking. |
| AT-25 | Time spent waiting for the user counts against no budget. `capability_time_budget_s` (default 600) bounds the agent's work in the phase; when it runs out, unfinished tools are treated as cancelled. At most `max_custom_tools` (default 3) tools per run. | Must | A run that waits 20 minutes for a decision still runs its full campaign. |
| AT-26 | `RunConfig.disabled_capabilities` removes built-in capabilities for a run (`elastic` removes every mechanical property). This is for demos and tests. | Must | With `elastic` disabled, a bulk modulus goal opens a gap. |
| AT-27 | The UI shows the plan and the code review as cards in the orchestrator log, with the four actions and a message box. The code card has a syntax-highlighted viewer. A tool library page lists approved and unapproved tools, and the settings panel exposes the `RunConfig` fields of this section. | Must | Every action can be taken from the UI. |
| AT-28 | The interactive CLI shows the plan and the code (paged, highlighted) and asks for the decision. `hephaestus tools list`, `review` and `remove` manage the library. Built on the CLI branch, through the Python API's `on_tool_proposal` callback and `tool_library`, `approve_tool` and `remove_tool`. | Must | Every action can be taken from the CLI. |
| AT-29 | The Python API takes a decision callback (`Experiment(goal, on_tool_proposal=fn)`) and exposes `exp.decide(proposal_id, action, message)` for step-by-step use. | Should | A scripted callback drives a run through both stages. |

### Functions

| Function | Inputs | Output | Notes |
| --- | --- | --- | --- |
| `find_gap` | SearchSpec, RunConfig, library | property name or None | AT-1, AT-26. Pure code. |
| `capability_phase` | RunState, Workspace, LLM, bus, decide | list\[CustomTool\] or a cancel | AT-2 to AT-13. Runs its own orchestrator conversation with the capability tools only. |
| `static_check` | source (str) | {passed, report} | AT-11. AST walk; never imports the source. |
| `run_sandboxed` | tool file, structure dict, inputs, toolkit server, timeout\_s | result dict or `{"error": ...}` | AT-14, AT-15. |
| `load_library` | tool\_library\_dir | {approved: list\[CustomTool\], unapproved: list\[path\]} | AT-20, AT-21. |
| `approve_tool` | path, sha256 shown to the user | CustomTool | Refuses if the file changed since it was shown. |
