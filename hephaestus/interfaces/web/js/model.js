// A run as the UI sees it: folded from ProgressEvents (live) and RunState (GET /runs/{id}).
// Nothing here decides a verdict, gate or stop: those arrive already decided in the payloads.

export const PHASES = [
  { key: "parse", name: "Parse goal" },
  { key: "tools", name: "Build tool" }, // only shown when the run has a capability gap
  { key: "campaign", name: "Campaign" },
  { key: "final", name: "Final re-check" },
  { key: "brief", name: "Lab brief" },
];

const LIT_TOOLS = new Set(["search_literature", "related_papers", "get_paper", "publication_trend", "note_evidence"]);
const DB_TOOLS = new Set(["lookup_materials_project"]);
const SIM_TOOLS = new Set(["build_structure", "relax", "energy_above_hull", "phonons", "run_md", "predict_property"]);

/**
 * Category of a tool, for icons and filters.
 * @param {string} tool Tool name.
 * @returns {"lit"|"db"|"sim"|"review"|"control"|"candidate"}
 */
export function toolKind(tool) {
  if (LIT_TOOLS.has(tool)) return "lit";
  if (DB_TOOLS.has(tool)) return "db";
  if (SIM_TOOLS.has(tool)) return "sim";
  if (tool === "request_review") return "review";
  if (tool === "propose_candidate") return "candidate";
  return "control";
}

export class RunModel {
  /**
   * @param {string} id Run or replay ID.
   */
  constructor(id) {
    this.id = id;
    this.isReplay = id.startsWith("replay-");
    this.reset();
  }

  /** Forget everything folded so far. */
  reset() {
    this.lastSeq = -1;
    this.events = [];
    this.state = null; // RunState from the server
    this.running = false;
    this.ended = false;
    this.goal = "";
    this.config = null;
    this.spec = null;
    this.questions = [];
    this.parseErrors = [];
    this.backend = null;
    this.model = null;
    this.calls = new Map(); // call_id -> call
    this.log = []; // ordered entries: {type, ...}
    this.candidates = new Map(); // candidate_id -> latest flat fields
    this.rounds = []; // per-round orchestrator results
    this.papers = new Map(); // paper_id -> {paper_id, title, year, citations}
    this.submitted = null;
    this.final = null; // {verdict, checks, notes} | {error}
    this.finalStarted = false;
    this.brief = null; // {pdf, cited}
    this.briefStarted = false;
    this.runResult = null; // payload of run finished/failed
    this.phase = { parse: "pending", tools: "pending", campaign: "pending", final: "pending", brief: "pending" };
    this.capability = null; // {gap, waiting: {proposal_id, stage} | null, registered, cancelled} once a gap opens
    this.startTime = null;
    this.lastTime = null;
    this.campaignStart = null; // first orchestrator start
    this.campaignEnd = null; // submission, final check start or run end
    this.finalStart = null;
    this.finalEnd = null;
    this.finalStep = null; // latest final-check tool call
    this.campaignUsage = null; // orchestrator tokens so far, from round events
    this._openFinal = null;
    this.round = 0;
    this.toolCount = 0;
    this.errors = 0;
  }

  /**
   * Fold one event in. Duplicates (an EventSource reconnect replays history) are ignored.
   * @param {object} e ProgressEvent.
   * @returns {boolean} Whether the event was new.
   */
  apply(e) {
    if (e.seq <= this.lastSeq) return false;
    this.lastSeq = e.seq;
    this.events.push(e);
    const t = new Date(e.time);
    if (!this.startTime) this.startTime = t;
    this.lastTime = t;
    if (e.iteration) this.round = Math.max(this.round, e.iteration);
    const p = e.payload || {};
    const handler = this[`on_${e.section}`];
    if (handler) handler.call(this, e, p);
    return true;
  }

  on_run(e, p) {
    if (e.status === "started") {
      this.running = true;
      this.ended = false;
      this.goal = p.goal || this.goal;
      this.config = p.config || this.config;
      this.log.push({ type: "marker", seq: e.seq, text: p.resume ? "Run resumed" : "Run started", time: e.time });
    } else {
      this.running = false;
      this.runResult = p;
      const t = new Date(e.time);
      if (this.campaignStart && !this.campaignEnd) this.campaignEnd = t;
      if (this.finalStart && !this.finalEnd) this.finalEnd = t;
      if (p.status === "stopped") {
        for (const ph of PHASES) if (this.phase[ph.key] === "pending" || this.phase[ph.key] === "active") this.phase[ph.key] = "skipped";
      }
      if (e.status === "failed") {
        this.errors += (p.errors || []).length || 1;
        const stuck = PHASES.find((ph) => (ph.key !== "tools" || this.capability) && this.phase[ph.key] !== "done" && this.phase[ph.key] !== "skipped");
        if (stuck && this.phase[stuck.key] !== "input") this.phase[stuck.key] = "failed";
      }
      this.log.push({
        type: "marker",
        seq: e.seq,
        text: e.status === "failed" ? "Run failed" : p.status === "stopped" ? "Run stopped" : "Run finished",
        status: e.status,
        detail: p,
        time: e.time,
      });
    }
  }

  on_goal_parser(e, p) {
    if (e.status === "started") this.phase.parse = "active";
    else if (e.status === "finished") {
      this.phase.parse = "done";
      this.spec = p.spec || this.spec;
    } else {
      this.phase.parse = p.questions && p.questions.length ? "input" : "failed";
      this.questions = p.questions || [];
      this.parseErrors = p.errors || [];
      this.spec = p.spec || this.spec;
    }
    this.log.push({ type: "parser", seq: e.seq, status: e.status, payload: p, time: e.time });
  }

  on_orchestrator(e, p) {
    if (e.status === "started") {
      if (this.phase.parse === "pending") this.phase.parse = "done";
      if (this.phase.tools === "active" || this.phase.tools === "input") this.phase.tools = "done";
      this.phase.campaign = "active";
      if (!this.campaignStart) this.campaignStart = new Date(e.time);
      this.campaignEnd = null; // a resume continues the campaign
      this.backend = p.backend;
      this.model = p.model;
      return;
    }
    if (p.usage) this.campaignUsage = p.usage;
    if (p.round != null) {
      const row = {
        round: p.round,
        best: p.best,
        stop: p.stop,
        rule: p.rule,
        gateNote: p.gate_note,
        score: (p.scores || [])[p.round - 1] ?? null,
        llm: (p.llm_scores || [])[p.round - 1] ?? null,
        scores: p.scores || [],
        llmScores: p.llm_scores || [],
      };
      this.rounds = this.rounds.filter((r) => r.round !== p.round).concat(row).sort((a, b) => a.round - b.round);
      this.log.push({ type: "round", seq: e.seq, round: row, time: e.time });
    }
    if (p.submitted !== undefined) {
      this.submitted = p.submitted;
      this.campaignEnd = new Date(e.time);
      this.phase.campaign = "done";
      this.log.push({ type: "submit", seq: e.seq, submitted: p.submitted, rule: p.rule, usage: p.usage, time: e.time });
    }
  }

  on_agent(e, p) {
    this.log.push({ type: p.kind === "thinking" ? "thinking" : "text", seq: e.seq, text: p.text || "", round: e.iteration, time: e.time });
  }

  on_tool(e, p) {
    if (this.phase.parse === "pending") this.phase.parse = "done";
    if (p.phase === "capability") {
      if (this.phase.tools === "pending") this.phase.tools = "active";
    } else if (this.phase.campaign === "pending") this.phase.campaign = "active";
    let call = this.calls.get(p.call_id);
    if (!call) {
      call = { id: p.call_id, tool: p.tool, input: p.input, status: "started", round: e.iteration, started: e.time };
      this.calls.set(p.call_id, call);
      this.log.push({ type: "tool", seq: e.seq, call, time: e.time });
    }
    if (e.status !== "started") {
      call.status = e.status;
      call.finished = e.time;
      call.result = p.result;
      call.review = p.review;
      call.payload = p;
      this.toolCount += 1;
      if (e.status === "failed" || (p.result && p.result.error)) {
        call.error = (p.result && p.result.error) || p.error || "failed";
        this.errors += 1;
      }
    }
    if (p.candidate_id) {
      const prev = this.candidates.get(p.candidate_id) || { candidate_id: p.candidate_id };
      const next = { ...prev };
      for (const k of [
        "composition", "cand_round", "e_hull", "thermo_stable", "dyn_stable", "min_freq", "verdict",
        "space_group", "value", "uncertainty", "novelty", "rationale", "evidence_ids",
      ]) {
        if (p[k] !== undefined && p[k] !== null) next[k] = p[k];
      }
      if ((p.tool === "predict_property" || String(p.tool).startsWith("custom_")) && p.result && !p.result.error) {
        next.units = p.result.units;
        next.property = p.result.property;
        next.model = p.result.model;
        next.target = p.result.target;
        next.agentBuilt = String(p.result.model || "").startsWith("agent-built:");
      }
      this.candidates.set(p.candidate_id, next);
    }
    for (const paper of p.papers || []) {
      if (!this.papers.has(paper.paper_id)) this.papers.set(paper.paper_id, paper);
    }
  }

  /** The capability phase: plan and code cards, checks, decisions (agent-built tools). */
  on_capability(e, p) {
    if (this.phase.parse === "pending") this.phase.parse = "done";
    const cap = (this.capability = this.capability || { gap: null, waiting: null, registered: null, cancelled: null });
    const kind = p.kind;
    if (kind === "gap") {
      cap.gap = p.property;
      this.phase.tools = "active";
    } else if (kind === "library") {
      cap.gap = cap.gap || p.property;
      cap.registered = { name: (p.reused || [])[0] };
      this.phase.tools = "done";
    } else if (kind === "plan" || kind === "code") {
      cap.waiting = { proposal_id: p.proposal.proposal_id, stage: kind };
      this.phase.tools = "input";
      this.log.push({ type: "capability", kind, seq: e.seq, proposal: p.proposal, allowed: p.allowed || [], decision: null, time: e.time });
      return;
    } else if (kind === "package" || kind === "question") {
      cap.waiting = { proposal_id: p.request.request_id, stage: kind };
      this.phase.tools = "input";
      this.log.push({ type: "capability", kind, seq: e.seq, request: p.request, allowed: p.allowed || [], decision: null, time: e.time });
      return;
    } else if (kind === "decision") {
      const d = p.decision;
      const idOf = (x) => (x.proposal ? x.proposal.proposal_id : x.request ? x.request.request_id : null);
      const card = [...this.log].reverse().find((x) => x.type === "capability" && idOf(x) === d.proposal_id && x.kind === d.stage && !x.decision);
      if (card) card.decision = d;
      if (cap.waiting && cap.waiting.proposal_id === d.proposal_id) cap.waiting = null;
      this.phase.tools = "active";
      if (card) return; // the card shows the decision
    } else if (kind === "registered") {
      cap.registered = p.tool;
      this.phase.tools = "done";
    } else if (kind === "cancelled") {
      cap.cancelled = p;
      cap.waiting = null;
      this.phase.tools = p.scope === "run" ? "skipped" : "done";
    }
    this.log.push({ type: "capability", kind, seq: e.seq, payload: p, time: e.time });
  }

  on_final_calculation(e, p) {
    if (e.status === "started") {
      this.finalStarted = true;
      this.phase.campaign = "done";
      this.phase.final = "active";
      this.finalStart = new Date(e.time);
      if (!this.campaignEnd) this.campaignEnd = this.finalStart;
    } else if (e.status === "finished") {
      this.final = p;
      this.phase.final = "done";
      this.finalEnd = new Date(e.time);
    } else {
      this.final = { error: p.error };
      this.phase.final = "failed";
      this.finalEnd = new Date(e.time);
    }
    this.log.push({ type: "final", seq: e.seq, status: e.status, payload: p, time: e.time });
  }

  on_final_step(e, p) {
    if (e.status === "started") {
      const entry = { type: "finalstep", seq: e.seq, tool: p.tool, info: p, status: "started", time: e.time, started: e.time };
      this.log.push(entry);
      this._openFinal = entry;
      this.finalStep = entry;
      return;
    }
    const entry = this._openFinal && this._openFinal.tool === p.tool ? this._openFinal : null;
    if (entry) {
      entry.status = e.status;
      entry.info = p;
      entry.finished = e.time;
      this._openFinal = null;
    }
  }

  /**
   * Tokens used: the run's stored totals when available, else the campaign's so far.
   * @returns {{input: number, output: number, fresh: number, cacheWrite: number, cacheRead: number}|null}
   */
  get tokens() {
    const stored = this.useState && this.state.usage;
    const u = stored && ("output_tokens" in stored || "input_tokens" in stored) ? stored : this.campaignUsage;
    if (!u) return null;
    const fresh = u.input_tokens || 0;
    const cacheWrite = u.cache_creation_input_tokens || 0;
    const cacheRead = u.cache_read_input_tokens || 0;
    return { input: fresh + cacheWrite + cacheRead, output: u.output_tokens || 0, fresh, cacheWrite, cacheRead };
  }

  /**
   * Seconds spent in a phase, live while it runs.
   * @param {Date|null} start Start.
   * @param {Date|null} end End, or null while running.
   * @returns {number|null}
   */
  span(start, end) {
    if (!start) return null;
    const stop = end || (this.running && !this.isReplay ? new Date() : this.lastTime);
    return Math.max(0, (stop - start) / 1000);
  }

  on_brief_generator(e, p) {
    if (e.status === "started") {
      this.briefStarted = true;
      if (this.phase.final === "pending") this.phase.final = "skipped";
      if (this.phase.campaign !== "done") this.phase.campaign = "done";
      this.phase.brief = "active";
    } else {
      this.brief = p;
      this.phase.brief = e.status === "finished" ? "done" : "failed";
    }
    this.log.push({ type: "brief", seq: e.seq, status: e.status, payload: p, time: e.time });
  }

  /**
   * Merge the stored RunState (richer than events: literature, iterations, validation, brief).
   * @param {object} state RunState JSON plus `running`.
   */
  setState(state) {
    this.state = state;
    this.goal = this.goal || state.raw_goal;
    this.config = this.config || state.config;
    this.spec = state.spec || this.spec;
    if (!this.events.length) {
      // No events (stored without history): infer phases from the status.
      const st = state.status;
      this.phase.parse = state.spec ? "done" : st === "needs_input" ? "input" : "pending";
      if (state.iterations.length) this.phase.campaign = "done";
      if (state.validation.length) this.phase.final = "done";
      if (state.brief) this.phase.brief = "done";
    }
    if (state.spec && this.phase.parse === "pending") this.phase.parse = "done"; // started from a parsed spec
    if (state.running) this.running = true;
  }

  // ------------------------------------------------------------------ derived views

  /** @returns {string} Status shown in the header. */
  get status() {
    if (this.running) return "running";
    if (this.runResult) return this.runResult.status || (this.phase.brief === "done" ? "finished" : "failed");
    return (this.state && this.state.status) || (this.events.length ? "interrupted" : "created");
  }

  /** @returns {boolean} Whether the run can be resumed. */
  get resumable() {
    return !this.isReplay && !this.running && this.state && !["finished", "created", "needs_input"].includes(this.state.status);
  }

  /** @returns {number} Seconds from the first to the last event (or now while running). */
  get elapsed() {
    if (!this.startTime) return null;
    const end = this.running && !this.isReplay ? new Date() : this.lastTime; // a replay follows the recorded clock
    return (end - this.startTime) / 1000;
  }

  /** @returns {string|null} The current best candidate ID. */
  get best() {
    if (this.submitted && this.submitted.candidate_id) return this.submitted.candidate_id;
    if (this.state && this.state.best && (this.ended || !this.isReplay)) return this.state.best;
    const last = this.rounds[this.rounds.length - 1];
    return last ? last.best : null;
  }

  /**
   * Everything known about a candidate: proposal, structure, simulation, predictions, live fields.
   * @param {string} cid Candidate ID.
   * @returns {object}
   */
  candidate(cid) {
    const out = { candidate_id: cid, ...(this.candidates.get(cid) || {}) };
    const st = this.useState ? this.state : null;
    if (st) {
      st.iterations.forEach((it, i) => {
        const c = it.candidates.find((x) => x.candidate_id === cid);
        if (c) Object.assign(out, { proposal: c, composition: c.composition, cand_round: i + 1 });
        const sim = it.sims.find((x) => x.candidate_id === cid);
        if (sim) out.sim = sim;
        const rec = it.structures.find((x) => x.candidate_id === cid);
        if (rec) out.structure = rec;
        const preds = it.predictions.filter((x) => x.candidate_id === cid);
        if (preds.length) out.predictions = preds;
      });
      if (out.sim) {
        out.e_hull ??= out.sim.energy_above_hull_eV_per_atom;
        out.thermo_stable ??= out.sim.thermo_stable;
        if (out.sim.phonons) {
          out.min_freq ??= out.sim.phonons.min_frequency_THz;
          out.dyn_stable ??= out.sim.phonons.stable;
        }
        out.space_group ??= out.sim.space_group;
      }
      const target = out.predictions && st.spec && out.predictions.find((p) => p.property === st.spec.target_property);
      if (target) {
        out.value ??= target.value.value;
        out.uncertainty ??= target.value.uncertainty;
        out.units ??= target.value.units;
        out.property ??= target.property;
        out.model ??= target.model;
      }
    }
    return out;
  }

  /** @returns {boolean} Whether the stored state may be shown (not mid-replay). */
  get useState() {
    // A replay shows the stored state only once it has replayed everything.
    return Boolean(this.state) && (!this.isReplay || (this.ended && !this.replayIncomplete));
  }

  /** @returns {string[]} Every candidate ID, in order of proposal. */
  get candidateIds() {
    const ids = new Set(this.candidates.keys());
    if (this.useState) for (const it of this.state.iterations) for (const c of it.candidates) ids.add(c.candidate_id);
    ids.delete("ref");
    return [...ids].sort((a, b) => {
      const [ra, na] = a.slice(1).split("-").map(Number);
      const [rb, nb] = b.slice(1).split("-").map(Number);
      return ra - rb || na - nb || a.localeCompare(b);
    });
  }

  /** @returns {object[]} Reviews: one per round, from events, else from the state. */
  get reviews() {
    const out = [];
    for (const call of this.calls.values()) if (call.tool === "request_review" && call.review) out.push(call.review);
    if (!out.length && this.useState) {
      for (const it of this.state.iterations) if (it.critique) out.push(critiqueAsReview(it.critique));
    }
    return out.sort((a, b) => (a.round || 0) - (b.round || 0));
  }

  /** @returns {object[]} Rounds with scores, from events, else rebuilt from the state. */
  get roundRows() {
    if (this.rounds.length) return this.rounds;
    if (!this.useState) return [];
    return this.state.iterations
      .map((it, i) => it.critique && { round: i + 1, best: it.critique.candidate_id, score: it.critique.score, llm: it.critique.llm_confidence, stop: it.critique.stop_recommended })
      .filter(Boolean);
  }

  /** @returns {object[]} Papers: the stored literature context, else what tool events showed. */
  get paperList() {
    const lit = this.useState && this.state.literature;
    if (lit && lit.papers.length) return lit.papers;
    return [...this.papers.values()].map((p) => ({ ...p, citation_count: p.citations }));
  }
}

/**
 * Turn a stored Critique into the shape of request_review's result.
 * @param {object} c Critique.
 * @returns {object}
 */
function critiqueAsReview(c) {
  return {
    round: c.iteration,
    best_of_round: c.candidate_id,
    score: c.score,
    score_breakdown: c.score_breakdown,
    checks: {
      target_met_central: c.target_met_central,
      target_met_conservative: c.target_met_conservative,
      thermo_stable: c.thermo_stable,
      dyn_stable: c.dyn_stable,
    },
    findings: c.findings,
    recommended_next: c.next_composition,
    why: c.next_rationale,
    llm_confidence: c.llm_confidence,
    llm_confidence_reason: c.llm_confidence_reason,
  };
}
