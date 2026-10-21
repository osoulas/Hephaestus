"""Brief facts, number check and assembly (Section 10).

The brief is the document a researcher takes to the lab: what to make, why, how to make it, how
to check it, and what could go wrong. Every computed result comes from the run state (EB-2). The
LLM writes the narrative parts with ``{placeholders}`` that code fills; it may also quote lab
conditions (temperatures, times, atmospheres) found in the evidence or in the abstracts it was
given, or mark a condition "inferred". Any other number makes that part fall back to its template.

Citations are written as ``[paper_id]`` and become numbered links with a reference list (EB-7).
"""

from __future__ import annotations

import json
import re
from typing import Any

from hephaestus.core.config import BAND_GAP_MODELS
from hephaestus.core.types import Brief, ErrorRecord, Paper, RunState, ValidationReport
from hephaestus.harness import gates as G
from hephaestus.llm.base import LLM, LLMError
from hephaestus.tools.registry import Workspace

SECTIONS = [
    "Recommended composition and structure",
    "Predicted properties against target",
    "Physical rationale",
    "Validation verdict",
    "Synthesis route",
    "Characterisation plan",
    "Risks and fallbacks",
    "Runner-up candidates",
    "Iteration history",
    "Literature support",
    "Scope and assumptions",
]
LLM_SECTIONS = [
    "Physical rationale",
    "Synthesis route",
    "Characterisation plan",
    "Risks and fallbacks",
    "Literature support",
]
NUMBER = re.compile(r"(?<![\w.])[-+\u2212]?\d+(?:\.\d+)?(?![\w.]*\d)")
DASH = "\u2013"  # shown for a missing value
CITATION = re.compile(r"\[([^\[\]]+)\](?!\()")
REQ_CODE = re.compile(r"\b(?:CG|DL|EB|FC|GP|LM|OR|PP|SC|SE|SG|SYS|UI)-\d{1,2}\b")
# Lab conditions an "inferred" step may state: temperatures, times, rates, pressures, amounts.
PROCESS_UNIT = re.compile(
    r"\s*(?:°\s?C|℃|K\b|C\b|h\b|hrs?\b|hours?\b|min\b|minutes?\b|days?\b|°C/min|K/min|MPa\b|bar\b|"
    r"atm\b|mbar\b|rpm\b|wt\s?%|mol\s?%|at\s?%|%|M\b|mm\b|µm\b|um\b|mL\b|ml\b|g\b)"
)
PROPERTY_NAMES = {
    "band_gap": "Band gap",
    "bulk_modulus": "Bulk modulus",
    "shear_modulus": "Shear modulus",
    "youngs_modulus": "Young's modulus",
    "debye_temperature": "Debye temperature",
}

BRIEF_SYSTEM = """\
You write parts of a lab brief: the document a materials scientist takes into the lab to make and \
test a candidate that may never have been made before. Be concrete, practical and brief. Plain \
English, short sentences, no filler.

Numbers. Computed results (energies, moduli, gaps, frequencies, scores) may appear only as \
{placeholders} from the facts list, never as digits. Lab conditions (temperatures, times, heating \
rates, atmospheres, pressures) may be written as digits only if they appear in the evidence or in \
a paper abstract you were given, cited with that paper's ID in square brackets, e.g. \
[arxiv:2101.00001]; otherwise give the condition and write "(inferred)" on that step.

Citations. Cite papers by ID in square brackets right after the claim they support. Only cite \
papers from the list given.

Synthesis route. Propose a route even if the compound has never been made: take it from the \
closest analogues in the evidence and abstracts (same structure family, same cations, end members, \
known solid solutions) and say which analogue each step follows. Format: one line starting \
"**Approach:**" naming the method and why it suits this compound, then numbered steps, each \
"**Step name.** what to do, conditions, source". Include precursors, mixing, calcination, \
sintering and atmosphere where relevant. End with one line starting "**Watch for:**" naming the \
competing phases from the facts (win_decomposes_to) and literature.

Characterisation plan. Numbered steps ordered by what most quickly confirms or kills the \
candidate, starting with phase identity (XRD). Each step: "**Technique.** what it tests; pass if \
..., stop if ...". Include a step that measures the target property itself with a suitable method.

Physical rationale. One short paragraph on why this composition should meet the target.

Risks. Up to six, most severe first; each a one-sentence risk, a severity, and a concrete fallback.

Literature notes. For each paper worth reading first, one short sentence on why it matters here.

Next step. One sentence: the first experiment to do."""


# ---------------------------------------------------------------------- facts


def _fmt(x: Any) -> str:
    """Format a fact for the brief.

    Parameters
    ----------
    x : Any
        Value.

    Returns
    -------
    str
        Floats with three significant decimals; ``n/a`` for ``None``.
    """
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.3g}" if abs(x) < 1000 else f"{x:.0f}"
    if isinstance(x, list):
        return ", ".join(str(v) for v in x) or "none"
    return str(x)


def property_name(key: str) -> str:
    """Readable name of a property key, e.g. ``bulk_modulus`` -> ``Bulk modulus``.

    Parameters
    ----------
    key : str
        Property key.

    Returns
    -------
    str
        Display name.
    """
    return PROPERTY_NAMES.get(key, key.replace("_", " ").capitalize())


def source_text(source: str | None) -> str:
    """Describe where a structure came from, e.g. ``template:double_perovskite``.

    Parameters
    ----------
    source : str or None
        ``mp:<id>`` or ``template:<name>``.

    Returns
    -------
    str
        A phrase such as "built from the double perovskite template".
    """
    if not source:
        return "of unrecorded origin"
    kind, _, name = source.partition(":")
    if kind == "mp":
        return f"taken from Materials Project entry [{name}](https://next-gen.materialsproject.org/materials/{name})"
    if kind == "template":
        return f"built from the {name.replace('_', ' ')} template"
    return f"from {source}"


def candidate_label(cid: str | None) -> str:
    """Readable candidate ID: ``c3-2`` -> ``candidate 3-2``.

    Parameters
    ----------
    cid : str or None
        Candidate ID.

    Returns
    -------
    str
        Label.
    """
    if not cid:
        return "n/a"
    m = re.match(r"^c(\d+)-(\d+)$", cid)
    return f"candidate {m.group(1)}-{m.group(2)}" if m else cid


def collect_facts(state: RunState, ws: Workspace) -> dict[str, Any]:
    """Every number and label the brief may show, from stored results (EB-2).

    Parameters
    ----------
    state : RunState
        Run state with the submission and validation.
    ws : Workspace
        Run workspace.

    Returns
    -------
    dict
        Fact name to value; candidate facts are prefixed ``win_`` and ``run_``.
    """
    spec = ws.spec
    out: dict[str, Any] = {
        "target_property": spec.target_property,
        "target_name": property_name(spec.target_property),
        "target_value": spec.target_value,
        "target_units": spec.target_units,
        "direction": spec.direction,
        "reference_material": spec.reference_material,
        "e_hull_threshold": spec.stability_thresholds.e_hull_max_eV_per_atom,
        "temperature_K": spec.conditions.temperature_K,
        "n_rounds": len(state.iterations),
        "n_candidates": len(ws.candidates),
        "n_papers": len(ws.papers),
    }
    sub = state.submission
    for prefix, cid in (
        ("win", sub.candidate_id if sub else None),
        ("run", sub.runner_up_id if sub else None),
    ):
        if cid and cid in ws.candidates:
            f = G.facts(ws, cid)
            for k in (
                "composition",
                "space_group",
                "e_hull",
                "min_freq",
                "value",
                "uncertainty",
                "units",
                "target_met_central",
                "target_met_conservative",
                "thermo_stable",
                "dyn_stable",
            ):
                out[f"{prefix}_{k}"] = f[k]
            out[f"{prefix}_id"] = cid
            out[f"{prefix}_source"] = f["source"]
            out[f"{prefix}_structure_type"] = ws.candidates[cid].candidate.structure_type
            out[f"{prefix}_decomposes_to"] = list(ws.candidates[cid].decomposes_to)
    out["gate_passed"] = bool(sub and sub.gate_passed)
    return out


# ---------------------------------------------------------------------- checks and links


def check_numbers(text: str, facts: dict[str, Any], extra: set[str] | None = None) -> list[str]:
    """Numbers in a draft that are not in the facts (EB-2).

    Formula subscripts, citation IDs and link targets are not numbers.

    Parameters
    ----------
    text : str
        Draft Markdown.
    facts : dict
        Allowed values.
    extra : set of str, optional
        Further allowed number strings (e.g. from cited evidence).

    Returns
    -------
    list of str
        Offending numbers; empty when the draft is clean.
    """
    allowed = set(extra or set())
    for v in facts.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            allowed |= {_fmt(float(v)), str(v), f"{v:g}"}
    allowed |= {str(i) for i in range(11)}
    scrubbed = re.sub(r"\]\([^)]*\)", "]", text)  # link targets
    scrubbed = CITATION.sub(" ", scrubbed)  # [paper ids]
    scrubbed = re.sub(r"https?://\S+", " ", scrubbed)
    bad = []
    for m in NUMBER.finditer(scrubbed):
        num = m.group().lstrip("+").replace("\u2212", "-")
        if num not in allowed and num.lstrip("-") not in allowed:
            bad.append(m.group())
    return bad


def _inferred_conditions_ok(text: str) -> set[str]:
    """Lab-condition numbers on lines marked "(inferred)", which the check allows.

    Parameters
    ----------
    text : str
        Draft Markdown.

    Returns
    -------
    set of str
        Number strings directly followed by a lab unit on an inferred line.
    """
    ok: set[str] = set()
    for line in text.splitlines():
        if "inferred" not in line.lower():
            continue
        for m in NUMBER.finditer(line):
            if PROCESS_UNIT.match(line, m.end()):
                ok.add(m.group().lstrip("+"))
    return ok


def fill(template: str, facts: dict[str, Any]) -> str:
    """Replace ``{name}`` placeholders with formatted facts; unknown names are left visible.

    Parameters
    ----------
    template : str
        Text with placeholders.
    facts : dict
        Values.

    Returns
    -------
    str
        Filled text.
    """
    return re.sub(
        r"\{(\w+)\}", lambda m: _fmt(facts[m.group(1)]) if m.group(1) in facts else m.group(0), template
    )


def paper_url(p: Paper) -> str | None:
    """Best link for a paper: its record URL, else its DOI.

    Parameters
    ----------
    p : Paper
        Paper.

    Returns
    -------
    str or None
        URL.
    """
    if p.url:
        return p.url
    if p.doi:
        return f"https://doi.org/{p.doi}"
    return None


def id_label(pid: str) -> str:
    """Readable label for a paper ID: "PubMed 37460465", "arXiv 0804.1681v1".

    Parameters
    ----------
    pid : str
        Paper ID.

    Returns
    -------
    str
        Label.
    """
    if pid.isdigit():
        return f"PubMed {pid}"
    if re.match(r"^(arxiv:)?\d{4}\.\d{4,5}(v\d+)?$", pid, re.I):
        return f"arXiv {pid.split(':')[-1]}"
    return pid


def link_citations(text: str, papers: dict[str, Paper], order: list[str] | None = None) -> str:
    """Turn ``[paper_id]`` (or ``[id1, id2]``) into numbered links, e.g. ``[[1]](url)`` (EB-7).

    Parameters
    ----------
    text : str
        Markdown.
    papers : dict
        Retrieved papers by ID.
    order : list of str, optional
        Citation order so far; filled in place with new IDs as they are first met.

    Returns
    -------
    str
        Linked Markdown; IDs that are not known papers are left as written.
    """
    order = [] if order is None else order

    def one(pid: str) -> str | None:
        """Link one ID.

        Parameters
        ----------
        pid : str
            Paper ID.

        Returns
        -------
        str or None
            ``[[n]](url)``, or ``None`` for an unknown ID.
        """
        p = papers.get(pid)
        if p is None:
            return None
        if pid not in order:
            order.append(pid)
        n = order.index(pid) + 1
        url = paper_url(p)
        return f"[[{n}]]({url})" if url else f"[{n}]"  # shows as "[n]" in any Markdown viewer

    def sub(m: re.Match[str]) -> str:
        """Link a bracketed group of IDs.

        Parameters
        ----------
        m : re.Match
            ``[...]`` match.

        Returns
        -------
        str
            Links, or the original text when any ID is unknown.
        """
        ids = [x.strip() for x in re.split(r"[;,]", m.group(1))]
        links = [one(pid) for pid in ids]
        if not ids or any(x is None for x in links):
            return m.group(0)
        return "".join(links)  # type: ignore[arg-type]

    return CITATION.sub(sub, text)


def strip_codes(text: str) -> str:
    """Remove requirement IDs such as "(FC-1, FC-2)" from reader-facing text.

    Parameters
    ----------
    text : str
        Text.

    Returns
    -------
    str
        Text without requirement IDs.
    """
    code_list = rf"{REQ_CODE.pattern}(?:\s*(?:,|;|and|to|\u2013|-)\s*{REQ_CODE.pattern})*"
    text = re.sub(rf"\s*\(\s*{code_list}\s*\)", "", text)
    text = re.sub(rf"\s*[,;]\s*{code_list}(?=\s*\))", "", text)
    text = re.sub(rf"\s*{code_list}", "", text)
    return re.sub(r"[ \t]+([.,;:])", r"\1", text)


# ---------------------------------------------------------------------- templates


CHARACTERISATION = {
    "band_gap": "**UV-vis diffuse reflectance.** Measure the optical gap with a Tauc plot; pass if it is below the target.",
    "bulk_modulus": "**High-pressure XRD (diamond-anvil cell) or resonant ultrasound.** Fit an equation of state for the bulk modulus; pass if it beats the reference measured the same way.",
    "shear_modulus": "**Resonant ultrasound or nanoindentation.** Measure the elastic constants; pass if the shear modulus beats the target.",
    "youngs_modulus": "**Nanoindentation or resonant ultrasound.** Measure Young's modulus on a dense pellet; pass if it beats the target.",
}


def template_sections(state: RunState, ws: Workspace, facts: dict[str, Any]) -> dict[str, str]:
    """Code-only text for every narrative section; used when the LLM is unavailable or fails a check.

    Parameters
    ----------
    state : RunState
        Run state.
    ws : Workspace
        Workspace.
    facts : dict
        Facts.

    Returns
    -------
    dict
        Section title to Markdown.
    """
    win = state.submission.candidate_id if state.submission else None
    cs = ws.candidates.get(win) if win else None
    synth = [i for i in ws.insights if i.finding_type == "synthesis"]
    neg = [i for i in ws.insights if i.finding_type == "negative"]
    s: dict[str, str] = {}
    s["Physical rationale"] = cs.candidate.rationale if cs else "No candidate was evaluated."

    steps = ["**Approach:** solid-state reaction from the oxides or carbonates (inferred).", ""]
    if synth:
        steps += [f"{n}. **From the literature.** {i.summary} [{i.paper_id}]" for n, i in enumerate(synth, 1)]
    else:
        steps += [
            "1. **Precursors.** Weigh stoichiometric amounts of the dried binary oxides or carbonates (inferred).",
            "2. **Mixing.** Ball-mill or grind in ethanol until homogeneous (inferred).",
            "3. **Calcination.** Calcine in air, regrind, and repeat until XRD shows a single phase (inferred).",
            "4. **Sintering.** Press pellets and sinter for dense samples (inferred).",
        ]
    comp = facts.get("win_decomposes_to") or []
    steps += ["", f"**Watch for:** {', '.join(comp)}." if comp else "**Watch for:** secondary phases in XRD."]
    s["Synthesis route"] = "\n".join(steps)

    plan = [
        "**XRD.** Confirm the structure and space group; stop if secondary phases dominate.",
        CHARACTERISATION.get(
            ws.spec.target_property,
            f"**Measure the {property_name(ws.spec.target_property).lower()}.** Compare it with the target.",
        ),
        "**Raman.** Check octahedral tilts and ordering the model may miss.",
        "**XPS.** Confirm the cation oxidation states.",
    ]
    s["Characterisation plan"] = "\n".join(f"{n}. {p}" for n, p in enumerate(plan, 1))

    rows: list[tuple[str, str, str]] = [(r, "major", "") for r in (cs.candidate.risks if cs else [])]
    rows += [(f"{i.summary} [{i.paper_id}]", "major", "") for i in neg]
    crit = state.iterations[-1].critique if state.iterations and state.iterations[-1].critique else None
    if crit:
        rows += [(_first_sentence(f.text), f.severity, _first_sentence(f.suggestion)) for f in crit.findings]
    if facts.get("run_id"):
        rows.append(
            (
                "The winner fails in the lab.",
                "major",
                f"Try the runner-up, {candidate_label(facts['run_id'])}.",
            )
        )
    s["Risks and fallbacks"] = risk_table(rows)

    cited = sorted(
        {pid for c in ws.candidates.values() for pid in c.candidate.evidence_ids if pid in ws.papers}
    )
    s["Literature support"] = "\n".join(f"- [{pid}]" for pid in cited)
    return s


def _first_sentence(text: str) -> str:
    """First sentence of a text, for compact tables.

    Parameters
    ----------
    text : str
        Text.

    Returns
    -------
    str
        Up to the first full stop.
    """
    text = (text or "").strip()
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    return (m.group(1) if m else text)[:240]


def risk_table(rows: list[tuple[str, str, str]]) -> str:
    """Risks as a table, most severe first (EB-6).

    Parameters
    ----------
    rows : list of tuple
        ``(risk, severity, fallback)``.

    Returns
    -------
    str
        Markdown table, or a line saying there are none.
    """
    if not rows:
        return "No risks recorded."
    rank = {"critical": 0, "major": 1, "minor": 2, "info": 3}
    rows = sorted(rows, key=lambda r: rank.get(r[1], 2))
    cell = lambda t: (t or DASH).replace("|", "/").replace("\n", " ")  # noqa: E731
    lines = ["| Risk | Severity | Fallback |", "| --- | --- | --- |"]
    lines += [f"| {cell(r)} | {sev.capitalize()} | {cell(fb)} |" for r, sev, fb in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------- LLM sections


def _llm_schema() -> dict[str, Any]:
    """JSON schema of the writer's answer.

    Returns
    -------
    dict
        Schema with prose fields, structured risks and literature notes.
    """
    text = {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "physical_rationale": text,
            "synthesis_route": text,
            "characterisation_plan": text,
            "risks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "risk": text,
                        "severity": {"type": "string", "enum": ["critical", "major", "minor"]},
                        "fallback": text,
                    },
                    "required": ["risk", "severity", "fallback"],
                    "additionalProperties": False,
                },
            },
            "literature_notes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"paper_id": text, "why": text},
                    "required": ["paper_id", "why"],
                    "additionalProperties": False,
                },
            },
            "next_step": text,
        },
        "required": [
            "physical_rationale",
            "synthesis_route",
            "characterisation_plan",
            "risks",
            "literature_notes",
            "next_step",
        ],
        "additionalProperties": False,
    }


def write_sections(
    llm: LLM | None, state: RunState, ws: Workspace, facts: dict[str, Any]
) -> tuple[dict[str, str], list[ErrorRecord]]:
    """LLM prose for the narrative sections, falling back to templates (EB-2, EB-4 to EB-6).

    Parameters
    ----------
    llm : LLM or None
        Writer model.
    state : RunState
        Run state.
    ws : Workspace
        Workspace.
    facts : dict
        Facts and their placeholder names.

    Returns
    -------
    sections : dict
        Title to Markdown, numbers filled, plus ``_notes`` (literature notes as JSON) and
        ``_next_step`` when the LLM gave them.
    errors : list of ErrorRecord
        Parts replaced by templates and why.
    """
    templates = template_sections(state, ws, facts)
    if llm is None:
        return templates, []
    # Numbers the evidence and the abstracts the writer read may quote (lab conditions).
    quotable = {_fmt(i.value.value) for i in ws.insights if i.value} | {
        str(int(i.value.value)) for i in ws.insights if i.value and float(i.value.value).is_integer()
    }
    for text in [f"{p.title} {p.abstract}" for p in ws.papers.values()] + [
        f"{i.summary} {i.conditions}" for i in ws.insights
    ]:
        quotable |= {m.group().lstrip("+") for m in NUMBER.finditer(text)}
    papers = sorted(ws.papers.values(), key=lambda p: (-p.relevance, -(p.citation_count or 0)))[:30]
    crit = state.iterations[-1].critique if state.iterations and state.iterations[-1].critique else None
    prompt = json.dumps(
        {
            "facts_placeholders": {k: _fmt(v) for k, v in facts.items()},
            "candidate": {
                "composition": facts.get("win_composition"),
                "structure_type": facts.get("win_structure_type"),
                "space_group": facts.get("win_space_group"),
                "competing_phases": facts.get("win_decomposes_to"),
                "proposal_rationale": templates["Physical rationale"],
                "risks_named_at_proposal": (
                    ws.candidates[facts["win_id"]].candidate.risks
                    if facts.get("win_id") in ws.candidates
                    else []
                ),
            },
            "justification": state.submission.justification if state.submission else "",
            "evidence": [i.model_dump(mode="json") for i in ws.insights],
            "papers": [
                {
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "year": p.year,
                    "abstract": (p.abstract or "")[:700],
                }
                for p in papers
            ],
            "last_review": crit.model_dump(mode="json") if crit else None,
            "final_check": [v.model_dump(mode="json") for v in state.validation],
        },
        ensure_ascii=False,
        default=str,
    )
    try:
        raw = llm.structured("brief", BRIEF_SYSTEM, prompt, _llm_schema())
    except LLMError as exc:
        return templates, [ErrorRecord(step="brief", code="llm_failed", message=str(exc)[:300])]

    errors: list[ErrorRecord] = []
    out = dict(templates)

    def clean(title: str, draft: str) -> str | None:
        """Number-check one part; ``None`` when it fails.

        Parameters
        ----------
        title : str
            Part name, for the error.
        draft : str
            Text.

        Returns
        -------
        str or None
            The draft with placeholders filled, or ``None``.
        """
        bad = check_numbers(draft, facts, quotable | _inferred_conditions_ok(draft))
        if bad:
            errors.append(ErrorRecord(step="brief", code="number_check", message=f"{title}: {bad[:5]}"))
            return None
        return fill(draft, facts)

    for title, key in (
        ("Physical rationale", "physical_rationale"),
        ("Synthesis route", "synthesis_route"),
        ("Characterisation plan", "characterisation_plan"),
    ):
        if raw.get(key) and (cleaned := clean(title, raw[key])):
            out[title] = cleaned
    risks = []
    for r in raw.get("risks") or []:
        risk = clean("Risks and fallbacks", f"{r.get('risk', '')} || {r.get('fallback', '')}")
        if risk is not None:
            a, _, b = risk.partition(" || ")
            risks.append((a, r.get("severity", "major"), b))
    if risks:
        if facts.get("run_id"):
            risks.append(
                (
                    "The winner fails in the lab.",
                    "major",
                    f"Try the runner-up, {candidate_label(facts['run_id'])}.",
                )
            )
        out["Risks and fallbacks"] = risk_table(risks)
    notes = {}
    for n in raw.get("literature_notes") or []:
        pid = n.get("paper_id", "").strip("[] ")
        if pid in ws.papers and (why := clean("Literature support", n.get("why", ""))):
            notes[pid] = why
    out["_notes"] = json.dumps(notes)
    if raw.get("next_step") and (step := clean("Next step", raw["next_step"])):
        out["_next_step"] = step
    return out, errors


# ---------------------------------------------------------------------- code-built parts


def property_table(facts: dict[str, Any]) -> str:
    """Markdown table: property, predicted ± uncertainty, target, met (EB-3).

    Parameters
    ----------
    facts : dict
        Output of ``collect_facts``.

    Returns
    -------
    str
        Table built in code.
    """
    arrow = {"minimise": "<", "maximise": ">", "hit": "≈"}[facts["direction"]]
    unc = facts.get("win_uncertainty")
    units = facts["target_units"]
    pred = (
        f"{_fmt(facts.get('win_value'))} ± {_fmt(unc)} {units}"
        if unc is not None
        else f"{_fmt(facts.get('win_value'))} {units} (no uncertainty available)"
    )
    ref = facts.get("reference_material")
    target = f"{arrow} {_fmt(facts['target_value'])} {units}" + (f" ({ref}, same model)" if ref else "")
    central, cons = facts.get("win_target_met_central"), facts.get("win_target_met_conservative")
    met = (
        "Yes, even at the low end of the error bar"
        if cons
        else "Yes, at the central value"
        if central
        else "No"
    )
    if arrow == "<" and cons:
        met = "Yes, even at the high end of the error bar"
    rows = [
        "| Property | Predicted | Target | Met |",
        "| --- | --- | --- | --- |",
        f"| {facts['target_name']} | {pred} | {target} | {met} |",
        f"| Energy above hull | {_fmt(facts.get('win_e_hull'))} eV/atom | ≤ {_fmt(facts['e_hull_threshold'])} eV/atom | "
        f"{'Yes' if facts.get('win_thermo_stable') else 'No'} |",
        f"| Lowest phonon frequency | {_fmt(facts.get('win_min_freq'))} THz | no unstable modes after mode mapping | "
        f"{'Yes' if facts.get('win_dyn_stable') else 'No'} |",
    ]
    return "\n".join(rows)


CHECK_NAMES = {
    "e_hull_eV_per_atom": "Energy above hull (eV/atom)",
    "thermo_stable": "On the hull",
    "dyn_stable": "Dynamically stable",
    "min_frequency_THz": "Lowest phonon (THz)",
    "md_stable": "Stable in MD",
    "target_met": "Target met",
    "target_met_conservative": "Target met within error",
    "disorder_spread_eV_per_atom": "Disorder spread (eV/atom)",
}
VERDICT_TEXT = {
    "confirmed": "**Confirmed.** The higher-fidelity re-check agrees with screening.",
    "weakened": "**Weakened.** The re-check moved some results the wrong way; see below.",
    "overturned": "**Overturned.** The re-check contradicts screening; do not take this candidate to the lab as is.",
}


def _cell(x: Any) -> str:
    """Format a re-check value for a table cell.

    Parameters
    ----------
    x : Any
        Value.

    Returns
    -------
    str
        "Yes", "No", a number, or a dash when not computed.
    """
    return DASH if x is None else _fmt(x).capitalize()


def _validation(reports: list[ValidationReport], spec_property: str = "", units: str = "") -> str:
    """Summarise the final re-check: verdict, a screening-versus-final table and notes.

    Parameters
    ----------
    reports : list of ValidationReport
        Final re-check results, winner first.
    spec_property : str
        Target property key, for its readable name.

    Returns
    -------
    str
        Markdown.
    """
    if not reports:
        return "Final re-check not run."
    out = []
    for i, r in enumerate(reports):
        if i:
            out += ["", f"Runner-up, {candidate_label(r.candidate_id)}: **{r.verdict}**."]
            continue
        out.append(VERDICT_TEXT.get(r.verdict, f"**{r.verdict.capitalize()}.**"))
        rows = [c for c in r.checks if c.screening_value is not None or c.final_value is not None]
        if rows:
            out += ["", "| Check | Screening | Final re-check |", "| --- | --- | --- |"]
            for c in rows:
                name = CHECK_NAMES.get(c.name) or (
                    f"{property_name(c.name)} ({units})"
                    if c.name == spec_property and units
                    else property_name(c.name)
                )
                out.append(f"| {name} | {_cell(c.screening_value)} | {_cell(c.final_value)} |")
        if r.notes:
            out += [""] + [f"- {n}" for n in r.notes]
    return "\n".join(out)


def _summary(facts: dict[str, Any], state: RunState, next_step: str | None) -> list[str]:
    """Build the "at a glance" block under the title.

    Parameters
    ----------
    facts : dict
        Facts.
    state : RunState
        Run state.
    next_step : str or None
        The writer's first experiment, if any.

    Returns
    -------
    list of str
        Markdown lines.
    """
    if not facts.get("win_id"):
        return []
    ref = facts.get("reference_material")
    pred = f"{_fmt(facts.get('win_value'))}" + (
        f" ± {_fmt(facts['win_uncertainty'])}" if facts.get("win_uncertainty") is not None else ""
    )
    target = f"{_fmt(facts['target_value'])} {facts['target_units']}" + (f" for {ref}" if ref else "")
    lines = [
        f"- **What to make:** {facts['win_composition']}, {facts.get('win_structure_type', '').replace('_', ' ')}"
        f"{', ' + facts['win_space_group'] if facts.get('win_space_group') else ''}.",
        f"- **Why:** predicted {facts['target_name'].lower()} {pred} {facts['target_units']} against {target} on the same model.",
        f"- **Stable?** {_fmt(facts.get('win_e_hull'))} eV/atom above the hull (limit {_fmt(facts['e_hull_threshold'])}); "
        + ("no unstable phonon modes." if facts.get("win_dyn_stable") else "unstable phonon modes remain."),
    ]
    if state.validation:
        v = state.validation[0]
        reason = next((n.split(": ", 1)[-1] for n in v.notes if n.lower().startswith(v.verdict)), "")
        lines.append(
            f"- **Final re-check:** {v.verdict}"
            + (f" ({reason[0].lower() + reason[1:].rstrip('.')})." if reason else ".")
        )
    if next_step:
        lines.append(f"- **First experiment:** {next_step}")
    return lines


def assemble_brief(state: RunState, ws: Workspace, llm: LLM | None = None) -> tuple[Brief, list[ErrorRecord]]:
    """Assemble the brief in the fixed section order (EB-1, EB-3, EB-8, EB-9, EB-11).

    Parameters
    ----------
    state : RunState
        Run state, synced, with submission and validation.
    ws : Workspace
        Workspace.
    llm : LLM, optional
        Writer for the narrative sections.

    Returns
    -------
    brief : Brief
        Markdown brief.
    errors : list of ErrorRecord
        Parts that fell back to templates.
    """
    facts = collect_facts(state, ws)
    sections, errors = write_sections(llm, state, ws, facts)
    notes: dict[str, str] = json.loads(sections.pop("_notes", "{}"))
    next_step = sections.pop("_next_step", None)
    sub = state.submission
    lines = [f"# Lab brief: {facts.get('win_composition', 'no candidate')}", ""]
    if not facts["gate_passed"]:
        lines += [
            "> **Target not met.** The candidate below is the best attempt; it failed at least one code check.",
            "",
        ]
    lines += [*_summary(facts, state, next_step), ""]

    body: dict[str, str] = {}
    body["Recommended composition and structure"] = (
        f"**{facts.get('win_composition', 'n/a')}** ({candidate_label(facts.get('win_id'))}): "
        f"{(facts.get('win_structure_type') or 'structure').replace('_', ' ')}, space group "
        f"{facts.get('win_space_group') or 'n/a'}, {source_text(facts.get('win_source'))}."
        + (f" Chosen by the {sub.picked_by}." if sub else "")
    )
    body["Predicted properties against target"] = property_table(facts) if sub else "No candidate."
    body["Validation verdict"] = _validation(state.validation, ws.spec.target_property, ws.spec.target_units)
    if facts.get("run_id"):
        body["Runner-up candidates"] = "\n".join(
            [
                f"| Candidate | {facts['target_name']} | Energy above hull | Dynamically stable |",
                "| --- | --- | --- | --- |",
                f"| {facts['run_composition']} ({candidate_label(facts['run_id'])}) | {_fmt(facts['run_value'])} ± "
                f"{_fmt(facts['run_uncertainty'])} {facts['target_units']} | {_fmt(facts['run_e_hull'])} eV/atom | "
                f"{'Yes' if facts['run_dyn_stable'] else 'No'} |",
            ]
        )
    else:
        body["Runner-up candidates"] = "None."
    hist = ["| Round | Candidates | Best | Rubric | Reviewer |", "| --- | --- | --- | --- | --- |"]
    for n, it in enumerate(state.iterations, start=1):
        c = it.critique
        comps = ", ".join(x.composition for x in it.candidates) or "none"
        best = next((x.composition for x in it.candidates if c and x.candidate_id == c.candidate_id), None)
        hist.append(
            f"| {n} | {comps} | {best or (candidate_label(c.candidate_id) if c else DASH)} | "
            f"{c.score if c else DASH} | {c.llm_confidence if c and c.llm_confidence is not None else DASH} |"
        )
    body["Iteration history"] = "\n".join(hist) if state.iterations else "No rounds."
    spec = ws.spec
    body["Scope and assumptions"] = "\n".join(
        [f"- {a}" for a in spec.assumptions]
        + [
            f"- Screening potential {ws.cfg.potential}; final re-check {ws.cfg.final_potential}.",
            f"- Band gaps from {BAND_GAP_MODELS[ws.cfg.band_gap_model]['label']}; property values are on each model's own scale.",
            *(
                [f"- {state.literature.coverage_note}"]
                if state.literature and state.literature.coverage_note
                else []
            ),
            "- Structures come from prototype templates or Materials Project polymorphs.",
        ]
    )
    body.update(sections)

    # Number citations in reading order, then list them with what each is for (EB-7).
    order: list[str] = []
    rendered = {}
    for title in SECTIONS:
        if title != "Literature support":
            rendered[title] = link_citations(body.get(title, ""), ws.papers, order)
    for pid in re.findall(r"\[([^\[\]]+)\]", body.get("Literature support", "")) + list(notes):
        if pid in ws.papers and pid not in order:
            order.append(pid)
    refs = []
    for n, pid in enumerate(order, 1):
        p = ws.papers[pid]
        url = paper_url(p)
        authors = (
            (p.authors[0].split(",")[0] + (" et al." if len(p.authors) > 1 else "")) if p.authors else ""
        )
        head = (
            f"{n}. **{p.title}**" + (f" — {authors}" if authors else "") + (f" ({p.year})" if p.year else "")
        )
        link = f" [{id_label(pid)}]({url})" if url else f" {id_label(pid)}"
        why = f" {link_citations(notes[pid], ws.papers, order)}" if pid in notes else ""
        refs.append(head + "." + link + why)
    rendered["Literature support"] = "\n".join(refs) or "No papers cited."

    for title in SECTIONS:
        lines += [f"## {title}", "", rendered[title], ""]
    md = strip_codes("\n".join(lines))
    return Brief(run_id=state.run_id, markdown=md, cited_paper_ids=list(order)), errors
