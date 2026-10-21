"""Brief facts, number check and assembly (Section 10).

Owner: A. Every number comes from the run state (EB-2). The LLM writes prose for a few
sections using ``{placeholders}`` that code fills; a section with any number not in the facts
is replaced by its template.
"""

from __future__ import annotations

import json
import re
from typing import Any

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
BRIEF_SYSTEM = """\
You write sections of an experimental brief for a lab researcher. Use only the facts given. \
Write numbers only as {placeholders} from the facts list, never as digits. Cite papers by their \
ID in square brackets, e.g. [arxiv:2101.00001]. Base the synthesis route on the synthesis \
evidence for the same or closest composition, and say "inferred" for any step without a source. \
Order the characterisation plan by what most quickly confirms or kills the candidate, starting \
with phase identity (XRD). Give each risk a severity and a concrete fallback for critical ones."""


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
    return str(x)


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
    out["gate_passed"] = bool(sub and sub.gate_passed)
    return out


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
    pred = (
        f"{_fmt(facts.get('win_value'))} ± {_fmt(unc)}"
        if unc is not None
        else f"{_fmt(facts.get('win_value'))} (no uncertainty available)"
    )
    rows = [
        "| Property | Predicted | Target | Met |",
        "| --- | --- | --- | --- |",
        f"| {facts['target_property']} ({facts['target_units']}) | {pred} | {arrow} {_fmt(facts['target_value'])} | "
        f"{_fmt(facts.get('win_target_met_central'))} (conservative: {_fmt(facts.get('win_target_met_conservative'))}) |",
        f"| Energy above hull (eV/atom) | {_fmt(facts.get('win_e_hull'))} | ≤ {_fmt(facts['e_hull_threshold'])} | "
        f"{_fmt(facts.get('win_thermo_stable'))} |",
        f"| Dynamic stability (min frequency, THz) | {_fmt(facts.get('win_min_freq'))} | mode mapping | "
        f"{_fmt(facts.get('win_dyn_stable'))} |",
    ]
    return "\n".join(rows)


def check_numbers(text: str, facts: dict[str, Any], extra: set[str] | None = None) -> list[str]:
    """Numbers in a draft that are not in the facts (EB-2). Formula subscripts are ignored.

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
    bad = []
    for m in NUMBER.finditer(text):
        num = m.group().lstrip("+").replace("\u2212", "-")
        if num not in allowed and num.lstrip("-") not in allowed:
            bad.append(m.group())
    return bad


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


def link_citations(text: str, papers: dict[str, Paper]) -> str:
    """Turn ``[paper_id]`` into Markdown links to the paper's Valency record (EB-7).

    Parameters
    ----------
    text : str
        Markdown.
    papers : dict
        Retrieved papers by ID.

    Returns
    -------
    str
        Linked Markdown.
    """

    def sub(m: re.Match[str]) -> str:
        """Link one citation.

        Parameters
        ----------
        m : re.Match
            ``[id]`` match.

        Returns
        -------
        str
            Link, or the original text for unknown IDs.
        """
        pid = m.group(1)
        p = papers.get(pid)
        if p is None:
            return m.group(0)
        return f"[{pid}]({p.url})" if p.url else f"[{pid}]"

    return re.sub(r"\[([^\]\[]+)\](?!\()", sub, text)


def template_sections(state: RunState, ws: Workspace, facts: dict[str, Any]) -> dict[str, str]:
    """Code-only text for every section; used when the LLM is unavailable or fails the number check.

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
    s["Synthesis route"] = (
        "\n".join(f"- {i.summary} [{i.paper_id}]" for i in synth)
        or "- No synthesis evidence recorded (inferred: solid-state reaction)."
    )
    s["Characterisation plan"] = "\n".join(
        [
            "1. XRD: confirm the perovskite phase and rule out secondary phases.",
            "2. UV-vis diffuse reflectance (Tauc plot): measure the optical gap.",
            "3. Raman: check for octahedral tilts the cubic model may miss.",
            "4. XPS: confirm dopant oxidation states.",
        ]
    )
    risks = [f"- {r}" for r in (cs.candidate.risks if cs else [])]
    risks += [f"- (literature) {i.summary} [{i.paper_id}]" for i in neg]
    crit = state.iterations[-1].critique if state.iterations and state.iterations[-1].critique else None
    if crit:
        risks += [
            f"- ({f.severity}) {f.text}" + (f" Fallback: {f.suggestion}" if f.suggestion else "")
            for f in crit.findings
        ]
    if facts.get("run_id"):
        risks.append("- Fallback: the runner-up candidate below.")
    s["Risks and fallbacks"] = "\n".join(risks) or "- None recorded."
    cited = sorted(
        {pid for c in ws.candidates.values() for pid in c.candidate.evidence_ids if pid in ws.papers}
    )
    s["Literature support"] = (
        "\n".join(f"- [{pid}] {ws.papers[pid].title}" for pid in cited) or "- No papers cited."
    )
    return s


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
        Title to Markdown, numbers filled.
    errors : list of ErrorRecord
        Sections replaced by templates and why.
    """
    templates = template_sections(state, ws, facts)
    if llm is None:
        return templates, []
    evidence_numbers = {_fmt(i.value.value) for i in ws.insights if i.value} | {
        str(int(i.value.value)) for i in ws.insights if i.value and float(i.value.value).is_integer()
    }
    schema = {
        "type": "object",
        "properties": {
            k.lower().replace(" ", "_").replace("-", "_"): {"type": "string"} for k in LLM_SECTIONS
        },
        "additionalProperties": False,
    }
    schema["required"] = list(schema["properties"])
    prompt = json.dumps(
        {
            "facts_placeholders": {k: _fmt(v) for k, v in facts.items()},
            "justification": state.submission.justification if state.submission else "",
            "rationale": templates["Physical rationale"],
            "evidence": [i.model_dump(mode="json") for i in ws.insights],
            "papers": [{"paper_id": p.paper_id, "title": p.title} for p in ws.papers.values()],
            "last_review": state.iterations[-1].critique.model_dump(mode="json")
            if state.iterations and state.iterations[-1].critique
            else None,
            "sections": LLM_SECTIONS,
        },
        ensure_ascii=False,
        default=str,
    )
    errors: list[ErrorRecord] = []
    try:
        raw = llm.structured("brief", BRIEF_SYSTEM, prompt, schema)
    except LLMError as exc:
        return templates, [ErrorRecord(step="brief", code="llm_failed", message=str(exc)[:300])]
    out = dict(templates)
    for title in LLM_SECTIONS:
        key = title.lower().replace(" ", "_").replace("-", "_")
        draft = raw.get(key)
        if not draft:
            continue
        bad = check_numbers(draft, {}, None)  # placeholders only: any literal number must come from evidence
        bad = [
            b for b in bad if b not in evidence_numbers and b.lstrip("-") not in {str(i) for i in range(11)}
        ]
        if bad:
            errors.append(ErrorRecord(step="brief", code="number_check", message=f"{title}: {bad[:5]}"))
            continue
        out[title] = fill(draft, facts)
    return out, errors


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
        Sections that fell back to templates.
    """
    facts = collect_facts(state, ws)
    sections, errors = write_sections(llm, state, ws, facts)
    sub = state.submission
    lines = [f"# Experimental brief: {facts.get('win_composition', 'no candidate')}", ""]
    if not facts["gate_passed"]:
        lines += [
            "> **Target not met.** The candidate below is the best attempt; it failed at least one code check.",
            "",
        ]
    body: dict[str, str] = {}
    body["Recommended composition and structure"] = (
        f"**{facts.get('win_composition', 'n/a')}** ({facts.get('win_id', 'n/a')}), space group "
        f"{facts.get('win_space_group') or 'n/a'}, structure from {facts.get('win_source') or 'n/a'}."
        + (f" Picked by {sub.picked_by}." if sub else "")
    )
    body["Predicted properties against target"] = property_table(facts) if sub else "No candidate."
    body["Validation verdict"] = _validation(state.validation)
    rows = []
    if facts.get("run_id"):
        rows.append(
            f"- {facts['run_composition']} ({facts['run_id']}): {facts['target_property']} {_fmt(facts['run_value'])} ± "
            f"{_fmt(facts['run_uncertainty'])} {facts['target_units']}, E_hull {_fmt(facts['run_e_hull'])} eV/atom, "
            f"dynamically stable: {_fmt(facts['run_dyn_stable'])}."
        )
    body["Runner-up candidates"] = "\n".join(rows) or "- None."
    hist = []
    for n, it in enumerate(state.iterations, start=1):
        c = it.critique
        comps = ", ".join(x.composition for x in it.candidates) or "none"
        hist.append(
            f"- Round {n}: {comps}; best {c.candidate_id if c else 'n/a'}, rubric {c.score if c else 'n/a'}, "
            f"reviewer confidence {c.llm_confidence if c and c.llm_confidence is not None else 'n/a'}."
        )
    body["Iteration history"] = "\n".join(hist) or "- No rounds."
    spec = ws.spec
    body["Scope and assumptions"] = "\n".join(
        [f"- {a}" for a in spec.assumptions]
        + [
            f"- Screening potential {ws.cfg.potential}; final re-check {ws.cfg.final_potential}.",
            f"- Band gap model at {ws.cfg.band_gap_fidelity.upper()} fidelity; values are on the model's scale.",
            f"- {state.literature.coverage_note if state.literature else ''}",
            "- Supported structures: cubic perovskite template and Materials Project polymorphs.",
        ]
    )
    body.update(sections)
    for title in SECTIONS:
        lines += [f"## {title}", "", body.get(title, ""), ""]
    md = link_citations("\n".join(lines), ws.papers)
    cited = sorted({m for m in re.findall(r"\[([^\]\[]+)\]", md) if m in ws.papers})
    return Brief(run_id=state.run_id, markdown=md, cited_paper_ids=cited), errors


def _validation(reports: list[ValidationReport]) -> str:
    """Summarise validation reports.

    Parameters
    ----------
    reports : list of ValidationReport
        Final re-check results.

    Returns
    -------
    str
        Markdown.
    """
    if not reports:
        return "Final re-check not run."
    out = []
    for r in reports:
        out.append(f"- {r.candidate_id}: **{r.verdict}**.")
        out += [f"  - {c.name}: {_fmt(c.screening_value)} → {_fmt(c.final_value)}" for c in r.checks]
        out += [f"  - {n}" for n in r.notes]
    return "\n".join(out)
