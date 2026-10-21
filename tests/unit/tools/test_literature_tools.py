"""Literature tool handlers against the ``docs/TOOLS.md`` contracts (fake backend, no network)."""

from typing import Any

import pytest

from hephaestus.tools import literature
from hephaestus.tools.literature import (
    HANDLERS,
    LITERATURE_TOOLS,
    check_novelty,
    get_paper,
    note_evidence,
    publication_trend,
    related_papers,
    search_literature,
)
from hephaestus.tools.registry import ToolError


def test_search_returns_compact_papers_and_records_query(valency: Any, ws: Any) -> None:
    """Results are compact, deduplicated, stored in the workspace, and the query is logged."""
    result = search_literature(ws, "SrTiO3 band gap", "semantic", 10)
    assert result["n"] == 2
    assert set(result["papers"][0]) == {"paper_id", "title", "year", "citations", "source", "abstract"}
    assert set(ws.papers) == {"2101.00001v2", "2203.04567v1"}
    assert ws.papers["2101.00001v2"].citation_count == 130
    assert ws.queries == [{"tool": "search_literature", "query": "SrTiO3 band gap", "mode": "semantic"}]


def test_search_limit_capped_at_15_and_abstract_at_700(valency: Any, ws: Any) -> None:
    """The limit sent to Valency is capped and abstracts are shortened for the LLM only."""
    valency.responses["semantic_search_papers"]["papers"][0]["abstract"] = "x" * 2000
    result = search_literature(ws, "q", "semantic", 50)
    assert valency.calls[-1][1]["limit"] == 15
    assert len(result["papers"][0]["abstract"]) == 700
    assert len(ws.papers["2101.00001v2"].abstract) == 2000


def test_lm5_abstract_mode_tags_negative(valency: Any, ws: Any) -> None:
    """Abstract mode is the failed-attempts search; its papers are tagged ``negative``."""
    search_literature(ws, "SrTiO3 Sn secondary phase", "abstract", 5)
    assert valency.calls[-1][0] == "search_by_abstract"
    assert ws.papers["1905.11111v1"].found_via == "negative"


def test_lm4_citations_mode(valency: Any, ws: Any) -> None:
    """Citations mode is a semantic search sorted by citations."""
    search_literature(ws, "SrTiO3", "citations", 5)
    assert valency.calls[-1] == (
        "semantic_search_papers",
        {"query": "SrTiO3", "limit": 5, "sort_by": "citations", "include_abstract": True, "max_authors": 6},
    )
    assert ws.papers["2101.00001v2"].found_via == "cited"


def test_lm3_distinct_queries_are_all_recorded(valency: Any, ws: Any) -> None:
    """Every query the agent makes is kept for the log (LM-3 needs at least four)."""
    for mode, query in [("semantic", "a"), ("abstract", "b"), ("citations", "c"), ("semantic", "d")]:
        search_literature(ws, query, mode, 5)
    assert len({q["query"] for q in ws.queries}) == 4


def test_unknown_mode_and_empty_query_are_tool_errors(valency: Any, ws: Any) -> None:
    """Bad inputs come back as ``ToolError`` without calling Valency."""
    with pytest.raises(ToolError, match="unknown mode"):
        search_literature(ws, "q", "fuzzy", 5)
    with pytest.raises(ToolError, match="empty"):
        search_literature(ws, "  ", "semantic", 5)
    assert valency.calls == []


def test_related_requires_known_seed(valency: Any, ws: Any) -> None:
    """The seed must have been retrieved first."""
    with pytest.raises(ToolError, match="unknown paper_id"):
        related_papers(ws, "nope", "similar", 5)


def test_lm6_lm7_related_similar_and_citing(valency: Any, ws: Any) -> None:
    """Similar and citing papers are added with their tags and 400-character abstracts."""
    search_literature(ws, "SrTiO3", "semantic", 5)
    similar = related_papers(ws, "2101.00001v2", "similar", 50)
    citing = related_papers(ws, "2101.00001v2", "citing", 3)
    assert valency.calls[-2][1]["limit"] == 10
    assert similar["papers"][0]["paper_id"] == "2008.22222v3"
    assert citing["papers"][0]["paper_id"] == "2310.33333v1"
    assert ws.papers["2008.22222v3"].found_via == "similar"
    assert ws.papers["2310.33333v1"].found_via == "citing"


def test_lm9_get_paper_reuses_workspace(valency: Any, ws: Any) -> None:
    """A paper already held with an abstract is not fetched again."""
    search_literature(ws, "SrTiO3", "semantic", 5)
    n_calls = len(valency.calls)
    result = get_paper(ws, "2101.00001v2")
    assert len(valency.calls) == n_calls
    assert result["doi"] == "10.1000/sto.2021.1"
    assert set(result) == {
        "paper_id",
        "title",
        "year",
        "citations",
        "source",
        "abstract",
        "doi",
        "url",
        "authors",
    }


def test_get_paper_fetches_unknown_paper_once(valency: Any, ws: Any) -> None:
    """An unknown paper is fetched, stored, and then served from the workspace."""
    get_paper(ws, "2401.44444v1")
    get_paper(ws, "2401.44444v1")
    assert [tool for tool, _ in valency.calls] == ["get_paper_by_id"]
    assert "2401.44444v1" in ws.papers


def test_publication_trend_fills_years_and_stores_trend(valency: Any, ws: Any) -> None:
    """The trend is returned with missing years filled and kept in the workspace."""
    result = publication_trend(ws, "SrTiO3 band gap")
    assert result == {"query": "SrTiO3 band gap", "papers_per_year": {2018: 12, 2019: 15, 2020: 0, 2021: 20}}
    assert ws.trend[2020] == 0


def test_note_evidence_unknown_paper_errors(ws: Any) -> None:
    """Evidence must cite a paper already retrieved."""
    with pytest.raises(ToolError, match="unknown paper_id"):
        note_evidence(ws, "nope", "band_gap", "positive", "s")


def test_lm14_note_evidence_keeps_value_in_abstract(valency: Any, ws: Any) -> None:
    """A value from the abstract is kept, with the paper as its source and a normalised composition."""
    search_literature(ws, "SrTiO3", "semantic", 5)
    result = note_evidence(
        ws,
        "2101.00001v2",
        "band_gap",
        "positive",
        "Sn lowers the gap",
        composition="Sr1Ti1O3",
        value=3.25,
        units="eV",
    )
    assert result == {"recorded": True, "value_dropped_not_in_abstract": False, "n_insights": 1}
    insight = ws.insights[0]
    assert insight.composition == "SrTiO3"
    assert insight.value.value == 3.25
    assert insight.value.source == "2101.00001v2"


def test_lm14_note_evidence_drops_invented_value(valency: Any, ws: Any) -> None:
    """A value not in the abstract is dropped by code; the finding is still recorded."""
    search_literature(ws, "SrTiO3", "semantic", 5)
    result = note_evidence(
        ws, "2101.00001v2", "band_gap", "positive", "s", composition="SrTiO3", value=2.71, units="eV"
    )
    assert result["value_dropped_not_in_abstract"] is True
    assert ws.insights[0].value is None


def test_check_novelty_hits_and_none(valency: Any) -> None:
    """A composition named in a result is a hit; one never named has none."""
    known = check_novelty("SrTi0.875Sn0.125O3")
    assert known["hits"] == 1
    assert known["paper_ids"] == ["2101.00001v2"]
    novel = check_novelty("BaHfS3")
    assert novel == {"composition": "BaHfS3", "hits": 0, "paper_ids": []}


def test_check_novelty_rejects_non_formula(valency: Any) -> None:
    """An unparseable composition is a tool error."""
    with pytest.raises(ToolError, match="cannot parse"):
        check_novelty("perovskite")


def test_schemas_match_handlers() -> None:
    """Every schema has a handler, a closed input object and only declared required fields."""
    assert {tool["name"] for tool in LITERATURE_TOOLS} == set(HANDLERS)
    for tool in LITERATURE_TOOLS:
        schema = tool["input_schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) <= set(schema["properties"])


def test_workspaces_do_not_share_state(valency: Any, ws: Any) -> None:
    """Handlers keep no state of their own: a second workspace sees none of the first's papers."""
    other = type(ws)()
    search_literature(ws, "SrTiO3", "semantic", 5)
    assert other.papers == {}
    assert literature.HANDLERS["search_literature"] is search_literature


def test_citing_paper_abstract_fetched_on_get_paper(valency: Any, ws: Any) -> None:
    """A citing paper arrives without an abstract; ``get_paper`` fetches it once, keeping its ID."""
    search_literature(ws, "SrTiO3", "semantic", 5)
    related_papers(ws, "2101.00001v2", "citing", 5)
    assert ws.papers["2310.33333v1"].abstract == ""
    valency.responses["get_paper_by_id"]["paper"].update(
        id="2310.33333v1", title="Revisiting the gap of Sn-doped SrTiO3"
    )
    result = get_paper(ws, "2310.33333v1")
    assert result["abstract"].startswith("SrTiO3 nanocubes")
    assert ws.papers["2310.33333v1"].found_via == "citing"
