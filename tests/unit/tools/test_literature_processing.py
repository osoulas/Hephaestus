"""Deterministic literature processing: duplicates, ranking, compositions, values, aggregation."""

from typing import Any

import pytest

from hephaestus.core.types import Insight, Quantity
from hephaestus.tools.literature import (
    COVERAGE_NOTE,
    _to_paper,
    aggregate_context,
    merge_duplicates,
    normalise_composition,
    numbers_in_text,
    rank_papers,
    value_in_abstract,
)


def _insight(
    composition: str | None, value: float | None, finding_type: str = "positive", **kw: Any
) -> Insight:
    """Build an insight.

    Parameters
    ----------
    composition : str or None
        Reduced formula.
    value : float or None
        Value in eV.
    finding_type : str
        Kind of finding.
    **kw : Any
        Other fields.

    Returns
    -------
    Insight
        Insight record.
    """
    fields: dict[str, Any] = {
        "composition": composition,
        "property": "band_gap",
        "value": Quantity(value=value, uncertainty=None, units="eV", source="p1")
        if value is not None
        else None,
        "conditions": "",
        "finding_type": finding_type,
        "summary": f"{composition} finding",
        "paper_id": "p1",
    }
    fields.update(kw)
    return Insight(**fields)


def test_lm11_duplicate_pair_merges_by_doi(make_paper: Any) -> None:
    """An arXiv and a PubMed copy with the same DOI (any case) yield one record."""
    arxiv = make_paper(paper_id="a", doi="10.1/X", abstract="long abstract text", citation_count=5)
    pubmed = make_paper(paper_id="b", doi="10.1/x", source="pubmed", abstract="short", citation_count=9)
    merged = merge_duplicates([arxiv, pubmed])
    assert len(merged) == 1
    assert merged[0].paper_id == "a"
    assert merged[0].abstract == "long abstract text"
    assert merged[0].citation_count == 9


def test_lm11_duplicate_pair_merges_by_title_and_surname(make_paper: Any) -> None:
    """Without a DOI, normalised title plus first author's surname identifies a duplicate."""
    a = make_paper(paper_id="a", title="Band gap of SrTiO3", authors=["Alice Smith"])
    b = make_paper(paper_id="b", title="Band Gap of SrTiO3.", authors=["Smith, Alice"])
    c = make_paper(paper_id="c", title="Band gap of SrTiO3", authors=["Bob Jones"])
    assert [p.paper_id for p in merge_duplicates([a, b, c])] == ["a", "c"]


def test_lm12_year_from_journal_ref_else_first_available() -> None:
    """The journal reference year wins over the first-available date."""
    with_ref = {
        "id": "x",
        "source": "arxiv",
        "journal_ref": "PRM 5, 1 (2021)",
        "dates": {"first_available": "2020-12-30"},
    }
    without = {"id": "y", "source": "arxiv", "dates": {"first_available": "2020-12-30"}}
    assert _to_paper(with_ref, "semantic", 0.5).year == 2021  # type: ignore[union-attr]
    assert _to_paper(without, "semantic", 0.5).year == 2020  # type: ignore[union-attr]


def test_lm1_sources_outside_paper_contract_are_dropped() -> None:
    """Records from corpora outside ``Paper.source`` are skipped; PMC maps to PubMed."""
    assert _to_paper({"id": "x", "source": "psyarxiv"}, "semantic", 0.5) is None
    assert _to_paper({"id": "x", "source": "PMC"}, "semantic", 0.5).source == "pubmed"  # type: ignore[union-attr]


@pytest.mark.parametrize("formula", ["SrTiO3", "Sr1Ti1O3", "Sr2Ti2O6", "O3TiSr"])
def test_lm15_sr1ti1o3_matches_srtio3(formula: str) -> None:
    """Different notations of one material map to one entry.

    Parameters
    ----------
    formula : str
        Notation to normalise.
    """
    assert normalise_composition(formula) == "SrTiO3"


@pytest.mark.parametrize("word", ["In", "CO", "No", "As", "be", "the", "Perovskite", "", None, "XyZ3"])
def test_lm15_rejects_sentence_words(word: str | None) -> None:
    """Sentence words and non-formulae are not compositions.

    Parameters
    ----------
    word : str or None
        Text to reject.
    """
    assert normalise_composition(word) is None


def test_numbers_ignore_formula_digits() -> None:
    """Subscripts in formulae are not numbers; signs, decimals and exponents are read."""
    assert numbers_in_text("SrTiO3 has a gap of 3.25 eV") == [(3.25, 2)]
    assert numbers_in_text("dE = \u22120.5 eV and n = 2.1\u00d710^19") == [(-0.5, 1), (2.1e19, 0)]


def test_lm14_value_absent_from_abstract_dropped() -> None:
    """A value appears only if some number in the abstract matches it to that number's precision."""
    abstract = "We report a band gap of 3.05 eV for SrTi0.875Sn0.125O3, down from 3.2 eV."
    assert value_in_abstract(3.05, abstract)
    assert value_in_abstract(3.2, abstract)
    assert value_in_abstract(3.20, abstract)
    assert not value_in_abstract(2.9, abstract)
    assert not value_in_abstract(3.0, abstract)
    assert not value_in_abstract(0.875, "SrTiO3")


def test_lm16_best_value_respects_direction() -> None:
    """The property map keeps the best value per composition for the target's direction."""
    insights = [
        _insight("SrTiO3", 3.2),
        _insight("SrTiO3", 3.0),
        _insight("SrTiO3", 9.9, property="bulk_modulus"),
        _insight("BaSnO3", 3.1),
    ]
    smallest = aggregate_context([], insights, target_property="band_gap", direction="minimise")
    largest = aggregate_context([], insights, target_property="band_gap", direction="maximise")
    hit = aggregate_context([], insights, target_property="band_gap", direction="hit", target_value=3.15)
    assert smallest.property_map["SrTiO3"].value == 3.0
    assert largest.property_map["SrTiO3"].value == 3.2
    assert hit.property_map["SrTiO3"].value == 3.2
    assert set(smallest.property_map) == {"SrTiO3", "BaSnO3"}


def test_lm18_caps_and_negative_bonus(make_paper: Any) -> None:
    """Ranking caps the list and a negative-result source outranks an equal positive one."""
    papers = [make_paper(paper_id=f"p{i}", relevance=0.5) for i in range(40)]
    papers.append(make_paper(paper_id="neg", relevance=0.5, found_via="negative"))
    ranked = rank_papers(papers, cap=30)
    assert len(ranked) == 30
    assert ranked[0].paper_id == "neg"


def test_lm18_insights_capped_per_type() -> None:
    """Each finding type keeps at most ``insight_cap`` insights."""
    insights = [_insight("SrTiO3", None, "negative") for _ in range(25)] + [
        _insight("SrTiO3", None, "synthesis")
    ]
    context = aggregate_context([], insights, insight_cap=20)
    assert len(context.failed_approaches) == 20
    assert len(context.synthesis_routes) == 1
    assert len(context.insights) == 21


def test_lm19_coverage_note() -> None:
    """Every context states what the corpus misses."""
    note = aggregate_context([], []).coverage_note
    assert note == COVERAGE_NOTE
    assert "Journal-only" in note


def test_lm11_pubmed_bracket_subscripts_match_arxiv_title(make_paper: Any) -> None:
    """PubMed's "SrTiO(3)" title matches the arXiv "SrTiO3" title without a DOI."""
    a = make_paper(paper_id="a", title="Strain in SrTiO3 films", authors=["Alice Smith"])
    b = make_paper(paper_id="b", title="Strain in SrTiO(3) films.", authors=["Smith, A"], source="pubmed")
    assert [p.paper_id for p in merge_duplicates([a, b])] == ["a"]


def test_lm15_pubmed_bracket_subscripts() -> None:
    """PubMed writes subscripts in brackets; they normalise like plain formulae."""
    assert normalise_composition("SrTiO(3)") == "SrTiO3"
    assert normalise_composition("SrTi(0.875)Sn(0.125)O(3)") == normalise_composition("SrTi0.875Sn0.125O3")


def test_lm12_year_from_structured_first_available() -> None:
    """Valency's ``dates.first_available`` is an object with a ``value``."""
    raw = {
        "id": "x",
        "source": "pubmed",
        "dates": {"first_available": {"value": "2011-09-30", "precision": "day"}},
    }
    assert _to_paper(raw, "semantic", 0.5).year == 2011  # type: ignore[union-attr]


def test_lm14_value_must_equal_written_number() -> None:
    """An integer in the abstract does not accept nearby values, and extra precision is not invented."""
    assert not value_in_abstract(2.7, "Samples were annealed for 3 h.")
    assert not value_in_abstract(3.24, "a gap of 3.2 eV")
    assert value_in_abstract(3.0, "annealed for 3 h")
    assert value_in_abstract(-0.5, "lowered by 0.5 eV")
