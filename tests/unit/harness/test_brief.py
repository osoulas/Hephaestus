"""Brief: numbers only from stored results (EB-2), citations linked (EB-7)."""

from hephaestus.core.types import Paper
from hephaestus.harness.brief import check_numbers, fill, link_citations


def test_eb2_number_check_flags_invented_numbers() -> None:
    """A number not in the facts is flagged; formula subscripts are not numbers."""
    facts = {"gap": 2.2, "e_hull": 0.02}
    assert check_numbers("SrTi0.75Fe0.25O3 has a gap of 2.2 eV and E_hull 0.02.", facts) == []
    assert check_numbers("The gap is 1.7 eV.", facts) == ["1.7"]


def test_eb2_placeholders_are_filled_from_facts() -> None:
    """Placeholders become formatted facts; unknown ones stay visible."""
    assert fill("gap {gap} eV, {nope}", {"gap": 2.2}) == "gap 2.2 eV, {nope}"


def test_eb7_citations_become_numbered_links() -> None:
    """Known IDs become numbered links in reading order; groups split; unknown IDs are left alone."""
    papers = {
        "p1": Paper(paper_id="p1", title="t", url="https://valency.io/p1"),
        "p2": Paper(paper_id="p2", title="u", doi="10.1/x"),
    }
    order: list[str] = []
    out = link_citations("see [p2] and [x], then [p1; p2]", papers, order)
    assert (
        out
        == "see [[1]](https://doi.org/10.1/x) and [x], then [[2]](https://valency.io/p1)[[1]](https://doi.org/10.1/x)"
    )
    assert order == ["p2", "p1"]


def test_eb2_citation_ids_and_formulas_are_not_numbers() -> None:
    """PubMed and arXiv IDs inside citations, links and ion charges never trip the number check."""
    text = "Stiffness model [37460465] and [arxiv:2511.00430]; see [MP](https://x.org/mp-149). Al3+ and Cr3+."
    assert check_numbers(text, {}) == []


def test_eb4_inferred_lab_conditions_are_allowed_but_results_are_not() -> None:
    """An inferred step may state a temperature and time; a computed result still needs a fact."""
    from hephaestus.harness.brief import _inferred_conditions_ok

    step = "3. **Calcination.** Calcine at 1250 °C for 12 h in air (inferred from YAlO3)."
    assert check_numbers(step, {}, _inferred_conditions_ok(step)) == []
    claim = "The bulk modulus is 201 GPa (inferred)."
    assert check_numbers(claim, {}, _inferred_conditions_ok(claim)) == ["201"]
