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


def test_eb7_citations_link_to_valency() -> None:
    """Known IDs become links; unknown ones are left alone."""
    papers = {"p1": Paper(paper_id="p1", title="t", url="https://valency.io/p1")}
    assert link_citations("see [p1] and [x]", papers) == "see [p1](https://valency.io/p1) and [x]"
