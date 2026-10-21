"""Real Valency Bond calls; run with ``pytest -m live`` after ``python -m hephaestus signin``."""

from collections.abc import Iterator

import pytest

from hephaestus.core.valency_auth import TokenStore
from hephaestus.tools import literature

pytestmark = pytest.mark.live


@pytest.fixture(scope="module", autouse=True)
def real_valency() -> Iterator[None]:
    """Use the real MCP client, skipping when nobody has signed in.

    Yields
    ------
    None
        After wiring the real backend.
    """
    if not TokenStore().status()["signed_in"]:
        pytest.skip("not signed in to Valency; run `python -m hephaestus signin`")
    literature.reset()
    literature.configure(backend=literature.McpValency(interactive=False))
    yield
    literature.reset()


def test_lm2_connect_uses_stored_token() -> None:
    """A signed-in machine connects and searches without opening a browser."""
    papers = literature.valency_semantic_search("SrTiO3 band gap", limit=5)
    assert papers
    assert all(0.0 <= p.relevance <= 1.0 for p in papers)


def test_every_wrapper_parses_real_results() -> None:
    """Each wrapper returns records in the shared ``Paper`` shape."""
    seeds = literature.valency_semantic_search("SrTiO3 band gap", limit=5)
    assert seeds and all(p.year and p.abstract and p.title for p in seeds)
    seed = max(seeds, key=lambda p: p.citation_count or 0)
    assert seed.citation_count, "need a cited seed to test citing papers"
    negative = literature.valency_abstract_search("SrTiO3 secondary phase", limit=5)
    assert negative and all(p.found_via == "negative" for p in negative)
    similar = literature.valency_similar(seed.paper_id, limit=3, seed_relevance=seed.relevance)
    assert similar and all(p.relevance <= seed.relevance + 1e-9 for p in similar)
    citing = literature.valency_citing(seed.paper_id, limit=10)
    assert citing, "a well-cited seed should have citing papers in Valency's corpus"
    fetched = literature.valency_get_paper(citing[0].paper_id)
    assert fetched is not None and fetched.abstract
    assert literature.normalise_composition("SrTiO(3)") == "SrTiO3"


def test_lm8_real_trend_is_contiguous() -> None:
    """The real trend covers every year from first to last."""
    trend = literature.valency_trend("SrTiO3 photocatalyst")
    assert trend
    assert list(trend) == list(range(min(trend), max(trend) + 1))
