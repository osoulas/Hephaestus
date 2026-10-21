"""Valency client: wrappers, retry, caching and result parsing (fake backend, no network)."""

from typing import Any

import pytest

from hephaestus.tools import literature
from hephaestus.tools.registry import ToolError


def test_lm8_trend_fills_missing_years(valency: Any) -> None:
    """Years Valency omits are filled with 0 between the first and last year."""
    assert literature.valency_trend("SrTiO3 band gap") == {2018: 12, 2019: 15, 2020: 0, 2021: 20}
    tool, args = valency.calls[-1]
    assert tool == "get_keyword_trends"
    assert args["query"] == "SrTiO3 band gap"
    assert args["granularity"] == "year"


def test_trend_accepts_standard_format(valency: Any) -> None:
    """The default ``standard`` format (a list of ``{period, paper_count}``) parses too."""
    valency.responses["get_keyword_trends"] = {
        "periods": [{"period": "2020", "paper_count": 3}, {"period": "2022", "paper_count": 1}]
    }
    assert literature.valency_trend("q") == {2020: 3, 2021: 0, 2022: 1}


def test_lm10_retries_once_then_succeeds(valency: Any) -> None:
    """One failure is retried."""
    valency.failures = 1
    assert literature.valency_semantic_search("SrTiO3")
    assert len(valency.calls) == 2


def test_lm10_retries_once_then_tool_error(valency: Any) -> None:
    """Two failures give a ``ToolError``, which the dispatcher reports and the run continues."""
    valency.failures = 2
    with pytest.raises(ToolError, match="after one retry"):
        literature.valency_semantic_search("SrTiO3")
    assert len(valency.calls) == 2


def test_lm2_signin_required_is_not_retried(valency: Any) -> None:
    """A missing sign-in fails at once with the sign-in hint."""
    from hephaestus.core.valency_auth import SignInRequired

    def no_token(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """Fail as a signed-out client would.

        Parameters
        ----------
        tool : str
            Ignored.
        args : dict
            Ignored.

        Returns
        -------
        dict
            Never returns.

        Raises
        ------
        ExceptionGroup
            Wrapping ``SignInRequired``, as the SDK's task group does.
        """
        raise ExceptionGroup("connect failed", [SignInRequired("no token")])

    valency.call = no_token
    with pytest.raises(ToolError, match="signin"):
        literature.valency_semantic_search("SrTiO3")


def test_dl3_second_identical_call_hits_cache(valency: Any) -> None:
    """With the store's cache configured, a repeated call makes no second backend call."""
    cache: dict[str, Any] = {}

    def cached(namespace: str, key_inputs: dict[str, Any], compute_fn: Any) -> Any:
        """Cache by exact inputs.

        Parameters
        ----------
        namespace : str
            Namespace.
        key_inputs : dict
            Inputs.
        compute_fn : callable
            Producer.

        Returns
        -------
        Any
            Result.
        """
        key = namespace + repr(sorted(key_inputs.items()))
        if key not in cache:
            cache[key] = compute_fn()
        return cache[key]

    literature.configure(cache=cached)
    first = literature.valency_semantic_search("SrTiO3")
    second = literature.valency_semantic_search("SrTiO3")
    assert first == second
    assert len(valency.calls) == 1


def test_dl5_calls_are_logged(valency: Any) -> None:
    """Every attempt is logged with service, operation and outcome."""
    log: list[tuple[Any, ...]] = []
    literature.configure(log=lambda *args: log.append(args))
    valency.failures = 1
    literature.valency_citing("2101.00001v2")
    assert [(entry[0], entry[1], entry[3]) for entry in log] == [
        ("valency", "get_citing_papers", False),
        ("valency", "get_citing_papers", True),
    ]


def test_semantic_search_parses_and_tags(valency: Any) -> None:
    """Semantic results become papers tagged ``semantic``; foreign corpora are dropped."""
    papers = literature.valency_semantic_search("SrTiO3 band gap", limit=5)
    assert [p.paper_id for p in papers] == ["2101.00001v2", "PMC1234567", "2203.04567v1"]
    assert papers[0].year == 2021
    assert papers[0].relevance == pytest.approx(0.91)
    assert {p.found_via for p in papers} == {"semantic"}
    assert valency.calls[-1][1]["limit"] == 5


def test_lm4_citation_pass_tagged_cited(valency: Any) -> None:
    """The citation-sorted pass asks Valency to sort by citations and tags the papers ``cited``."""
    papers = literature.valency_semantic_search("SrTiO3", sort_by="citations")
    assert valency.calls[-1][1]["sort_by"] == "citations"
    assert {p.found_via for p in papers} == {"cited"}


def test_abstract_hits_rank_below_good_semantic_matches(valency: Any) -> None:
    """Keyword hits get a fixed, lower relevance because Valency gives no comparable score."""
    (paper,) = literature.valency_abstract_search("SrTiO3 secondary phase")
    assert paper.found_via == "negative"
    assert paper.relevance == literature.KEYWORD_RELEVANCE


def test_similar_relevance_scaled_by_seed(valency: Any) -> None:
    """Similarity to the seed is scaled by the seed's own relevance."""
    (paper,) = literature.valency_similar("2101.00001v2", seed_relevance=0.5)
    assert paper.relevance == pytest.approx(0.4)
    assert paper.found_via == "similar"


def test_lm7_citing_sorted_by_citations(valency: Any) -> None:
    """Citing papers are requested most-cited first and tagged ``citing``."""
    (paper,) = literature.valency_citing("2101.00001v2")
    assert valency.calls[-1][1]["sort_by"] == "citations"
    assert paper.found_via == "citing"


def test_get_paper_parses_bare_record(valency: Any) -> None:
    """``get_paper_by_id`` returns a bare record, which parses to a paper."""
    paper = literature.valency_get_paper("2401.44444v1")
    assert paper is not None
    assert paper.doi == "10.1000/syn.2024.3"


def test_mcp_valency_resolves_tool_errors_without_reconnecting() -> None:
    """A tool-level error from the server becomes an exception for that call only."""
    import asyncio
    import concurrent.futures
    from types import SimpleNamespace

    class Client:
        """Fake connected session."""

        async def call_tool(self, tool: str, args: dict[str, Any]) -> Any:
            """Return an error result for ``bad`` and JSON text otherwise.

            Parameters
            ----------
            tool : str
                Tool name.
            args : dict
                Ignored.

            Returns
            -------
            SimpleNamespace
                Shaped like ``CallToolResult``.
            """
            if tool == "bad":
                return SimpleNamespace(
                    content=[SimpleNamespace(text="nope")], is_error=True, structured_content=None
                )
            return SimpleNamespace(
                content=[SimpleNamespace(text='{"papers": []}')], is_error=False, structured_content=None
            )

    bad: concurrent.futures.Future[Any] = concurrent.futures.Future()
    good: concurrent.futures.Future[Any] = concurrent.futures.Future()
    asyncio.run(literature.McpValency._handle(Client(), ("bad", {}, bad)))
    asyncio.run(literature.McpValency._handle(Client(), ("good", {}, good)))
    with pytest.raises(RuntimeError, match="nope"):
        bad.result()
    assert good.result() == {"papers": []}


def test_lm7_citing_keeps_only_corpus_papers(valency: Any) -> None:
    """Citing records come from OpenAlex; only those in Valency's corpus are kept, with their IDs."""
    papers = literature.valency_citing("2101.00001v2")
    assert [p.paper_id for p in papers] == ["2310.33333v1"]
    assert papers[0].year == 2023
    assert papers[0].citation_count == 2
    assert papers[0].abstract == ""


@pytest.mark.parametrize(
    ("paper_id", "source"),
    [
        ("25352066", "pubmed"),
        ("PMC3855844", "pubmed"),
        ("1109.1271v1", "arxiv"),
        ("cond-mat/0601001", "arxiv"),
        ("10.1101/2020.01.01.123", None),
    ],
)
def test_source_from_id(paper_id: str, source: str | None) -> None:
    """Citing records carry no source, so it is inferred from the ID shape.

    Parameters
    ----------
    paper_id : str
        Valency ID.
    source : str or None
        Expected corpus.
    """
    assert literature._source_from_id(paper_id) == source
