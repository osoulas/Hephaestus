"""Valency Bond client, evidence notes and novelty search (Section 2, LM-*).

Four layers, bottom up:

1. **Client.** :class:`McpValency` runs the async MCP session on one background thread. :func:`_call`
   adds the cache, one retry (LM-10) and call logging. The ``valency_*`` wrappers turn raw results
   into :class:`~hephaestus.core.types.Paper` records.
2. **Processing.** Deterministic functions: duplicates (LM-11), ranking (LM-18), compositions
   (LM-15), the value-in-abstract check (LM-14), aggregation (LM-16, LM-19) and novelty (SC-11).
3. **Tool handlers.** ``search_literature``, ``related_papers``, ``get_paper``,
   ``publication_trend`` and ``note_evidence``, as specified in ``docs/TOOLS.md``.
   Each is registered with :func:`~hephaestus.tools.registry.register`; the schemas live in
   ``registry.TOOLS``.

``find_open_questions`` (LM-17) needs an LLM call, so it lives in ``harness`` (tools may not import
``llm``).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import math
import re
import threading
import time
from collections.abc import Callable, Iterable
from contextlib import AsyncExitStack
from typing import Any, Protocol

from pymatgen.core import Composition, Element

from hephaestus.core.store import cached
from hephaestus.core.types import Insight, LiteratureContext, Paper, Quantity
from hephaestus.tools.registry import ToolError, register

# Bump when the wrappers change what they send, so stale cache entries are not reused (DL-3).
VALENCY_CLIENT_VERSION = "1"
SOURCES = {"arxiv", "pubmed", "biorxiv", "medrxiv"}
# search_by_abstract and get_citing_papers give no comparable score; rank them below good
# semantic matches (docs/valency_tools.md).
KEYWORD_RELEVANCE = 0.5
MAX_AUTHORS = 6
COVERAGE_NOTE = (
    "Literature comes from Valency Bond, which indexes arXiv, bioRxiv, medRxiv and PubMed records. "
    "Journal-only papers with no preprint or PubMed record are not covered."
)
CALL_TIMEOUT_S = 330.0
# Literature calls run in parallel (``registry.lock_key`` is None for them), so changes to the
# shared workspace collections and the lazy backend are guarded.
_WS_LOCK = threading.Lock()
_BACKEND_LOCK = threading.Lock()


# ---------------------------------------------------------------------------------------------
# 1. Client
# ---------------------------------------------------------------------------------------------


class ValencyBackend(Protocol):
    """Anything that can call a Valency tool synchronously and return its JSON result."""

    def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """Call one Valency tool.

        Parameters
        ----------
        tool : str
            Valency tool name, e.g. ``"semantic_search_papers"``.
        args : dict
            Tool arguments.

        Returns
        -------
        dict
            Parsed JSON result.
        """
        ...


CacheFn = Callable[[str, dict[str, Any], Callable[[], Any]], Any]
LogFn = Callable[..., None]


def _no_cache(namespace: str, key_inputs: dict[str, Any], compute_fn: Callable[[], Any]) -> Any:
    """Compute without caching; tests use it so recorded responses are never reused across tests.

    Parameters
    ----------
    namespace : str
        Cache namespace (unused).
    key_inputs : dict
        Cache key inputs (unused).
    compute_fn : callable
        Produces the result.

    Returns
    -------
    Any
        ``compute_fn()``.
    """
    return compute_fn()


class _Config:
    """Module-level wiring: the Valency backend, the cache and the call logger."""

    backend: ValencyBackend | None = None
    cache: CacheFn = staticmethod(cached)  # type: ignore[assignment]
    log: LogFn | None = None


def configure(
    *,
    backend: ValencyBackend | None = None,
    cache: CacheFn | None = None,
    log: LogFn | None = None,
) -> None:
    """Wire the literature tools to a backend, cache and logger.

    Valency results are cached on disk by exact inputs by default (``core.store.cached``, DL-3), so a
    repeated call and an offline replay make no network request. Tests pass a fake backend and
    ``_no_cache``.

    Parameters
    ----------
    backend : ValencyBackend or None
        Valency client; :class:`McpValency` is created lazily if never set.
    cache : callable or None
        ``cached(namespace, key_inputs, compute_fn)``; defaults to ``core.store.cached``.
    log : callable or None
        ``log_external_call(service, operation, duration_s, ok, error)``.
    """
    if backend is not None:
        _Config.backend = backend
    if cache is not None:
        _Config.cache = cache
    if log is not None:
        _Config.log = log


def reset() -> None:
    """Restore the default wiring (no backend yet, the disk cache, no logger)."""
    _Config.backend = None
    _Config.cache = cached
    _Config.log = None


def get_backend() -> ValencyBackend:
    """Return the configured backend, creating the real MCP client on first use.

    Returns
    -------
    ValencyBackend
        The Valency client.
    """
    with _BACKEND_LOCK:
        if _Config.backend is None:
            _Config.backend = McpValency()
        return _Config.backend


class McpValency:
    """Synchronous Valency client over one long-lived MCP session (``valency_connect``, LM-2).

    The MCP SDK is async and its session must be opened and closed in the same task, so a single
    owner task on a background event loop holds the session and serves requests from a queue.
    A transport failure closes the session; the next request reconnects.

    Parameters
    ----------
    interactive : bool
        Allow a browser sign-in on connect. Runs use ``False`` and fail with a sign-in hint.
    """

    def __init__(self, interactive: bool = False) -> None:
        """Start the background event loop; the session opens on the first call.

        Parameters
        ----------
        interactive : bool
            Allow a browser sign-in on connect.
        """
        self._interactive = interactive
        self._loop = asyncio.new_event_loop()
        self._queue: asyncio.Queue[tuple[str, dict[str, Any], concurrent.futures.Future[Any]]] | None = None
        threading.Thread(target=self._loop.run_forever, daemon=True, name="valency-mcp").start()
        self._server_task = asyncio.run_coroutine_threadsafe(self._serve(), self._loop)

    def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """Call one Valency tool and wait for its result.

        Parameters
        ----------
        tool : str
            Valency tool name.
        args : dict
            Tool arguments.

        Returns
        -------
        dict
            Parsed JSON result.
        """
        fut: concurrent.futures.Future[Any] = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(self._put, (tool, args, fut))
        result: dict[str, Any] = fut.result(timeout=CALL_TIMEOUT_S)
        return result

    def _put(self, item: tuple[str, dict[str, Any], concurrent.futures.Future[Any]]) -> None:
        """Queue a request; runs on the event loop.

        Parameters
        ----------
        item : tuple
            Tool name, arguments and the future to resolve.
        """
        if self._queue is None:
            self._queue = asyncio.Queue()
        self._queue.put_nowait(item)

    async def _serve(self) -> None:
        """Own the session: connect on demand, serve requests, reconnect after a failure."""
        from hephaestus.core.valency_auth import open_client

        if self._queue is None:
            self._queue = asyncio.Queue()
        queue = self._queue
        while True:
            item = await queue.get()
            try:
                async with AsyncExitStack() as stack:
                    client = await open_client(stack, interactive=self._interactive)
                    await self._handle(client, item)
                    while True:
                        item = await queue.get()
                        await self._handle(client, item)
            except BaseException as exc:  # surfaced through the caller's future
                if isinstance(exc, asyncio.CancelledError):
                    raise
                if not item[2].done():
                    item[2].set_exception(exc)

    @staticmethod
    async def _handle(client: Any, item: tuple[str, dict[str, Any], concurrent.futures.Future[Any]]) -> None:
        """Run one request; a tool-level error resolves the future, a transport error propagates.

        Parameters
        ----------
        client : mcp.Client
            Connected session.
        item : tuple
            Tool name, arguments and the future to resolve.
        """
        tool, args, fut = item
        result = await client.call_tool(tool, args)
        text = "".join(getattr(block, "text", "") for block in result.content)
        if result.is_error:
            fut.set_exception(RuntimeError(f"Valency {tool} error: {text[:300]}"))
            return
        if isinstance(result.structured_content, dict):
            fut.set_result(result.structured_content)
            return
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            fut.set_exception(RuntimeError(f"Valency {tool} returned non-JSON content"))
            return
        fut.set_result(data if isinstance(data, dict) else {"result": data})


def _log_call(tool: str, started: float, ok: bool, error: str | None) -> None:
    """Record one external call if a logger is configured (DL-5).

    Parameters
    ----------
    tool : str
        Valency tool name.
    started : float
        ``time.monotonic()`` when the call started, in seconds.
    ok : bool
        Whether the call succeeded.
    error : str or None
        Error message on failure.
    """
    if _Config.log is not None:
        _Config.log("valency", tool, time.monotonic() - started, ok, error)


def _call(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call Valency through the cache, retrying once on failure (DL-3, LM-10).

    Parameters
    ----------
    tool : str
        Valency tool name.
    args : dict
        Tool arguments; together with ``tool`` they form the cache key.

    Returns
    -------
    dict
        Parsed JSON result.

    Raises
    ------
    ToolError
        If the second attempt also fails, or a sign-in is needed. The run continues (SYS-6).
    """
    from hephaestus.core.valency_auth import SIGNIN_HINT, SignInRequired, find_exception

    def compute() -> dict[str, Any]:
        """Call the backend with one retry.

        Returns
        -------
        dict
            Parsed JSON result.
        """
        last: BaseException | None = None
        for _ in range(2):
            started = time.monotonic()
            try:
                result = get_backend().call(tool, args)
            except Exception as exc:
                if find_exception(exc, SignInRequired) is not None:
                    _log_call(tool, started, False, "sign-in required")
                    raise ToolError(SIGNIN_HINT) from None
                _log_call(tool, started, False, str(exc)[:300])
                last = exc
                continue
            _log_call(tool, started, True, None)
            return result
        raise ToolError(f"Valency unavailable for {tool} after one retry: {last}")

    key = {"tool": tool, "args": args, "client_version": VALENCY_CLIENT_VERSION}
    result: dict[str, Any] = _Config.cache("valency", key, compute)
    return result


def _year(raw: dict[str, Any]) -> int | None:
    """Return a paper's year from its journal reference, else its first-available date (LM-12).

    Parameters
    ----------
    raw : dict
        One Valency paper record.

    Returns
    -------
    int or None
        Four-digit year, or ``None`` if no date is given.
    """
    match = re.search(r"\b(19|20)\d{2}\b", str(raw.get("journal_ref") or ""))
    if match:
        return int(match.group())
    first_available = (raw.get("dates") or {}).get("first_available")
    if isinstance(first_available, dict):
        first_available = first_available.get("value")
    for value in (
        first_available,
        raw.get("first_submitted"),
        raw.get("datestamp"),
        raw.get("publication_year"),
    ):
        found = re.match(r"(19|20)\d{2}", str(value or ""))
        if found:
            return int(found.group())
    return None


def _to_paper(raw: dict[str, Any], found_via: str, relevance: float) -> Paper | None:
    """Convert one Valency record to a :class:`Paper`.

    Parameters
    ----------
    raw : dict
        Valency paper record.
    found_via : str
        How it was found: semantic, cited, negative, similar or citing.
    relevance : float
        Relevance before clamping to 0 to 1.

    Returns
    -------
    Paper or None
        ``None`` for records with no ID or from a corpus outside ``SOURCES``.
    """
    source = str(raw.get("source") or "").lower()
    if source == "pmc":
        source = "pubmed"
    paper_id = raw.get("id") or raw.get("paper_id")
    if source not in SOURCES or not paper_id:
        return None
    citations = raw.get("citation_count")
    return Paper(
        paper_id=str(paper_id),
        title=str(raw.get("title") or "").strip(),
        authors=[str(a) for a in raw.get("authors") or []],
        year=_year(raw),
        doi=raw.get("doi") or None,
        url=raw.get("url") or None,
        source=source,  # type: ignore[arg-type]
        abstract=str(raw.get("abstract") or ""),
        citation_count=int(citations) if citations is not None else None,
        relevance=min(1.0, max(0.0, float(relevance))),
        found_via=found_via,  # type: ignore[arg-type]
    )


def _papers(data: dict[str, Any], found_via: str, score: Callable[[dict[str, Any]], float]) -> list[Paper]:
    """Convert a Valency result list to papers.

    Parameters
    ----------
    data : dict
        Valency result with a ``papers`` list.
    found_via : str
        Tag for every paper.
    score : callable
        Relevance for one raw record.

    Returns
    -------
    list of Paper
        Papers in Valency's order, skipping unusable records.
    """
    records = data.get("papers") or data.get("results") or data.get("result") or []
    papers = (_to_paper(raw, found_via, score(raw)) for raw in records if isinstance(raw, dict))
    return [p for p in papers if p is not None]


def _score(raw: dict[str, Any], default: float = KEYWORD_RELEVANCE) -> float:
    """Return Valency's relevance score for a record, or ``default`` if it has none.

    Parameters
    ----------
    raw : dict
        Valency paper record.
    default : float
        Fallback relevance.

    Returns
    -------
    float
        Relevance score.
    """
    for key in ("relevance_score", "similarity", "similarity_score", "score"):
        if isinstance(raw.get(key), int | float):
            return float(raw[key])
    return default


def valency_semantic_search(query: str, limit: int = 15, sort_by: str = "relevance") -> list[Paper]:
    """Search Valency by meaning (``semantic_search_papers``; LM-3, LM-4).

    Parameters
    ----------
    query : str
        Natural-language query.
    limit : int
        Maximum papers.
    sort_by : {"relevance", "citations"}
        ``"citations"`` is the foundational-papers pass, tagged ``cited`` (LM-4).

    Returns
    -------
    list of Paper
        Matching papers.
    """
    data = _call(
        "semantic_search_papers",
        {
            "query": query,
            "limit": limit,
            "sort_by": sort_by,
            "include_abstract": True,
            "max_authors": MAX_AUTHORS,
        },
    )
    return _papers(data, "cited" if sort_by == "citations" else "semantic", _score)


def valency_abstract_search(query: str, limit: int = 15, found_via: str = "negative") -> list[Paper]:
    """Keyword search in abstracts (``search_by_abstract``), used for failed attempts (LM-5).

    Parameters
    ----------
    query : str
        Keywords, e.g. ``"SrTiO3 secondary phase solubility limit"``.
    limit : int
        Maximum papers.
    found_via : str
        Tag for the results; ``"negative"`` for failed-attempt searches.

    Returns
    -------
    list of Paper
        Matching papers, all at ``KEYWORD_RELEVANCE``.
    """
    data = _call(
        "search_by_abstract",
        {"query": query, "limit": limit, "include_abstract": True, "max_authors": MAX_AUTHORS},
    )
    return _papers(data, found_via, lambda raw: KEYWORD_RELEVANCE)


def valency_similar(paper_id: str, limit: int = 8, seed_relevance: float = 1.0) -> list[Paper]:
    """Papers similar to a seed (``find_similar_papers``; LM-6).

    Parameters
    ----------
    paper_id : str
        Seed paper's Valency ID.
    limit : int
        Maximum papers.
    seed_relevance : float
        The seed's relevance; similarity is to the seed, so it is scaled by this.

    Returns
    -------
    list of Paper
        Similar papers, tagged ``similar``.
    """
    data = _call(
        "find_similar_papers",
        {"paper_id": paper_id, "limit": limit, "include_abstract": True, "max_authors": MAX_AUTHORS},
    )
    return _papers(data, "similar", lambda raw: _score(raw) * seed_relevance)


def valency_citing(paper_id: str, limit: int = 8) -> list[Paper]:
    """Papers citing a seed, most-cited first (``get_citing_papers``; LM-7).

    Parameters
    ----------
    paper_id : str
        Seed paper's Valency ID.
    limit : int
        Maximum papers.

    Returns
    -------
    list of Paper
        Citing papers that are in Valency's corpus, tagged ``citing``. They carry no abstract;
        ``get_paper`` fetches it.

    Notes
    -----
    Valency answers with OpenAlex records under ``results``: ``title``, ``authors``,
    ``publication_year``, ``cited_by_count``, ``in_corpus`` and ``valency_paper_id``. Records not
    in the corpus are dropped, because they cannot be fetched, checked or linked (EB-7).
    """
    data = _call(
        "get_citing_papers",
        {"paper_id": paper_id, "limit": limit, "sort_by": "citations", "max_authors": MAX_AUTHORS},
    )
    records = []
    for raw in data.get("results") or data.get("papers") or []:
        valency_id = raw.get("valency_paper_id") or raw.get("id")
        source = raw.get("source") or _source_from_id(str(valency_id or ""))
        if not valency_id or raw.get("in_corpus") is False or source is None:
            continue
        records.append(
            {
                **raw,
                "id": valency_id,
                "source": source,
                "citation_count": raw.get("citation_count", raw.get("cited_by_count")),
            }
        )
    return _papers({"papers": records}, "citing", lambda raw: KEYWORD_RELEVANCE)


def _source_from_id(paper_id: str) -> str | None:
    """Infer a paper's corpus from the shape of its Valency ID.

    Parameters
    ----------
    paper_id : str
        Valency ID, e.g. ``"25352066"``, ``"PMC3855844"`` or ``"1109.1271v1"``.

    Returns
    -------
    str or None
        ``"pubmed"`` or ``"arxiv"``, or ``None`` if the shape is not recognised.
    """
    if re.fullmatch(r"(PMC)?\d+", paper_id):
        return "pubmed"
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?|[a-z-]+(\.[A-Z]{2})?/\d{7}(v\d+)?", paper_id):
        return "arxiv"
    return None


def valency_get_paper(paper_id: str, found_via: str = "semantic") -> Paper | None:
    """Fetch one paper by ID (``get_paper_by_id``).

    Parameters
    ----------
    paper_id : str
        Versioned, base or PMC ID.
    found_via : str
        Tag for the record if it is new to the run.

    Returns
    -------
    Paper or None
        The paper, or ``None`` if Valency has no usable record.
    """
    data = _call(
        "get_paper_by_id", {"paper_id": paper_id, "include_abstract": True, "max_authors": MAX_AUTHORS}
    )
    nested = data.get("paper")
    raw: dict[str, Any] = nested if isinstance(nested, dict) else data
    return _to_paper(raw, found_via, _score(raw))


def valency_trend(query: str) -> dict[int, int]:
    """Papers per year for a keyword query, with missing years filled with 0 (``get_keyword_trends``; LM-8).

    Parameters
    ----------
    query : str
        Abstract full-text query.

    Returns
    -------
    dict of int to int
        Year to paper count, covering every year from the first to the last.
    """
    data = _call("get_keyword_trends", {"query": query, "granularity": "year", "format": "compact"})
    periods: Any = data.get("periods", data.get("result", data))
    counts: dict[int, int] = {}
    pairs: Iterable[tuple[Any, Any]]
    if isinstance(periods, dict):
        pairs = periods.items()
    else:
        pairs = ((p.get("period"), p.get("paper_count")) for p in periods if isinstance(p, dict))
    for period, n in pairs:
        match = re.match(r"\d{4}", str(period))
        if match and isinstance(n, int | float):
            counts[int(match.group())] = counts.get(int(match.group()), 0) + int(n)
    if not counts:
        return {}
    return {year: counts.get(year, 0) for year in range(min(counts), max(counts) + 1)}


# ---------------------------------------------------------------------------------------------
# 2. Processing
# ---------------------------------------------------------------------------------------------


def _norm_title(title: str) -> str:
    """Lower-case a title and strip punctuation and extra spaces.

    Parameters
    ----------
    title : str
        Paper title.

    Returns
    -------
    str
        Normalised title.
    """
    title = re.sub(r"\((\d+(?:\.\d+)?)\)", r"\1", title)  # PubMed's "SrTiO(3)"
    return " ".join(re.sub(r"[^a-z0-9]+", " ", title.lower()).split())


def _first_surname(paper: Paper) -> str:
    """Return the first author's surname, lower-cased.

    Parameters
    ----------
    paper : Paper
        Paper record.

    Returns
    -------
    str
        Surname, or ``""`` if there are no authors.
    """
    if not paper.authors:
        return ""
    name = paper.authors[0].strip()
    surname = name.split(",")[0] if "," in name else (name.split() or [""])[-1]
    return re.sub(r"[^a-z]", "", surname.lower())


def _dedupe_keys(paper: Paper) -> list[str]:
    """Return the keys that identify a paper across corpora (LM-11).

    Parameters
    ----------
    paper : Paper
        Paper record.

    Returns
    -------
    list of str
        ``doi:<doi>`` and ``title:<normalised title>|<surname>`` when available.
    """
    keys = []
    if paper.doi:
        keys.append("doi:" + paper.doi.strip().lower().removeprefix("https://doi.org/"))
    title = _norm_title(paper.title)
    if title:
        keys.append(f"title:{title}|{_first_surname(paper)}")
    return keys


def _merge_pair(kept: Paper, other: Paper) -> Paper:
    """Merge a duplicate into the record already kept, keeping its ID and tag.

    Parameters
    ----------
    kept : Paper
        Record seen first; its ``paper_id`` and ``found_via`` survive.
    other : Paper
        Duplicate record.

    Returns
    -------
    Paper
        Merged record with the longer abstract and the larger counts.
    """
    counts = [c for c in (kept.citation_count, other.citation_count) if c is not None]
    return kept.model_copy(
        update={
            "abstract": kept.abstract if len(kept.abstract) >= len(other.abstract) else other.abstract,
            "authors": kept.authors or other.authors,
            "year": kept.year if kept.year is not None else other.year,
            "doi": kept.doi or other.doi,
            "url": kept.url or other.url,
            "citation_count": max(counts) if counts else None,
            "relevance": max(kept.relevance, other.relevance),
        }
    )


def _add_paper(by_id: dict[str, Paper], index: dict[str, str], paper: Paper) -> str:
    """Add a paper to ``by_id``, merging it into a duplicate if one exists (LM-9, LM-11).

    Parameters
    ----------
    by_id : dict of str to Paper
        Papers by ID; updated in place.
    index : dict of str to str
        Dedupe key to paper ID; updated in place.
    paper : Paper
        Paper to add.

    Returns
    -------
    str
        ID under which the paper is now stored.
    """
    keys = _dedupe_keys(paper)
    existing = (
        paper.paper_id if paper.paper_id in by_id else next((index[k] for k in keys if k in index), None)
    )
    if existing is None:
        by_id[paper.paper_id] = paper
        existing = paper.paper_id
    else:
        by_id[existing] = _merge_pair(by_id[existing], paper)
    for key in keys:
        index.setdefault(key, existing)
    return existing


def merge_duplicates(papers: Iterable[Paper]) -> list[Paper]:
    """Merge duplicate records by DOI, or by normalised title plus first author's surname (LM-11).

    Parameters
    ----------
    papers : iterable of Paper
        Papers, possibly with arXiv and PubMed copies of one paper.

    Returns
    -------
    list of Paper
        One record per paper, in first-seen order.
    """
    by_id: dict[str, Paper] = {}
    index: dict[str, str] = {}
    for paper in papers:
        _add_paper(by_id, index, paper)
    return list(by_id.values())


def paper_score(paper: Paper) -> float:
    """Ranking score: 0.7 * relevance + 0.3 * capped citations + 0.1 for negative results (LM-18).

    Citations count as ``log10(1 + n) / 3``, capped at 1 (1000 citations scores fully).

    Parameters
    ----------
    paper : Paper
        Paper record.

    Returns
    -------
    float
        Score, 0 to 1.1.
    """
    citations = min(1.0, math.log10(1 + (paper.citation_count or 0)) / 3)
    bonus = 0.1 if paper.found_via == "negative" else 0.0
    return 0.7 * paper.relevance + 0.3 * citations + bonus


def rank_papers(papers: Iterable[Paper], cap: int = 30) -> list[Paper]:
    """Rank papers by :func:`paper_score` and keep the top ``cap`` (LM-18).

    Parameters
    ----------
    papers : iterable of Paper
        Papers to rank.
    cap : int
        Maximum papers to keep.

    Returns
    -------
    list of Paper
        Best first.
    """
    return sorted(papers, key=paper_score, reverse=True)[:cap]


# Words that parse as element formulae but are almost always English in a sentence (LM-15).
_WORD_FORMULAE = {"in", "as", "at", "be", "he", "no", "co", "on", "so", "am", "re", "i", "s", "b", "w"}
_FORMULA_RE = re.compile(r"^(?:\(?[A-Z][a-z]?(?:\d+(?:\.\d+)?)?\)?(?:\d+(?:\.\d+)?)?)+$")


def normalise_composition(formula: str | None) -> str | None:
    """Normalise a formula to pymatgen reduced notation (LM-15).

    Parameters
    ----------
    formula : str or None
        Formula as written, e.g. ``"Sr1Ti1O3"``.

    Returns
    -------
    str or None
        Reduced formula, e.g. ``"SrTiO3"``; ``None`` for empty input, sentence words such as
        ``"In"`` or ``"CO"``, or text that is not a formula.
    """
    text = (formula or "").strip().rstrip(".,;:")
    # PubMed titles write subscripts in brackets: "SrTiO(3)" means SrTiO3.
    text = re.sub(r"\((\d+(?:\.\d+)?)\)", r"\1", text)
    if not text or not _FORMULA_RE.match(text):
        return None
    if not any(ch.isdigit() for ch in text) and text.lower() in _WORD_FORMULAE:
        return None
    try:
        composition = Composition(text)
    except (ValueError, KeyError, TypeError):
        return None
    if not composition.elements or not all(isinstance(el, Element) for el in composition.elements):
        return None
    return composition.reduced_formula


_NUMBER_RE = re.compile(
    r"(?<![\w.])([-\u2212\u2013]?\d+(?:\.\d+)?)(?:\s*(?:[\u00d7x]\s*10\^?\s*|[eE])([-\u2212\u2013]?\d+))?"
)


def numbers_in_text(text: str) -> list[tuple[float, int]]:
    """Extract numbers from text with the decimal places they were written with.

    Digits inside chemical formulae (``SrTiO3``) are not numbers.

    Parameters
    ----------
    text : str
        Abstract text.

    Returns
    -------
    list of (float, int)
        Value and decimal places, e.g. ``(3.2, 1)`` for ``"3.2"``.
    """
    found = []
    for mantissa, exponent in _NUMBER_RE.findall(text):
        mantissa = mantissa.replace("\u2212", "-").replace("\u2013", "-")
        places = len(mantissa.split(".")[1]) if "." in mantissa else 0
        value = float(mantissa)
        if exponent:
            power = int(exponent.replace("\u2212", "-").replace("\u2013", "-"))
            value *= 10.0**power
            places = max(0, places - power)
        found.append((value, places))
    return found


def value_in_abstract(value: float, abstract: str) -> bool:
    """Whether ``value`` is a number written in the abstract (LM-14).

    The value must equal the number as written: ``3.20`` matches ``3.2``, but ``2.7`` does not
    match ``3`` and ``3.24`` does not match ``3.2``. Signs are ignored, because abstracts often
    write a negative quantity as a magnitude.

    Parameters
    ----------
    value : float
        Value the LLM wants to record.
    abstract : str
        The paper's full stored abstract (not the shortened copy shown to the LLM).

    Returns
    -------
    bool
        True if some number in the abstract equals ``value``.
    """
    target = abs(value)
    return any(
        math.isclose(target, abs(number), rel_tol=1e-9, abs_tol=1e-12)
        for number, _ in numbers_in_text(abstract)
    )


def _better(new: float, old: float, direction: str, target: float | None) -> bool:
    """Whether ``new`` beats ``old`` for the target's direction (LM-16).

    Parameters
    ----------
    new, old : float
        Values to compare.
    direction : {"maximise", "minimise", "hit"}
        Target direction from ``SearchSpec``.
    target : float or None
        Target value, used when ``direction`` is ``"hit"``.

    Returns
    -------
    bool
        True if ``new`` is better.
    """
    if direction == "minimise":
        return new < old
    if direction == "hit" and target is not None:
        return abs(new - target) < abs(old - target)
    return new > old


def aggregate_context(
    papers: Iterable[Paper],
    insights: Iterable[Insight],
    field_trend: dict[int, int] | None = None,
    *,
    target_property: str | None = None,
    direction: str = "maximise",
    target_value: float | None = None,
    open_questions: list[str] | None = None,
    paper_cap: int = 30,
    insight_cap: int = 20,
) -> LiteratureContext:
    """Aggregate a run's papers and evidence into a :class:`LiteratureContext` (LM-16, LM-18, LM-19).

    Parameters
    ----------
    papers : iterable of Paper
        Every paper retrieved in the run.
    insights : iterable of Insight
        Every insight recorded with ``note_evidence``.
    field_trend : dict of int to int or None
        Papers per year from ``publication_trend``.
    target_property : str or None
        ``SearchSpec.target_property``; only insights about it enter the property map.
        ``None`` uses every insight with a value.
    direction : {"maximise", "minimise", "hit"}
        ``SearchSpec.direction``.
    target_value : float or None
        ``SearchSpec.target_value``, for ``"hit"``.
    open_questions : list of str or None
        From the harness's ``find_open_questions`` (LM-17).
    paper_cap : int
        Maximum papers kept (LM-18).
    insight_cap : int
        Maximum insights kept per finding type (LM-18).

    Returns
    -------
    LiteratureContext
        Aggregated context.
    """
    insights = list(insights)
    property_map: dict[str, Quantity] = {}
    for ins in insights:
        if ins.composition is None or ins.value is None:
            continue
        if target_property is not None and ins.property.lower() != target_property.lower():
            continue
        old = property_map.get(ins.composition)
        if old is None or _better(ins.value.value, old.value, direction, target_value):
            property_map[ins.composition] = ins.value

    kept: list[Insight] = []
    by_type: dict[str, list[str]] = {"negative": [], "mechanism": [], "synthesis": []}
    counts: dict[str, int] = {}
    for ins in insights:
        counts[ins.finding_type] = counts.get(ins.finding_type, 0) + 1
        if counts[ins.finding_type] > insight_cap:
            continue
        kept.append(ins)
        if ins.finding_type in by_type:
            by_type[ins.finding_type].append(f"{ins.summary} [{ins.paper_id}]")

    return LiteratureContext(
        papers=rank_papers(papers, paper_cap),
        insights=kept,
        property_map=property_map,
        failed_approaches=by_type["negative"],
        mechanisms=by_type["mechanism"],
        synthesis_routes=by_type["synthesis"],
        open_questions=open_questions or [],
        field_trend=dict(field_trend or {}),
        coverage_note=COVERAGE_NOTE,
    )


# Formula-like tokens, including digit-free (GaN), mixed-site (Sr(Ti,Sn)O3) and variable
# (SrTi1-xSnxO3) notation.
_FORMULA_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Za-z0-9.()\[\],+\-\u2212\u03b4]*")
_ELEMENT_RE = re.compile(r"[A-Z][a-z]?")
_DOPED_RE = r"(?<![A-Za-z]){el}(?:[\s-]*(?:doped|doping|substituted|substitution|incorporated|alloyed)\b)"


def _elements_in(token: str) -> set[str]:
    """Element symbols in a formula token, ignoring variables such as ``x`` and ``δ``.

    Parameters
    ----------
    token : str
        Formula-like token.

    Returns
    -------
    set of str
        Symbols that are real elements; empty if any capitalised part is not an element.
    """
    found = set()
    for symbol in _ELEMENT_RE.findall(token):
        try:
            Element(symbol)
        except ValueError:
            return set()
        found.add(symbol)
    return found


def _host_and_dopants(composition: str) -> tuple[str, set[str]]:
    """Split a substituted composition into its parent compound and its dopant elements.

    A dopant is an element with a fractional amount below 0.5 per formula unit; the host keeps
    the other elements with their amounts rounded to integers.

    Parameters
    ----------
    composition : str
        Reduced formula, e.g. ``"SrTi0.875Sn0.125O3"``.

    Returns
    -------
    host : str
        Parent reduced formula, e.g. ``"SrTiO3"``; the composition itself if nothing is fractional.
    dopants : set of str
        Dopant symbols, e.g. ``{"Sn"}``.
    """
    comp = Composition(composition)
    amounts = comp.get_el_amt_dict()
    dopants = {el for el, n in amounts.items() if n < 0.5 and abs(n - round(n)) > 1e-6}
    if not dopants:
        return comp.reduced_formula, set()
    host = {el: max(1, round(n)) for el, n in amounts.items() if el not in dopants}
    return Composition(host).reduced_formula, dopants


def _formula_tokens(text: str) -> list[str]:
    """Formula-like tokens in text, with PubMed's bracketed subscripts unwrapped.

    Parameters
    ----------
    text : str
        Title and abstract.

    Returns
    -------
    list of str
        Candidate tokens, trailing punctuation removed; hyphenated tokens (``SrTiO3-based``)
        also contribute their parts.
    """
    text = re.sub(r"\((\d+(?:\.\d+)?)\)", r"\1", text)
    tokens = []
    for tok in _FORMULA_TOKEN_RE.findall(text):
        tok = tok.rstrip(".,;:-")
        tokens.append(tok)
        parts = [part for part in re.split(r"[-\u2212]", tok) if part]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


def mentions_composition(paper: Paper, composition: str) -> bool:
    """Whether a paper reports ``composition`` or, for a substituted one, the same substitution.

    A paper counts if its title or abstract has

    - a formula that normalises to ``composition``; or
    - for a substituted composition, a mixed-site or variable formula with exactly the same
      elements (``Sr(Ti,Sn)O3``, ``SrTi1-xSnxO3``), or the parent formula together with every
      dopant named as doped or substituted (``Sn-doped SrTiO3``).

    Parameters
    ----------
    paper : Paper
        Paper to scan.
    composition : str
        Reduced formula, already normalised.

    Returns
    -------
    bool
        True if the paper reports it.
    """
    text = f"{paper.title} {paper.abstract}"
    tokens = _formula_tokens(text)
    if any(normalise_composition(tok) == composition for tok in tokens):
        return True
    host, dopants = _host_and_dopants(composition)
    if not dopants:
        return False
    elements = {str(el) for el in Composition(composition).elements}
    for tok in tokens:
        mixed = any(ch in tok for ch in ",x\u03b4") or bool(re.search(r"\d\.\d", tok))
        if mixed and _elements_in(tok) == elements:
            return True
    if not any(normalise_composition(tok) == host for tok in tokens):
        return False
    return all(re.search(_DOPED_RE.format(el=el), text, re.IGNORECASE) for el in dopants)


def check_novelty(ws: Any, composition: str, property: str | None = None, limit: int = 15) -> dict[str, Any]:
    """Search for prior reports of a composition (SC-11, CG-12, the novelty gate).

    Runs a semantic and an abstract search for the formula, plus an abstract search for the
    parent and dopants of a substituted composition. A hit is a paper that names the composition,
    or the same substitution in mixed-site, variable or "X-doped" notation (see
    :func:`mentions_composition`).

    Parameters
    ----------
    ws : Workspace
        Per-run workspace (unused; the harness passes it to every novelty function).
    composition : str
        Candidate formula.
    property : str or None
        Target property (unused; reports of any property count as prior reports).
    limit : int
        Papers per search.

    Returns
    -------
    dict
        ``{composition, hits, paper_ids}``; ``hits`` is 0 when nothing reports it.

    Raises
    ------
    ToolError
        If the formula cannot be parsed or Valency is unavailable.
    """
    target = normalise_composition(composition)
    if target is None:
        raise ToolError(f"cannot parse composition {composition!r}")
    found = valency_semantic_search(composition, limit=limit) + valency_abstract_search(
        composition, limit=limit, found_via="semantic"
    )
    host, dopants = _host_and_dopants(target)
    if dopants:
        query = f"{host} {' '.join(sorted(dopants))}"
        found += valency_abstract_search(query, limit=limit, found_via="semantic")
    ids = [p.paper_id for p in merge_duplicates(found) if mentions_composition(p, target)]
    return {"composition": target, "hits": len(ids), "paper_ids": ids}


# ---------------------------------------------------------------------------------------------
# 3. Tool handlers (docs/TOOLS.md)
# ---------------------------------------------------------------------------------------------


def _remember(ws: Any, papers: Iterable[Paper]) -> list[Paper]:
    """Add papers to ``ws.papers``, merging duplicates into records already held (LM-9, LM-11).

    Parameters
    ----------
    ws : Workspace
        Per-run workspace.
    papers : iterable of Paper
        Newly retrieved papers.

    Returns
    -------
    list of Paper
        The stored records for these papers, in order, without repeats.
    """
    with _WS_LOCK:
        index: dict[str, str] = {}
        for pid, held in ws.papers.items():
            for key in _dedupe_keys(held):
                index.setdefault(key, pid)
        ids: list[str] = []
        for paper in papers:
            pid = _add_paper(ws.papers, index, paper)
            if pid not in ids:
                ids.append(pid)
        return [ws.papers[pid] for pid in ids]


def _log_query(ws: Any, entry: str) -> None:
    """Record a query in ``ws.queries`` (LM-3).

    Parameters
    ----------
    ws : Workspace
        Per-run workspace.
    entry : str
        ``"<tool>:<detail>"``, e.g. ``"search_literature:abstract:SrTiO3 secondary phase"``.
    """
    with _WS_LOCK:
        ws.queries.append(entry)


def _brief(paper: Paper, abstract_chars: int) -> dict[str, Any]:
    """Compact view of a paper for the LLM.

    Parameters
    ----------
    paper : Paper
        Paper record.
    abstract_chars : int
        Maximum abstract length shown.

    Returns
    -------
    dict
        ``paper_id``, ``title``, ``year``, ``citations``, ``source`` and ``abstract``.
    """
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "year": paper.year,
        "citations": paper.citation_count,
        "source": paper.source,
        "abstract": paper.abstract[:abstract_chars],
    }


def _cap(limit: Any, default: int, maximum: int) -> int:
    """Clamp a requested result count.

    Parameters
    ----------
    limit : Any
        Requested count, possibly ``None``.
    default : int
        Used when ``limit`` is missing.
    maximum : int
        Upper bound.

    Returns
    -------
    int
        Count from 1 to ``maximum``.
    """
    return max(1, min(int(limit) if limit is not None else default, maximum))


@register("search_literature")
def search_literature(
    ws: Any, query: str, mode: str = "semantic", limit: int | None = None
) -> dict[str, Any]:
    """Search Valency by concept, abstract keywords or citations (LM-3 to LM-5).

    Parameters
    ----------
    ws : Workspace
        Per-run workspace; papers are added to ``ws.papers`` and the query to ``ws.queries``
        as ``"search_literature:<mode>:<query>"``.
    query : str
        Search text.
    mode : {"semantic", "abstract", "citations"}
        ``abstract`` finds failed attempts (tagged ``negative``); ``citations`` finds
        foundational work (tagged ``cited``).
    limit : int or None
        Maximum papers, capped at 15.

    Returns
    -------
    dict
        ``{n, papers: [{paper_id, title, year, citations, source, abstract (≤ 700 chars)}]}``.

    Raises
    ------
    ToolError
        For an unknown mode, an empty query or a Valency failure.
    """
    if not query or not query.strip():
        raise ToolError("query is empty")
    n = _cap(limit, 10, 15)
    if mode == "semantic":
        found = valency_semantic_search(query, n)
    elif mode == "citations":
        found = valency_semantic_search(query, n, sort_by="citations")
    elif mode == "abstract":
        found = valency_abstract_search(query, n)
    else:
        raise ToolError(f"unknown mode {mode!r}; use semantic, abstract or citations")
    _log_query(ws, f"search_literature:{mode}:{query}")
    papers = _remember(ws, found)
    return {"n": len(papers), "papers": [_brief(p, 700) for p in papers]}


@register("related_papers")
def related_papers(
    ws: Any, paper_id: str, relation: str = "similar", limit: int | None = None
) -> dict[str, Any]:
    """Papers similar to, or citing, a paper already retrieved (LM-6, LM-7).

    Parameters
    ----------
    ws : Workspace
        Per-run workspace.
    paper_id : str
        Seed paper; must be in ``ws.papers``.
    relation : {"similar", "citing"}
        Kind of relation.
    limit : int or None
        Maximum papers, capped at 10.

    Returns
    -------
    dict
        ``{n, papers: [...]}`` with abstracts of at most 400 characters.

    Raises
    ------
    ToolError
        For an unknown seed or relation, or a Valency failure.
    """
    seed = ws.papers.get(paper_id)
    if seed is None:
        raise ToolError(f"unknown paper_id {paper_id!r}; use an ID returned by search_literature")
    n = _cap(limit, 8, 10)
    if relation == "similar":
        found = valency_similar(paper_id, n, seed_relevance=seed.relevance)
    elif relation == "citing":
        found = valency_citing(paper_id, n)
    else:
        raise ToolError(f"unknown relation {relation!r}; use similar or citing")
    _log_query(ws, f"related_papers:{relation}:{paper_id}")
    papers = [p for p in _remember(ws, found) if p.paper_id != paper_id]
    return {"n": len(papers), "papers": [_brief(p, 400) for p in papers]}


@register("get_paper")
def get_paper(ws: Any, paper_id: str) -> dict[str, Any]:
    """Full record of one paper, reusing the workspace copy when it has an abstract (LM-9).

    Parameters
    ----------
    ws : Workspace
        Per-run workspace.
    paper_id : str
        Valency paper ID.

    Returns
    -------
    dict
        ``{paper_id, title, year, citations, source, abstract (≤ 3000), doi, url, authors}``.

    Raises
    ------
    ToolError
        If Valency has no such paper or is unavailable.
    """
    paper = ws.papers.get(paper_id)
    if paper is None or not paper.abstract:
        fetched = valency_get_paper(paper_id)
        if fetched is None:
            raise ToolError(f"no paper {paper_id!r} in Valency")
        if paper is not None:
            fetched = fetched.model_copy(update={"paper_id": paper_id})
        paper = _remember(ws, [fetched])[0]
    return {**_brief(paper, 3000), "doi": paper.doi, "url": paper.url, "authors": paper.authors}


@register("publication_trend")
def publication_trend(ws: Any, query: str) -> dict[str, Any]:
    """Papers per year for a keyword query (LM-8).

    Parameters
    ----------
    ws : Workspace
        Per-run workspace; the trend is stored in ``ws.trend``.
    query : str
        Abstract full-text query, e.g. ``"perovskite photocatalyst"``.

    Returns
    -------
    dict
        ``{query, papers_per_year: {year: n}}`` with missing years filled with 0.

    Raises
    ------
    ToolError
        For an empty query or a Valency failure.
    """
    if not query or not query.strip():
        raise ToolError("query is empty")
    counts = valency_trend(query)
    with _WS_LOCK:
        ws.trend.clear()
        ws.trend.update(counts)
    _log_query(ws, f"publication_trend:{query}")
    return {"query": query, "papers_per_year": counts}


@register("note_evidence")
def note_evidence(
    ws: Any,
    paper_id: str,
    property: str,
    finding_type: str,
    summary: str,
    composition: str | None = None,
    value: float | None = None,
    units: str | None = None,
    conditions: str = "",
) -> dict[str, Any]:
    """Record a finding for the reviewer and the brief (LM-13, LM-14, LM-15).

    Parameters
    ----------
    ws : Workspace
        Per-run workspace; the insight is appended to ``ws.insights``.
    paper_id : str
        Source paper; must already be in ``ws.papers``.
    property : str
        Property the finding is about.
    finding_type : {"positive", "negative", "mechanism", "synthesis", "method"}
        Kind of finding.
    summary : str
        One-line summary.
    composition : str or None
        Formula; normalised to pymatgen notation, or dropped if it is not a formula.
    value : float or None
        Reported value; dropped by code unless it appears in the paper's abstract (LM-14).
    units : str or None
        Units of ``value``.
    conditions : str
        Measurement or synthesis conditions.

    Returns
    -------
    dict
        ``{recorded, value_dropped_not_in_abstract, n_insights}``.

    Raises
    ------
    ToolError
        For an unknown paper or finding type.
    """
    paper = ws.papers.get(paper_id)
    if paper is None:
        raise ToolError(f"unknown paper_id {paper_id!r}; retrieve the paper before noting evidence")
    if finding_type not in {"positive", "negative", "mechanism", "synthesis", "method"}:
        raise ToolError(f"unknown finding_type {finding_type!r}")
    dropped = value is not None and not value_in_abstract(float(value), paper.abstract)
    quantity = None
    if value is not None and not dropped:
        quantity = Quantity(value=float(value), uncertainty=None, units=units or "", source=paper_id)
    insight = Insight(
        composition=normalise_composition(composition),
        property=property,
        value=quantity,
        conditions=conditions,
        finding_type=finding_type,  # type: ignore[arg-type]
        summary=summary,
        paper_id=paper_id,
    )
    with _WS_LOCK:
        ws.insights.append(insight)
        n = len(ws.insights)
    return {"recorded": True, "value_dropped_not_in_abstract": dropped, "n_insights": n}
