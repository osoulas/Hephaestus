# Valency Bond tool schemas

Listed live from `https://mcp.valency.io/` on 2026-10-21 (server 0.4.0, MCP SDK 2.x). List the tools again if
Valency changes.

## Differences from the requirements table

- All six tools the Literature Miner needs exist under the names in REQUIREMENTS.md (`semantic_search_papers`, `search_by_abstract`, `find_similar_papers`, `get_citing_papers`, `get_paper_by_id`, `get_keyword_trends`).
- `get_keyword_trends` takes **`query`** (abstract full-text search), not a `term`; yearly bins need `granularity: "year"` (default is month). Counts come back under `periods`: a list of `{period, paper_count}` in the default `standard` format, a `{"YYYY": n}` dict in `compact`. Years with no papers are omitted, so the client fills them with 0 (LM-8).
- Keyword tools (`search_by_abstract`) return no comparable relevance score; the client ranks those hits below good semantic matches. `find_similar_papers` scores are similarity to the seed paper, so the client multiplies them by the seed's relevance.
- `get_citing_papers.sort_by` is `citations | year | year_asc` (no `relevance`).
- Search results are `{papers: [...], confidence, top_similarity, ...}`; each paper has `id` (versioned arXiv ID or PMC/PubMed ID), `base_id`, `title`, `abstract`, `authors` (strings), `doi`, `url`, `source`, `citation_count`, `relevance_score`, `datestamp`, `first_submitted`, `dates.first_available`. Year comes from `journal_ref` when present, else `dates.first_available` (LM-12).
- MCP SDK 2.x exposes tool schemas as `Tool.input_schema` / `output_schema` (snake_case); 1.x used `inputSchema`.

## Tools used by the Literature Miner

### `semantic_search_papers`

Wrapper: `Session.semantic_search -> valency_semantic_search`. Searches papers by semantic meaning using AI embeddings Find papers conceptually similar to your query using AI embeddings. Unlike keyword search, this understands meaning - 'neural network image recognition' finds papers about 'convolutional deep learning for visual classification'. Returns confidence level indicating match quality. Note: the optional category filter is source-specific. arxiv uses dotted codes (cs.LG, q-bio.QM); biorxiv and medrxiv use plain-English labels (biorxiv 'genetics', medrxiv 'infectious diseases'); the OSF providers (socarxiv/psyarxiv/eartharxiv) use bepress Title-Case discipline labels (e.g. 'Sociology'). Passing an arxiv category code while omitting source still restricts results to arxiv; omit the category filter (and source) for true cross-corpus reach.

| Argument | Type | Required | Default | Allowed / notes |
| --- | --- | --- | --- | --- |
| `query` | string | yes |  | Natural language search query describing the research topic or concept (at most 5,000 characters) |
| `category` | string |  |  | Optional category filter to limit results (e.g., cs.AI, cs.LG, math.GT) |
| `limit` | integer |  | 20 | Maximum number of results to return (default: 20, max: 100) |
| `start_date` | string |  |  | Optional: Start date as YYYY, YYYY-MM, or YYYY-MM-DD. Reduced-precision values expand to the start of their ca… |
| `end_date` | string |  |  | Optional: End date as YYYY, YYYY-MM, or YYYY-MM-DD. Reduced-precision values expand to the end of their calend… |
| `partial_date_policy` | string |  | "contained" | contained, overlap |
| `source` | string |  |  | Paper source (e.g., 'pubmed', 'arxiv', 'biorxiv', 'medrxiv', 'psyarxiv', 'socarxiv', 'eartharxiv'). Omit to se… |
| `max_authors` | integer |  | 20 | Maximum number of authors to return per paper (default: 20, max: 500). Author lists longer than this value are… |
| `include_abstract` | boolean |  | true | Whether to include paper abstracts in results (default: true). Set to false to omit abstracts and reduce respo… |
| `sort_by` | string |  | "relevance" | relevance, citations |
| `enrich_citations` | boolean |  | true | Whether to fetch citation counts for each result (default: true — adds a citation_count field, usually populat… |
| `phrase_mode` | boolean |  | false | Optional: when true, the BM25 reranking component (5% of the hybrid score) treats the query as a strict phrase… |

### `search_by_abstract`

Wrapper: `Session.abstract_search -> valency_abstract_search`. Finds papers by abstract content Query the Valency corpus by abstract content. Returns papers with matching text in abstracts. Optionally filter by date range, include or exclude categories — pass exclude_categories to drop unrelated domains for polysemous keywords (e.g. exclude_categories=['q-bio'] when searching 'nuclear cycle' to avoid biological-cycle hits).

| Argument | Type | Required | Default | Allowed / notes |
| --- | --- | --- | --- | --- |
| `query` | string | yes |  | Keywords to search for (at most 1,000 characters) |
| `category` | string |  |  | Optional: category filter to limit results (e.g., cs.AI, astro-ph.HE) |
| `exclude_categories` | array |  |  | Optional: list of category codes to exclude from results (e.g., ['q-bio', 'physics.bio-ph']). A bare parent co… |
| `limit` | integer |  | 10 | Maximum number of results to return (default: 10, max: 200) |
| `start_date` | string |  |  | Optional: Start date as YYYY, YYYY-MM, or YYYY-MM-DD. Reduced-precision values expand to the start of their ca… |
| `end_date` | string |  |  | Optional: End date as YYYY, YYYY-MM, or YYYY-MM-DD. Reduced-precision values expand to the end of their calend… |
| `partial_date_policy` | string |  | "contained" | contained, overlap |
| `source` | string |  |  | Paper source (e.g., 'pubmed', 'arxiv', 'biorxiv', 'medrxiv', 'psyarxiv', 'socarxiv', 'eartharxiv'). Omit to se… |
| `max_authors` | integer |  | 20 | Maximum number of authors to return per paper (default: 20, max: 500). Author lists longer than this value are… |
| `include_abstract` | boolean |  | true | Whether to include paper abstracts in results (default: true). Set to false to omit abstracts and reduce respo… |
| `sort_by` | string |  | "relevance" | relevance, citations |
| `enrich_citations` | boolean |  | true | Whether to fetch citation counts for each result (default: true — adds a citation_count field, usually populat… |
| `author` | string |  |  | Optional: restrict results to papers with at least one author whose name matches. Pairs well with the keyword … |
| `strict_mode` | string |  | "prefix" | fuzzy, exact, prefix |
| `phrase_mode` | boolean |  | false | Optional: when true, the entire query is matched as a strict contiguous phrase (default: false). Use for polys… |

### `find_similar_papers`

Wrapper: `Session.similar -> valency_similar`. Finds papers semantically similar to a given paper Given a paper ID, find other papers with similar content using vector similarity. Uses AI embeddings to find conceptually related work. Great for literature discovery and finding related research you might have missed.

| Argument | Type | Required | Default | Allowed / notes |
| --- | --- | --- | --- | --- |
| `paper_id` | string | yes |  | Paper ID (e.g., '1706.03762v7' or '1706.03762') |
| `limit` | integer |  | 10 | Maximum number of similar papers to return (default: 10, max: 50) |
| `source` | string |  |  | Source of the seed paper (e.g., 'pubmed', 'arxiv', 'biorxiv', 'medrxiv', 'psyarxiv', 'socarxiv', 'eartharxiv')… |
| `result_source` | string |  |  | Filter results to a specific source (e.g., 'biorxiv'). Overrides source for result filtering. Omit to fall bac… |
| `start_date` | string |  |  | Optional: Start date as YYYY, YYYY-MM, or YYYY-MM-DD. Reduced-precision values expand to the start of their ca… |
| `end_date` | string |  |  | Optional: End date as YYYY, YYYY-MM, or YYYY-MM-DD. Reduced-precision values expand to the end of their calend… |
| `partial_date_policy` | string |  | "contained" | contained, overlap |
| `max_authors` | integer |  | 20 | Maximum number of authors to return per paper (default: 20, max: 500). Author lists longer than this value are… |
| `include_abstract` | boolean |  | true | Whether to include paper abstracts in results (default: true). Set to false to omit abstracts and reduce respo… |
| `enrich_citations` | boolean |  | true | Whether to fetch citation counts for each result (default: true — adds a citation_count field, usually populat… |

### `get_citing_papers`

Wrapper: `Session.citing -> valency_citing`. Finds papers that cite a given paper — its forward citations, 'cited by', later/subsequent work that references it Find the later work that references or cites a paper — its forward citations ('cited by'). Given a Valency paper ID, find papers that cite it using the upstream citation graph. Returns citing papers with their titles, authors, publication year, and citation counts. Each result includes an in_corpus flag indicating whether it exists in the Valency corpus for further exploration. Supports sorting by citation count, recency, or chronological order.

| Argument | Type | Required | Default | Allowed / notes |
| --- | --- | --- | --- | --- |
| `paper_id` | string | yes |  | Valency paper ID (e.g., '1706.03762v7', '10.1101/2024.03.15.585123v1') |
| `limit` | integer |  | 20 | Maximum citing papers to return (default: 20, max: 100) |
| `sort_by` | string |  | "citations" | citations, year, year_asc |
| `source` | string |  |  | Paper source (e.g., 'pubmed', 'arxiv', 'biorxiv', 'medrxiv', 'psyarxiv', 'socarxiv', 'eartharxiv'; ordered lar… |
| `max_authors` | integer |  | 20 | Maximum number of authors to return per paper (default: 20, max: 500). Author lists longer than this value are… |

### `get_paper_by_id`

Wrapper: `Session.get_paper -> valency_get_paper`. Retrieves a single paper by its ID Direct lookup of a paper by its ID. Accepts three shapes: versioned IDs (e.g., '2301.07041v3'), base IDs (e.g., '2301.07041' → latest version), and PubMed Central identifiers (e.g., 'PMC3855844' → the pubmed paper with that PMC deposit). Much faster than search_by_title for known paper IDs.

| Argument | Type | Required | Default | Allowed / notes |
| --- | --- | --- | --- | --- |
| `paper_id` | string | yes |  | Paper ID — a versioned ID (e.g., '2301.07041v3'), a base ID (e.g., '2301.07041', returns latest version), or a… |
| `source` | string |  |  | Paper source (e.g., 'pubmed', 'arxiv', 'biorxiv', 'medrxiv', 'psyarxiv', 'socarxiv', 'eartharxiv'; ordered lar… |
| `max_authors` | integer |  | 20 | Maximum number of authors to return per paper (default: 20, max: 500). Author lists longer than this value are… |
| `include_abstract` | boolean |  | true | Whether to include paper abstracts in results (default: true). Set to false to omit abstracts and reduce respo… |
| `enrich_citations` | boolean |  | true | Whether to fetch citation counts for each result (default: true — adds a citation_count field, usually populat… |

### `get_keyword_trends`

Wrapper: `Session.trend -> valency_trend`. Returns paper counts over time for a keyword/abstract search Get publication volume trends filtered by abstract keyword search, binned by month, year, or week. Combines full-text search with time-series aggregation in a single query. Use this instead of calling count_papers repeatedly for each time period. Optionally filter by category. Supports standard and compact response formats.

| Argument | Type | Required | Default | Allowed / notes |
| --- | --- | --- | --- | --- |
| `query` | string | yes |  | Keywords to search for in paper abstracts (full-text search), at most 500 characters |
| `category` | string |  |  | Optional: category filter (e.g., 'cs.AI') |
| `granularity` | string |  | "month" | month, year, week |
| `start_date` | string |  |  | Optional: Start date as YYYY, YYYY-MM, or YYYY-MM-DD. Reduced-precision values expand to the start of their ca… |
| `end_date` | string |  |  | Optional: End date as YYYY, YYYY-MM, or YYYY-MM-DD. Reduced-precision values expand to the end of their calend… |
| `format` | string |  | "standard" | standard, compact |
| `source` | string |  |  | Paper source (e.g., 'pubmed', 'arxiv', 'biorxiv', 'medrxiv', 'psyarxiv', 'socarxiv', 'eartharxiv'; ordered lar… |

### `search_by_title`

Wrapper: `(not used by the miner)`. Finds academic papers by title keywords Query the Valency corpus by title keywords. Returns papers with matching titles. Optionally filter by date range. For known exact titles, this tool finds them quickly. For conceptual queries, use semantic_search_papers instead.

| Argument | Type | Required | Default | Allowed / notes |
| --- | --- | --- | --- | --- |
| `query` | string | yes |  | Keywords to search for (at most 1,000 characters) |
| `limit` | integer |  | 10 | Maximum number of results to return (default: 10, max: 200) |
| `start_date` | string |  |  | Optional: Start date as YYYY, YYYY-MM, or YYYY-MM-DD. Reduced-precision values expand to the start of their ca… |
| `end_date` | string |  |  | Optional: End date as YYYY, YYYY-MM, or YYYY-MM-DD. Reduced-precision values expand to the end of their calend… |
| `partial_date_policy` | string |  | "contained" | contained, overlap |
| `source` | string |  |  | Paper source (e.g., 'pubmed', 'arxiv', 'biorxiv', 'medrxiv', 'psyarxiv', 'socarxiv', 'eartharxiv'). Omit to se… |
| `max_authors` | integer |  | 20 | Maximum number of authors to return per paper (default: 20, max: 500). Author lists longer than this value are… |
| `include_abstract` | boolean |  | true | Whether to include paper abstracts in results (default: true). Set to false to omit abstracts and reduce respo… |
| `sort_by` | string |  | "relevance" | relevance, citations |
| `enrich_citations` | boolean |  | true | Whether to fetch citation counts for each result (default: true — adds a citation_count field, usually populat… |
| `semantic_fallback` | boolean |  | true | Whether to fall back to semantic search when keyword search returns insufficient results (default: true). Set … |
| `author` | string |  |  | Optional: restrict results to papers with at least one author whose name matches. Pairs well with the keyword … |
| `category` | string |  |  | Optional: category filter (e.g., 'astro-ph', 'cs.AI'). A bare parent code ('astro-ph') also matches its subcat… |
| `strict_mode` | string |  | "prefix" | fuzzy, exact, prefix |

## All tools on the server

| Tool | Required arguments | Summary |
| --- | --- | --- |
| `search_by_title` | `query` | Finds academic papers by title keywords Query the Valency corpus by title keywords. Returns papers with matching titles. |
| `search_by_abstract` | `query` | Finds papers by abstract content Query the Valency corpus by abstract content. Returns papers with matching text in abst |
| `search_by_abstract_batch` | `queries` | Runs multiple abstract FTS queries in one call with deduplicated results and per-paper provenance Run 1..10 abstract key |
| `search_by_author` | `author` | Finds papers by author name (name matching only, no identity resolution) Retrieve publications by author name within the |
| `search_by_category` | `category` | Finds papers in a research domain (recency-ordered) Query the Valency corpus by research domain (e.g., 'cs.AI', 'math.GT |
| `filter_by_date_range` | `start_date`, `end_date` | Filters papers by scholarly availability date (recency-ordered by default) Filter the Valency corpus by temporal range.  |
| `filter_by_categories` | `categories` | Filters papers across multiple domains (recency-ordered) Filter the Valency corpus across multiple research domains. Ret |
| `filter_papers_with_doi` |  | Finds papers with DOI identifiers Retrieve papers from the Valency corpus with DOI metadata. Returns only papers with va |
| `analyze_corpus_metrics` |  | Computes corpus-wide statistics Compute corpus-wide descriptive statistics for the Valency research paper collection. Re |
| `identify_research_domains` |  | Ranks research domains by volume Identify predominant research domains within the Valency corpus by publication volume.  |
| `identify_prolific_authors` |  | Ranks author-name strings by publication count, with author-identity disambiguation flags Identify most prolific contrib |
| `count_papers` |  | Counts papers matching filters without returning paper records Count how many papers match the given filters. Returns a  |
| `semantic_search_papers` | `query` | Searches papers by semantic meaning using AI embeddings Find papers conceptually similar to your query using AI embeddin |
| `find_similar_papers` | `paper_id` | Finds papers semantically similar to a given paper Given a paper ID, find other papers with similar content using vector |
| `get_citing_papers` | `paper_id` | Finds papers that cite a given paper — its forward citations, 'cited by', later/subsequent work that references it Find  |
| `export_papers_json` | `paper_ids` | Exports papers to JSON format Export a collection of papers from the Valency corpus to JSON format. Returns the JSON con |
| `export_papers_csv` | `paper_ids` | Exports papers to CSV format Export a collection of papers from the Valency corpus to CSV format. Returns the CSV conten |
| `export_papers_bibtex` | `paper_ids` | Exports papers as BibTeX references Export a collection of papers from the Valency corpus as BibTeX references. Returns  |
| `get_paper_versions` | `paper_id` | Returns all versions of a paper showing its revision history Given a paper ID (base or versioned), returns all versions  |
| `get_paper_by_id` | `paper_id` | Retrieves a single paper by its ID Direct lookup of a paper by its ID. Accepts three shapes: versioned IDs (e.g., '2301. |
| `search_by_venue` | `venue` | Searches papers by publication venue — the journal or conference a paper was published in Find papers published in a spe |
| `search_by_comments` | `query` | Searches the arxiv `comments` field for venue names, acceptance notes, and author notes Full-text search over the paper  |
| `filter_by_license` | `license` | Filters papers by license type (CC-BY, CC0, arxiv default, etc.) Filter papers by license. Every source spells the same  |
| `get_publication_trends` | `category` | Returns paper counts over time for a category, binned by month or year Get publication volume trends for a research cate |
| `get_publication_trends_batch` | `categories` | Returns paper counts over time for multiple categories in one call Get publication volume trends for multiple research c |
| `get_keyword_trends` | `query` | Returns paper counts over time for a keyword/abstract search Get publication volume trends filtered by abstract keyword  |
| `get_author_profile` |  | Returns a comprehensive summary profile for a researcher Get a researcher's profile in three blocks: who they are (perso |
| `compare_authors` | `authors` | Compares multiple researchers side by side (researcher-vs-researcher comparison) Compare several researchers side by sid |
| `batch_author_categories` | `authors` | Returns category distributions for multiple authors in one call Get the research category distributions for up to 50 aut |
| `search_cross_category` | `query`, `categories` | Searches abstracts across multiple categories simultaneously Search for papers across multiple research categories at on |
| `export_from_filter` |  | Exports papers matching a filter as in-memory content Combine a filter/search operation with export in a single call. In |
| `get_field_coverage` | `field` | Returns most common values for a metadata field Get the top N most common values for a metadata field (e.g., journal_ref |
| `list_sources` |  | Lists available paper sources, with paper counts and date ranges Find out which paper sources and databases are availabl |
| `submit_feedback` |  | Submit client feedback about Valency search quality, bugs, and/or feature requests Submit feedback about your experience |
| `get_author_identity` |  | Returns a comprehensive summary profile for a researcher Same tool as get_author_profile, under its earlier name. Get a  |
| `find_papers_by_researcher` |  | Finds papers by a specific disambiguated researcher (more precise than search_by_author) Find papers in our corpus by a  |
| `find_coauthors` |  | Finds an author's most frequent collaborators Given a researcher (name, orcid, author_id, doi or paper_id), returns thei |
| `resolve_orcid` | `orcid` | Looks up a researcher by ORCID and returns their profile Given an ORCID, return that researcher's profile in three block |
