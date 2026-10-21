"""Shared data types: the frozen contracts between sections (SYS-5).

Pydantic v2 models that round-trip through JSON with no loss. Agreed by the whole team in hour 0;
see the *Shared data types* table in ``docs/REQUIREMENTS.md``. Changing a type needs all three
people to agree.
"""

from typing import Literal

from pydantic import BaseModel, Field

PaperSource = Literal["arxiv", "pubmed", "biorxiv", "medrxiv"]
FoundVia = Literal["semantic", "cited", "negative", "similar", "citing"]
FindingType = Literal["positive", "negative", "mechanism", "synthesis"]


class Quantity(BaseModel):
    """A number with its uncertainty, units and source (SYS-4).

    Attributes
    ----------
    value : float
        The number, in ``units``.
    uncertainty : float or None
        One-sigma uncertainty in ``units``; ``None`` means the source gives none.
    units : str
        Units of ``value`` and ``uncertainty``, e.g. ``"eV"``.
    source : str
        Model name, Valency paper ID or database ID the value came from.
    """

    value: float
    uncertainty: float | None
    units: str
    source: str


class Paper(BaseModel):
    """One literature record retrieved from Valency Bond (Section 2).

    Attributes
    ----------
    paper_id : str
        Valency's ID as given (versioned arXiv ID, PMC or PubMed ID).
    title : str
        Paper title.
    authors : list of str
        Author names in order.
    year : int or None
        Publication year from the journal reference, else first availability (LM-12).
    doi : str or None
        DOI when known.
    url : str or None
        Link to the record.
    source : {"arxiv", "pubmed", "biorxiv", "medrxiv"}
        Corpus the record came from.
    abstract : str
        Full abstract text; values recorded as evidence must appear here (LM-14).
    citation_count : int or None
        Citations when Valency reports them.
    relevance : float
        Relevance to the query, 0 to 1.
    found_via : {"semantic", "cited", "negative", "similar", "citing"}
        How the paper was found (LM-4 to LM-7).
    """

    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    source: PaperSource
    abstract: str = ""
    citation_count: int | None = None
    relevance: float = Field(ge=0.0, le=1.0)
    found_via: FoundVia


class Insight(BaseModel):
    """One structured finding recorded with ``note_evidence`` (LM-13).

    Attributes
    ----------
    composition : str or None
        Composition in pymatgen reduced notation (LM-15), or ``None`` if the finding is general.
    property : str
        Property the finding is about, e.g. ``"band_gap"``.
    value : Quantity or None
        Reported value; ``None`` when none was given or it failed the abstract check (LM-14).
    conditions : str
        Measurement or synthesis conditions.
    finding_type : {"positive", "negative", "mechanism", "synthesis"}
        Kind of finding.
    summary : str
        One-line summary.
    paper_id : str
        Valency ID of the source paper.
    """

    composition: str | None
    property: str
    value: Quantity | None
    conditions: str
    finding_type: FindingType
    summary: str
    paper_id: str


class LiteratureContext(BaseModel):
    """Literature aggregated over a run for the reviewer and the brief (Section 2).

    Attributes
    ----------
    papers : list of Paper
        Ranked, capped papers (LM-18).
    insights : list of Insight
        Recorded insights, capped per finding type (LM-18).
    property_map : dict of str to Quantity
        Best reported value per composition (LM-16).
    failed_approaches : list of str
        Summaries of negative findings (LM-5).
    mechanisms : list of str
        Summaries of mechanism findings.
    synthesis_routes : list of str
        Summaries of synthesis findings.
    open_questions : list of str
        Three to five open questions (LM-17).
    field_trend : dict of int to int
        Papers per year, with missing years filled with 0 (LM-8).
    coverage_note : str
        What the corpus does and does not cover (LM-19).
    """

    papers: list[Paper] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)
    property_map: dict[str, Quantity] = Field(default_factory=dict)
    failed_approaches: list[str] = Field(default_factory=list)
    mechanisms: list[str] = Field(default_factory=list)
    synthesis_routes: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    field_trend: dict[int, int] = Field(default_factory=dict)
    coverage_note: str = ""
