"""Typed contracts shared by every component (ledger, engine, API, downstream modules)."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Source = Literal["news", "social", "filing"]
Status = Literal["provisional", "confirmed"]


class Document(BaseModel):
    """One unit of unstructured input. Immutable once written to the ledger."""

    model_config = ConfigDict(frozen=True)

    id: str
    source: Source
    publisher: str                      # normalised domain, or "twitter"
    ts: datetime                        # timezone-aware UTC
    text: str
    url: str = ""
    lang: str = "en"


class LedgerEntry(BaseModel):
    offset: int
    doc: Document
    prev_hash: str
    hash: str


class Evidence(BaseModel):
    doc_id: str
    source: Source
    publisher: str
    ts: datetime
    text: str
    url: str = ""
    sentiment: float
    tier: Literal["lexicon", "finbert"]
    echo_of: str | None = None          # set when the item is a syndicated copy of an earlier one


class ObligorHit(BaseModel):
    entity_id: str
    name: str
    role: Literal["SELF", "GROUP", "PEER", "MARKET"]
    edge: str
    path: list[str]
    exposure_usd_m: float
    expected_loss_usd_m: float          # median stressed loss on this obligor
    loss_p95_usd_m: float


class LossRange(BaseModel):
    p5: float
    p50: float
    p95: float
    mean: float
    var99: float
    es99: float


class AnalogueUsed(BaseModel):
    id: str
    title: str
    date: str
    cls: str
    weight: float
    similarity: float
    severity: float


class ObligorPoint(BaseModel):
    entity_id: str
    name: str
    sector: str
    region: str
    exposure_usd_m: float
    loss_p50_usd_m: float
    role: Literal["SELF", "GROUP", "PEER", "MARKET"] | None = None


class Signal(BaseModel):
    """The engine's structured output, consumed by downstream modules (Module B) and the API."""

    signal_id: str
    event_id: str
    version: int
    as_of: datetime
    ledger_offset: int
    config_hash: str
    book_hash: str
    # --- fields required by the brief --------------------------------------------------------------
    entities: list[str]
    sentiment: float = Field(ge=-1.0, le=1.0)
    event_class: str
    impact: float = Field(ge=1.0, le=10.0)
    # --- risk-signal extensions --------------------------------------------------------------------
    class_probs: dict[str, float]
    status: Status
    confidence: float = Field(ge=0.0, le=1.0)
    independent_sources: int
    corroboration_weight: float
    n_docs: int
    n_social: int
    n_news: int
    novelty: float
    headline: str
    # --- stress result -----------------------------------------------------------------------------
    book_value_usd_m: float
    stressed_value_usd_m: float
    loss: LossRange
    ess: float                          # effective number of analogues behind the range
    analogues: list[AnalogueUsed]
    obligors_hit: list[ObligorHit]
    n_obligors_hit: int
    n_obligors_named: int
    shock: dict[str, float]             # headline factor shocks (median)
    evidence: list[Evidence]
    hist_edges: list[float] = []
    hist_counts: list[int] = []
    components: dict[str, float] = {}
    fallback_share: dict[str, float] = {}
    obligor_map: list[ObligorPoint] = []
    hash: str                           # sha256 over the canonical, rounded signal core
