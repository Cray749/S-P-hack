"""Streaming event tracker.

Turns a stream of scored documents into *Event* objects:
  * clustering on the shared embedding (same story -> same event),
  * syndication detection (near-identical text from another publisher is an echo, not new evidence),
  * independence-weighted corroboration (volume never counts, distinct publishers do),
  * the provisional -> confirmed lifecycle,
  * credibility-weighted sentiment, class posterior, entity set, novelty and confidence.

Every number is a pure function of the documents seen so far, in ledger order.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from . import config as C
from .nlp import LABELS
from .schemas import Document

NON_OTHER = len(LABELS) - 1          # LABELS[-1] == "OTHER"
SIM_WITH_ENTITY = 0.55               # cosine needed to join an event when a named entity is shared
SIM_NO_ENTITY = 0.64                 # cosine needed (plus same top class) when no entity is shared
MAX_EVENT_AGE_H = 40                 # an event stops accepting documents after this long; a continuing story starts a new event
SOCIAL_WEIGHT = 0.25                 # a tweet counts for a quarter of a news item in aggregates
OTHER_GATE = 0.5                     # documents that look routine (analyst notes, chatter) never form events


@dataclass
class DocState:
    offset: int
    doc: Document
    emb: np.ndarray
    sentiment: float
    tier: str
    probs: np.ndarray                 # over LABELS (last = OTHER)
    mentions: list[str]               # entity ids, in order of appearance

    @property
    def weight(self) -> float:
        if self.doc.source == "social":
            return SOCIAL_WEIGHT
        return C.PUBLISHER_TIER.get(self.doc.publisher, C.DEFAULT_PUBLISHER_WEIGHT)


@dataclass
class Event:
    id: str
    first_offset: int
    last_offset: int
    first_ts: datetime
    last_ts: datetime
    novelty: float
    members: list[DocState] = field(default_factory=list)
    echo_of: dict[str, str] = field(default_factory=dict)       # doc_id -> original doc_id (syndicated copy)
    _sum_emb: np.ndarray | None = None
    _w_probs: np.ndarray = field(default_factory=lambda: np.zeros(len(LABELS)))
    _w_sent: float = 0.0
    _w_sum: float = 0.0
    _ent: Counter = field(default_factory=Counter)
    _dsu: dict[str, str] = field(default_factory=dict)
    _cent: np.ndarray | None = None

    # ---- incremental updates ------------------------------------------------------------------------
    def _find(self, p: str) -> str:
        self._dsu.setdefault(p, p)
        while self._dsu[p] != p:
            self._dsu[p] = self._dsu[self._dsu[p]]
            p = self._dsu[p]
        return p

    def add(self, ds: DocState, echo_orig: DocState | None = None) -> None:
        self.members.append(ds)
        self._cent = None
        self.last_offset, self.last_ts = ds.offset, ds.doc.ts
        self._sum_emb = ds.emb.copy() if self._sum_emb is None else self._sum_emb + ds.emb
        if echo_orig is not None:
            self.echo_of[ds.doc.id] = echo_orig.doc.id
            if ds.doc.source == "news" and echo_orig.doc.source == "news":
                a, b = self._find(ds.doc.publisher), self._find(echo_orig.doc.publisher)
                if a != b:
                    self._dsu[a] = b                      # same wire copy -> one independence group
            return                                         # an echo adds no new evidence weight
        w = ds.weight
        self._w_probs += w * ds.probs
        self._w_sent += w * ds.sentiment
        self._w_sum += w
        for e in set(ds.mentions):
            self._ent[e] += w
        if ds.doc.source == "news":
            self._find(ds.doc.publisher)

    # ---- derived state ------------------------------------------------------------------------------
    @property
    def centroid(self) -> np.ndarray:
        assert self._sum_emb is not None
        if self._cent is None:
            self._cent = self._sum_emb / max(float(np.linalg.norm(self._sum_emb)), 1e-9)
        return self._cent

    @property
    def n_docs(self) -> int:
        return len(self.members)

    @property
    def n_news(self) -> int:
        return sum(1 for m in self.members if m.doc.source == "news")

    @property
    def n_social(self) -> int:
        return sum(1 for m in self.members if m.doc.source == "social")

    @property
    def independent_sources(self) -> int:
        news_pubs = {m.doc.publisher for m in self.members if m.doc.source == "news"}
        return len({self._find(p) for p in news_pubs})

    @property
    def corroboration_news(self) -> float:
        best: dict[str, float] = {}
        for m in self.members:
            if m.doc.source == "news":
                g = self._find(m.doc.publisher)
                best[g] = max(best.get(g, 0.0), m.weight)
        return float(sum(best.values()))

    @property
    def corroboration_social(self) -> float:
        n = sum(1 for m in self.members if m.doc.source == "social" and m.doc.id not in self.echo_of)
        return min(C.SOCIAL_CAP, 0.1 * math.log2(1 + n))

    @property
    def corroboration(self) -> float:
        return self.corroboration_news + self.corroboration_social

    @property
    def status(self) -> str:
        return "confirmed" if (self.independent_sources >= 2 and self.corroboration_news >= C.CONFIRM_WEIGHT) else "provisional"

    @property
    def class_probs(self) -> np.ndarray:
        """Posterior over the 10 event classes (OTHER removed, renormalised)."""
        p = self._w_probs[:NON_OTHER]
        s = p.sum()
        return p / s if s > 0 else np.full(NON_OTHER, 1.0 / NON_OTHER)

    @property
    def top_class(self) -> str:
        return C.EVENT_CLASSES[int(np.argmax(self.class_probs))]

    @property
    def class_margin(self) -> float:
        p = np.sort(self.class_probs)[::-1]
        return float(p[0] - p[1])

    @property
    def sentiment(self) -> float:
        return float(np.clip(self._w_sent / self._w_sum, -1, 1)) if self._w_sum else 0.0

    def named_entities(self) -> list[str]:
        if not self._ent:
            return []
        # an entity is *named* by the event only if it carries a real share of the evidence mass
        floor = 0.25 * self._w_sum
        keep = [e for e, c in self._ent.most_common() if c >= floor]
        return keep[:6]

    @property
    def confidence(self) -> float:
        z = (-2.0 + 1.1 * self.corroboration_news + 3.0 * self.corroboration_social
             + 2.5 * self.class_margin + (0.8 if self._ent else 0.0))
        return float(1.0 / (1.0 + math.exp(-z)))

    @property
    def intensity(self) -> float:
        """Observable-evidence intensity in [0,1] used only as a soft prior on analogue severity."""
        z = 0.30 * math.log2(1 + sum(1 for m in self.members if m.doc.id not in self.echo_of)) \
            + 0.45 * self.corroboration + 0.9 * abs(self.sentiment)
        return float(1.0 - math.exp(-z / 1.6))

    def headline(self) -> str:
        news = [m for m in self.members if m.doc.source == "news" and m.doc.id not in self.echo_of]
        pool = news or self.members
        best = max(pool, key=lambda m: (m.weight, -m.offset))
        return best.doc.text


class EventTracker:
    """Online clusterer. Feed it DocStates in ledger order."""

    def __init__(self) -> None:
        self.events: dict[str, Event] = {}
        self._active: list[str] = []
        self._hist_ids: list[str] = []
        self._hist_cent: np.ndarray | None = None
        self.ignored = 0
        self._n = 0

    # ---------------------------------------------------------------------------------------------
    def _expire(self, now: datetime) -> None:
        cutoff = now - timedelta(hours=C.EVENT_WINDOW_HOURS)
        keep = []
        for eid in self._active:
            ev = self.events[eid]
            if ev.last_ts < cutoff:
                self._hist_ids.append(eid)
                c = ev.centroid[None, :]
                self._hist_cent = c if self._hist_cent is None else np.vstack([self._hist_cent, c])
            else:
                keep.append(eid)
        self._active = keep

    def _novelty(self, emb: np.ndarray, now: datetime) -> float:
        if self._hist_cent is None:
            return 1.0
        sims = self._hist_cent @ emb
        recent = np.array([(now - self.events[i].last_ts).days <= 14 for i in self._hist_ids])
        if not recent.any():
            return 1.0
        m = float(sims[recent].max())
        return float(1.0 - np.clip((m - 0.5) / 0.4, 0.0, 1.0))

    # ---------------------------------------------------------------------------------------------
    def ingest(self, ds: DocState) -> tuple[Event | None, bool]:
        """Returns (event touched or None, created)."""
        if ds.probs[-1] >= OTHER_GATE:
            self.ignored += 1
            return None, False
        self._expire(ds.doc.ts)
        best, best_sim = None, -1.0
        top = int(np.argmax(ds.probs[:NON_OTHER]))
        ments = set(ds.mentions)
        for eid in self._active:
            ev = self.events[eid]
            if (ds.doc.ts - ev.first_ts).total_seconds() > MAX_EVENT_AGE_H * 3600:
                continue
            sim = float(ev.centroid @ ds.emb)
            shared = bool(ments & set(ev.named_entities()))
            need = SIM_WITH_ENTITY if shared else SIM_NO_ENTITY
            if not shared and C.EVENT_CLASSES[top] != ev.top_class:
                continue
            if sim >= need and sim > best_sim:
                best, best_sim = ev, sim
        if best is None:
            self._n += 1
            ev = Event(id=f"EV{self._n:05d}", first_offset=ds.offset, last_offset=ds.offset,
                       first_ts=ds.doc.ts, last_ts=ds.doc.ts, novelty=self._novelty(ds.emb, ds.doc.ts))
            ev.add(ds)
            self.events[ev.id] = ev
            self._active.append(ev.id)
            return ev, True
        # syndication test: near-identical text to an existing member from a *different* publisher
        echo = None
        for m in best.members:
            if m.doc.publisher != ds.doc.publisher or ds.doc.source == "social":
                if float(m.emb @ ds.emb) >= C.DUPLICATE_SIM and m.doc.id not in best.echo_of:
                    echo = m
                    break
            elif float(m.emb @ ds.emb) >= C.DUPLICATE_SIM and m.doc.id not in best.echo_of:
                echo = m                                    # same-publisher repost
                break
        best.add(ds, echo_orig=echo)
        return best, False
