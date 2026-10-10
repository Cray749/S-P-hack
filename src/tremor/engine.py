"""The TREMOR engine.

    Components   immutable, shared: models, analogue library, book, graph
    Pipeline     replayable state: event tracker + emitted signals; a pure function of a ledger prefix
    Engine       owns the ledger and one live Pipeline; `replay()` rebuilds a fresh Pipeline to any offset

Because Pipeline state depends only on (ledger prefix, book snapshot, config hash), any past signal can be
re-derived byte for byte, which is what the dashboard's "replay" button demonstrates.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config as C
from .analogues import AnalogueLibrary
from .book import Book, build_book
from .entities import ExposureGraph, load_entities
from .events import Event, EventTracker
from .ingest import from_post, replay_corpus
from .ledger import Ledger
from .schemas import AnalogueUsed, Document, Evidence, LedgerEntry, LossRange, ObligorHit, Signal
from .scoring import Scorer
from .stress import StressModel

log = logging.getLogger("tremor.engine")
MIN_SOCIAL_EVIDENCE = 3


class SignalBus:
    """Minimal in-process pub/sub. In production this is Kafka / Redis Streams; consumers do not change."""

    def __init__(self) -> None:
        self._subs: list[Callable[[Signal], None]] = []

    def subscribe(self, fn: Callable[[Signal], None]) -> None:
        self._subs.append(fn)

    def publish(self, sig: Signal) -> None:
        for fn in self._subs:
            fn(sig)


class Components:
    def __init__(self) -> None:
        self.scorer = Scorer()
        self.lib = AnalogueLibrary(self.scorer.embedder)
        self.book: Book = build_book()
        self.stress = StressModel(self.book, self.lib)
        self.graph = ExposureGraph()
        self.entities = load_entities()
        self.config_hash = C.config_hash()
        committed = C.DATA / "book" / "book.json"
        if committed.exists():
            ref = json.loads(committed.read_text(encoding="utf-8")).get("hash")
            if ref != self.book.hash:
                log.warning("book hash %s differs from committed snapshot %s", self.book.hash, ref)


class Pipeline:
    """Event state + emitted signals for one pass over the ledger."""

    def __init__(self, comp: Components, bus: SignalBus | None = None) -> None:
        self.c = comp
        self.tracker = EventTracker()
        self.versions: dict[str, list[Signal]] = defaultdict(list)
        self._key: dict[str, tuple] = {}
        self.bus = bus
        self.offset = 0
        self.processed = 0

    # ---------------------------------------------------------------------------------------------
    def process(self, entry: LedgerEntry) -> list[Signal]:
        ds = self.c.scorer.state(entry.offset, entry.doc)
        self.offset = entry.offset
        self.processed += 1
        ev, _ = self.tracker.ingest(ds)
        if ev is None:
            return []
        if ev.n_news == 0 and ev.n_social < MIN_SOCIAL_EVIDENCE:
            return []
        key = (ev.status, ev.independent_sources, int(np.log2(max(ev.n_docs, 1))))
        if self._key.get(ev.id) == key:
            return []
        self._key[ev.id] = key
        sig = self.make_signal(ev, entry.offset)
        if sig is None:
            return []
        self.versions[ev.id].append(sig)
        if self.bus:
            self.bus.publish(sig)
        return [sig]

    # ---------------------------------------------------------------------------------------------
    def make_signal(self, ev: Event, offset: int) -> Signal | None:
        c = self.c
        cp = ev.class_probs
        post = c.lib.posterior(ev.centroid, cp, ev.intensity, ev.last_ts)
        if post is None:
            return None
        named = ev.named_entities()
        exposures = c.graph.propagate(named)
        version = len(self.versions[ev.id]) + 1
        seed = int(hashlib.sha256(f"{ev.id}|{offset}|{c.config_hash}|{c.book.hash}".encode()).hexdigest()[:8], 16)
        res = c.stress.run(post, exposures, seed)
        q = StressModel.summarise(res.loss_total)

        ents = c.entities.by_id
        hits: list[ObligorHit] = []
        for eid, x in exposures.items():
            exp = float(c.stress.expo.get(eid, 0.0))
            if exp <= 0 or eid not in res.by_obligor:
                continue
            arr = res.by_obligor[eid]
            hits.append(ObligorHit(entity_id=eid, name=ents[eid].name, role=x.role, edge=x.edge, path=x.path,  # type: ignore[arg-type]
                                   exposure_usd_m=round(exp, 3), expected_loss_usd_m=round(float(np.median(arr)), 4),
                                   loss_p95_usd_m=round(float(np.percentile(arr, 95)), 4)))
        hits.sort(key=lambda h: -h.expected_loss_usd_m)
        named_obl = [e for e in named if e in res.by_obligor or e in c.stress.oid]

        evid = []
        ranked = sorted(ev.members, key=lambda m: (-m.weight, m.offset))[:14]
        for m in sorted(ranked, key=lambda m: m.offset):
            evid.append(Evidence(doc_id=m.doc.id, source=m.doc.source, publisher=m.doc.publisher, ts=m.doc.ts,
                                 text=m.doc.text, url=m.doc.url, sentiment=round(m.sentiment, 4),
                                 tier=m.tier, echo_of=ev.echo_of.get(m.doc.id)))  # type: ignore[arg-type]

        probs = {C.EVENT_CLASSES[i]: round(float(p), 4) for i, p in enumerate(cp)}
        core = {
            "event_id": ev.id, "version": version, "offset": offset, "entities": named, "cls": ev.top_class,
            "sentiment": round(ev.sentiment, 4), "impact": round(post.impact, 2), "status": ev.status,
            "confidence": round(ev.confidence, 3), "n_docs": ev.n_docs, "src": ev.independent_sources,
            "p5": round(q["p5"], 2), "p50": round(q["p50"], 2), "p95": round(q["p95"], 2),
            "book": c.book.hash, "cfg": c.config_hash,
        }
        h = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]
        return Signal(
            signal_id=f"{ev.id}.v{version}", event_id=ev.id, version=version, as_of=ev.last_ts, ledger_offset=offset,
            config_hash=c.config_hash, book_hash=c.book.hash, entities=named, sentiment=ev.sentiment,
            event_class=ev.top_class, impact=round(post.impact, 1), class_probs=probs, status=ev.status,  # type: ignore[arg-type]
            confidence=round(ev.confidence, 3), independent_sources=ev.independent_sources,
            corroboration_weight=round(ev.corroboration, 3), n_docs=ev.n_docs, n_social=ev.n_social,
            n_news=ev.n_news, novelty=round(ev.novelty, 3), headline=ev.headline(),
            book_value_usd_m=round(res.book_value, 2), stressed_value_usd_m=round(res.book_value - q["p50"], 2),
            loss=LossRange(**{k: round(v, 3) for k, v in q.items()}), ess=round(post.ess, 2),
            analogues=[AnalogueUsed(**a) for a in c.lib.describe(post)], obligors_hit=hits[:25],
            n_obligors_hit=len(hits), n_obligors_named=len(named_obl), shock={k: round(v, 3) for k, v in res.shock.items()},
            evidence=evid, hash=h,
        )


# =================================================================================================
class Engine:
    def __init__(self, ledger_path: str | Path | None = None, comp: Components | None = None) -> None:
        self.c = comp or Components()
        self.ledger = Ledger(ledger_path or ":memory:")
        self.bus = SignalBus()
        self.pipe = Pipeline(self.c, self.bus)

    # ---- loading / replay -----------------------------------------------------------------------------
    def load_corpus(self, docs: list[Document] | None = None) -> int:
        docs = docs or replay_corpus()
        self.c.scorer.prepare(docs, log=log.info)
        return self.ledger.append_many(docs)

    def reset(self) -> None:
        self.pipe = Pipeline(self.c, self.bus)

    def advance(self, to_offset: int) -> list[Signal]:
        """Process ledger entries (pipe.offset, to_offset]."""
        out: list[Signal] = []
        for e in self.ledger.read(upto=to_offset, start=self.pipe.offset + 1):
            out.extend(self.pipe.process(e))
        return out

    def replay(self, to_offset: int) -> Pipeline:
        """Fresh pipeline over the ledger prefix. Same inputs -> identical signals (verified by hash)."""
        p = Pipeline(self.c, None)
        for e in self.ledger.read(upto=to_offset):
            p.process(e)
        return p

    def verify_signal(self, sig: Signal) -> dict:
        p = self.replay(sig.ledger_offset)
        redo = next((s for s in p.versions.get(sig.event_id, []) if s.ledger_offset == sig.ledger_offset), None)
        return {"signal_id": sig.signal_id, "original_hash": sig.hash, "replayed_hash": redo.hash if redo else None,
                "match": bool(redo and redo.hash == sig.hash), "ledger_entries_replayed": sig.ledger_offset,
                "ledger_chain_valid": self.ledger.verify(sig.ledger_offset)}

    # ---- live path ----------------------------------------------------------------------------------------
    def ingest_live(self, text: str, source: str = "news", publisher: str = "api.local",
                    ts: datetime | None = None) -> list[Signal]:
        doc = from_post(text, source, publisher, ts)
        self.ledger.append(doc)
        return self.advance(self.ledger.head()[0])

    # ---- views --------------------------------------------------------------------------------------------
    def latest(self) -> list[Signal]:
        return [v[-1] for v in self.pipe.versions.values() if v]
