"""Scores documents once (embedding, sentiment cascade, class posterior, entity mentions) and caches the result."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import config as C
from .entities import get_linker
from .events import DocState
from .nlp import DocCache, Embedder, EventClassifier, SentimentCascade
from .schemas import Document


class Scorer:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.embedder = Embedder()
        self.sent = SentimentCascade()
        self.clf = EventClassifier(self.embedder)
        self.linker = get_linker()
        fp = hashlib.sha256(
            f"{self.embedder.name}|{'finbert' if self.sent.neural else 'lex'}|v1".encode()
        ).hexdigest()[:10]
        self.fingerprint = fp
        self.cache = DocCache((cache_dir or C.CACHE) / f"doccache_{fp}.npz", fp)

    # --------------------------------------------------------------------------------------------
    def prepare(self, docs: list[Document], batch: int = 256, log=print) -> None:
        """Compute and cache embeddings + sentiment for any document not yet cached."""
        todo = [d for d in docs if d.id not in self.cache.emb or d.id not in self.cache.sent]
        if not todo:
            return
        log(f"scoring {len(todo)} new documents (embeddings + sentiment cascade)")
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            texts = [d.text for d in chunk]
            emb = self.embedder.embed(texts)
            sent = self.sent.score(texts)
            for d, e, s in zip(chunk, emb, sent):
                self.cache.emb[d.id] = e
                self.cache.sent[d.id] = (s.score, s.tier)
            if (i // batch) % 10 == 0:
                log(f"  {min(i + batch, len(todo))}/{len(todo)}")
        self.cache.save()

    def state(self, offset: int, doc: Document) -> DocState:
        if doc.id not in self.cache.emb:
            self.prepare([doc], log=lambda *_: None)
        emb = self.cache.emb[doc.id]
        sc, tier = self.cache.sent[doc.id]
        probs = self.clf.probs(emb[None, :], [doc.text])[0]
        ments = [m.entity_id for m in self.linker.link(doc.text)]
        return DocState(offset=offset, doc=doc, emb=emb, sentiment=sc, tier=tier, probs=probs, mentions=ments)

    def cascade_stats(self) -> dict[str, float]:
        tiers = [t for _, t in self.cache.sent.values()]
        n = max(1, len(tiers))
        return {"documents": float(len(tiers)), "escalated_to_finbert": float(sum(t == "finbert" for t in tiers)),
                "escalation_rate": sum(t == "finbert" for t in tiers) / n}
