"""Pre-compute and cache embeddings + sentiment for the replay corpus (run at image build time).

    python -m tremor.warm
"""
from __future__ import annotations

import time

from .ingest import replay_corpus
from .scoring import Scorer


def main() -> None:
    t0 = time.time()
    docs = replay_corpus()
    sc = Scorer()
    print(f"corpus={len(docs)} embedder={sc.embedder.name} finbert={sc.sent.neural}")
    sc.prepare(docs)
    print(sc.cascade_stats(), f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
