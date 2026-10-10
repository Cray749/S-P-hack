"""Source adapters. Every source (replay corpus, live feeds, API posts) normalises into `Document`."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from . import config as C
from .schemas import Document

_WS = re.compile(r"\s+")


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def read_jsonl_gz(path: Path) -> Iterator[Document]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            yield Document(id=r["id"], source=r["source"], publisher=r["publisher"], ts=_parse_ts(r["ts"]),
                           text=_WS.sub(" ", r["text"]).strip(), url=r.get("url", ""))


def replay_corpus(root: Path | None = None) -> list[Document]:
    """Both sources merged into one chronological stream (ties broken by id for determinism)."""
    root = root or (C.DATA / "corpus")
    docs = [d for f in ("news.jsonl.gz", "social.jsonl.gz") if (root / f).exists() for d in read_jsonl_gz(root / f)]
    docs.sort(key=lambda d: (d.ts, d.id))
    return docs


def from_post(text: str, source: str = "news", publisher: str = "api.local", ts: datetime | None = None,
              url: str = "") -> Document:
    """Adapter for documents submitted through the API (the live path)."""
    ts = ts or datetime.now(timezone.utc)
    did = "api-" + hashlib.sha256(f"{text}|{publisher}|{ts.isoformat()}".encode()).hexdigest()[:12]
    return Document(id=did, source=source, publisher=publisher, ts=ts, text=_WS.sub(" ", text).strip(), url=url)  # type: ignore[arg-type]
