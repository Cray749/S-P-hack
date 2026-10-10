"""Append-only, hash-chained event ledger.

The ledger is the single source of truth. Live ingestion and historical replay both write here, and every
downstream state (events, signals, stress results) is a pure function of a ledger prefix. UPDATE and DELETE
are rejected by database triggers, and each entry commits to its predecessor's hash.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path

from .schemas import Document, LedgerEntry

GENESIS = "0" * 64


def canonical(doc: Document) -> str:
    return json.dumps(doc.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def chain_hash(prev: str, payload: str) -> str:
    return hashlib.sha256((prev + payload).encode("utf-8")).hexdigest()


class Ledger:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS entries(
                offset INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT UNIQUE NOT NULL,
                ts TEXT NOT NULL,
                payload TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                hash TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS entries_no_update BEFORE UPDATE ON entries
              BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
            CREATE TRIGGER IF NOT EXISTS entries_no_delete BEFORE DELETE ON entries
              BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
            """
        )

    # ------------------------------------------------------------------------------------------------
    def head(self) -> tuple[int, str]:
        row = self.db.execute("SELECT offset, hash FROM entries ORDER BY offset DESC LIMIT 1").fetchone()
        return (row[0], row[1]) if row else (0, GENESIS)

    def __len__(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    def append_many(self, docs: Iterable[Document]) -> int:
        """Append documents in order; duplicates (same id) are ignored, so ingestion is idempotent."""
        _, prev = self.head()
        added = 0
        with self.db:
            for d in docs:
                if self.db.execute("SELECT 1 FROM entries WHERE doc_id=?", (d.id,)).fetchone():
                    continue
                payload = canonical(d)
                h = chain_hash(prev, payload)
                self.db.execute(
                    "INSERT INTO entries(doc_id, ts, payload, prev_hash, hash) VALUES (?,?,?,?,?)",
                    (d.id, d.ts.isoformat(), payload, prev, h),
                )
                prev = h
                added += 1
        return added

    def append(self, doc: Document) -> int:
        self.append_many([doc])
        row = self.db.execute("SELECT offset FROM entries WHERE doc_id=?", (doc.id,)).fetchone()
        return int(row[0])

    def read(self, upto: int | None = None, start: int = 1) -> Iterator[LedgerEntry]:
        q = "SELECT offset, payload, prev_hash, hash FROM entries WHERE offset>=?"
        args: list[int] = [start]
        if upto is not None:
            q += " AND offset<=?"
            args.append(upto)
        for off, payload, prev, h in self.db.execute(q + " ORDER BY offset", args):
            yield LedgerEntry(offset=off, doc=Document.model_validate_json(payload), prev_hash=prev, hash=h)

    def get(self, doc_id: str) -> tuple[int, Document] | None:
        row = self.db.execute("SELECT offset, payload FROM entries WHERE doc_id=?", (doc_id,)).fetchone()
        return (row[0], Document.model_validate_json(row[1])) if row else None

    def hash_at(self, offset: int) -> str:
        row = self.db.execute("SELECT hash FROM entries WHERE offset=?", (offset,)).fetchone()
        return row[0] if row else GENESIS

    def verify(self, upto: int | None = None) -> bool:
        """Recompute the whole hash chain; True only if no entry has been altered."""
        prev = GENESIS
        for e in self.read(upto):
            if e.prev_hash != prev or chain_hash(prev, canonical(e.doc)) != e.hash:
                return False
            prev = e.hash
        return True
