"""FastAPI service: signals for downstream consumers + everything the dashboard needs."""
from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from . import config as C
from .engine import Engine
from .modules import ModuleA, ModuleB
from .schemas import Signal

log = logging.getLogger("tremor.api")
WEB = Path(__file__).resolve().parent / "web"
RESULTS = C.DATA / "results"


class State:
    engine: Engine
    mod_a: ModuleA
    mod_b: ModuleB
    scenarios: list[dict]
    lock = threading.RLock()
    scenario: str | None = None
    cursor_ts: datetime | None = None


S = State()


def _summary(s: Signal) -> dict:
    return {
        "signal_id": s.signal_id, "event_id": s.event_id, "version": s.version, "as_of": s.as_of.isoformat(),
        "headline": s.headline, "entities": s.entities, "event_class": s.event_class, "impact": s.impact,
        "sentiment": round(s.sentiment, 3), "status": s.status, "confidence": s.confidence,
        "independent_sources": s.independent_sources, "n_docs": s.n_docs, "n_news": s.n_news, "n_social": s.n_social,
        "loss": s.loss.model_dump(), "book_value_usd_m": s.book_value_usd_m, "n_obligors_hit": s.n_obligors_hit,
        "ledger_offset": s.ledger_offset, "hash": s.hash, "ess": s.ess,
    }


def _scenario_offsets(eng: Engine, scenarios: list[dict]) -> None:
    for sc in scenarios:
        a = datetime.combine(sc["start"], datetime.min.time(), tzinfo=timezone.utc)
        b = datetime.combine(sc["end"], datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1)
        row = eng.ledger.db.execute("SELECT MIN(offset), MAX(offset), COUNT(*) FROM entries WHERE ts>=? AND ts<?",
                                    (a.isoformat(), b.isoformat())).fetchone()
        sc["first_offset"], sc["last_offset"], sc["n_docs"] = row
        sc["start_ts"], sc["end_ts"] = a.isoformat(), b.isoformat()
        sc["start"], sc["end"] = str(sc["start"]), str(sc["end"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    C.CACHE.mkdir(parents=True, exist_ok=True)
    eng = Engine(os.environ.get("TREMOR_LEDGER", C.DATA / "ledger.db"))
    added = eng.load_corpus()
    log.info("ledger ready: %d entries (%d new), chain valid=%s", len(eng.ledger), added, eng.ledger.verify())
    S.engine = eng
    S.mod_a, S.mod_b = ModuleA(), ModuleB()
    eng.bus.subscribe(S.mod_b.on_signal)
    eng.bus.subscribe(S.mod_a.on_signal)
    S.scenarios = yaml.safe_load((C.DATA / "scenarios.yaml").read_text(encoding="utf-8"))["scenarios"]
    _scenario_offsets(eng, S.scenarios)
    yield


app = FastAPI(title="TREMOR Risk Engine", version=__version__, lifespan=lifespan)


# =================================================================================================
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "ledger_entries": len(S.engine.ledger),
            "neural": S.engine.c.scorer.embedder.neural and S.engine.c.scorer.sent.neural}


@app.get("/api/state")
def state() -> dict:
    e = S.engine
    off, h = e.ledger.head()
    book = e.c.book
    return {
        "scenario": S.scenario, "cursor_offset": e.pipe.offset, "cursor_ts": S.cursor_ts.isoformat() if S.cursor_ts else None,
        "ledger_head": off, "ledger_hash": h, "config_hash": e.c.config_hash, "book_hash": book.hash,
        "book_value_usd_m": round(book.value, 1), "n_events": len(e.pipe.tracker.events), "n_signals": sum(len(v) for v in e.pipe.versions.values()),
        "docs_processed": e.pipe.processed, "docs_ignored_routine": e.pipe.tracker.ignored,
        "cascade": e.c.scorer.cascade_stats(), "embedder": e.c.scorer.embedder.name, "kappa": e.c.stress.kappa,
        "analogues": e.c.lib.A, "mc_draws": C.MC_DRAWS,
    }


@app.get("/api/scenarios")
def scenarios() -> list[dict]:
    return S.scenarios


class StartReq(BaseModel):
    scenario: str


@app.post("/api/session/start")
def session_start(req: StartReq) -> dict:
    sc = next((s for s in S.scenarios if s["id"] == req.scenario), None)
    if not sc:
        raise HTTPException(404, "unknown scenario")
    with S.lock:
        e = S.engine
        e.reset()
        S.mod_a.reset()
        S.mod_b.reset()
        e.advance(sc["first_offset"] - 1, emit=False)       # silent warm-up from the ledger start
        S.scenario = sc["id"]
        S.cursor_ts = datetime.fromisoformat(sc["start_ts"])
    return {"scenario": sc["id"], "cursor_offset": e.pipe.offset, "first_offset": sc["first_offset"], "last_offset": sc["last_offset"],
            "start_ts": sc["start_ts"], "end_ts": sc["end_ts"]}


class AdvanceReq(BaseModel):
    to_ts: str | None = None
    docs: int | None = None


@app.post("/api/session/advance")
def session_advance(req: AdvanceReq) -> dict:
    with S.lock:
        e = S.engine
        db = e.ledger.db
        if req.to_ts:
            row = db.execute("SELECT MAX(offset) FROM entries WHERE ts<=?", (datetime.fromisoformat(req.to_ts).isoformat(),)).fetchone()
            target = row[0] or e.pipe.offset
            S.cursor_ts = datetime.fromisoformat(req.to_ts)
        else:
            target = min(e.ledger.head()[0], e.pipe.offset + (req.docs or 100))
            r = db.execute("SELECT ts FROM entries WHERE offset=?", (target,)).fetchone()
            S.cursor_ts = datetime.fromisoformat(r[0]) if r else S.cursor_ts
        new = e.advance(target) if target > e.pipe.offset else []
    return {"cursor_offset": e.pipe.offset, "cursor_ts": S.cursor_ts.isoformat() if S.cursor_ts else None,
            "new_signals": [_summary(s) for s in new], "n_events": len(e.pipe.tracker.events)}


@app.get("/api/signals")
def signals(limit: int = 60, min_impact: float = 0.0, status: str | None = None) -> list[dict]:
    out = [s for s in S.engine.latest() if s.impact >= min_impact and (status is None or s.status == status)]
    out.sort(key=lambda s: s.as_of, reverse=True)
    return [_summary(s) for s in out[:limit]]


def _find(signal_id: str) -> Signal:
    ev = signal_id.split(".v")[0]
    for s in S.engine.pipe.versions.get(ev, []):
        if s.signal_id == signal_id:
            return s
    raise HTTPException(404, "signal not found")


@app.get("/api/events/{event_id}")
def event(event_id: str) -> dict:
    vs = S.engine.pipe.versions.get(event_id)
    if not vs:
        raise HTTPException(404, "event not found")
    return {"event_id": event_id, "latest": vs[-1].model_dump(mode="json"),
            "timeline": [_summary(s) for s in vs]}


@app.get("/api/signals/{signal_id}")
def signal(signal_id: str) -> dict:
    return _find(signal_id).model_dump(mode="json")


@app.post("/api/signals/{signal_id}/replay")
def replay(signal_id: str) -> dict:
    s = _find(signal_id)
    import time
    t0 = time.time()
    with S.lock:
        r = S.engine.verify_signal(s)
    r["seconds"] = round(time.time() - t0, 2)
    return r


@app.get("/api/signals/{signal_id}/reverse-stress")
def reverse_stress(signal_id: str, limit_pct: float = 2.0) -> dict:
    s = _find(signal_id)
    ctx = S.engine.pipe.ctx.get(signal_id)
    if not ctx:
        raise HTTPException(409, "what-if context not available for this signal")
    post, exposures, seed, p_sys = ctx
    limit = limit_pct / 100.0 * s.book_value_usd_m
    with S.lock:
        out = S.engine.c.stress.reverse_stress(post, exposures, seed, limit, p_sys=p_sys)
    out["limit_pct"] = limit_pct
    return out


class IngestReq(BaseModel):
    text: str
    source: str = "news"
    publisher: str = "api.local"


@app.post("/api/ingest")
def ingest(req: IngestReq) -> dict:
    """Live path: any text posted here goes to the ledger and through the same pipeline as replayed data."""
    if req.source not in ("news", "social"):
        raise HTTPException(422, "source must be news or social")
    with S.lock:
        new = S.engine.ingest_live(req.text, req.source, req.publisher, datetime.now(timezone.utc))
    return {"ledger_head": S.engine.ledger.head()[0], "new_signals": [_summary(s) for s in new]}


@app.get("/api/ledger")
def ledger_info() -> dict:
    off, h = S.engine.ledger.head()
    return {"entries": off, "head_hash": h, "chain_valid": S.engine.ledger.verify()}


@app.get("/api/module-b/log")
def module_b() -> list[dict]:
    return [{**t.__dict__, "as_of": t.as_of.isoformat()} for t in S.mod_b.log][-50:]


@app.get("/api/module-a/weights")
def module_a() -> dict:
    a = S.mod_a
    return {"names": a.names, "weights": a.weights, "history": a.history[-400:], "turnover": a.turnover,
            "params": {"lambda": a.lam, "half_life_h": a.half_life_h, "w_min": a.w_min, "w_max": a.w_max, "band": a.band}}


@app.get("/api/obligors")
def obligors() -> list[dict]:
    st = S.engine.c.stress
    ents = S.engine.c.entities.by_id
    return [{"entity_id": e, "name": ents[e].name, "sector": st.ob_sector[j], "region": st.ob_region[j],
             "exposure_usd_m": round(float(st.expo.get(e, 0.0)), 2)} for j, e in enumerate(st.ob_ids)]


@app.get("/api/book")
def book() -> dict:
    b = S.engine.c.book
    p = b.positions.merge(b.obligors[["entity_id", "sector", "rating", "region"]], on="entity_id")
    by = lambda col: {k: round(float(v), 1) for k, v in p.groupby(col)["value_usd_m"].sum().sort_values(ascending=False).items()}  # noqa: E731
    return {"hash": b.hash, "value_usd_m": round(b.value, 1), "positions": len(p), "obligors": len(b.obligors),
            "by_type": by("type"), "by_sector": by("sector"), "by_rating": by("rating"), "by_region": by("region"),
            "synthetic": True}


@app.get("/api/results/{name}")
def results(name: str) -> JSONResponse:
    f = RESULTS / f"{name}.json"
    if not f.exists() or "/" in name or ".." in name:
        raise HTTPException(404, "result not found")
    return JSONResponse(json.loads(f.read_text(encoding="utf-8")))


@app.get("/api/entities")
def entities() -> list[dict]:
    t = S.engine.c.entities
    return [{"id": e.id, "name": e.name, "kind": e.kind, "sector": e.sector, "parent": e.parent_id, "group": e.group_id}
            for e in t.by_id.values() if not e.watch_only]


# ---- static dashboard --------------------------------------------------------------------------------
if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB / "index.html")


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("tremor.api:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
