"""End-to-end tests on a tiny synthetic corpus: replay exactness, signal invariants, API surface."""
import gzip
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tremor import config as C
from tremor.engine import Components, Engine
from tremor.ingest import replay_corpus

HERTZ = [
    ("news", "benzinga.com", "2020-04-16T15:00:00Z", "Hertz Seeks Coronavirus Bailout; Could be Forced to File for Bankruptcy by Summer"),
    ("news", "benzinga.com", "2020-04-16T15:00:00Z", "Avis Budget Group shares are trading lower in sympathy with Hertz bankruptcy warning"),
    ("news", "nasdaq.com", "2020-04-16T15:00:00Z", "Car rental firm Hertz may file for bankruptcy as it asks for a coronavirus bailout, report says"),
]
OIL = [
    ("news", "nasdaq.com", "2020-04-20T15:00:00Z", "Oil crashes as crude futures turn negative for the first time and Exxon and Chevron cut spending"),
    ("news", "benzinga.com", "2020-04-20T15:00:00Z", "WTI crude collapses below zero as OPEC talks fail, energy stocks plunge"),
]
NOISE = [("news", "zacks.com", "2020-04-17T15:00:00Z", "Deutsche Bank maintains hold rating and lowers price target on Hertz")]


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    rows = []
    for i, (src, pub, ts, text) in enumerate(HERTZ + OIL + NOISE):
        rows.append({"id": f"nw{i:03d}", "source": src, "publisher": pub, "ts": ts, "text": text, "url": ""})
    for i in range(6):
        rows.append({"id": f"so{i:03d}", "source": "social", "publisher": "twitter", "ts": f"2020-04-16T{10 + i}:00:00Z",
                     "text": f"$HTZ bankruptcy rumour: Hertz could file for chapter 11 soon, bailout talks {i}", "url": ""})
    with gzip.open(d / "news.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return d


@pytest.fixture(scope="module")
def engine(corpus_dir, monkeypatch_module):
    monkeypatch_module.setenv("TREMOR_CORPUS_DIR", str(corpus_dir))
    eng = Engine(comp=Components())
    eng.load_corpus(replay_corpus())
    eng.advance(len(eng.ledger))
    return eng


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_engine_emits_valid_signals(engine):
    sigs = engine.latest()
    assert sigs, "expected at least one signal from the Hertz and oil stories"
    for s in sigs:
        assert 1.0 <= s.impact <= 10.0 and -1.0 <= s.sentiment <= 1.0
        assert s.loss.p5 <= s.loss.p50 <= s.loss.p95
        assert sum(s.hist_counts) == C.MC_DRAWS and len(s.hist_edges) == len(s.hist_counts) + 1
        assert abs(sum(s.class_probs.values()) - 1.0) < 1e-3
        assert s.config_hash == engine.c.config_hash and s.book_hash == engine.c.book.hash
        assert len(s.obligor_map) == engine.c.stress.O


def test_hertz_story_reaches_the_subsidiaries_the_headline_never_named(engine):
    hz = [s for s in engine.latest() if "htz" in s.entities]
    assert hz, "Hertz should be a named entity"
    hit = {h.entity_id: h for h in hz[0].obligors_hit}
    assert "hertzcorp" in hit and hit["hertzcorp"].edge == "CONTROL"
    assert "hertzcorp" not in hz[0].entities                      # never named in the text


def test_social_only_signals_stay_provisional(engine):
    for s in engine.latest():
        if s.n_news == 0:
            assert s.status == "provisional" and s.independent_sources == 0


def test_replay_reproduces_every_signal_hash(engine):
    """The central guarantee: a signal is a pure function of (ledger prefix, book, config)."""
    checked = 0
    for ev_id, versions in engine.pipe.versions.items():
        for sig in versions:
            r = engine.verify_signal(sig)
            assert r["match"], f"{sig.signal_id}: {r}"
            assert r["ledger_chain_valid"]
            checked += 1
    assert checked >= 2


def test_replay_detects_a_changed_ledger(engine):
    sig = engine.latest()[0]
    fake = sig.model_copy(update={"hash": "0" * 16})
    assert not engine.verify_signal(fake)["match"]


def test_module_b_and_a_subscribe_without_engine_changes(engine):
    from tremor.modules import ModuleA, ModuleB

    b, a = ModuleB(), ModuleA()
    for s in engine.latest():
        b.on_signal(s)
        a.on_signal(s)
    assert abs(sum(a.weights.values()) - 1.0) < 1e-9
    assert all(a.w_min - 1e-9 <= w <= a.w_max + 1e-9 for w in a.weights.values())


def test_stress_is_deterministic_given_seed(engine):
    sig = engine.latest()[0]
    post, expo, seed, p_sys = engine.pipe.ctx[sig.signal_id]
    a = engine.c.stress.run(post, expo, seed, p_sys=p_sys).loss_total
    b = engine.c.stress.run(post, expo, seed, p_sys=p_sys).loss_total
    assert np.array_equal(a, b)


def test_idiosyncratic_events_do_not_inherit_crisis_sized_market_shocks(engine):
    """An earnings-type event (P(systemic)=0.05) must not draw Lehman-sized market moves."""
    sig = engine.latest()[0]
    post, expo, seed, _ = engine.pipe.ctx[sig.signal_id]
    low = engine.c.stress.run(post, expo, seed, p_sys=0.05).loss_total
    high = engine.c.stress.run(post, expo, seed, p_sys=0.95).loss_total
    assert np.percentile(low, 95) < np.percentile(high, 95)


# ---- API ----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client(corpus_dir, tmp_path_factory, monkeypatch_module):
    monkeypatch_module.setenv("TREMOR_CORPUS_DIR", str(corpus_dir))
    monkeypatch_module.setenv("TREMOR_LEDGER", str(tmp_path_factory.mktemp("led") / "ledger.db"))
    from tremor.api import app

    with TestClient(app) as c:
        yield c


def test_api_health_state_and_book(client):
    assert client.get("/api/health").json()["status"] == "ok"
    st = client.get("/api/state").json()
    assert st["ledger_head"] >= 12 and st["analogues"] == 36
    b = client.get("/api/book").json()
    assert b["synthetic"] is True and b["positions"] > 100


def test_api_session_signals_replay_and_reverse_stress(client):
    r = client.post("/api/session/start", json={"scenario": "hertz_apr2020"})
    assert r.status_code == 200
    adv = client.post("/api/session/advance", json={"to_ts": "2020-04-22T00:00:00+00:00"}).json()
    assert adv["new_signals"], "the Hertz window should produce signals"
    sid = adv["new_signals"][-1]["signal_id"]
    full = client.get(f"/api/signals/{sid}").json()
    assert full["hash"] and full["loss"]["p5"] <= full["loss"]["p95"]
    rep = client.post(f"/api/signals/{sid}/replay").json()
    assert rep["match"] is True and rep["ledger_chain_valid"] is True
    rev = client.get(f"/api/signals/{sid}/reverse-stress?limit_pct=2").json()
    assert rev["limit_usd_m"] > 0 and len(rev["curve"]) >= 5


def test_api_live_ingest_goes_through_the_same_pipeline(client):
    before = client.get("/api/ledger").json()["entries"]
    r = client.post("/api/ingest", json={"text": "Hertz Global Holdings to file for Chapter 11 bankruptcy protection, Reuters reports",
                                         "source": "news", "publisher": "reuters.com"}).json()
    assert r["ledger_head"] == before + 1
    ledger = client.get("/api/ledger").json()
    assert ledger["chain_valid"] is True


def test_api_rejects_bad_input(client):
    assert client.post("/api/session/start", json={"scenario": "nope"}).status_code == 404
    assert client.post("/api/ingest", json={"text": "x", "source": "email"}).status_code == 422
    assert client.get("/api/results/..%2Fsecret").status_code in (404, 422)
