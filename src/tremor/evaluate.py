"""Measured evaluation of the NLP engine and of the signals it emits. Writes data/results/eval.json.

    python -m tremor.evaluate

Sections (all numbers are produced here, none are typed by hand):
  sentiment     FinBERT-only vs lexicon-only vs cascade on a public labelled set (+ escalation rate = cost saving)
  event_class   macro-F1 on a public labelled topic set mapped to our taxonomy (+ routine-vs-event F1)
  entity_link   recall of tagged tickers on news headlines (FNSPID ticker tags)
  pipeline      documents -> events, syndicated echoes, routine/spam filtered, throughput
  event_study   do signals line up with realised abnormal returns? (concurrent, n in the hundreds)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

RAW = C.DATA / "_raw"
OUT = C.DATA / "results"

TOPIC_MAP = {
    "Analyst Update": "OTHER", "Fed | Central Banks": "MACROECONOMIC", "Company | Product News": "PRODUCT_LAUNCH",
    "Treasuries | Corporate Debt": "CREDIT_EVENT", "Dividend": "OTHER", "Earnings": "EARNINGS_GUIDANCE",
    "Energy | Oil": "COMMODITY_SHOCK", "Financials": "OTHER", "Currencies": "MACROECONOMIC",
    "General News | Opinion": "OTHER", "Gold | Metals | Materials": "COMMODITY_SHOCK", "IPO": "OTHER",
    "Legal | Regulation": "REGULATORY_LEGAL", "M&A | Investments": "MERGER_ACQUISITION", "Macro": "MACROECONOMIC",
    "Markets": "OTHER", "Politics": "GEOPOLITICAL", "Personnel Change": "OTHER", "Stock Commentary": "OTHER",
    "Stock Movement": "OTHER",
}


# label ids as documented on the dataset card (zeroshot/twitter-financial-news-topic README)
TOPIC_IDS = {
    0: "Analyst Update", 1: "Fed | Central Banks", 2: "Company | Product News", 3: "Treasuries | Corporate Debt",
    4: "Dividend", 5: "Earnings", 6: "Energy | Oil", 7: "Financials", 8: "Currencies", 9: "General News | Opinion",
    10: "Gold | Metals | Materials", 11: "IPO", 12: "Legal | Regulation", 13: "M&A | Investments", 14: "Macro",
    15: "Markets", 16: "Politics", 17: "Personnel Change", 18: "Stock Commentary", 19: "Stock Movement",
}


def _fetch(repo: str, name: str) -> Path:
    from huggingface_hub import hf_hub_download

    RAW.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo, name, repo_type="dataset"))


def macro_f1(y: list[str], p: list[str]) -> tuple[float, float]:
    from sklearn.metrics import accuracy_score, f1_score

    labs = sorted(set(y))
    return float(f1_score(y, p, labels=labs, average="macro", zero_division=0)), float(accuracy_score(y, p))


# -------------------------------------------------------------------------------------------------------------
def eval_sentiment(sc) -> dict:
    df = pd.read_csv(_fetch("zeroshot/twitter-financial-news-sentiment", "sent_valid.csv"))
    names = {0: "bearish", 1: "bullish", 2: "neutral"}
    y = [names[i] for i in df["label"]]
    texts = df["text"].tolist()
    from .nlp import lexicon_score

    def lab(s: float) -> str:
        return "bullish" if s > 0.15 else "bearish" if s < -0.15 else "neutral"

    lex = [lab(lexicon_score(t)[0] * (0.6 if lexicon_score(t)[1] < 2 else 0.8)) for t in texts]
    out = {"dataset": "zeroshot/twitter-financial-news-sentiment (validation, MIT)", "n": len(texts)}
    out["lexicon_only"] = dict(zip(("macro_f1", "accuracy"), macro_f1(y, lex)))
    if sc.sent.neural:
        fb = sc.sent.finbert(texts)
        out["finbert_only"] = dict(zip(("macro_f1", "accuracy"), macro_f1(y, [lab(s) for s in fb])))
        casc = sc.sent.score(texts)
        cl = [lab(r.score) for r in casc]
        esc = float(np.mean([r.tier == "finbert" for r in casc]))
        out["cascade"] = dict(zip(("macro_f1", "accuracy"), macro_f1(y, cl)))
        out["cascade"]["escalation_rate"] = esc
        out["cascade"]["finbert_calls_saved"] = 1.0 - esc
        out["cascade"]["agreement_with_finbert_only"] = float(np.mean([a == b for a, b in zip(cl, [lab(s) for s in fb])]))
    return out


def eval_event_class(sc) -> dict:
    df = pd.read_csv(_fetch("zeroshot/twitter-financial-news-topic", "topic_valid.csv"))
    id2name = TOPIC_IDS
    y = [TOPIC_MAP[id2name[i]] for i in df["label"]]
    texts = df["text"].tolist()
    emb = sc.embedder.embed(texts)
    P = sc.clf.probs(emb, texts)
    from .nlp import LABELS

    pred = [LABELS[i] for i in P.argmax(axis=1)]
    f1, acc = macro_f1(y, pred)
    ev_y = ["event" if t != "OTHER" else "routine" for t in y]
    ev_p = ["event" if t != "OTHER" else "routine" for t in pred]
    from sklearn.metrics import f1_score, precision_score, recall_score

    per = {}
    for lab in sorted(set(y)):
        per[lab] = round(float(f1_score([a == lab for a in y], [b == lab for b in pred], zero_division=0)), 3)
    return {"dataset": "zeroshot/twitter-financial-news-topic (validation, MIT), mapped to our taxonomy", "n": len(texts),
            "macro_f1": f1, "accuracy": acc, "per_class_f1": per,
            "event_vs_routine": {"f1_event": float(f1_score(ev_y, ev_p, pos_label="event", zero_division=0)),
                                 "precision_event": float(precision_score(ev_y, ev_p, pos_label="event", zero_division=0)),
                                 "recall_event": float(recall_score(ev_y, ev_p, pos_label="event", zero_division=0))},
            "caveat": "The topic taxonomy is not ours; the mapping is approximate (e.g. 'Markets' -> routine), so absolute "
                      "scores understate the classifier. Reported for reproducibility, not as a headline."}


def eval_entity_link(n: int = 4000) -> dict:
    """Recall of FNSPID ticker tags: if a headline is tagged with an in-universe ticker, do we link that entity?"""
    from .entities import get_linker, load_entities

    df = pd.read_parquet(RAW / "fnspid39.parquet")
    t = load_entities()
    tick2id = {e.ticker.split(".")[0]: e.id for e in t.by_id.values() if e.ticker and e.country != "IN" and e.kind == "listed"}
    df = df[df.Stock_symbol.isin(tick2id)].dropna(subset=["Article_title"]).sample(min(n, len(df)), random_state=7)
    lk = get_linker()
    hit = tot = 0
    extra = 0
    misses: list[str] = []
    for r in df.itertuples():
        ids = {m.entity_id for m in lk.link(r.Article_title)}
        tot += 1
        if tick2id[r.Stock_symbol] in ids:
            hit += 1
        elif len(misses) < 8:
            misses.append(f"{r.Stock_symbol}: {r.Article_title[:90]}")
        extra += len(ids - {tick2id[r.Stock_symbol]})
    return {"dataset": "FNSPID ticker tags on headlines for in-universe US tickers", "n": tot,
            "recall_of_tagged_ticker": hit / max(tot, 1), "avg_additional_links_per_headline": extra / max(tot, 1),
            "note": "Headline-only linking (article bodies are not used), so recall is a conservative floor. Precision is not "
                    "measurable from tags because untagged mentions can be correct.", "sample_misses": misses}


# -------------------------------------------------------------------------------------------------------------
def run_pipeline():
    from .engine import Engine

    eng = Engine()
    eng.load_corpus()
    t0 = time.time()
    eng.advance(len(eng.ledger))
    dt = time.time() - t0
    return eng, dt


def eval_pipeline(eng, dt: float) -> dict:
    tr = eng.pipe.tracker
    ev = list(tr.events.values())
    n_doc = eng.pipe.processed
    echoes = sum(len(e.echo_of) for e in ev)
    clustered = sum(e.n_docs for e in ev)
    signals = sum(len(v) for v in eng.pipe.versions.values())
    return {"documents": n_doc, "routine_or_spam_filtered": tr.ignored, "documents_in_events": clustered, "events": len(ev),
            "docs_per_event": clustered / max(len(ev), 1), "syndicated_echoes": echoes,
            "echo_share_of_news": echoes / max(1, sum(e.n_news for e in ev)), "signals_emitted": signals,
            "seconds": dt, "ms_per_doc_including_stress": 1000 * dt / max(n_doc, 1),
            "finbert_escalation_rate": eng.c.scorer.cascade_stats()["escalation_rate"],
            "confirmed_events": sum(1 for e in ev if e.status == "confirmed"),
            "provisional_events": sum(1 for e in ev if e.status == "provisional")}


def eval_event_study(eng) -> dict:
    """Do signals line up with what the market actually did? Concurrent (same-day) abnormal return of the named entity."""
    from scipy.stats import spearmanr

    prices = pd.read_parquet(RAW / "prices.parquet")
    betas = json.loads((C.DATA / "market" / "betas.json").read_text(encoding="utf-8"))
    ents = eng.c.entities.by_id
    spx = prices["^GSPC"].dropna()
    rows = []
    for ev in eng.pipe.tracker.events.values():
        named = ev.named_entities()
        if len(named) != 1 or ev.n_docs < 2:
            continue
        e = ents.get(named[0])
        if not e or not e.ticker or e.ticker not in prices or e.country == "IN":
            continue
        ser = prices[e.ticker].dropna()
        d = pd.Timestamp(ev.first_ts.date())
        if d not in ser.index or d not in spx.index:
            continue
        i = ser.index.get_loc(d)
        j = spx.index.get_loc(d)
        if i < 1 or j < 1 or i + 1 >= len(ser) or j + 1 >= len(spx):
            continue
        b = betas.get(e.id, 1.0)
        ar0 = float(ser.iloc[i] / ser.iloc[i - 1] - 1 - b * (spx.iloc[j] / spx.iloc[j - 1] - 1))
        ar1 = float(ser.iloc[i + 1] / ser.iloc[i] - 1 - b * (spx.iloc[j + 1] / spx.iloc[j] - 1))
        sig = eng.pipe.versions.get(ev.id, [])
        impact = sig[-1].impact if sig else np.nan
        rows.append({"event": ev.id, "entity": e.id, "sentiment": ev.sentiment, "impact": impact, "ar0": ar0, "ar1": ar1,
                     "status": ev.status, "n_docs": ev.n_docs, "intensity": ev.intensity, "cls": ev.top_class})
    df = pd.DataFrame(rows).dropna(subset=["impact"])
    if len(df) < 20:
        return {"n_events": int(len(df)), "note": "too few entity-linked events with price data"}
    strong = df[(df.sentiment.abs() > 0.25) & (df.ar0.abs() > 0.005)]
    dir_acc = float(np.mean(np.sign(strong.sentiment) == np.sign(strong.ar0))) if len(strong) else float("nan")
    # binomial 95% interval for direction accuracy
    se = float(np.sqrt(dir_acc * (1 - dir_acc) / max(len(strong), 1))) if len(strong) else float("nan")
    r_imp = spearmanr(df.impact, df.ar0.abs())
    r_int = spearmanr(df.intensity, df.ar0.abs())
    r_next = spearmanr(df.sentiment, df.ar1)
    conf, prov = df[df.status == "confirmed"], df[df.status == "provisional"]
    # first half / second half stability (chronological by event id)
    h = len(df) // 2
    stab = [float(spearmanr(df.iloc[:h].impact, df.iloc[:h].ar0.abs()).statistic), float(spearmanr(df.iloc[h:].impact, df.iloc[h:].ar0.abs()).statistic)]
    return {"n_events": int(len(df)), "entity_linked_single_name_events": True,
            "direction_accuracy_same_day": {"value": dir_acc, "n": int(len(strong)), "ci95": [dir_acc - 1.96 * se, dir_acc + 1.96 * se]},
            "impact_vs_abs_abnormal_return": {"spearman": float(r_imp.statistic), "p": float(r_imp.pvalue)},
            "intensity_vs_abs_abnormal_return": {"spearman": float(r_int.statistic), "p": float(r_int.pvalue)},
            "sentiment_vs_next_day_return": {"spearman": float(r_next.statistic), "p": float(r_next.pvalue)},
            "median_abs_ar_confirmed": float(conf.ar0.abs().median()) if len(conf) else None,
            "median_abs_ar_provisional": float(prov.ar0.abs().median()) if len(prov) else None,
            "n_confirmed": int(len(conf)), "n_provisional": int(len(prov)),
            "impact_vs_abs_ar_first_half_second_half": stab,
            "caveat": "Concurrent validation, not forecasting. News is day-resolution, social posts partly react to the move. "
                      "Abnormal return = stock return minus beta x S&P 500 return, same day."}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    from .scoring import Scorer

    sc = Scorer()
    print("sentiment ...")
    res["sentiment"] = eval_sentiment(sc)
    print("event class ...")
    res["event_class"] = eval_event_class(sc)
    print("entity linking ...")
    res["entity_link"] = eval_entity_link()
    print("pipeline over the full corpus ...")
    eng, dt = run_pipeline()
    res["pipeline"] = eval_pipeline(eng, dt)
    print("event study ...")
    res["event_study"] = eval_event_study(eng)
    (OUT / "eval.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "event_class"}, indent=1, default=float)[:3500])
    print("event_class macro_f1", res["event_class"]["macro_f1"], "acc", res["event_class"]["accuracy"], res["event_class"]["event_vs_routine"])


if __name__ == "__main__":
    main()
