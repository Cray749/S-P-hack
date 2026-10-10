"""Build the replay corpus (news + social) from public datasets.

Sources (see data/NOTICE.md):
  * FNSPID financial news headlines  (CC BY-NC 4.0)  - shard covering 2020-02..2021-06
  * stock-market-tweets-data         (CC BY 4.0)     - 2020-04-09 .. 2020-07-16

Output: data/corpus/news.jsonl.gz, data/corpus/social.jsonl.gz  (small, committed)
    python scripts/build_corpus.py
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tremor.entities import get_linker

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "_raw"
OUT = ROOT / "data" / "corpus"
WIN_START, WIN_END = pd.Timestamp("2020-04-09", tz="UTC"), pd.Timestamp("2020-07-17", tz="UTC")
NEWS_HOUR_UTC = 15          # news dates are day-resolution; assume publication mid US session
BACKGROUND_NEWS, BACKGROUND_SOCIAL = 2600, 3200
PER_TITLE_CAP = 6           # keep a few syndicated copies (that is the dedup test), not hundreds
MACRO = re.compile(
    r"\b(oil|crude|brent|wti|opec|fed|federal reserve|powell|treasury|yields?|recession|unemployment|jobs report|"
    r"gdp|inflation|stimulus|lockdown|coronavirus|covid|pandemic|bankruptcy|chapter 11|default|downgrade|tariffs?|"
    r"china|trade war|sanctions?|bailout|layoffs?|furlough|rate cut|selloff|sell-off|plunge|crash)\b", re.I)
URL_RE = re.compile(r"https?://\S+")

SCENARIOS = yaml.safe_load((ROOT / "data" / "scenarios.yaml").read_text(encoding="utf-8"))["scenarios"]


def load_news() -> pd.DataFrame:
    cache = RAW / "fnspid39.parquet"
    if not cache.exists():
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq
        RAW.mkdir(parents=True, exist_ok=True)
        fs = HfFileSystem()
        path = "datasets/sabareesh88/FNSPID_nasdaq_sorted/data/train-00039-of-00044.parquet"
        with fs.open(path, "rb") as f:
            pq.read_table(f, columns=["Date", "Article_title", "Stock_symbol", "Url", "Publisher"]).to_pandas().to_parquet(cache)
    df = pd.read_parquet(cache)
    df["ts"] = pd.to_datetime(df["Date"], utc=True) + pd.Timedelta(hours=NEWS_HOUR_UTC)
    df = df[(df.ts >= WIN_START) & (df.ts < WIN_END)].copy()
    df["publisher"] = df["Url"].str.extract(r"https?://(?:www\.)?([^/]+)")[0].fillna("unknown")
    df["text"] = df["Article_title"].fillna("").str.strip()
    return df[df.text.str.len() > 25]


def load_tweets() -> pd.DataFrame:
    cache = RAW / "tweets.parquet"
    if not cache.exists():
        from huggingface_hub import hf_hub_download
        RAW.mkdir(parents=True, exist_ok=True)
        p = hf_hub_download("StephanAkkerman/stock-market-tweets-data", "stock-market-tweets-data.csv", repo_type="dataset")
        pd.read_csv(p).to_parquet(cache)
    df = pd.read_parquet(cache)
    df["ts"] = pd.to_datetime(df["created_at"], utc=True)
    df["text"] = df["text"].fillna("").map(lambda t: re.sub(r"\s+", " ", URL_RE.sub("", t)).strip())
    df = df[(df.text.str.len() > 25)]
    df["key"] = df.text.str.lower().str.replace(r"[^a-z0-9$ ]", "", regex=True)
    return df.drop_duplicates("key").copy()


def relevant(texts: pd.Series) -> np.ndarray:
    linker = get_linker()
    return np.array([bool(linker.link(t)) or bool(MACRO.search(t)) for t in texts])


def pick(df: pd.DataFrame, rng: np.random.Generator, n_bg: int, scen_cap: int) -> pd.DataFrame:
    df = df[relevant(df.text)].copy()
    keep = np.zeros(len(df), dtype=bool)
    for s in SCENARIOS:
        a, b = pd.Timestamp(s["start"], tz="UTC"), pd.Timestamp(s["end"], tz="UTC") + pd.Timedelta(days=1)
        in_win = (df.ts >= a).to_numpy() & (df.ts < b).to_numpy()
        focus = df.text.str.contains(s["focus"], case=False, regex=True).to_numpy()
        keep[np.where(in_win & focus)[0]] = True            # never drop on-topic items
        idx = np.where(in_win & ~focus)[0]                   # sample the rest as realistic noise
        if len(idx) > scen_cap:
            idx = rng.choice(idx, scen_cap, replace=False)
        keep[idx] = True
    rest = np.where(~keep)[0]
    keep[rng.choice(rest, min(n_bg, len(rest)), replace=False)] = True
    return df[keep].sort_values("ts")


def write(df: pd.DataFrame, name: str, source: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT / f"{name}.jsonl.gz", "wt", encoding="utf-8") as fh:
        for i, r in enumerate(df.itertuples()):
            fh.write(json.dumps({
                "id": f"{source[:2]}{i:06d}", "source": source, "publisher": r.publisher,
                "ts": r.ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "text": r.text,
                "url": getattr(r, "Url", "") if source == "news" else "",
            }, ensure_ascii=False) + "\n")


def main() -> None:
    rng = np.random.default_rng(2020)
    news = load_news()
    # cap identical headlines (Reuters BUZZ items are syndicated dozens of times)
    news = news.assign(n=news.groupby("text").cumcount()).query("n < @PER_TITLE_CAP").drop(columns="n")
    news = pick(news, rng, BACKGROUND_NEWS, 500)
    tw = load_tweets()
    tw["publisher"] = "twitter"
    tweets = pick(tw, rng, BACKGROUND_SOCIAL, 700)
    write(news, "news", "news")
    write(tweets, "social", "social")
    print(f"news={len(news)} social={len(tweets)}")
    for s in SCENARIOS:
        a, b = pd.Timestamp(s["start"], tz="UTC"), pd.Timestamp(s["end"], tz="UTC") + pd.Timedelta(days=1)
        print(s["id"], "news", int(((news.ts >= a) & (news.ts < b)).sum()), "social", int(((tweets.ts >= a) & (tweets.ts < b)).sum()))
    for f in ("news", "social"):
        print(f, round((OUT / f"{f}.jsonl.gz").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
