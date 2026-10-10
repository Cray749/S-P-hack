"""Download daily adjusted closes for every ticker the project needs.

Output: data/_raw/prices.parquet (git-ignored cache). Run once; later steps are offline.
    python scripts/fetch_market.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "_raw"

FACTORS = [
    "^GSPC", "^NSEI", "^NSEBANK", "^VIX", "^TNX", "^FVX", "HYG", "LQD", "IEF",
    "BZ=F", "GC=F", "DX-Y.NYB", "INR=X", "GBPUSD=X", "^N225", "^HSI", "^STOXX50E",
]
SECTOR_ETFS = ["XLE", "XLF", "XLK", "XLY", "XLI", "XLV", "XLP", "XLU", "XLB", "XLC", "XLRE", "KRE"]


def universe() -> list[str]:
    ents = pd.read_csv(ROOT / "data" / "entities.csv")
    tickers = set(ents["ticker"].dropna())
    eps = yaml.safe_load((ROOT / "data" / "analogues" / "episodes.yaml").read_text(encoding="utf-8"))["episodes"]
    for e in eps:
        for k in ("self", "group", "peers"):
            tickers.update(str(t) for t in e.get(k, []))
    return sorted(tickers | set(FACTORS) | set(SECTOR_ETFS))


def fetch_batch(batch: list[str], start: str, tries: int = 4) -> pd.DataFrame:
    for attempt in range(tries):
        try:
            df = yf.download(batch, start=start, auto_adjust=True, progress=False,
                             group_by="column", threads=False)
            if df is not None and len(df):
                close = df["Close"] if "Close" in df.columns.get_level_values(0) else df
                if isinstance(close, pd.Series):
                    close = close.to_frame(batch[0])
                return close
        except Exception as exc:  # network / rate limit
            print(f"  retry {attempt + 1}: {str(exc)[:80]}", file=sys.stderr)
        time.sleep(4 * (attempt + 1))
    return pd.DataFrame()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    tickers = universe()
    cache = RAW / "prices.parquet"
    existing = pd.read_parquet(cache) if cache.exists() else pd.DataFrame()
    have = set(existing.columns[existing.notna().sum() > 50]) if len(existing) else set()
    todo = [t for t in tickers if t not in have]
    print(f"{len(tickers)} tickers, {len(todo)} to fetch")
    if not todo:
        return
    tickers = todo
    frames = [existing] if len(existing) else []
    for i in range(0, len(tickers), 20):
        batch = tickers[i:i + 20]
        print(f"batch {i // 20 + 1}: {batch[0]} .. {batch[-1]}")
        frames.append(fetch_batch(batch, "2007-06-01"))
        time.sleep(2)
    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.dropna(how="all")
    prices.to_parquet(RAW / "prices.parquet")
    have = set(prices.columns[prices.notna().sum() > 50])
    missing = [t for t in tickers if t not in have]
    print(f"saved {prices.shape}; missing/empty ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
