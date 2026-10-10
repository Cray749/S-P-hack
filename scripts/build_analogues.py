"""Compute the analogue library from market data.

Reads  data/analogues/episodes.yaml  and  data/_raw/prices.parquet
Writes data/analogues/library.json, data/market/betas.json, data/market/calibration.json

Every shock stored in the library is *measured* here; nothing is typed by hand.
    python scripts/build_analogues.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tremor import config as C

ROOT = Path(__file__).resolve().parents[1]
H = C.HORIZON_DAYS

SECTOR_ETF = {
    "Energy": "XLE", "Financials": "XLF", "Technology": "XLK", "ConsumerDisc": "XLY",
    "Industrials": "XLI", "Healthcare": "XLV", "ConsumerStaples": "XLP", "Utilities": "XLU",
    "Materials": "XLB", "Communication": "XLC", "RealEstate": "XLRE",
}
# sectors for tickers that only appear in the analogue library (not in the book)
EXTRA_SECTOR = {
    "META": "Communication", "GOOGL": "Communication", "NFLX": "Communication", "DIS": "Communication",
    "AMZN": "ConsumerDisc", "NVDA": "Technology", "AMD": "Technology", "INTC": "Technology",
    "AVGO": "Technology", "MU": "Technology", "AAPL": "Technology", "PFE": "Healthcare",
    "BA": "Industrials", "LMT": "Industrials", "RTX": "Industrials", "CVX": "Energy", "OXY": "Energy",
    "HTZ": "ConsumerDisc", "CAR": "ConsumerDisc", "UBS": "Financials", "DB": "Financials",
    "HSBC": "Financials", "BCS": "Financials", "NMR": "Financials", "MS": "Financials",
    "GS": "Financials", "C": "Financials", "BAC": "Financials", "JPM": "Financials", "WFC": "Financials",
    "WAL": "Financials", "ZION": "Financials", "KEY": "Financials", "CFG": "Financials",
    "FITB": "Financials", "RF": "Financials", "HBAN": "Financials",
}
LOCAL_BENCH_SUFFIX = {".NS": "^NSEI", ".HK": "^HSI", ".MI": "^STOXX50E", ".VI": "^STOXX50E"}


def cum_path(s: pd.Series, day: pd.Timestamp) -> np.ndarray | None:
    """Cumulative return path c_k = P_k / P_base - 1 for k = 1..H (base = last close before `day`)."""
    s = s.dropna()
    if s.empty:
        return None
    pos = int(s.index.searchsorted(day))
    i0, end = pos - 1, pos + H - 1
    if i0 < 0 or end >= len(s) or (s.index[pos] - day).days > 4:
        return None
    base = float(s.iloc[i0])
    if base <= 0:
        return None
    return s.iloc[pos:end + 1].to_numpy(dtype=float) / base - 1.0


def lvl_path(s: pd.Series, day: pd.Timestamp) -> np.ndarray | None:
    s = s.dropna()
    if s.empty:
        return None
    pos = int(s.index.searchsorted(day))
    i0, end = pos - 1, pos + H - 1
    if i0 < 0 or end >= len(s) or (s.index[pos] - day).days > 4:
        return None
    return s.iloc[pos:end + 1].to_numpy(dtype=float) - float(s.iloc[i0])


def bench_series_name(ticker: str, sectors: dict[str, str]) -> str:
    for suf, idx in LOCAL_BENCH_SUFFIX.items():
        if ticker.endswith(suf):
            return "^NSEBANK" if (suf == ".NS" and sectors.get(ticker) == "Financials") else idx
    etf = SECTOR_ETF.get(sectors.get(ticker, ""))
    return etf if etf else "^GSPC"


def pct_rank(values: list[float | None], x: float | None) -> float | None:
    vs = [v for v in values if v is not None]
    if x is None or not vs:
        return None
    return sum(v <= x for v in vs) / len(vs)


def main() -> None:
    prices = pd.read_parquet(ROOT / "data" / "_raw" / "prices.parquet")
    ents = pd.read_csv(ROOT / "data" / "entities.csv")
    sectors = {r.ticker: r.sector for r in ents.itertuples() if isinstance(r.ticker, str)}
    sectors.update(EXTRA_SECTOR)
    eps = yaml.safe_load((ROOT / "data" / "analogues" / "episodes.yaml").read_text(encoding="utf-8"))["episodes"]

    out = []
    for e in eps:
        d = pd.Timestamp(e["date"])
        P = prices

        def cp(t: str) -> np.ndarray | None:
            return cum_path(P[t], d) if t in P else None

        def lp(t: str) -> np.ndarray | None:
            return lvl_path(P[t], d) if t in P else None

        # common measurement day k*: where the subject asset (else the home market) peaks in absolute move
        subj_t = next((str(t) for t in e.get("self", []) if cp(str(t)) is not None), None)
        home = "^NSEI" if any(str(t).endswith(".NS") for t in e.get("self", []) + e.get("peers", [])) else "^GSPC"
        subj = cp(subj_t) if subj_t else cp(home)
        k = int(np.argmax(np.abs(subj))) if subj is not None else H - 1

        def r(t: str) -> float | None:
            a_ = cp(t)
            return None if a_ is None else float(a_[k])

        def lv(t: str) -> float | None:
            a_ = lp(t)
            return None if a_ is None else float(a_[k])

        spx = r("^GSPC")
        f: dict[str, float | None] = {
            "spx": spx, "nifty": r("^NSEI"), "vix_pts": lv("^VIX"),
            "brent": r("BZ=F"), "gold": r("GC=F"), "usd": r("DX-Y.NYB"), "inr": r("INR=X"), "gbp": r("GBPUSD=X"),
        }
        for key, t in (("d10y_bps", "^TNX"), ("d5y_bps", "^FVX")):
            x = lv(t)
            f[key] = None if x is None else x * 100.0
        ief, hyg, lqd = r("IEF"), r("HYG"), r("LQD")
        f["hy_bps"] = None if None in (ief, hyg) else -(hyg - ief) / C.SPREAD_DURATION_HY * 1e4
        f["ig_bps"] = None if None in (ief, lqd) else -(lqd - ief) / C.SPREAD_DURATION_IG * 1e4

        sec_excess = {}
        for sec, etf in SECTOR_ETF.items():
            x = r(etf)
            sec_excess[sec] = None if (x is None or spx is None) else x - spx
        nb, nf = r("^NSEBANK"), r("^NSEI")
        sec_excess["IN_Financials"] = None if None in (nb, nf) else nb - nf
        kre, xlf = r("KRE"), r("XLF")
        sec_excess["RegionalBanks"] = None if None in (kre, xlf) else kre - xlf

        roles: dict[str, dict[str, dict[str, float]]] = {"SELF": {}, "GROUP": {}, "PEER": {}}
        missing = []
        for role, key in (("SELF", "self"), ("GROUP", "group"), ("PEER", "peers")):
            for t in e.get(key, []):
                t = str(t)
                x = r(t)
                bn = bench_series_name(t, sectors)
                b_ = r(bn)
                if b_ is None:  # sector ETF not yet listed at that date -> fall back to the broad market
                    b_ = r("^GSPC") if bn not in ("^NSEI", "^NSEBANK", "^HSI", "^STOXX50E") else None
                if x is None or b_ is None:
                    missing.append(t)
                    continue
                roles[role][t] = {"ret": round(x, 5), "excess": round(x - b_, 5)}

        out.append({
            "id": e["id"], "date": str(e["date"]), "cls": e["cls"], "title": e["title"], "text": e["text"],
            "k_star": k + 1, "subject": subj_t or home,
            "factors": {a_: (None if v is None else round(float(v), 5)) for a_, v in f.items()},
            "sector_excess": {a_: (None if v is None else round(float(v), 5)) for a_, v in sec_excess.items()},
            "roles": roles, "missing": missing,
        })

    # severity percentile: worst of {equity drawdown, VIX jump, HY-spread jump, |Brent move|}, or a
    # direct-hit entity loss scaled by 0.6 so one name collapsing never outranks a market-wide crash
    comps = {
        "eq": [-(o["factors"]["spx"]) if o["factors"]["spx"] is not None else None for o in out],
        "vix": [o["factors"]["vix_pts"] for o in out],
        "hy": [o["factors"]["hy_bps"] for o in out],
        "oil": [abs(o["factors"]["brent"]) if o["factors"]["brent"] is not None else None for o in out],
    }
    hit = [(-min([v["ret"] for v in o["roles"]["SELF"].values()], default=0.0)) for o in out]
    for i, o in enumerate(out):
        pm = max([pct_rank(v, v[i]) or 0.0 for v in comps.values()])
        ph = pct_rank(hit, hit[i]) if o["roles"]["SELF"] else None
        o["severity"] = round(max(pm, 0.6 * (ph or 0.0)), 4)

    (ROOT / "data" / "analogues").mkdir(exist_ok=True)
    (ROOT / "data" / "analogues" / "library.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    # ---- credit sensitivity kappa: ln(1 + dHY/S0) = -kappa * spx  (pooled, through origin) -------------
    xs = np.array([o["factors"]["spx"] for o in out if o["factors"]["spx"] is not None and o["factors"]["hy_bps"] is not None])
    ys = np.array([math.log(max(1e-3, 1 + o["factors"]["hy_bps"] / C.S0_HY_BPS)) for o in out
                   if o["factors"]["spx"] is not None and o["factors"]["hy_bps"] is not None])
    kappa = float(-(xs @ ys) / (xs @ xs)) if len(xs) > 3 and (xs @ xs) > 0 else C.KAPPA_DEFAULT
    kappa_c = float(np.clip(kappa, *C.KAPPA_BOUNDS))
    ss_res = float(((ys + kappa * xs) ** 2).sum())
    ss_tot = float((ys ** 2).sum())
    (ROOT / "data" / "market").mkdir(exist_ok=True)
    (ROOT / "data" / "market" / "calibration.json").write_text(json.dumps({
        "kappa_fitted": round(kappa, 3), "kappa_used": round(kappa_c, 3), "n": int(len(xs)),
        "r2_through_origin": round(1 - ss_res / ss_tot, 3) if ss_tot else None,
        "note": "ln(1+dHY/S0) = -kappa*spx over analogue episodes; HY spread is an ETF-based proxy.",
    }, indent=1), encoding="utf-8")

    # ---- betas (Blume-adjusted) of every listed entity to its local benchmark ----------------------------
    rets = prices.pct_change(fill_method=None)
    rets = rets.loc["2019-01-01":"2025-06-30"]
    betas = {}
    for r in ents.itertuples():
        t = r.ticker
        if not isinstance(t, str) or t not in rets:
            continue
        bm = rets["^NSEI" if t.endswith(".NS") else "^GSPC"]
        pair = pd.concat([rets[t], bm], axis=1).dropna()
        if len(pair) < 250:
            continue
        b = float(np.cov(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1] / np.var(pair.iloc[:, 1], ddof=1))
        betas[r.id] = round(0.67 * b + 0.33, 3)
    (ROOT / "data" / "market" / "betas.json").write_text(json.dumps(betas, indent=1), encoding="utf-8")

    n_roles = {k: sum(1 for o in out if o["roles"][k]) for k in ("SELF", "GROUP", "PEER")}
    print(f"episodes={len(out)} roles-with-data={n_roles} kappa_fit={kappa:.2f} used={kappa_c:.2f} betas={len(betas)}")
    for o in out:
        print(f"{o['date']} {o['id']:<22} sev={o['severity']:.2f} spx={o['factors']['spx']} "
              f"self={list(o['roles']['SELF'])} missing={o['missing']}")


if __name__ == "__main__":
    main()
