"""Synthetic wholesale banking book (loans, bonds, interest-rate swaps).

Entirely synthetic: company names are real public companies used only as labels, while amounts, ratings,
tenors and swap sides are invented by a seeded generator. See data/NOTICE.md.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .entities import load_entities

SEED = 20261011
BANK_INDUSTRIES = {"BanksGlobal", "InvestmentBanks", "BanksSuperRegional", "BanksRegional", "BanksIN",
                   "Cards", "Brokerage", "AutoFinance", "NBFC"}
SIZE_BY_RATING = {"AAA": 1.4, "AA": 1.3, "A": 1.2, "BBB": 1.0, "BB": 0.7, "B": 0.45, "CCC": 0.3}


@dataclass
class Book:
    positions: pd.DataFrame
    obligors: pd.DataFrame
    hash: str

    @property
    def value(self) -> float:
        return float(self.positions["value_usd_m"].sum())


def _canon(df: pd.DataFrame) -> str:
    return json.dumps(json.loads(df.round(6).to_json(orient="records")), sort_keys=True, separators=(",", ":"))


def build_book(seed: int = SEED) -> Book:
    rng = np.random.default_rng(seed)
    ents = load_entities()
    rows = []
    obl = []
    for e in sorted(ents.by_id.values(), key=lambda x: x.id):
        if e.kind == "group" or e.watch_only or not e.rating:
            continue
        s0 = C.RATING_SPREAD_BPS[e.rating] * float(rng.uniform(0.9, 1.1))
        obl.append({"entity_id": e.id, "name": e.name, "sector": e.sector, "industry": e.industry,
                    "region": e.region, "rating": e.rating, "spread_bps": round(s0, 1)})
        size = SIZE_BY_RATING[e.rating]
        is_bank = e.industry in BANK_INDUSTRIES
        plan: list[str] = []
        if is_bank:
            plan = ["SWAP", "SWAP", "BOND"] if e.kind == "listed" else ["SWAP", "BOND"]
        else:
            plan = ["LOAN", "BOND"] if e.kind == "listed" else ["LOAN"]
            if e.kind == "listed" and rng.random() < 0.35:
                plan.append("LOAN")
            if rng.random() < 0.30:
                plan.append("SWAP")
        for k, typ in enumerate(plan):
            pid = f"{e.id}-{typ[0]}{k + 1}"
            if typ == "LOAN":
                tenor = int(rng.integers(2, 8))
                n = float(np.exp(rng.normal(np.log(120), 0.55)) * size)
                lgd = 0.35 if rng.random() < 0.5 else C.LGD_SENIOR_UNSECURED
                rows.append(dict(pos_id=pid, entity_id=e.id, type="LOAN", notional_usd_m=n, tenor_y=tenor,
                                 lgd=lgd, duration=0.0, convexity=0.0, dv01_usd_m=0.0, value_usd_m=n))
            elif typ == "BOND":
                tenor = int(rng.integers(3, 11))
                n = float(np.exp(rng.normal(np.log(60), 0.5)) * size)
                d = 0.85 * tenor
                rows.append(dict(pos_id=pid, entity_id=e.id, type="BOND", notional_usd_m=n, tenor_y=tenor,
                                 lgd=C.LGD_SENIOR_UNSECURED, duration=d, convexity=0.8 * d * d, dv01_usd_m=0.0,
                                 value_usd_m=n))
            else:
                tenor = int(rng.integers(3, 11))
                n = float(np.exp(rng.normal(np.log(400), 0.5)) * size)
                side = 1 if rng.random() < 0.5 else -1               # +1 = bank pays fixed (gains when rates rise)
                annuity = tenor * 0.93
                dv01 = side * n * annuity * 1e-4                      # USD m per basis point
                mtm = float(side * n * rng.normal(0, 0.01))
                rows.append(dict(pos_id=pid, entity_id=e.id, type="SWAP", notional_usd_m=n, tenor_y=tenor,
                                 lgd=C.LGD_SENIOR_UNSECURED, duration=0.0, convexity=0.0, dv01_usd_m=dv01,
                                 value_usd_m=mtm))
    pos = pd.DataFrame(rows)
    pos = pos.round({"notional_usd_m": 3, "dv01_usd_m": 6, "value_usd_m": 3, "lgd": 3, "duration": 3, "convexity": 3})
    ob = pd.DataFrame(obl)
    h = hashlib.sha256((_canon(pos) + _canon(ob)).encode()).hexdigest()[:16]
    return Book(positions=pos, obligors=ob, hash=h)


def exposure_by_obligor(book: Book) -> pd.Series:
    """Gross exposure per obligor: loan EAD + bond market value + swap PFE proxy."""
    p = book.positions.copy()
    pfe = 0.006 * p["notional_usd_m"] * np.sqrt(p["tenor_y"])
    p["exposure"] = np.where(p["type"] == "SWAP", pfe.clip(lower=0) + p["value_usd_m"].clip(lower=0), p["value_usd_m"])
    return p.groupby("entity_id")["exposure"].sum()


def write_snapshot() -> Book:
    """Persist the generated book so reviewers can inspect the exact synthetic exposures."""
    b = build_book()
    out = C.DATA / "book"
    out.mkdir(parents=True, exist_ok=True)
    (out / "book.json").write_text(json.dumps({
        "hash": b.hash, "seed": SEED, "note": "SYNTHETIC. Real company names are labels only; amounts are invented.",
        "obligors": json.loads(b.obligors.to_json(orient="records")),
        "positions": json.loads(b.positions.to_json(orient="records")),
    }, indent=1), encoding="utf-8")
    return b


if __name__ == "__main__":
    bk = write_snapshot()
    print(f"book {bk.hash}: {len(bk.positions)} positions, {len(bk.obligors)} obligors, USD {bk.value:,.0f}m")
