"""Historical analogue library and retrieval.

A new event is compared to documented past episodes. The posterior weight of each analogue is

    w_i  ∝  exp(sim_i / T)          [only if the episode's outcome window had already closed: no look-ahead]

with a deliberately soft temperature. What the backtest (docs/EVALUATION.md) showed:
  * sharper text/class/severity conditioning does NOT beat the unconditional historical spread at n=36;
  * what text DOES tell us reliably is whether an event is systemic or idiosyncratic. That is applied in
    `tremor.stress`: with probability P(systemic | class) the market-wide move is drawn from the crisis library,
    otherwise from the ordinary-week reference distribution (`ordinary.json`).

The spread of measured historical moves *is* the uncertainty. Nothing is typed in by hand.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from . import config as C
from .nlp import Embedder

FACTOR_KEYS = ["spx", "nifty", "d10y_bps", "d5y_bps", "hy_bps", "ig_bps", "vix_pts", "brent", "usd", "inr", "gold"]
ROLES = ("SELF", "GROUP", "PEER")


@dataclass
class Posterior:
    weights: np.ndarray          # (A,) sums to 1 over eligible analogues
    sims: np.ndarray             # (A,)
    eligible: np.ndarray         # (A,) bool
    ess: float                   # effective sample size 1 / sum(w^2)
    impact: float                # 1..10
    median_severity: float


class AnalogueLibrary:
    def __init__(self, embedder: Embedder):
        raw = json.loads((C.DATA / "analogues" / "library.json").read_text(encoding="utf-8"))
        raw.sort(key=lambda r: r["date"])
        self.items = raw
        self.A = len(raw)
        self.ids = [r["id"] for r in raw]
        self.dates = np.array([datetime.fromisoformat(r["date"]) for r in raw])
        self.emb = embedder.embed([f"{r['title']}. {r['text']}" for r in raw])
        self.cls_idx = np.array([C.EVENT_CLASSES.index(r["cls"]) for r in raw])
        self.sev = np.array([r["severity"] for r in raw], dtype=float)

        nan = np.nan
        self.fac = {k: np.array([nan if r["factors"].get(k) is None else r["factors"][k] for r in raw]) for k in FACTOR_KEYS}
        self.sec_names = list(raw[0]["sector_excess"].keys())
        self.sec = np.array([[nan if r["sector_excess"].get(s) is None else r["sector_excess"][s]
                              for s in self.sec_names] for r in raw])
        # ordinary (non-event) weeks: the reference for idiosyncratic events
        od = json.loads((C.DATA / "analogues" / "ordinary.json").read_text(encoding="utf-8"))
        self.ord = {k: np.array([np.nan if v is None else v for v in vals]) for k, vals in od.items()}
        self.n_ord = len(self.ord["spx"])
        self.ord_sec = np.column_stack([self.ord[s] for s in self.sec_names])
        # role excess returns, padded per analogue
        self.role_pad: dict[str, np.ndarray] = {}
        self.role_cnt: dict[str, np.ndarray] = {}
        for role in ROLES:
            lists = [[v["excess"] for v in r["roles"][role].values()] for r in raw]
            m = max(1, max(len(x) for x in lists))
            pad = np.full((self.A, m), np.nan)
            for i, x in enumerate(lists):
                pad[i, :len(x)] = x
            self.role_pad[role] = pad
            self.role_cnt[role] = np.array([len(x) for x in lists])

    # ---------------------------------------------------------------------------------------------
    def posterior(self, emb: np.ndarray, class_probs: np.ndarray, intensity: float, as_of: datetime,
                  *, use_text: bool = True, use_class: bool = False, use_severity: bool = False,
                  eligible_override: np.ndarray | None = None) -> Posterior | None:
        as_of_naive = as_of.replace(tzinfo=None)
        eligible = np.array([d + timedelta(days=C.HORIZON_DAYS + 3) <= as_of_naive for d in self.dates])
        if eligible_override is not None:
            eligible = eligible_override
        if not eligible.any():
            return None
        sims = self.emb @ emb
        logit = np.zeros(self.A)
        if use_text:
            logit += (sims - sims[eligible].max()) / C.SIM_TEMPERATURE
        w = np.exp(logit)
        if use_class:
            w *= 0.05 + class_probs[self.cls_idx]
        if use_severity:
            w *= np.exp(-((intensity - self.sev) ** 2) / (2 * 0.4 ** 2))
        w = np.where(eligible, w, 0.0)
        w = w / w.sum()
        ess = float(1.0 / np.sum(w ** 2))
        order = np.argsort(self.sev)
        cw = np.cumsum(w[order])
        med = float(self.sev[order][np.searchsorted(cw, 0.5)])
        return Posterior(w, sims, eligible, ess, 1.0 + 9.0 * med, med)

    @staticmethod
    def p_systemic(class_probs: np.ndarray) -> float:
        """P(event is market-wide) = sum_c P(c) * prior_c, using the pre-registered table in config."""
        return float(class_probs @ np.array([C.SYSTEMIC_PRIOR[c] for c in C.EVENT_CLASSES]))

    def describe(self, post: Posterior, top: int = 6) -> list[dict]:
        idx = np.argsort(-post.weights)[:top]
        return [{"id": self.ids[i], "title": self.items[i]["title"], "date": self.items[i]["date"],
                 "cls": self.items[i]["cls"], "weight": float(post.weights[i]), "similarity": float(post.sims[i]),
                 "severity": float(self.sev[i])} for i in idx if post.weights[i] > 0]
