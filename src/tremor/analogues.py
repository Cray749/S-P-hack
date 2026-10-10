"""Historical analogue library and retrieval.

A new event is compared to documented past episodes. The posterior weight of each analogue is

    w_i  ∝  exp(sim_i / T)  *  (floor + P(class_i))  *  K_sev(intensity, severity_i)   [only if the outcome is known]

* sim_i     cosine similarity of the event text to the episode description (shared embedding)
* P(class)  the event's class posterior - a vague rumour has a flat posterior, so many analogues stay in play
* K_sev     soft prior tying observable evidence intensity to the severity percentile of the analogue
* causality only analogues whose outcome window had *already closed* before the event date are eligible

The spread of the analogues' measured market moves *is* the uncertainty. Nothing is typed in by hand.
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
                  *, use_text: bool = True, use_class: bool = True, use_severity: bool = True) -> Posterior | None:
        as_of_naive = as_of.replace(tzinfo=None)
        eligible = np.array([d + timedelta(days=C.HORIZON_DAYS + 3) <= as_of_naive for d in self.dates])
        if not eligible.any():
            return None
        sims = self.emb @ emb
        logit = np.zeros(self.A)
        if use_text:
            logit += (sims - sims[eligible].max()) / C.SIM_TEMPERATURE
        w = np.exp(logit)
        if use_class:
            w *= C.CLASS_PRIOR_FLOOR + class_probs[self.cls_idx]
        if use_severity:
            w *= np.exp(-((intensity - self.sev) ** 2) / (2 * C.SEVERITY_BANDWIDTH ** 2))
        w = np.where(eligible, w, 0.0)
        w = w / w.sum()
        ess = float(1.0 / np.sum(w ** 2))
        order = np.argsort(self.sev)
        cw = np.cumsum(w[order])
        med = float(self.sev[order][np.searchsorted(cw, 0.5)])
        return Posterior(w, sims, eligible, ess, 1.0 + 9.0 * med, med)

    def describe(self, post: Posterior, top: int = 6) -> list[dict]:
        idx = np.argsort(-post.weights)[:top]
        return [{"id": self.ids[i], "title": self.items[i]["title"], "date": self.items[i]["date"],
                 "cls": self.items[i]["cls"], "weight": float(post.weights[i]), "similarity": float(post.sims[i]),
                 "severity": float(self.sev[i])} for i in idx if post.weights[i] > 0]
