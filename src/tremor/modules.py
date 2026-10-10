"""Downstream consumers of the signal bus.

ModuleB  strategic stress-testing trigger: records every signal whose impact or tail loss crosses a threshold
         (the stress itself is already attached to the signal; this module owns the alerting policy).
ModuleA  tactical index rebalancer: tilts a mock 20-stock index by decayed, confidence-weighted sentiment.

Both subscribe to the same bus, and neither required a change to the engine, which is the point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from .schemas import Signal

TRIGGER_IMPACT = 6.0          # brief: "when a high-impact event is detected ... trigger a stress test"
TRIGGER_TAIL_PCT = 0.5        # or P95 loss above this % of book value


@dataclass
class StressTrigger:
    signal_id: str
    event_id: str
    as_of: datetime
    event_class: str
    impact: float
    status: str
    headline: str
    loss_p50: float
    loss_p95: float
    reason: str


class ModuleB:
    def __init__(self) -> None:
        self.log: list[StressTrigger] = []

    def on_signal(self, s: Signal) -> None:
        pct95 = 100.0 * s.loss.p95 / max(s.book_value_usd_m, 1.0)
        reason = None
        if s.impact >= TRIGGER_IMPACT:
            reason = f"impact {s.impact:.1f} >= {TRIGGER_IMPACT:.0f}"
        elif pct95 >= TRIGGER_TAIL_PCT:
            reason = f"P95 loss {pct95:.2f}% of book >= {TRIGGER_TAIL_PCT}%"
        if reason:
            self.log.append(StressTrigger(s.signal_id, s.event_id, s.as_of, s.event_class, s.impact, s.status,
                                          s.headline, s.loss.p50, s.loss.p95, reason))

    def reset(self) -> None:
        self.log.clear()


# ---------------------------------------------------------------------------------------------------
INDEX = ["aapl", "msft", "amzn", "googl", "meta", "nvda", "jpm", "bac", "xom", "cvx",
         "jnj", "pfe", "wmt", "unh", "ba", "cat", "t", "vz", "f", "gm"]


@dataclass
class ModuleA:
    """Weight_i  ∝  base * exp(lambda * score_i), capped, with a no-trade band to avoid churn."""

    lam: float = 1.6
    half_life_h: float = 36.0
    w_min: float = 0.02
    w_max: float = 0.10
    band: float = 0.004
    names: list[str] = field(default_factory=lambda: list(INDEX))
    _score: dict[str, float] = field(default_factory=dict)
    _last_ts: dict[str, datetime] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    turnover: float = 0.0

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        n = len(self.names)
        self._score = {k: 0.0 for k in self.names}
        self._last_ts = {}
        self.weights = {k: 1.0 / n for k in self.names}
        self.history = []
        self.turnover = 0.0

    def _decayed(self, k: str, now: datetime) -> float:
        t0 = self._last_ts.get(k)
        if not t0:
            return self._score[k]
        dh = max(0.0, (now - t0).total_seconds() / 3600.0)
        return self._score[k] * math.exp(-math.log(2) * dh / self.half_life_h)

    def on_signal(self, s: Signal) -> None:
        touched = [e for e in s.entities if e in self._score]
        if not touched:
            return
        for k in touched:
            self._score[k] = self._decayed(k, s.as_of) + s.sentiment * s.confidence * (s.impact / 10.0)
            self._last_ts[k] = s.as_of
        self._rebalance(s.as_of, f"{s.event_class} · {', '.join(touched).upper()}")

    def _rebalance(self, now: datetime, why: str) -> None:
        n = len(self.names)
        raw = {k: math.exp(self.lam * self._decayed(k, now)) / n for k in self.names}
        # iterative capped normalisation
        tgt = dict(raw)
        for _ in range(20):
            z = sum(tgt.values())
            tgt = {k: v / z for k, v in tgt.items()}
            over = {k for k, v in tgt.items() if v > self.w_max}
            under = {k for k, v in tgt.items() if v < self.w_min}
            if not over and not under:
                break
            for k in over:
                tgt[k] = self.w_max
            for k in under:
                tgt[k] = self.w_min
        z = sum(tgt.values())
        tgt = {k: v / z for k, v in tgt.items()}
        moved = 0.0
        for k in self.names:
            if abs(tgt[k] - self.weights[k]) > self.band:
                moved += abs(tgt[k] - self.weights[k])
                self.weights[k] = tgt[k]
        z = sum(self.weights.values())
        self.weights = {k: v / z for k, v in self.weights.items()}
        self.turnover += moved / 2.0
        self.history.append({"ts": now.isoformat(), "why": why, "weights": dict(self.weights),
                             "turnover": round(self.turnover, 4)})
