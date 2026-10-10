"""Module B: strategic, event-driven stress test.

Pipeline (all vectorised, one coherent historical scenario per Monte-Carlo draw):

  analogue posterior -> draw an episode -> market / rates / sector / role-excess moves
  -> obligor equity return  r = g * ( beta * market + sector_excess + role_excess )
  -> obligor spread         s1 = s0 * exp(-kappa * r)                    (kappa calibrated on the library)
  -> position P&L           loans: change in expected credit loss (credit triangle: spread ~ PD x LGD)
                            bonds: -D*dy - SD*ds + 1/2 C dy^2
                            swaps: DV01*dy  minus change in CVA
  -> total loss distribution (P5/P50/P95, VaR99, ES99), obligor drill-down, reverse stress.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .analogues import AnalogueLibrary, Posterior
from .book import Book, exposure_by_obligor
from .entities import Exposure, load_entities

SPREAD_UP_CAP = 15.0     # a spread can at most multiply by 15x within the horizon (keeps ratings-implied PD < 100%)


@dataclass
class StressResult:
    loss_total: np.ndarray                  # (n,) loss in USD m, positive = loss
    by_obligor: dict[str, np.ndarray]       # entity_id -> (n,) loss, only for obligors with a role
    book_value: float
    shock: dict[str, float]
    fallback_share: dict[str, float]
    components: dict[str, float] = field(default_factory=dict)   # median loss split: loans/bonds/swaps/cva
    ob_median: np.ndarray | None = None                          # (O,) median loss for every obligor
    ob_p95: np.ndarray | None = None                             # (O,) 95th-percentile loss for every obligor


class StressModel:
    def __init__(self, book: Book, lib: AnalogueLibrary):
        self.book, self.lib = book, lib
        ents = load_entities()
        ob = book.obligors.reset_index(drop=True)
        self.O = len(ob)
        self.oid = {e: i for i, e in enumerate(ob["entity_id"])}
        self.ob = ob
        self.ob_ids = ob["entity_id"].tolist()
        self.ob_sector = ob["sector"].tolist()
        self.ob_region = ob["region"].tolist()
        calib = json.loads((C.DATA / "market" / "calibration.json").read_text(encoding="utf-8"))
        self.kappa = float(calib["kappa_used"])
        betas = json.loads((C.DATA / "market" / "betas.json").read_text(encoding="utf-8"))
        self.s0 = ob["spread_bps"].to_numpy(float)
        self.region_in = (ob["region"] == "IN").to_numpy()
        self.beta = np.array([betas.get(e, betas.get(ents.ultimate_parent(e), 1.0)) for e in ob["entity_id"]])
        # sector columns (up to two per obligor: sector, plus regional-bank / India-financial add-on)
        sidx = {s: i for i, s in enumerate(lib.sec_names)}
        self.sec_cols = np.full((self.O, 2), -1, dtype=int)
        for i, r in ob.iterrows():
            if r["region"] == "IN":
                if r["sector"] == "Financials":
                    self.sec_cols[i, 0] = sidx["IN_Financials"]
            else:
                self.sec_cols[i, 0] = sidx.get(r["sector"], -1)
                if r["industry"] == "BanksRegional":
                    self.sec_cols[i, 1] = sidx["RegionalBanks"]
        # positions
        p = book.positions.reset_index(drop=True)
        self.P = len(p)
        self.p = p
        self.p_obl = p["entity_id"].map(self.oid).to_numpy()
        self.M = np.zeros((self.P, self.O))
        self.M[np.arange(self.P), self.p_obl] = 1.0
        t = p["type"].to_numpy()
        self.is_loan, self.is_bond, self.is_swap = t == "LOAN", t == "BOND", t == "SWAP"
        n = p["notional_usd_m"].to_numpy(float)
        ten = p["tenor_y"].to_numpy(float)
        self.val = p["value_usd_m"].to_numpy(float)
        self.loan_amt = np.where(self.is_loan, self.val, 0.0) * np.minimum(ten, 5.0)
        self.bond_mv = np.where(self.is_bond, self.val, 0.0)
        self.bond_D = p["duration"].to_numpy(float)
        self.bond_C = p["convexity"].to_numpy(float)
        self.swap_dv01 = np.where(self.is_swap, p["dv01_usd_m"].to_numpy(float), 0.0)
        pfe = 0.006 * n * np.sqrt(ten) + np.maximum(self.val, 0)
        self.cva_w = np.where(self.is_swap, pfe * ten, 0.0)
        self.expo = exposure_by_obligor(book)

    # ---------------------------------------------------------------------------------------------
    def _role_matrix(self, exposures: dict[str, Exposure]) -> np.ndarray:
        role = np.zeros(self.O, dtype=int)
        for e, x in exposures.items():
            if e in self.oid:
                role[self.oid[e]] = {"SELF": 1, "GROUP": 2, "PEER": 3}[x.role]
        return role

    def _role_excess(self, post: Posterior, idx: np.ndarray, role: np.ndarray, rng: np.random.Generator):
        lib, n = self.lib, len(idx)
        out = np.zeros((n, self.O))
        fb: dict[str, float] = {}
        for code, name in ((1, "SELF"), (2, "GROUP"), (3, "PEER")):
            cols = np.where(role == code)[0]
            if not len(cols):
                continue
            pad, cnt = lib.role_pad[name], lib.role_cnt[name]
            c = cnt[idx]                                          # per-draw count of observed names
            j = np.minimum((rng.random((n, len(cols))) * np.maximum(c, 1)[:, None]).astype(int), pad.shape[1] - 1)
            direct = pad[idx[:, None], j]
            # pooled fallback when the drawn analogue has no data for this role
            have = cnt > 0
            wsum = post.weights[have]
            if wsum.sum() > 0:
                vals, probs = [], []
                for i in np.where(have)[0]:
                    v = pad[i, :cnt[i]]
                    vals.append(v)
                    probs.append(np.full(len(v), post.weights[i] / len(v)))
                vals_a, probs_a = np.concatenate(vals), np.concatenate(probs)
                probs_a = probs_a / probs_a.sum()
                pooled = rng.choice(vals_a, size=(n, len(cols)), p=probs_a)
            else:
                pooled = np.zeros((n, len(cols)))
            use_direct = (c > 0)[:, None]
            out[:, cols] = np.where(use_direct, direct, pooled)
            fb[name] = float(1.0 - use_direct.mean())
        return out, fb

    def run(self, post: Posterior, exposures: dict[str, Exposure], seed: int, n: int | None = None,
            scale: float = 1.0, use_graph: bool = True, p_sys: float = 1.0) -> StressResult:
        n = n or C.MC_DRAWS
        lib = self.lib
        rng = np.random.default_rng(seed)
        idx = rng.choice(lib.A, size=n, p=post.weights)
        g = np.exp(rng.normal(0.0, C.KERNEL_NOISE, n)) * scale
        nz = lambda a: np.nan_to_num(a, nan=0.0)  # noqa: E731

        # systemic draws use the crisis library; idiosyncratic draws use an ordinary historical week
        systemic = rng.random(n) < p_sys
        oidx = rng.integers(0, lib.n_ord, n)
        pick = lambda crisis, ordinary: np.where(systemic, crisis[idx], ordinary[oidx])  # noqa: E731
        spx = nz(pick(lib.fac["spx"], lib.ord["spx"]))
        nif_c = np.where(np.isnan(lib.fac["nifty"]), 0.7 * lib.fac["spx"], lib.fac["nifty"])
        nif_o = np.where(np.isnan(lib.ord["nifty"]), 0.7 * lib.ord["spx"], lib.ord["nifty"])
        nif = nz(pick(nif_c, nif_o))
        d10, d5 = nz(pick(lib.fac["d10y_bps"], lib.ord["d10y_bps"])), nz(pick(lib.fac["d5y_bps"], lib.ord["d5y_bps"]))
        mkt = np.where(self.region_in[None, :], nif[:, None], spx[:, None])            # (n, O)

        sec_draw = np.where(systemic[:, None], lib.sec[idx], lib.ord_sec[oidx])
        sec = np.zeros((n, self.O))
        for k in (0, 1):
            cols = self.sec_cols[:, k]
            ok = cols >= 0
            if ok.any():
                sec[:, ok] += nz(sec_draw[:, cols[ok]])

        role = self._role_matrix(exposures) if use_graph else np.zeros(self.O, dtype=int)
        if not use_graph:
            # no-graph baseline: only the directly named entity (if it is an obligor) is treated as hit
            for e, x in exposures.items():
                if x.edge == "SELF" and e in self.oid:
                    role[self.oid[e]] = 1
        rex, fb = self._role_excess(post, idx, role, rng)

        r = g[:, None] * (self.beta[None, :] * mkt + sec + rex)
        r = np.clip(r, -0.95, 1.0)
        ds = np.clip(self.s0[None, :] * (np.exp(-self.kappa * r) - 1.0), -0.5 * self.s0[None, :], SPREAD_UP_CAP * self.s0[None, :])
        ds_p = ds[:, self.p_obl]                                                       # (n, P) in bps
        dy10, dy5 = (g * d10)[:, None], (g * d5)[:, None]

        ecl = self.loan_amt[None, :] * ds_p / 1e4
        bond = self.bond_mv[None, :] * (-self.bond_D[None, :] * dy10 / 1e4 - self.bond_D[None, :] * ds_p / 1e4
                                        + 0.5 * self.bond_C[None, :] * (dy10 / 1e4) ** 2)
        swap = self.swap_dv01[None, :] * dy5
        cva = self.cva_w[None, :] * ds_p / 1e4
        pnl = bond + swap - ecl - cva
        loss_pos = -pnl
        total = loss_pos.sum(axis=1)
        by_ob_all = loss_pos @ self.M                                                  # (n, O)
        by_ob = {self.ob.loc[j, "entity_id"]: by_ob_all[:, j] for j in np.where(role > 0)[0]}

        w = post.weights
        shock = {
            "equity_pct": float(np.median(spx * g) * 100),
            "rates_10y_bps": float(np.median(d10 * g)),
            "hy_spread_bps": float(np.median(nz(pick(lib.fac["hy_bps"], lib.ord["hy_bps"])) * g)),
            "oil_pct": float(np.median(nz(pick(lib.fac["brent"], lib.ord["brent"])) * g) * 100),
            "p_systemic": float(p_sys),
        }
        comp = {"loans_ecl": float(np.median(ecl.sum(1))), "bonds_mtm": float(np.median(-bond.sum(1))),
                "swaps_mtm": float(np.median(-swap.sum(1))), "swaps_cva": float(np.median(cva.sum(1)))}
        del w
        return StressResult(total, by_ob, self.book.value, shock, fb, comp, np.median(by_ob_all, axis=0),
                            np.percentile(by_ob_all, 95, axis=0))

    # ---------------------------------------------------------------------------------------------
    @staticmethod
    def summarise(loss: np.ndarray) -> dict[str, float]:
        p5, p50, p95, p99 = np.percentile(loss, [5, 50, 95, 99])
        tail = loss[loss >= p99]
        return {"p5": float(p5), "p50": float(p50), "p95": float(p95), "mean": float(loss.mean()),
                "var99": float(p99), "es99": float(tail.mean() if len(tail) else p99)}

    def reverse_stress(self, post: Posterior, exposures: dict[str, Exposure], seed: int, limit_usd_m: float,
                       scales: tuple[float, ...] = (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8), p_sys: float = 1.0) -> dict:
        """How many times the typical analogue shock is needed before the median loss breaches `limit`?"""
        curve = []
        for s in scales:
            res = self.run(post, exposures, seed, n=1500, scale=s, p_sys=p_sys)
            q = self.summarise(res.loss_total)
            curve.append({"scale": s, "p50": q["p50"], "p95": q["p95"]})
        breach = next((c["scale"] for c in curve if c["p50"] >= limit_usd_m), None)
        if breach is not None and breach != scales[0]:
            lo = next(c for c in reversed(curve) if c["scale"] < breach)
            hi = next(c for c in curve if c["scale"] == breach)
            f = (limit_usd_m - lo["p50"]) / max(hi["p50"] - lo["p50"], 1e-9)
            breach = lo["scale"] + f * (hi["scale"] - lo["scale"])
        return {"limit_usd_m": limit_usd_m, "breach_scale": breach, "curve": curve}
