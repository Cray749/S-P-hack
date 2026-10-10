"""Time-machine backtest of the analogue model.

For every historical episode j (oldest to newest) the model sees ONLY episodes whose outcome window had already
closed before j (an expanding window, so nothing from the future leaks in). It is given nothing but the
episode's one-line description. We then compare the predicted distribution with what actually happened:

  1. market targets  - S&P 500, VIX, 10y yield, HY-spread proxy, Brent
       coverage of the central 50/80/90% bands, and pinball-loss skill against an unconditional baseline
  2. role targets    - excess returns of directly hit / group / peer names
       with-graph (role-conditioned) vs no-graph (no role information) prediction error

Small sample: ~30 tests. Results are indicative of "not obviously miscalibrated", not proof of calibration.
Leave-one-out (uses future episodes) is reported separately and only as a stability check.

    python -m tremor.backtest
"""
from __future__ import annotations

import json
import math

import numpy as np

from . import config as C
from .analogues import AnalogueLibrary, Posterior
from .nlp import Embedder, EventClassifier

TARGETS = {"spx": "S&P 500 move", "vix_pts": "VIX change (pts)", "d10y_bps": "10y yield (bps)",
           "hy_bps": "HY spread proxy (bps)", "brent": "Brent crude move"}
QS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
MIN_PRIORS = 6
N_DRAW = 4000


def pinball(y: float, qpred: np.ndarray) -> float:
    d = y - qpred
    return float(np.mean(np.maximum(QS * d, (QS - 1) * d)))


def weighted_samples(vals: np.ndarray, w: np.ndarray, rng: np.random.Generator, noise: float = C.KERNEL_NOISE) -> np.ndarray:
    ok = ~np.isnan(vals) & (w > 0)
    if ok.sum() == 0:
        return np.array([])
    p = w[ok] / w[ok].sum()
    draws = rng.choice(vals[ok], size=N_DRAW, p=p)
    return draws * np.exp(rng.normal(0, noise, N_DRAW))


def band(draws: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    return float(np.percentile(draws, lo)), float(np.percentile(draws, hi))


def run(loo: bool = False, seed: int = 11) -> dict:
    emb = Embedder()
    lib = AnalogueLibrary(emb)
    clf = EventClassifier(emb)
    rng = np.random.default_rng(seed)
    K = {"50": (25, 75), "80": (10, 90), "90": (5, 95)}
    hits = {t: {k: [] for k in K} for t in TARGETS}
    skill_model: dict[str, list[float]] = {t: [] for t in TARGETS}
    skill_base: dict[str, list[float]] = {t: [] for t in TARGETS}
    skill_ord: dict[str, list[float]] = {t: [] for t in TARGETS}
    scale = {t: float(np.nanstd(lib.fac[t])) or 1.0 for t in TARGETS}
    role_rows = []
    per_episode = []
    n_tests = 0

    for j in range(lib.A):
        if loo:
            elig = np.ones(lib.A, dtype=bool)
            elig[j] = False
        else:
            elig = None
        text = f"{lib.items[j]['title']}. {lib.items[j]['text']}"
        e = lib.emb[j]
        cp = clf.probs(e[None, :], [text])[0][:-1]
        cp = cp / cp.sum()
        post: Posterior | None = lib.posterior(e, cp, 0.5, lib.dates[j], eligible_override=elig)
        if post is None or int((post.weights > 0).sum()) < MIN_PRIORS:
            continue
        n_tests += 1
        w = post.weights
        w_base = (w > 0).astype(float)                      # unconditional baseline: all priors, equal weight
        row = {"id": lib.ids[j], "date": lib.items[j]["date"], "ess": round(post.ess, 1), "cls": lib.items[j]["cls"],
               "predicted_class": C.EVENT_CLASSES[int(np.argmax(cp))], "targets": {}}
        psys = lib.p_systemic(cp)
        row["p_systemic"] = round(psys, 3)
        for t in TARGETS:
            y = lib.fac[t][j]
            if np.isnan(y):
                continue
            analog = weighted_samples(lib.fac[t], w, rng)
            uncond = weighted_samples(lib.fac[t], w_base, rng)
            orow = lib.ord[t][~np.isnan(lib.ord[t])]
            if not len(analog) or not len(uncond) or not len(orow):
                continue
            ordd = rng.choice(orow, N_DRAW) * np.exp(rng.normal(0, C.KERNEL_NOISE, N_DRAW))
            d = np.where(rng.random(N_DRAW) < psys, analog, ordd)          # production model: systemic mixture
            for k, (lo, hi) in K.items():
                a_, c_ = band(d, lo, hi)
                hits[t][k].append(bool(a_ <= y <= c_))
            skill_model[t].append(pinball(y, np.percentile(d, QS * 100)) / scale[t])
            skill_base[t].append(pinball(y, np.percentile(uncond, QS * 100)) / scale[t])
            skill_ord[t].append(pinball(y, np.percentile(ordd, QS * 100)) / scale[t])
            a_, c_ = band(d, 5, 95)
            row["targets"][t] = {"actual": round(float(y), 4), "p5": round(a_, 4), "p50": round(float(np.median(d)), 4),
                                 "p95": round(c_, 4), "covered90": bool(a_ <= y <= c_)}
        # ---- role targets: with-graph vs no-graph --------------------------------------------------------
        for role in ("GROUP", "PEER"):
            actual = [v["excess"] for v in lib.items[j]["roles"][role].values()]
            if not actual:
                continue
            pad, cnt = lib.role_pad[role], lib.role_cnt[role]
            have = (cnt > 0) & (w > 0)
            if have.sum() == 0:
                role_rows.append({"id": lib.ids[j], "role": role, "n_names": len(actual), "evaluable": False})
                continue
            vals, probs = [], []
            for i in np.where(have)[0]:
                v = pad[i, :cnt[i]]
                vals.append(v)
                probs.append(np.full(len(v), w[i] / len(v)))
            vals_a, probs_a = np.concatenate(vals), np.concatenate(probs)
            probs_a /= probs_a.sum()
            graph_draws = rng.choice(vals_a, size=N_DRAW, p=probs_a)
            # no-graph: no role information -> ordinary idiosyncratic noise around zero
            pool_sd = float(np.sqrt(np.sum(probs_a * vals_a ** 2)))
            peer_vals = [lib.role_pad["PEER"][i, :lib.role_cnt["PEER"][i]]
                         for i in np.where((lib.role_cnt["PEER"] > 0) & post.eligible)[0]]
            sd0 = float(np.std(np.concatenate(peer_vals))) if peer_vals else 0.05
            none_draws = rng.normal(0.0, max(sd0, 1e-3), N_DRAW)
            g_mae = float(np.mean([abs(a - np.median(graph_draws)) for a in actual]))
            n_mae = float(np.mean([abs(a - np.median(none_draws)) for a in actual]))
            g_pin = float(np.mean([pinball(a, np.percentile(graph_draws, QS * 100)) for a in actual]))
            n_pin = float(np.mean([pinball(a, np.percentile(none_draws, QS * 100)) for a in actual]))
            g_cov = float(np.mean([np.percentile(graph_draws, 5) <= a <= np.percentile(graph_draws, 95) for a in actual]))
            role_rows.append({"id": lib.ids[j], "role": role, "n_names": len(actual), "evaluable": True,
                              "mae_graph": round(g_mae, 4), "mae_nograph": round(n_mae, 4),
                              "pinball_graph": round(g_pin, 5), "pinball_nograph": round(n_pin, 5),
                              "coverage90_graph": round(g_cov, 3), "actual_mean": round(float(np.mean(actual)), 4),
                              "graph_median": round(float(np.median(graph_draws)), 4), "pool_rms": round(pool_sd, 4)})
        per_episode.append(row)

    summary = {}
    for t, label in TARGETS.items():
        n = len(hits[t]["90"])
        if not n:
            continue
        sm, sb, so = float(np.mean(skill_model[t])), float(np.mean(skill_base[t])), float(np.mean(skill_ord[t]))
        summary[t] = {"label": label, "n": n,
                      "coverage": {k: round(float(np.mean(v)), 3) for k, v in hits[t].items()},
                      "pinball_model": round(sm, 4), "pinball_unconditional": round(sb, 4), "pinball_ordinary": round(so, 4),
                      "skill_vs_unconditional": round(1 - sm / sb, 3) if sb > 0 else None,
                      "skill_vs_ordinary": round(1 - sm / so, 3) if so > 0 else None}
    tot_m = float(np.mean([np.mean(v) for v in skill_model.values() if v]))
    tot_b = float(np.mean([np.mean(v) for v in skill_base.values() if v]))
    tot_o = float(np.mean([np.mean(v) for v in skill_ord.values() if v]))
    ev = [r for r in role_rows if r.get("evaluable")]
    graph_summary = {}
    for role in ("PEER", "GROUP"):
        rr = [r for r in ev if r["role"] == role]
        allr = [r for r in role_rows if r["role"] == role]
        if rr:
            graph_summary[role] = {"episodes_evaluable": len(rr), "episodes_total": len(allr),
                                   "mae_graph": round(float(np.mean([r["mae_graph"] for r in rr])), 4),
                                   "mae_nograph": round(float(np.mean([r["mae_nograph"] for r in rr])), 4),
                                   "pinball_graph": round(float(np.mean([r["pinball_graph"] for r in rr])), 5),
                                   "pinball_nograph": round(float(np.mean([r["pinball_nograph"] for r in rr])), 5),
                                   "coverage90_graph": round(float(np.mean([r["coverage90_graph"] for r in rr])), 3)}
        else:
            graph_summary[role] = {"episodes_evaluable": 0, "episodes_total": len(allr),
                                   "note": "no earlier analogue with this role - cannot be backtested"}
    overall = {k: round(float(np.mean([s["coverage"][k] for s in summary.values()])), 3) for k in K}
    n_int = {k: int(sum(len(hits[t][k]) for t in hits)) for k in K}
    return {"mode": "leave-one-out (stability only, uses future episodes)" if loo else "expanding window (no look-ahead)",
            "episodes_tested": n_tests, "episodes_in_library": lib.A, "min_priors": MIN_PRIORS,
            "overall_coverage": overall, "nominal": {"50": 0.5, "80": 0.8, "90": 0.9}, "intervals_scored": n_int,
            "binomial_se_90": round(math.sqrt(0.9 * 0.1 / max(1, n_int["90"])), 3),
            "overall_skill_vs_unconditional": round(1 - tot_m / tot_b, 3), "overall_skill_vs_ordinary": round(1 - tot_m / tot_o, 3),
            "targets": summary, "graph_ablation": graph_summary, "episodes": per_episode,
            "caveat": "Small sample, overlapping episodes (not independent). Indicates the method is not obviously "
                      "miscalibrated; it does not prove calibration."}


def main() -> None:
    out = C.DATA / "results"
    out.mkdir(parents=True, exist_ok=True)
    res = {"expanding": run(False), "loo": run(True)}
    (out / "backtest.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    e, lo = res["expanding"], res["loo"]
    print(f"expanding window: tested {e['episodes_tested']} episodes, coverage {e['overall_coverage']} (nominal 50/80/90), "
          f"intervals {e['intervals_scored']['90']}, se90={e['binomial_se_90']}")
    for t, s in e["targets"].items():
        print(f"  {s['label']:<24} n={s['n']:<3} cov50/80/90={s['coverage']['50']}/{s['coverage']['80']}/{s['coverage']['90']} "
              f"skill vs unconditional={s['skill_vs_unconditional']} vs ordinary={s['skill_vs_ordinary']}")
    print(f"overall skill vs unconditional crisis library: {e['overall_skill_vs_unconditional']}  vs ordinary weeks: {e['overall_skill_vs_ordinary']}")
    print("graph ablation:", json.dumps(e["graph_ablation"]))
    print(f"leave-one-out coverage {lo['overall_coverage']}")


if __name__ == "__main__":
    main()
