"""Build docs/presentation.pdf (7 slides) from the measured results. Every number is read from data/results/*.json.

    python scripts/make_deck.py
"""
from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BG, INK, MUTE = "#0a0810", "#f3eee4", "#a39cb3"
ORANGE, PINK, VIOLET, ICE, MINT, GOLD = "#ff6a1f", "#ff2e7e", "#8a45ff", "#7fd4ff", "#33f0c0", "#ffc23d"

bt = json.loads((ROOT / "data/results/backtest.json").read_text(encoding="utf-8"))["expanding"]
ev_path = ROOT / "data/results/eval.json"
ev = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else {}
pc = lambda x, d=0: "n/a" if x is None else f"{x * 100:.{d}f}%"  # noqa: E731
sg = lambda x, d=1: "n/a" if x is None else f"{x * 100:+.{d}f}%"  # noqa: E731


def page(title: str, kicker: str = ""):
    fig = plt.figure(figsize=(13.333, 7.5), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56.25)
    ax.axis("off")
    ax.set_facecolor(BG)
    if kicker:
        ax.text(4, 51.2, kicker.upper(), color=ORANGE, fontsize=11, fontweight="bold")
    ax.text(4, 47, title, color=INK, fontsize=27, fontweight="bold", va="center")
    ax.text(96, 2.2, "TREMOR · Lakshya Manchanda · Thapar Institute of Engineering and Technology", color=MUTE, fontsize=8, ha="right")
    return fig, ax


def card(ax, x, y, w, h, head, body, c=ORANGE, bs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc="#15111f", ec=c, lw=1.6))
    ax.text(x + 1.5, y + h - 2.4, head, color=INK, fontsize=12.5, fontweight="bold", va="center")
    ax.text(x + 1.5, y + h - 5.2, body, color=MUTE, fontsize=bs, va="top", linespacing=1.45)


def main() -> None:
    out = ROOT / "docs" / "presentation.pdf"
    prev = os.environ.get("DECK_PREVIEW")
    n_page = [0]
    with PdfPages(out) as pdf:

        def save(fig):
            pdf.savefig(fig, facecolor=BG)
            if prev:
                n_page[0] += 1
                fig.savefig(Path(prev) / f"slide{n_page[0]}.png", facecolor=BG, dpi=70)

        # 1 ---------------------------------------------------------------------------------------------
        fig, ax = page("", "S&P Global x Crisil Campus Hackathon 2026 · Case study")
        ax.text(4, 33, "TREMOR", color=ORANGE, fontsize=84, fontweight="bold")
        ax.text(4, 24.5, "A headline names one company.\nWe show a bank the loss on its whole book, including the\nobligors the headline never named, with a range and a trail to the sources.",
                color=INK, fontsize=19, va="top", linespacing=1.4)
        ax.text(4, 9, "Lakshya Manchanda  ·  Thapar Institute of Engineering and Technology", color=MUTE, fontsize=13)
        ax.text(4, 5.5, "Engine: sentiment · event class · impact  |  Module B: event-driven portfolio stress test  |  Module A: rebalancer (second subscriber)", color=MUTE, fontsize=10.5)
        save(fig)
        plt.close(fig)

        # 2 ---------------------------------------------------------------------------------------------
        fig, ax = page("Problem and approach", "2 · business problem in our words")
        card(ax, 4, 26, 29, 17, "The gap", "Text signals are generic (one score per ticker).\nReposts count as evidence. Rumours fire alerts.\nStress tests are periodic and detached from news.\nNo uncertainty, no audit trail.", PINK)
        card(ax, 35, 26, 29, 17, "The question", "\"If this headline is real, what does it do to\nOUR book, how sure are we, and can I show\nthe auditor exactly how we got the number?\"", VIOLET)
        card(ax, 66, 26, 30, 17, "One claim", "Every risk signal is a loss range for a specific\nbook, derived from historical analogues,\ntraceable to its sources, and exactly\nreproducible.", ORANGE)
        card(ax, 4, 5, 92, 18, "Approach", "text  →  event (clustered, syndication-aware, independence-weighted)  →  exposure graph (control · promoter group · industry · text cue)\n"
             "→  scenario draws from 36 documented historical episodes + ordinary weeks (P(systemic))  →  Monte-Carlo revaluation of a synthetic\nwholesale book (loans, bonds, swaps)  →  P5/P50/P95 loss, reverse stress, evidence lineage, byte-exact replay.", MINT, 11)
        save(fig)
        plt.close(fig)

        # 3 ---------------------------------------------------------------------------------------------
        fig, ax = page("System design", "3 · architecture and data flow")
        img = mpimg.imread(ROOT / "docs" / "architecture.png")
        ax2 = fig.add_axes([0.03, 0.04, 0.94, 0.80])
        ax2.imshow(img)
        ax2.axis("off")
        save(fig)
        plt.close(fig)

        # 4 ---------------------------------------------------------------------------------------------
        fig, ax = page("Implementation highlights", "4 · key modules, tech choices and why")
        card(ax, 4, 28, 29, 16, "Append-only ledger", "SQLite + SHA-256 hash chain; UPDATE/DELETE\nrejected by triggers. Replay = live path.\nWhy: audit trail and exact reproducibility.", ORANGE)
        card(ax, 35, 28, 29, 16, "Score once", "One MiniLM embedding feeds clustering, class,\nnovelty and analogues. Sentiment cascade:\nlexicon first, FinBERT only when unsure.", VIOLET)
        card(ax, 66, 28, 30, 16, "Events, not articles", "Echo detection (syndicated copies are not\nindependent), publisher-independence weights,\nsocial volume capped. provisional → confirmed.", VIOLET)
        card(ax, 4, 9, 29, 17, "Exposure graph", "Parent / subsidiary, promoter group, industry\npeers and rule-based text cues. Reaches the\nobligors the headline never named.", PINK)
        card(ax, 35, 9, 29, 17, "Module B stress", "Credit triangle (spread ≈ PD×LGD); loans ΔECL,\nbonds duration/convexity, swaps DV01 + CVA.\nVectorised Monte Carlo; reverse stress.", PINK)
        card(ax, 66, 9, 30, 17, "Engineering", "FastAPI · Pydantic contracts · pytest (36 tests,\nincl. replay exactness) · ruff · CI · one-container\nDocker, models baked, offline at runtime.", MINT)
        save(fig)
        plt.close(fig)

        # 5 ---------------------------------------------------------------------------------------------
        fig, ax = page("Key results (measured, out of sample)", "5 · outputs, metrics, efficiency vs naive")
        cov = bt["overall_coverage"]
        stats = [
            (pc(cov["90"]), "of realised market moves inside the predicted 90% band", f"{bt['intervals_scored']['90']} intervals · chronological · no look-ahead", ORANGE),
            (sg(bt["overall_skill_vs_unconditional"]), "pinball skill vs drawing every event from the crisis library", f"{sg(bt['overall_skill_vs_ordinary'])} vs ordinary weeks", PINK),
        ]
        s = ev.get("sentiment", {}).get("cascade")
        if s:
            stats.append((pc(s["finbert_calls_saved"]), "FinBERT calls avoided by the cascade", f"accuracy {pc(s['accuracy'], 1)} vs FinBERT {pc(ev['sentiment']['finbert_only']['accuracy'], 1)}, lexicon {pc(ev['sentiment']['lexicon_only']['accuracy'], 1)}", VIOLET))
        p = ev.get("pipeline")
        if p:
            es = ev.get("event_study", {})
            stats.append((f"{p['documents']:,} → {p['events']:,}", "docs → events: routine/spam filtered, echoes collapsed", f"{pc(p['echo_share_of_news'])} of news were syndicated copies · {p['ms_per_doc_including_stress']:.1f} ms/doc", MINT))
            if es.get("n_events"):
                stats.append((f"ρ {es['impact_vs_abs_abnormal_return']['spearman']:.2f}", "Impact vs realised |abnormal return|", f"n={es['n_events']} single-name events, p={es['impact_vs_abs_abnormal_return']['p']:.3f}", GOLD))
        for i, (v, a, b, c) in enumerate(stats[:5]):
            card(ax, 4 + i * 18.6, 30.5, 17.6, 14.5, v, textwrap.fill(a, 31) + "\n" + textwrap.fill(b, 31), c, 8)
        ax2 = fig.add_axes([0.06, 0.11, 0.36, 0.40], facecolor="#15111f")
        xs = [0.5, 0.8, 0.9]
        ax2.plot([0, 1], [0, 1], "--", color=MUTE, lw=1)
        ax2.plot(xs, [cov["50"], cov["80"], cov["90"]], "-o", color=GOLD, lw=2.2, ms=8)
        ax2.set_title("Calibration: nominal vs actual coverage", color=INK, fontsize=11)
        ax2.set_xlabel("nominal", color=MUTE)
        ax2.set_ylabel("actual", color=MUTE)
        ax2.tick_params(colors=MUTE)
        for sp in ax2.spines.values():
            sp.set_color("#333")
        ax3 = fig.add_axes([0.52, 0.11, 0.42, 0.40], facecolor="#15111f")
        names = [t["label"].replace(" move", "").replace(" (bps)", "").replace(" (pts)", "") for t in bt["targets"].values()]
        ax3.barh(names, [t["skill_vs_unconditional"] * 100 for t in bt["targets"].values()], color=ORANGE, label="vs crisis library")
        ax3.barh(names, [t["skill_vs_ordinary"] * 100 for t in bt["targets"].values()], color=VIOLET, alpha=0.6, height=0.4, label="vs ordinary weeks")
        ax3.set_title("Skill by target (%, higher is better)", color=INK, fontsize=11)
        ax3.tick_params(colors=MUTE)
        ax3.legend(facecolor="#15111f", labelcolor=INK, fontsize=8)
        for sp in ax3.spines.values():
            sp.set_color("#333")
        save(fig)
        plt.close(fig)

        # 6 ---------------------------------------------------------------------------------------------
        fig, ax = page("Domain impact", "6 · why it matters")
        card(ax, 4, 26, 29, 17, "Time to exposure", "From headline to a book-specific loss range in\nseconds, with the obligors and paths that carry\nit. Triage first, analyst judgement second.", MINT)
        card(ax, 35, 26, 29, 17, "Fewer false alarms", "Syndicated copies and tweet volume never count as\nconfirmation. Provisional signals carry wide\nranges that tighten as independent sources agree.", ORANGE)
        card(ax, 66, 26, 30, 17, "Audit-ready", "Every number traces to source documents and\nre-derives byte for byte from the ledger, the\nbook snapshot and the config hash.", VIOLET)
        card(ax, 4, 5, 92, 18, "Positioning for risk teams", "Complements, does not replace, surveillance and supervisory stress testing: it is the fast, transparent first estimate that tells an analyst where to look.\n"
             "Regulatory scenarios are periodic and fixed; static group-exposure aggregation exists, but it does not react to a headline, weigh evidence or give a range.\n"
             "Module A (rebalancer) was added as a second bus subscriber without changing the engine, showing how consumers plug in.", PINK, 10.5)
        save(fig)
        plt.close(fig)

        # 7 ---------------------------------------------------------------------------------------------
        fig, ax = page("Limitations and next steps", "7 · what we found, stated plainly")
        card(ax, 4, 24, 44, 20, "What did not work", "Sharper analogue retrieval (text / class / severity) did NOT\nbeat the unconditional history at n=36, so we keep it soft and\nuse what did work: P(systemic | class).\nImpact is a transparent triage score, not a calibrated forecast.\nPromoter-group contagion cannot be backtested (1 episode).", PINK, 10)
        card(ax, 52, 24, 44, 20, "Assumptions", "News is day-resolution (no minute-level lead/lag claims).\nSynthetic book and ratings; real names are labels only.\nCredit spreads are ETF proxies; credit triangle, LGD 45%.\nHorizon: 5-day peak excursion. Ranges are illustrative.", GOLD, 10)
        card(ax, 4, 4, 92, 17, "Next steps", "Grow the episode library (hundreds) so conditioning can be tested properly · intraday timestamps and a licensed live feed ·\nmultilingual local-press ingestion · GLEIF parent-child data for real group graphs · Kafka / Redis Streams behind the same bus interface.", MINT, 10.5)
        save(fig)
        plt.close(fig)
    print("wrote", out, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
