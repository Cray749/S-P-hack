"""Render docs/architecture.png (high resolution) in the dashboard's visual language.

    python scripts/make_architecture.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"
BG, INK, MUTE = "#0a0810", "#f3eee4", "#a39cb3"
ORANGE, PINK, VIOLET, ICE, MINT, GOLD = "#ff6a1f", "#ff2e7e", "#8a45ff", "#7fd4ff", "#33f0c0", "#ffc23d"


def main() -> None:
    fig, ax = plt.subplots(figsize=(20, 12.6), dpi=170)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1200)
    ax.set_ylim(780, 0)
    ax.axis("off")

    def box(x, y, w, h, title, sub="", c=ORANGE, fs=13, fill="#15111f"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14", fc=fill, ec=c, lw=1.8))
        ax.text(x + w / 2, y + h / 2 - (15 if "\n" in sub else 7 if sub else 0), title, color=INK, ha="center", va="center", fontsize=fs, fontweight="bold")
        if sub:
            ax.text(x + w / 2, y + h / 2 + (20 if "\n" in sub else 13), sub, color=MUTE, ha="center", va="center", fontsize=8.6)

    def arrow(x1, y1, x2, y2, c=ORANGE, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14, color=c, lw=1.8, linestyle=ls))

    def label(x, y, t, c=MUTE, fs=9, **kw):
        ax.text(x, y, t, color=c, fontsize=fs, **kw)

    ax.text(30, 30, "TREMOR", color=ORANGE, fontsize=30, fontweight="bold")
    ax.text(30, 56, "event  →  exposure  →  loss range    |    kappa-style, event-sourced, replayable byte for byte", color=MUTE, fontsize=11)

    label(30, 92, "SOURCES", fs=9.5)
    box(30, 102, 250, 62, "News headlines", "FNSPID replay · live-feed adapter", ICE)
    box(300, 102, 250, 62, "Social posts", "public tweet corpus replay", VIOLET)
    box(570, 102, 250, 62, "POST /api/ingest", "live path · same pipeline", MINT)
    for x in (155, 425, 695):
        arrow(x, 164, 425 if x != 695 else 560, 218, ORANGE)

    ax.add_patch(FancyBboxPatch((30, 218), 790, 92, boxstyle="round,pad=0,rounding_size=18", fc="#ff6a1f14", ec=ORANGE, lw=2.6))
    ax.text(425, 246, "APPEND-ONLY LEDGER  ·  the contract", color=INK, ha="center", fontsize=17, fontweight="bold")
    ax.text(425, 268, "SQLite · SHA-256 hash chain · UPDATE / DELETE rejected by database triggers · replay = live", color=MUTE, ha="center", fontsize=9.5)
    ax.text(425, 290, "every state below is a pure function of  (ledger prefix, book snapshot, config hash)", color=GOLD, ha="center", fontsize=9.5)

    arrow(425, 310, 425, 352)
    label(30, 346, "SCORE ONCE, REUSE EVERYWHERE", fs=9.5)
    box(30, 358, 190, 70, "Embed once", "MiniLM · cluster · class\nanalogues", VIOLET)
    box(236, 358, 190, 70, "Sentiment cascade", "lexicon → FinBERT\nonly if unsure", VIOLET)
    box(442, 358, 190, 70, "Event class", "10 classes + routine gate", VIOLET)
    box(648, 358, 172, 70, "Entity linker", "trie regex · aliases\ncashtags", VIOLET)
    arrow(425, 428, 425, 470)

    ax.add_patch(FancyBboxPatch((30, 470), 790, 84, boxstyle="round,pad=0,rounding_size=18", fc="#8a45ff14", ec=VIOLET, lw=2))
    ax.text(425, 498, "EVENT TRACKER", color=INK, ha="center", fontsize=15, fontweight="bold")
    ax.text(425, 520, "cluster · syndication-echo detection · independence-weighted corroboration · provisional → confirmed", color=MUTE, ha="center", fontsize=9.5)
    ax.text(425, 540, "output: sentiment · event class · impact 1-10 · confidence · evidence ids", color=MUTE, ha="center", fontsize=9.5)
    arrow(425, 554, 425, 596)

    label(30, 590, "MODULE B  ·  STRATEGIC STRESS TEST  (a subscriber on the bus)", fs=9.5)
    box(30, 602, 190, 84, "Exposure graph", "control · promoter group\nindustry · text cue", PINK)
    box(236, 602, 190, 84, "Scenario draws", "36 analogues + 1,200\nordinary weeks · P(systemic)", PINK)
    box(442, 602, 190, 84, "Book revaluation", "loans ΔECL · bonds D/C\nswaps DV01 + CVA", PINK)
    box(648, 602, 172, 84, "Loss range", "P5/P50/P95 · VaR99\nES99 · reverse stress", PINK)
    for a, b in ((220, 236), (426, 442), (632, 648)):
        arrow(a, 644, b, 644, PINK)

    arrow(820, 644, 900, 644, MINT)
    label(900, 92, "CONSUMERS", fs=9.5)
    box(900, 602, 270, 84, "Signal bus", "pub/sub · Kafka / Redis\nStreams in production", MINT)
    box(900, 262, 270, 80, "Replay verifier", "re-derive any signal · hash must match", GOLD)
    box(900, 372, 270, 80, "Module A · rebalancer", "decayed sentiment tilt\ncaps · no-trade band", MINT)
    box(900, 482, 270, 80, "Dashboard + REST API", "control room · evidence\nreplay · reverse stress", MINT)
    arrow(1035, 602, 1035, 562, MINT)
    arrow(1150, 602, 1150, 452, MINT, ls=(0, (4, 4)))
    arrow(820, 266, 900, 302, GOLD, style="-|>", ls=(0, (4, 4)))

    label(30, 722, "One container · one command · models baked at build · runs offline · 36 analogue episodes · 20.8k documents · 252 synthetic positions", fs=10)
    label(30, 746, "Innovation is the contract: append-only ledger + versioned inputs  ⇒  byte-exact replay  ⇒  a defensible audit trail for every number.", c=GOLD, fs=10.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
    print("wrote", OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
