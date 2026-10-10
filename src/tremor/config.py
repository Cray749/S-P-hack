"""Central configuration. Every modelling assumption lives here so it can be audited and hashed."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("TREMOR_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
CACHE = DATA / "cache"

# ---- event taxonomy -------------------------------------------------------------------------
EVENT_CLASSES = [
    "GEOPOLITICAL",
    "MACROECONOMIC",
    "CREDIT_EVENT",
    "MERGER_ACQUISITION",
    "PRODUCT_LAUNCH",
    "REGULATORY_LEGAL",
    "OPERATIONAL",
    "EARNINGS_GUIDANCE",
    "COMMODITY_SHOCK",
    "PANDEMIC_DISASTER",
]

# ---- analogue / shock model ------------------------------------------------------------------
HORIZON_DAYS = 5                 # trading days over which analogue moves are measured
SPREAD_DURATION_HY = 3.8         # proxy: HYG spread duration (years)
SPREAD_DURATION_IG = 8.5         # proxy: LQD spread duration (years)
S0_HY_BPS = 450.0                # assumed baseline HY OAS used to log-scale the credit proxy
SIM_TEMPERATURE = 0.12           # softmax temperature on text-embedding similarity
SEVERITY_BANDWIDTH = 0.40        # kernel width on the severity-percentile prior
CLASS_PRIOR_FLOOR = 0.05         # keeps off-class analogues in play when the class is uncertain
KERNEL_NOISE = 0.20              # multiplicative jitter so a handful of analogues form a smooth range
MC_DRAWS = 4000

# ---- credit mapping (documented simplifications; see docs/LIMITATIONS.md) ---------------------
RATING_SPREAD_BPS = {"AAA": 40, "AA": 55, "A": 80, "BBB": 130, "BB": 280, "B": 450, "CCC": 900}
RATING_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
LGD_SENIOR_UNSECURED = 0.45      # Basel foundation-IRB supervisory LGD for senior unsecured
LGD_SECURED = 0.35
KAPPA_DEFAULT = 2.0              # spread sensitivity to equity return; recalibrated if the data supports it
KAPPA_BOUNDS = (1.0, 5.0)
CREDIT_TRIANGLE = "spread ~= PD x LGD"

# ---- evidence / corroboration ------------------------------------------------------------------
CLUSTER_SIM = 0.58               # cosine threshold to join an existing event
DUPLICATE_SIM = 0.93             # cosine above which two items are treated as syndicated copies
EVENT_WINDOW_HOURS = 72
CONFIRM_WEIGHT = 1.6             # independent-publisher weight needed to promote provisional -> confirmed
PUBLISHER_TIER = {               # credibility weights (design prior, not fitted)
    "reuters.com": 1.0, "bloomberg.com": 1.0, "wsj.com": 1.0, "ft.com": 1.0, "cnbc.com": 0.9,
    "apnews.com": 1.0, "nasdaq.com": 0.9, "benzinga.com": 0.8, "marketwatch.com": 0.8,
    "investors.com": 0.8, "seekingalpha.com": 0.6, "zacks.com": 0.6, "gurufocus.com": 0.5,
    "talkmarkets.com": 0.4,
}
DEFAULT_PUBLISHER_WEIGHT = 0.45
SOCIAL_CAP = 0.5                 # social volume can never contribute more than this to corroboration

# ---- models -------------------------------------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENT_MODEL = "ProsusAI/finbert"
LEXICON_ACCEPT = 0.55            # |lexicon score| needed to skip FinBERT


def config_hash() -> str:
    """Stable hash of every tunable above; stamped on each emitted signal."""
    blob = {
        k: (v if not isinstance(v, Path) else str(v))
        for k, v in sorted(globals().items())
        if k.isupper() and k not in {"ROOT", "DATA", "CACHE"}
    }
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:12]
