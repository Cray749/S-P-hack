"""NLP layer: one embedding model, a two-tier sentiment cascade and an event classifier.

Design notes
* The embedding is computed ONCE per document and reused for clustering, event classification, novelty and
  analogue retrieval. This is the main efficiency lever.
* Sentiment is a cascade: a finance lexicon resolves the easy cases; only ambiguous items escalate to FinBERT.
  The escalation rate and the agreement with FinBERT-only labels are measured by `tremor.evaluate`.
* Everything degrades gracefully: if the neural models are unavailable (or TREMOR_LIGHT=1) a deterministic
  hashing embedder and the lexicon alone keep the pipeline running, so tests and CI stay fast.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as C

LIGHT = os.environ.get("TREMOR_LIGHT") == "1"
log = logging.getLogger("tremor.nlp")


# =================================================================================================
# Embeddings
# =================================================================================================
class HashEmbedder:
    """Deterministic, dependency-free fallback: hashed word + char-trigram bag, L2-normalised."""

    name = "hash-384"
    dim = 384

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            t = t.lower()
            toks = re.findall(r"[a-z0-9$]+", t)
            feats = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])] + [t[j:j + 3] for j in range(0, max(0, len(t) - 2), 2)]
            for f in feats:
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)


class Embedder:
    """Sentence-transformer wrapper with a hashing fallback."""

    def __init__(self) -> None:
        self._model = None
        self._fallback = HashEmbedder()
        self.name = self._fallback.name
        self.dim = self._fallback.dim
        if not LIGHT:
            try:
                from sentence_transformers import SentenceTransformer

                try:
                    self._model = SentenceTransformer(C.EMBED_MODEL, device="cpu", local_files_only=True)
                except Exception:
                    self._model = SentenceTransformer(C.EMBED_MODEL, device="cpu")
                self.name = C.EMBED_MODEL.split("/")[-1]
                self.dim = int(self._model.get_sentence_embedding_dimension())
            except Exception as exc:  # offline / not installed
                log.warning("neural embedder unavailable, using hashing fallback: %r", exc)
                self._model = None

    @property
    def neural(self) -> bool:
        return self._model is not None

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._model is None:
            return self._fallback.embed(texts)
        v = self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)


# =================================================================================================
# Sentiment cascade
# =================================================================================================
_POS = {
    "beat", "beats", "surge", "surges", "surged", "soar", "soars", "soared", "jump", "jumps", "jumped", "rally",
    "rallies", "rallied", "gain", "gains", "gained", "rise", "rises", "rose", "upgrade", "upgraded", "bullish",
    "record", "strong", "growth", "profit", "profits", "outperform", "raise", "raises", "raised", "boost",
    "boosts", "approval", "approved", "rebound", "rebounds", "recover", "recovery", "optimism", "optimistic",
    "win", "wins", "breakthrough", "positive", "upside", "higher", "bull", "moon", "squeeze", "improve",
    "improves", "improved", "stimulus", "bailout", "accretive", "robust", "exceed", "exceeds", "exceeded",
}
_NEG = {
    "miss", "misses", "missed", "plunge", "plunges", "plunged", "plummet", "plummets", "tumble", "tumbles",
    "tumbled", "crash", "crashes", "crashed", "fall", "falls", "fell", "drop", "drops", "dropped", "slump",
    "slumps", "sink", "sinks", "sank", "loss", "losses", "downgrade", "downgraded", "bearish", "default",
    "defaults", "bankruptcy", "bankrupt", "lawsuit", "fraud", "probe", "investigation", "fined", "fine",
    "recall", "layoffs", "layoff", "furlough", "cut", "cuts", "warns", "warning", "weak", "decline", "declines",
    "declined", "fear", "fears", "selloff", "sell-off", "crisis", "collapse", "collapses", "collapsed", "risk",
    "negative", "concern", "concerns", "lower", "slash", "slashes", "struggle", "struggles", "shutdown",
    "outage", "tank", "tanks", "tanked", "bear", "worst", "recession", "delist", "insolvency", "halt", "halted",
    "trouble", "pressure", "worries", "worry", "slides", "slide", "slid", "sell", "shorts", "short",
}
_NEGATORS = {"not", "no", "never", "without", "fails", "failed", "unlikely", "avoid", "avoids", "avoided"}
_TOK = re.compile(r"[a-z][a-z'\-]+|\$[a-z]+")


@dataclass
class SentimentResult:
    score: float          # [-1, 1]
    tier: str             # "lexicon" | "finbert"


def lexicon_score(text: str) -> tuple[float, int]:
    """Return (score in [-1,1], number of sentiment hits)."""
    toks = _TOK.findall(text.lower())
    pos = neg = 0
    for i, t in enumerate(toks):
        s = 1 if t in _POS else -1 if t in _NEG else 0
        if not s:
            continue
        if any(x in _NEGATORS for x in toks[max(0, i - 3):i]):
            s = -s
        pos += s > 0
        neg += s < 0
    n = pos + neg
    return ((pos - neg) / n if n else 0.0), n


class SentimentCascade:
    """Lexicon first; FinBERT only when the lexicon is not decisive."""

    def __init__(self) -> None:
        self._tok = None
        self._model = None
        self.calls_finbert = 0
        self.calls_total = 0
        if not LIGHT:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                try:  # prefer the local cache: deterministic and works offline
                    self._tok = AutoTokenizer.from_pretrained(C.SENT_MODEL, local_files_only=True)
                    self._model = AutoModelForSequenceClassification.from_pretrained(C.SENT_MODEL, local_files_only=True).eval()
                except Exception:
                    self._tok = AutoTokenizer.from_pretrained(C.SENT_MODEL)
                    self._model = AutoModelForSequenceClassification.from_pretrained(C.SENT_MODEL).eval()
                self._torch = torch
                torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
            except Exception as exc:
                log.warning("FinBERT unavailable, lexicon-only sentiment: %r", exc)
                self._model = None

    @property
    def neural(self) -> bool:
        return self._model is not None

    def finbert(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """P(pos) - P(neg) for each text."""
        assert self._model is not None and self._tok is not None
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self._tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=64, return_tensors="pt")
            with self._torch.no_grad():
                p = self._torch.softmax(self._model(**enc).logits, dim=-1).numpy()
            out.append(p[:, 0] - p[:, 1])          # ProsusAI/finbert label order: positive, negative, neutral
        return np.concatenate(out) if out else np.zeros(0)

    @staticmethod
    def decisive(score: float, hits: int) -> bool:
        """Lexicon is trusted when it saw nothing (neutral) or a clear, multi-cue polarity."""
        return hits == 0 or (hits >= 2 and abs(score) >= C.LEXICON_ACCEPT)

    def score(self, texts: list[str], force_finbert: bool = False) -> list[SentimentResult]:
        lex = [lexicon_score(t) for t in texts]
        self.calls_total += len(texts)
        res: list[SentimentResult | None] = [None] * len(texts)
        need = []
        for i, (s, h) in enumerate(lex):
            if self._model is not None and (force_finbert or not self.decisive(s, h)):
                need.append(i)
            else:
                res[i] = SentimentResult(float(np.clip(s * (0.6 if h < 2 else 0.8), -1, 1)), "lexicon")
        if need:
            fb = self.finbert([texts[i] for i in need])
            self.calls_finbert += len(need)
            for i, v in zip(need, fb):
                res[i] = SentimentResult(float(v), "finbert")
        return [r for r in res if r is not None]


# =================================================================================================
# Event classification
# =================================================================================================
EXEMPLARS: dict[str, list[str]] = {
    "GEOPOLITICAL": [
        "War breaks out and sanctions are imposed on the country",
        "Military strikes escalate tensions in the region",
        "Trade war escalates as new tariffs are announced",
        "Government announces embargo and sanctions",
        "Election result triggers political uncertainty",
        "Terror attack raises geopolitical risk",
        "Diplomatic crisis between the two nations deepens",
        "Border conflict and missile launch rattle markets",
    ],
    "MACROECONOMIC": [
        "Federal Reserve cuts interest rates",
        "Inflation data comes in hotter than expected",
        "Unemployment rate rises as jobs report disappoints",
        "GDP contracts as the economy slips into recession",
        "Central bank signals interest rate hikes",
        "Government unveils fiscal stimulus package",
        "Treasury yields surge on bond market sell-off",
        "Retail sales and consumer confidence fall sharply",
        "Stocks plunge as the Dow falls hundreds of points in a broad market sell-off",
        "Wall Street suffers its worst day as investors fear a second wave",
    ],
    "CREDIT_EVENT": [
        "Company files for Chapter 11 bankruptcy protection",
        "Firm defaults on its bond payments",
        "Credit rating downgraded to junk by agency",
        "Lender warns of loan losses and debt restructuring",
        "Bank run as depositors withdraw funds",
        "Company seeks bailout and could be forced to file for bankruptcy",
        "Missed interest payment raises default fears",
        "Liquidity crisis forces emergency funding",
    ],
    "MERGER_ACQUISITION": [
        "Company agrees to acquire rival in billion dollar deal",
        "Merger announced between two firms",
        "Takeover bid launched for the company",
        "Private equity firm to buy company in leveraged buyout",
        "Shareholders approve the acquisition",
        "Spin-off and divestiture of business unit",
        "Antitrust regulators review the proposed merger",
        "Activist investor builds a stake in the company",
    ],
    "PRODUCT_LAUNCH": [
        "Company unveils new product at launch event",
        "Firm launches new service and platform",
        "New smartphone model released with upgraded features",
        "Drug receives regulatory approval for launch",
        "Company announces breakthrough technology",
        "Electric vehicle maker unveils new model",
        "New AI model released by startup",
        "Streaming service launches in new markets",
    ],
    "REGULATORY_LEGAL": [
        "Regulator opens investigation into the company",
        "Company hit with lawsuit and fined",
        "Short seller alleges accounting fraud",
        "Court rules against the firm in antitrust case",
        "Government proposes new regulation on the industry",
        "SEC charges company executives",
        "Data privacy scandal triggers regulatory scrutiny",
        "Settlement reached to resolve litigation",
    ],
    "OPERATIONAL": [
        "Factory shut down after fire and production halted",
        "Cyber attack disrupts company systems",
        "Plane crash leads to grounding of fleet",
        "Product recall announced over safety defect",
        "Supply chain disruption halts output",
        "Outage knocks out services for customers",
        "Company announces layoffs and workforce reductions",
        "Mine accident and strike halts operations",
    ],
    "EARNINGS_GUIDANCE": [
        "Company reports quarterly earnings above estimates",
        "Earnings miss expectations and shares drop",
        "Firm cuts full-year guidance",
        "Revenue beats forecasts and raises outlook",
        "Company withdraws guidance amid uncertainty",
        "Profit falls as costs rise",
        "Dividend cut announced after weak results",
        "Quarterly results show slowing growth",
    ],
    "COMMODITY_SHOCK": [
        "Oil prices crash as demand collapses",
        "Crude futures turn negative as storage fills",
        "OPEC agrees to production cuts",
        "Gold surges to record high",
        "Copper and metal prices plunge",
        "Natural gas prices spike",
        "Wheat and grain prices surge on supply fears",
        "Energy stocks fall as oil slumps",
    ],
    "PANDEMIC_DISASTER": [
        "Coronavirus outbreak spreads and lockdowns are imposed",
        "Pandemic triggers travel bans and market panic",
        "Hurricane causes widespread damage",
        "Earthquake disrupts production",
        "Vaccine trial results announced",
        "Wildfires and floods hit the region",
        "Health authorities warn of a second wave of infections",
        "Countries reopen economies after lockdown",
    ],
    "OTHER": [
        "Analyst reiterates buy rating and price target",
        "Stock trading higher in premarket session",
        "Technical analysis chart shows resistance and support",
        "Company announces dividend payment date",
        "Insider sells shares in routine filing",
        "Top gainers and losers of the day",
        "Options traders place bets on the stock",
        "What to watch in tomorrow's trading",
        "Deutsche Bank maintains hold rating and lowers price target",
        "Morgan Stanley raises price target on shares and keeps overweight",
        "Analyst upgrades stock to buy and initiates coverage",
        "Stocks on the move: top premarket gainers",
        "Credit Suisse maintains outperform rating on the stock and lowers price target",
        "Analyst reiterates market perform and sets a new price target",
        "Jefferies upgrades shares to buy with a higher target",
        "Benzinga's top upgrades, downgrades and initiations for the day",
    ],
}
LABELS = C.EVENT_CLASSES + ["OTHER"]

_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"chapter 11|bankrupt|default|downgrad|junk|restructur|insolven|bailout", re.I), "CREDIT_EVENT"),
    (re.compile(r"\bacquir|merger|takeover|buyout|to buy\b|bid for", re.I), "MERGER_ACQUISITION"),
    (re.compile(r"\boil\b|crude|opec|brent|\bwti\b|natural gas|\bgold\b", re.I), "COMMODITY_SHOCK"),
    (re.compile(r"\bfed\b|federal reserve|rate cut|inflation|\bgdp\b|unemployment|jobs report|treasury yield", re.I), "MACROECONOMIC"),
    (re.compile(r"sanction|invasion|\bwar\b|missile|tariff|trade war|embargo", re.I), "GEOPOLITICAL"),
    (re.compile(r"coronavirus|covid|pandemic|lockdown|hurricane|earthquake|vaccine|outbreak", re.I), "PANDEMIC_DISASTER"),
    (re.compile(r"investigation|lawsuit|fined|\bsec\b|fraud|\bprobe\b|antitrust|regulator", re.I), "REGULATORY_LEGAL"),
    (re.compile(r"recall|outage|cyber|layoff|furlough|shutdown|plant clos|grounded", re.I), "OPERATIONAL"),
    (re.compile(r"earnings|\beps\b|guidance|quarterly results|revenue", re.I), "EARNINGS_GUIDANCE"),
    (re.compile(r"\blaunch|unveil|debut|new product|approval", re.I), "PRODUCT_LAUNCH"),
    # routine analyst chatter: never an event (checked last so it can outvote the cue words above)
    (re.compile(r"price target|maintains|reiterat|initiates|outperform|underperform|market perform|overweight|"
                r"underweight|\banalyst|top gainers|premarket|upgrades? .{0,40}to (buy|hold|neutral)|"
                r"downgrades? .{0,40}to (hold|neutral|sell)", re.I), "OTHER"),
]
RULE_BONUS = 0.10
OTHER_BONUS = 0.28
CLASS_TEMP = 0.06


class EventClassifier:
    """Prototype (nearest-centroid) classifier on the shared embedding, nudged by high-precision cue words."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        flat = [(lab, s) for lab in LABELS for s in EXEMPLARS[lab]]
        vecs = embedder.embed([s for _, s in flat])
        self.protos = np.zeros((len(LABELS), vecs.shape[1]), dtype=np.float32)
        for i, lab in enumerate(LABELS):
            rows = [j for j, (l, _) in enumerate(flat) if l == lab]
            c = vecs[rows].mean(axis=0)
            self.protos[i] = c / np.linalg.norm(c)

    def probs(self, emb: np.ndarray, texts: list[str]) -> np.ndarray:
        """Posterior over LABELS (last = OTHER) for each row."""
        logits = emb @ self.protos.T
        bonus = np.zeros_like(logits)
        for i, t in enumerate(texts):
            for rx, lab in _RULES:
                if rx.search(t):
                    bonus[i, LABELS.index(lab)] += OTHER_BONUS if lab == "OTHER" else RULE_BONUS
        z = (logits + bonus) / CLASS_TEMP
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)


# =================================================================================================
# Persistent per-document cache (embeddings + sentiment), so warm starts and replays are instant
# =================================================================================================
class DocCache:
    def __init__(self, path: Path, fingerprint: str):
        self.path = path
        self.fp = fingerprint
        self.emb: dict[str, np.ndarray] = {}
        self.sent: dict[str, tuple[float, str]] = {}
        if path.exists():
            z = np.load(path, allow_pickle=True)
            if str(z["fp"]) == fingerprint:
                ids = list(z["ids"])
                self.emb = dict(zip(ids, z["emb"]))
                self.sent = {i: (float(s), str(t)) for i, s, t in zip(ids, z["sent"], z["tier"])}

    def save(self) -> None:
        ids = [i for i in self.emb if i in self.sent]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.path, fp=self.fp, ids=np.array(ids, dtype=object),
            emb=np.stack([self.emb[i] for i in ids]) if ids else np.zeros((0, 1)),
            sent=np.array([self.sent[i][0] for i in ids], dtype=np.float32),
            tier=np.array([self.sent[i][1] for i in ids], dtype=object),
        )
