# Evaluation methodology

Everything is reproducible: `python -m tremor.backtest` writes `data/results/backtest.json`; `python -m tremor.evaluate`
writes `data/results/eval.json`. The dashboard's **Proof** tab renders those files directly.

## 1. Analogue / scenario model (backtest)

* **Design.** 36 historical episodes sorted by date. For episode *j* the model sees only episodes whose outcome window closed before *j*
  (expanding window, no look-ahead) and only the episode's one-line description. Class is predicted by our classifier, not taken from the label.
* **Targets.** S&P 500, VIX, 10y yield, HY-spread proxy and Brent, measured as the peak excursion within five trading days.
* **Metrics.** Coverage of the central 50/80/90% bands; pinball loss normalised by each target's standard deviation.
* **Baselines.** (a) unconditional crisis library (equal weights over eligible episodes), (b) ordinary weeks only.
* **Result (30 tested episodes, 150 intervals).** Coverage 45% / 74% / 85% (nominal 50/80/90, binomial s.e. ≈ 2.4% at 90%).
  Skill vs unconditional crisis library +7.5%, vs ordinary weeks +18.3%; positive on every target (10y yield ≈ 0).
* **Pre-registration.** `config.SYSTEMIC_PRIOR` was fixed before the mixture was evaluated. Earlier ablations (class-only, text-only at several
  temperatures, severity kernel, first-day-reaction matching) showed no skill over the unconditional baseline and are why retrieval stays soft.
* **Graph ablation.** Peer excess returns predicted with role-conditioned draws vs a no-information baseline: pinball 0.0213 vs 0.0232 and MAE 0.0563 vs 0.0595 over 17 episodes.
  Promoter-group contagion has one episode in the library and is not evaluable.
* **Stability check.** Leave-one-out (uses future episodes, so *not* a forecast evaluation) gives similar coverage.

## 2. NLP engine

| Component | Data | Result |
|---|---|---|
| Sentiment (3 classes) | `zeroshot/twitter-financial-news-sentiment` validation, n = 2,388 | lexicon 73.7% acc / F1 0.63; FinBERT 65.5% / 0.62; **cascade 74.8% / 0.63**, FinBERT used on 23.8% of items |
| Event class | `zeroshot/twitter-financial-news-topic` validation (different taxonomy, mapped) | macro-F1 0.44, event-vs-routine F1 0.78 (P 0.69, R 0.91). Treat as a sanity check, not a headline. |
| Entity linking | (a) FNSPID ticker tags, n = 4,000; (b) manual audit, n = 70 | (a) recall 32% (tags are noisy and often name tickers the headline never mentions); (b) 70/70 identities correct, 60/70 on-subject. Audit judged by the author with AI assistance. |
| Pipeline | full replay corpus | 20,836 docs → 11,096 routine/spam filtered, 9,740 clustered into 6,123 events; 1,232 syndicated echoes (32% of news); 957 signals; 5.1 ms/doc including Monte Carlo |

## 3. Event study (concurrent validation, not forecasting)

For events naming exactly one US-listed company with price data (n = 103): abnormal return = stock return − β × S&P 500 return on the event day.

* Impact vs |abnormal return|: Spearman 0.23 (p = 0.022); 0.18 and 0.30 in the first and second half (stable sign).
* Evidence intensity vs |abnormal return|: 0.14 (p = 0.16, not significant).
* Sentiment sign vs same-day abnormal return: 56% (n = 16, 95% CI 32-81%); inconclusive.
* Sentiment vs next-day return: ρ = 0.07 (p = 0.50): no predictive claim.
* Confirmed events (n = 8) have a larger median |abnormal return| than provisional (n = 95), 2.7% vs 2.1%, but n is small.

Caveats: news is day-resolution and social posts partly react to the move, so this shows the engine flags the right names on the right days; it is not alpha.

## 4. Replay guarantee

`tests/test_engine_api.py::test_replay_reproduces_every_signal_hash` re-derives every emitted signal from a fresh pipeline over the ledger prefix and
asserts identical hashes; `tests/test_ledger.py` proves the hash chain detects tampering even if the database triggers are bypassed.
