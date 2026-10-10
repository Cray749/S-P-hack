# Demo video script (target 4:30, under both the 5-minute and 10-minute limits)

Record at 1440x900, browser zoom 100%, dark room theme is the app's own. Have the container already built, then show the start live.

| Time | Screen | Say (the *what, how, why*) |
|---|---|---|
| 0:00-0:30 | Title card, then the control room | **Problem.** "A headline names one company. A bank's risk is spread across a whole book: subsidiaries, promoter groups, industry peers. Risk teams find out late, and nobody can audit the number. TREMOR turns text into a loss range for a specific book, and proves how it got it." |
| 0:30-1:00 | Terminal: `docker compose up --build`, then open `localhost:8000` | "One container, one command, models baked in, runs offline. Everything you will see is reproducible from the README." |
| 1:00-2:15 | Pick **Crude oil goes negative**, play at 12h/s. Watch the seismograph, feed and map. Click the top oil tremor. | **Engine.** "Two sources, news and social, flow into an append-only, hash-chained ledger. Each item is embedded once, scored by a sentiment cascade (lexicon first, FinBERT only when unsure), classified, and linked to companies. Items are clustered into events. Notice the status: provisional until independent publishers agree; syndicated copies and tweet volume never count as confirmation." |
| 2:15-3:15 | Same tremor: the loss fan, range-over-time, "Obligor paths" tab | **Module B.** "Impact, event class and sentiment come out. Module B maps the event to the book: here are the obligors the headline never named, and the path to each. Scenario draws come from 36 documented historical episodes. The range is the spread of what actually happened, mixed by P(systemic): single-name events don't inherit crisis-size market moves. The loan, bond, swap and CVA split shows what drives the loss. Reverse stress tells us how large a shock breaks a limit." |
| 3:15-3:55 | Switch to **Hertz bankruptcy warning**, open it, then **Replay & integrity -> Replay signal** | **Trust.** "Every number is a pure function of the ledger, the book and the config. Replay throws away all state, re-reads the ledger, re-clusters, re-runs the Monte Carlo and compares hashes: identical." |
| 3:55-4:20 | **Proof & evaluation** tab | **Honesty.** "We backtested out of sample. The mixture beats the unconditional history; sharper retrieval did not, so we say so. Coverage and the graph ablation are on screen. Limitations are in the repo." |
| 4:20-4:30 | **Live wire**: post a headline | "Anything posted goes through the same pipeline. Module A, the rebalancer, is a second subscriber added without touching the engine." |

## Pre-flight checklist

* Container built and `docker compose up` shows `Application startup complete`.
* Run the oil scenario once beforehand to confirm the selected tremor and numbers you will click.
* Replay button: confirm it prints `IDENTICAL`.
* Upload to YouTube as **Unlisted**, test in an incognito window, paste the link in README.
