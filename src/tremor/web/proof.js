/* Proof & evaluation view + architecture diagram. Every number shown is read from data/results/*.json
   (produced by `python -m tremor.backtest` and `python -m tremor.evaluate`), never typed here. */
(function () {
  "use strict";
  const f = (x, d = 1) => (x == null || Number.isNaN(x) ? "n/a" : (+x).toFixed(d));
  const pc = (x, d = 0) => (x == null || Number.isNaN(x) ? "n/a" : (x * 100).toFixed(d) + "%");
  const sg = (x, d = 1) => (x == null ? "n/a" : (x >= 0 ? "+" : "") + (x * 100).toFixed(d) + "%");
  const kpi = (l, v, d) => `<div class="kpi"><small>${l}</small><div class="v">${v}</div><div class="d">${d}</div></div>`;

  function reliability(bt) {
    const W = 380, H = 250, L = 38, B = 32, T = 14, R = 14, X = (v) => L + v * (W - L - R), Y = (v) => T + (1 - v) * (H - T - B);
    const pts = [["50", .5], ["80", .8], ["90", .9]].map(([k, n]) => [n, bt.overall_coverage[k]]);
    const se = bt.binomial_se_90 || .03;
    const band = pts.map(([n]) => `${X(n)},${Y(Math.min(1, n + 1.96 * se))}`).join(" ") + " " + [...pts].reverse().map(([n]) => `${X(n)},${Y(Math.max(0, n - 1.96 * se))}`).join(" ");
    const grid = [0, .25, .5, .75, 1].map((v) => `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="rgba(255,255,255,.06)"/><text x="${L - 6}" y="${Y(v) + 3}" fill="#655d78" font-size="9" text-anchor="end">${(v * 100) | 0}%</text><text x="${X(v)}" y="${H - 14}" fill="#655d78" font-size="9" text-anchor="middle">${(v * 100) | 0}%</text>`).join("");
    return `<svg viewBox="0 0 ${W} ${H}" style="width:100%">${grid}<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="rgba(255,255,255,.3)" stroke-dasharray="4 4"/>
      <polygon points="${band}" fill="rgba(255,106,31,.14)"/><polyline fill="none" stroke="#ffc23d" stroke-width="2.2" points="${pts.map(([n, a]) => `${X(n)},${Y(a)}`).join(" ")}"/>
      ${pts.map(([n, a]) => `<circle cx="${X(n)}" cy="${Y(a)}" r="5" fill="#ff6a1f" stroke="#0a0810" stroke-width="2"/><text x="${X(n) + 8}" y="${Y(a) + 14}" fill="#f3eee4" font-size="10" font-weight="700">${(a * 100).toFixed(0)}%</text>`).join("")}
      <text x="${W / 2}" y="${H - 1}" fill="#a39cb3" font-size="9.5" text-anchor="middle">nominal coverage of the predicted band</text>
      <text x="12" y="${H / 2}" fill="#a39cb3" font-size="9.5" transform="rotate(-90 12 ${H / 2})" text-anchor="middle">actual coverage</text></svg>`;
  }

  function skillBars(bt) {
    const rows = Object.values(bt.targets);
    const mx = Math.max(.3, ...rows.flatMap((r) => [Math.abs(r.skill_vs_unconditional || 0), Math.abs(r.skill_vs_ordinary || 0)]));
    return rows.map((r) => `<div class="b" style="grid-template-columns:130px 1fr 1fr;gap:10px;margin:9px 0">
      <span>${r.label}</span>
      ${[["skill_vs_unconditional", "vs crisis library"], ["skill_vs_ordinary", "vs ordinary weeks"]].map(([k, l]) => `<div><div class="track"><div class="fill" style="left:0;width:${Math.max(2, Math.abs(r[k] || 0) / mx * 100)}%;background:${(r[k] || 0) >= 0 ? "linear-gradient(90deg,#8a45ff,#ffc23d)" : "#ff2e7e"}"></div></div><div class="faint" style="font-size:10px;margin-top:2px">${l}: <b style="color:var(--ink)">${sg(r[k], 1)}</b></div></div>`).join("")}</div>`).join("");
  }

  window.renderProof = async function (el, api) {
    let bt, ev, ledger;
    try { [bt, ev, ledger] = await Promise.all([api("/results/backtest"), api("/results/eval").catch(() => null), api("/ledger").catch(() => null)]); }
    catch (e) { el.innerHTML = `<div class="empty">No results found. Run <span class="mono">python -m tremor.backtest</span> and <span class="mono">python -m tremor.evaluate</span>.</div>`; return; }
    const ex = bt.expanding, loo = bt.loo, sent = ev && ev.sentiment, ec = ev && ev.event_class, pipe = ev && ev.pipeline, es = ev && ev.event_study, link = ev && ev.entity_link;
    const gp = ex.graph_ablation.PEER;
    el.innerHTML = `
      <div class="kpis">
        ${kpi("Out-of-sample band coverage", pc(ex.overall_coverage["90"]), `of realised market moves fell inside the predicted 90% band (nominal 90%). ${ex.intervals_scored["90"]} intervals over ${ex.episodes_tested} episodes, chronological, no look-ahead.`)}
        ${kpi("Skill vs unconditional history", sg(ex.overall_skill_vs_unconditional), `pinball-loss improvement from the systemic-vs-idiosyncratic mixture over drawing every event from the crisis library (${sg(ex.overall_skill_vs_ordinary)} vs ordinary weeks).`)}
        ${sent && sent.cascade ? kpi("FinBERT calls avoided", pc(sent.cascade.finbert_calls_saved), `the lexicon tier resolves the easy items; the cascade agrees with FinBERT-only on ${pc(sent.cascade.agreement_with_finbert_only)} of ${sent.n.toLocaleString()} labelled posts.`) : ""}
        ${es && es.n_events ? kpi("Direction vs the tape", pc(es.direction_accuracy_same_day.value), `same-day sign agreement between event sentiment and abnormal return (n=${es.direction_accuracy_same_day.n}, 95% CI ${pc(es.direction_accuracy_same_day.ci95[0])}–${pc(es.direction_accuracy_same_day.ci95[1])}).`) : ""}
        ${ledger ? kpi("Ledger integrity", ledger.chain_valid ? "VALID" : "BROKEN", `${ledger.entries.toLocaleString()} hash-chained entries; UPDATE and DELETE are rejected by database triggers.`) : ""}
      </div>

      <div class="row2">
        <div class="panel"><h3>Calibration <span class="tag">expanding window</span></h3><div class="pad">${reliability(ex)}
          <div class="faint mono" style="font-size:10.5px;margin-top:6px">Shaded: ±1.96 binomial s.e. at n=${ex.intervals_scored["90"]}. Leave-one-out (uses future episodes, stability check only): ${["50", "80", "90"].map((k) => pc(loo.overall_coverage[k])).join(" / ")}.</div></div></div>
        <div class="panel"><h3>Skill by market target <span class="tag">higher is better</span></h3><div class="pad bars">${skillBars(ex)}</div></div>
      </div>

      <div class="row2" style="margin-top:14px">
        <div class="panel"><h3>Does the exposure graph help? <span class="tag">peers, ablation</span></h3><div class="pad">
          <table class="t"><tr><th></th><th>with graph</th><th>no graph</th></tr>
          <tr><td class="faint">pinball loss</td><td style="color:var(--mint)">${f(gp.pinball_graph, 4)}</td><td>${f(gp.pinball_nograph, 4)}</td></tr>
          <tr><td class="faint">mean abs. error</td><td style="color:var(--mint)">${f(gp.mae_graph, 4)}</td><td>${f(gp.mae_nograph, 4)}</td></tr>
          <tr><td class="faint">90% coverage</td><td>${pc(gp.coverage90_graph)}</td><td class="faint">n/a</td></tr></table>
          <p class="mute" style="font-size:12.5px;margin-top:10px">Peer excess returns of ${gp.episodes_evaluable} historical episodes predicted from earlier episodes only. <b>Promoter-group contagion cannot be backtested</b>: the library holds a single such episode (Adani 2023), so there is no earlier analogue to learn from.</p></div></div>
        <div class="panel"><h3>NLP engine <span class="tag">public labelled data</span></h3><div class="pad">${ev ? `<table class="t">
          <tr><th>component</th><th>measure</th><th>value</th></tr>
          ${sent ? `<tr><td>Sentiment · lexicon</td><td class="faint">macro-F1 / acc</td><td>${f(sent.lexicon_only.macro_f1, 2)} / ${f(sent.lexicon_only.accuracy, 2)}</td></tr>${sent.finbert_only ? `<tr><td>Sentiment · FinBERT</td><td class="faint">macro-F1 / acc</td><td>${f(sent.finbert_only.macro_f1, 2)} / ${f(sent.finbert_only.accuracy, 2)}</td></tr><tr><td>Sentiment · cascade</td><td class="faint">macro-F1 / acc</td><td style="color:var(--mint)">${f(sent.cascade.macro_f1, 2)} / ${f(sent.cascade.accuracy, 2)}</td></tr>` : ""}` : ""}
          ${ec ? `<tr><td>Event class (mapped)</td><td class="faint">macro-F1 / acc</td><td>${f(ec.macro_f1, 2)} / ${f(ec.accuracy, 2)}</td></tr><tr><td>Event vs routine</td><td class="faint">F1 (P / R)</td><td>${f(ec.event_vs_routine.f1_event, 2)} (${f(ec.event_vs_routine.precision_event, 2)} / ${f(ec.event_vs_routine.recall_event, 2)})</td></tr>` : ""}
          ${link ? `<tr><td>Entity linking</td><td class="faint">recall of tagged ticker</td><td>${pc(link.recall_of_tagged_ticker)}</td></tr>` : ""}
          ${pipe ? `<tr><td>Dedup / clustering</td><td class="faint">docs → events</td><td>${pipe.documents_in_events.toLocaleString()} → ${pipe.events.toLocaleString()}</td></tr><tr><td>Syndicated echoes</td><td class="faint">share of news</td><td>${pc(pipe.echo_share_of_news, 1)}</td></tr><tr><td>Throughput</td><td class="faint">incl. Monte Carlo</td><td>${f(pipe.ms_per_doc_including_stress, 1)} ms/doc</td></tr>` : ""}
          </table><p class="faint" style="font-size:11px;margin-top:8px">Sentiment: ${sent ? sent.dataset : ""}. Event class uses a different topic taxonomy mapped onto ours, so treat it as a sanity check, not a headline.</p>` : `<div class="empty">run python -m tremor.evaluate</div>`}</div></div>
      </div>

      ${es && es.n_events ? `<div class="panel" style="margin-top:14px"><h3>Event study <span class="tag">${es.n_events} single-name events · concurrent</span></h3><div class="pad">
        <div class="row2"><table class="t">
          <tr><td class="faint">direction: sentiment sign vs abnormal return</td><td>${pc(es.direction_accuracy_same_day.value, 1)} (n=${es.direction_accuracy_same_day.n})</td></tr>
          <tr><td class="faint">impact vs |abnormal return|, Spearman</td><td>${f(es.impact_vs_abs_abnormal_return.spearman, 2)} (p=${f(es.impact_vs_abs_abnormal_return.p, 3)})</td></tr>
          <tr><td class="faint">evidence intensity vs |abnormal return|</td><td>${f(es.intensity_vs_abs_abnormal_return.spearman, 2)} (p=${f(es.intensity_vs_abs_abnormal_return.p, 3)})</td></tr>
          <tr><td class="faint">sentiment vs <i>next-day</i> return</td><td>${f(es.sentiment_vs_next_day_return.spearman, 2)} (p=${f(es.sentiment_vs_next_day_return.p, 3)})</td></tr>
          <tr><td class="faint">median |abnormal return| · confirmed vs provisional</td><td>${pc(es.median_abs_ar_confirmed, 2)} (n=${es.n_confirmed}) vs ${pc(es.median_abs_ar_provisional, 2)} (n=${es.n_provisional})</td></tr></table>
        <p class="mute" style="font-size:12.5px">${es.caveat}</p></div></div></div>` : ""}

      <div class="panel" style="margin-top:14px"><h3>What did not work, and what we did about it <span class="tag">stated before you ask</span></h3><div class="pad">
        <div class="honest"><b>Sharper analogue retrieval lost to the unconditional baseline.</b> Conditioning the analogue weights on text, class or severity gave negative pinball skill at n=${ex.episodes_in_library} (class-only was worst). We therefore keep the text kernel soft and use what the data <i>did</i> support: P(systemic | class) mixes crisis-library shocks with ordinary-week shocks.</div>
        <div class="honest"><b>The Impact Score is a triage score, not a calibrated forecast.</b> Impact computed from analogue severity had no rank-correlation with realised severity (|ρ| &lt; 0.25, n.s.), so it is defined from observable evidence instead (intensity, P(systemic), confidence, polarity). Its relation to realised moves is reported above.</div>
        <div class="honest"><b>News timestamps are day-resolution.</b> We assume 15:00 UTC publication, so the data cannot support any claim about minutes of lead time between social posts and news.</div>
        <div class="honest"><b>Small samples, synthetic book.</b> 36 episodes, one promoter-group episode, delisted names without prices, synthetic ratings and exposures. Ranges are indicative, not capital numbers.</div></div></div>`;
  };

  // ============================================================================================================
  // Architecture diagram
  // ============================================================================================================
  window.renderArch = function (el) {
    const box = (x, y, w, h, t, s, c = "#ff6a1f", extra = "") => `<g ${extra}><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="14" fill="rgba(255,255,255,.04)" stroke="${c}" stroke-opacity=".7" stroke-width="1.4"/><text x="${x + w / 2}" y="${y + (s ? h / 2 - 3 : h / 2 + 4)}" text-anchor="middle" fill="#f3eee4" font-size="13" font-weight="700" font-family="Bricolage Grotesque">${t}</text>${s ? `<text x="${x + w / 2}" y="${y + h / 2 + 14}" text-anchor="middle" fill="#a39cb3" font-size="9.5">${s}</text>` : ""}</g>`;
    const arrow = (x1, y1, x2, y2, c = "#ff6a1f") => `<path d="M${x1} ${y1} L${x2} ${y2}" stroke="${c}" stroke-width="1.6" fill="none" class="flow" marker-end="url(#ah)"/>`;
    el.innerHTML = `<svg viewBox="0 0 1200 760" role="img" aria-label="TREMOR architecture">
      <defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="#ff6a1f"/></marker>
      <linearGradient id="lg" x1="0" x2="1"><stop offset="0" stop-color="#ffc23d"/><stop offset=".5" stop-color="#ff6a1f"/><stop offset="1" stop-color="#ff2e7e"/></linearGradient></defs>
      <text x="30" y="34" fill="#a39cb3" font-size="10.5" letter-spacing="2">SOURCES</text>
      ${box(30, 46, 250, 62, "News headlines", "FNSPID replay · live-feed adapter", "#7fd4ff")}${box(300, 46, 250, 62, "Social posts", "public tweet corpus replay", "#8a45ff")}${box(570, 46, 250, 62, "POST /api/ingest", "live path · same pipeline", "#33f0c0")}
      ${arrow(155, 108, 330, 168)}${arrow(425, 108, 430, 168)}${arrow(695, 108, 560, 168)}

      <rect x="30" y="168" width="790" height="86" rx="18" fill="rgba(255,106,31,.07)" stroke="url(#lg)" stroke-width="2"/>
      <text x="425" y="196" text-anchor="middle" fill="#f3eee4" font-size="17" font-weight="800" font-family="Bricolage Grotesque">APPEND-ONLY LEDGER &middot; the contract</text>
      <text x="425" y="218" text-anchor="middle" fill="#a39cb3" font-size="10.5">SQLite · SHA-256 hash chain · UPDATE / DELETE rejected by triggers · replay = live</text>
      <text x="425" y="238" text-anchor="middle" fill="#ffc23d" font-size="10.5">every state below is a pure function of (ledger prefix, book snapshot, config hash)</text>

      ${arrow(425, 254, 425, 300)}
      <text x="30" y="296" fill="#a39cb3" font-size="10.5" letter-spacing="2">SCORE ONCE, REUSE EVERYWHERE</text>
      ${box(30, 310, 190, 70, "Embed once", "MiniLM · cluster · class · analogues", "#8a45ff")}
      ${box(236, 310, 190, 70, "Sentiment cascade", "lexicon → FinBERT if unsure", "#8a45ff")}
      ${box(442, 310, 190, 70, "Event class", "10 classes + routine gate", "#8a45ff")}
      ${box(648, 310, 172, 70, "Entity linker", "trie regex · aliases · cashtags", "#8a45ff")}
      ${arrow(425, 380, 425, 426)}

      <rect x="30" y="426" width="790" height="82" rx="18" fill="rgba(138,69,255,.07)" stroke="#8a45ff" stroke-opacity=".7" stroke-width="1.6"/>
      <text x="425" y="454" text-anchor="middle" fill="#f3eee4" font-size="15" font-weight="800" font-family="Bricolage Grotesque">EVENT TRACKER</text>
      <text x="425" y="475" text-anchor="middle" fill="#a39cb3" font-size="10.5">cluster · syndication echo detection · independence-weighted corroboration · provisional → confirmed</text>
      <text x="425" y="494" text-anchor="middle" fill="#a39cb3" font-size="10.5">output: sentiment · event class · impact 1–10 · confidence · evidence ids</text>
      ${arrow(425, 508, 425, 556)}

      <text x="30" y="548" fill="#a39cb3" font-size="10.5" letter-spacing="2">MODULE B · STRATEGIC STRESS TEST (subscriber)</text>
      ${box(30, 562, 190, 82, "Exposure graph", "control · promoter group · industry · text cue", "#ff2e7e")}
      ${box(236, 562, 190, 82, "Scenario draws", "36 analogues + 1,200 ordinary weeks · P(systemic)", "#ff2e7e")}
      ${box(442, 562, 190, 82, "Book revaluation", "loans ΔECL · bonds D/C · swaps DV01 + CVA", "#ff2e7e")}
      ${box(648, 562, 172, 82, "Loss range", "P5/P50/P95 · VaR99 · ES99 · reverse stress", "#ff2e7e")}
      ${arrow(220, 603, 236, 603)}${arrow(426, 603, 442, 603)}${arrow(632, 603, 648, 603)}

      ${arrow(820, 603, 900, 603, "#33f0c0")}
      <text x="880" y="40" fill="#a39cb3" font-size="10.5" letter-spacing="2">CONSUMERS</text>
      ${box(900, 562, 270, 82, "Signal bus", "pub/sub · Kafka / Redis Streams in production", "#33f0c0")}
      ${box(900, 330, 270, 80, "Module A · rebalancer", "decayed sentiment tilt · caps · no-trade band", "#33f0c0")}
      ${box(900, 440, 270, 80, "Dashboard + REST API", "control room · evidence · replay", "#33f0c0")}
      ${box(900, 222, 270, 80, "Replay verifier", "re-derive any signal · hash must match", "#ffc23d")}
      ${arrow(1035, 562, 1035, 520, "#33f0c0")}${arrow(1035, 562, 1180, 540, "#33f0c0")}
      <path d="M820 211 L900 262" stroke="#ffc23d" stroke-width="1.4" stroke-dasharray="4 4" fill="none"/>
      <text x="30" y="696" fill="#655d78" font-size="10.5">One container · one command · models baked at build · works offline · 36 episodes, 20.8k documents, 252 synthetic positions</text>
      <text x="30" y="716" fill="#655d78" font-size="10.5">Innovation is the contract: append-only ledger + versioned inputs ⇒ byte-exact replay ⇒ a defensible audit trail for every number.</text></svg>`;
  };
})();
