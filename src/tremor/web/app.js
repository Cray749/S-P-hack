/* TREMOR dashboard logic */
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const api = async (p, o) => { const r = await fetch("/api" + p, o); if (!r.ok) throw new Error((await r.text()).slice(0, 300)); return r.json(); };
  const post = (p, b) => api(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b || {}) });
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const S = { scens: [], scen: null, running: false, speedH: 4, cursor: null, start: 0, end: 0, events: new Map(), selected: null, detail: null, timeline: [], dock: "evidence", busy: false, state: null, expected: [], pulsed: new Set(), minImp: 4.5 };
  let seismo, constel;
  const CLS = { GEOPOLITICAL: "Geopolitical", MACROECONOMIC: "Macroeconomic", CREDIT_EVENT: "Credit event", MERGER_ACQUISITION: "M&A", PRODUCT_LAUNCH: "Product launch", REGULATORY_LEGAL: "Legal / regulatory", OPERATIONAL: "Operational", EARNINGS_GUIDANCE: "Earnings", COMMODITY_SHOCK: "Commodity shock", PANDEMIC_DISASTER: "Pandemic / disaster" };

  function toast(msg, ms = 4200) { const t = document.createElement("div"); t.className = "toast"; t.innerHTML = msg; document.body.appendChild(t); setTimeout(() => t.remove(), ms); }
  const fmtClock = (ms) => new Date(ms).toISOString().replace("T", " ").slice(0, 16);

  // ---------------------------------------------------------------------------------------------------------
  // magnitude ring
  function ring(imp, size = 56, label = true) {
    const r = size / 2 - 5, c = 2 * Math.PI * r, k = Math.max(.05, Math.min(1, imp / 10)), id = "g" + Math.random().toString(36).slice(2, 7);
    return `<svg class="mag" viewBox="0 0 ${size} ${size}" width="${size}" height="${size}"><defs><linearGradient id="${id}" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stop-color="#8a45ff"/><stop offset=".5" stop-color="#ff2e7e"/><stop offset="1" stop-color="#ffc23d"/></linearGradient></defs>
      <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="4.5"/>
      <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="url(#${id})" stroke-width="4.5" stroke-linecap="round" stroke-dasharray="${c * k} ${c}" transform="rotate(-90 ${size / 2} ${size / 2})"/>
      <text x="50%" y="${label ? "52%" : "56%"}" text-anchor="middle" dominant-baseline="middle" fill="#f3eee4" font-size="${size * .3}" font-weight="700" font-family="JetBrains Mono">${imp.toFixed(1)}</text>${label ? `<text x="50%" y="76%" text-anchor="middle" fill="#a39cb3" font-size="6.5" letter-spacing=".12em" font-family="JetBrains Mono">IMPACT</text>` : ""}</svg>`;
  }

  // ---------------------------------------------------------------------------------------------------------
  // feed
  function renderFeed() {
    const all = [...S.events.values()], list = all.filter((s) => s.impact >= S.minImp).sort((a, b) => b.as_of.localeCompare(a.as_of)).slice(0, 60);
    $("#feedcount").textContent = `${list.length} shown · ${all.length} events`;
    const el = $("#feed");
    if (!list.length) { el.innerHTML = `<div class="empty">Waiting for the first events…<br>press play</div>`; return; }
    const seen = new Set($$(".card", el).map((c) => c.dataset.id + ":" + c.dataset.v));
    el.innerHTML = list.map((s) => {
      const tot = Math.max(1, s.n_docs), nn = s.n_news / tot * 100, ns = s.n_social / tot * 100, key = s.event_id + ":" + s.version;
      return `<div class="card ${S.selected === s.event_id ? "sel" : ""}" data-id="${s.event_id}" data-v="${s.version}" style="${seen.has(key) ? "animation:none" : ""}">
        ${ring(s.impact)}
        <div class="cls"><span>${CLS[s.event_class] || s.event_class}</span><span class="faint">${s.as_of.slice(5, 16).replace("T", " ")}</span></div>
        <div class="hl">${esc(s.headline)}</div>
        <div class="row"><span class="pill ${s.status === "confirmed" ? "conf" : "prov"}">${s.status}</span>
          ${s.entities.slice(0, 2).map((e) => `<span class="chip ent">${esc(e)}</span>`).join("")}
          <span class="chip">${s.n_docs} docs · ${s.independent_sources} src</span><span class="loss">${V.fmtM(s.loss.p50, 0)}</span></div>
        <div class="srcbar"><i class="n" style="width:${nn}%"></i><i class="s" style="width:${ns}%"></i></div></div>`;
    }).join("");
  }
  $("#minimp").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; S.minImp = +b.dataset.m; $$("#minimp button").forEach((x) => x.classList.toggle("on", x === b)); renderFeed(); });
  $("#feed").addEventListener("click", (e) => { const c = e.target.closest(".card"); if (c) select(c.dataset.id); });

  // ---------------------------------------------------------------------------------------------------------
  // selection + impact panel
  async function select(eid, quiet) {
    S.selected = eid;
    try {
      const d = await api("/events/" + eid);
      S.detail = d.latest; S.timeline = d.timeline;
    } catch (e) { return; }
    renderFeed(); renderImpact(); constel.setFocus(S.detail); if (!quiet) constel.ping();
    $("#epihint").textContent = S.detail.entities.length ? "epicentre: " + S.detail.entities.join(", ") : "market-wide event · no named entity";
    $("#stagetag").textContent = `${S.detail.n_obligors_hit} of ${S.detail.obligor_map.length} obligors hit`;
    renderDock();
  }

  function bars(c) {
    const items = [["Loans · ΔECL", c.loans_ecl], ["Bonds · MtM", c.bonds_mtm], ["Swaps · MtM", c.swaps_mtm], ["Swaps · CVA", c.swaps_cva]];
    const mx = Math.max(...items.map((i) => Math.abs(i[1])), 1);
    return items.map(([l, v]) => `<div class="b"><span>${l}</span><div class="track"><div class="fill" style="${v >= 0 ? `left:50%;width:${Math.abs(v) / mx * 50}%;background:linear-gradient(90deg,#ff6a1f,#ff2e7e)` : `right:50%;width:${Math.abs(v) / mx * 50}%;background:#33f0c0`}"></div><div style="position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:rgba(255,255,255,.25)"></div></div><span style="text-align:right">${V.fmtM(v, 1)}</span></div>`).join("");
  }

  function renderImpact() {
    const s = S.detail; if (!s) return;
    $("#sigid").textContent = s.signal_id;
    const after = s.stressed_value_usd_m;
    $("#impact").innerHTML = `
      <div style="display:flex;gap:12px;align-items:center;margin-bottom:12px">${ring(s.impact, 64)}
        <div><div style="font-weight:700;font-size:14px;line-height:1.3">${esc(s.headline.slice(0, 110))}</div>
        <div style="margin-top:6px;display:flex;gap:6px;flex-wrap:wrap"><span class="pill ${s.status === "confirmed" ? "conf" : "prov"}">${s.status}</span><span class="chip">${CLS[s.event_class]}</span><span class="chip">conf ${(s.confidence * 100).toFixed(0)}%</span><span class="chip">novelty ${(s.novelty * 100).toFixed(0)}%</span></div></div></div>
      <div class="hero">
        <div class="stat"><small>Book value · before</small><div class="v mono" id="v-before">${V.fmtM(s.book_value_usd_m, 0)}</div><div class="s">${s.n_obligors_hit} obligors hit · ${s.n_obligors_named} named</div></div>
        <div class="stat"><small>Median stressed value</small><div class="v mono" id="v-after">${V.fmtM(after, 0)}</div><div class="s">after ${s.event_class.toLowerCase().replace("_", " ")}</div></div>
        <div class="stat"><small>Median loss · P50</small><div class="v loss mono" id="v-loss" data-v="0">$0.0m</div><div class="s">${V.pct(s.loss.p50 / s.book_value_usd_m, 2)} of book</div></div>
        <div class="stat"><small>Tail · P95 / ES99</small><div class="v mono" style="font-size:19px;line-height:2">${V.fmtM(s.loss.p95, 0)} <span class="faint">/</span> ${V.fmtM(s.loss.es99, 0)}</div><div class="s">VaR99 ${V.fmtM(s.loss.var99, 0)}</div></div>
      </div>
      <div class="sub">Loss distribution <em>${s.ess.toFixed(1)} effective analogues · ${s.shock.p_systemic != null ? (s.shock.p_systemic * 100).toFixed(0) : "--"}% systemic</em></div>
      <div class="range" id="fan">${V.lossFan(s)}</div>
      <div class="sub">Range over time <em>${S.timeline.length} versions · hollow = provisional</em></div>
      <div class="range">${V.rangeTimeline(S.timeline)}</div>
      <div class="sub">What drives the loss <em>median, USD m</em></div><div class="bars">${bars(s.components)}</div>
      <div class="sub">Typical shock behind it <em>from analogue draws</em></div>
      <div class="mono" style="font-size:11.5px;color:var(--mute);line-height:1.8">equity <b style="color:var(--ink)">${s.shock.equity_pct.toFixed(1)}%</b> &nbsp; 10y <b style="color:var(--ink)">${s.shock.rates_10y_bps >= 0 ? "+" : ""}${s.shock.rates_10y_bps.toFixed(0)}bp</b> &nbsp; HY spread <b style="color:var(--ink)">${s.shock.hy_spread_bps >= 0 ? "+" : ""}${s.shock.hy_spread_bps.toFixed(0)}bp</b> &nbsp; oil <b style="color:var(--ink)">${s.shock.oil_pct.toFixed(1)}%</b></div>
      <div class="sub">Reverse stress <em>what would break the limit?</em></div>
      <div class="slider"><span>limit</span><input type="range" id="lim" min="0.5" max="10" step="0.5" value="2"><b id="limv" class="mono" style="color:var(--ink)">2.0% of book</b></div>
      <div class="callout" id="rev">computing…</div>`;
    V.countTo($("#v-loss"), s.loss.p50, (v) => V.fmtM(v, 1), 800);
    const lim = $("#lim"); let to; lim.oninput = () => { $("#limv").textContent = (+lim.value).toFixed(1) + "% of book"; clearTimeout(to); to = setTimeout(reverse, 250); };
    reverse();
  }
  async function reverse() {
    const s = S.detail; if (!s || !$("#lim")) return;
    try {
      const lim = +$("#lim").value, r = await api(`/signals/${s.signal_id}/reverse-stress?limit_pct=${lim}`);
      const el = $("#rev"); if (!el || S.detail.signal_id !== s.signal_id) return;
      el.innerHTML = r.breach_scale == null
        ? `Even <b>8×</b> the typical historical shock keeps the median loss under <b>${V.fmtM(r.limit_usd_m, 0)}</b> (${lim}% of book). The limit is not threatened by this event class.`
        : r.breach_scale <= 1 ? `The median loss already exceeds the <b>${V.fmtM(r.limit_usd_m, 0)}</b> limit at the typical shock size. <b>Breach.</b>`
          : `The median loss reaches the <b>${V.fmtM(r.limit_usd_m, 0)}</b> limit (${lim}% of book) at about <b>${r.breach_scale.toFixed(1)}×</b> the typical historical shock for this kind of event.`;
    } catch (e) { /* ignore */ }
  }

  // ---------------------------------------------------------------------------------------------------------
  // dock
  $("#dockbar").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; S.dock = b.dataset.d; $$("#dockbar button").forEach((x) => x.classList.toggle("on", x === b)); renderDock(); });
  const sentCol = (v) => v > .15 ? ["rgba(51,240,192,.14)", "#33f0c0"] : v < -.15 ? ["rgba(255,46,126,.16)", "#ff6a9a"] : ["rgba(255,255,255,.07)", "#a39cb3"];
  async function renderDock() {
    const d = $("#dock"), s = S.detail;
    if (S.dock === "trigger") {
      const log = await api("/module-b/log").catch(() => []);
      d.innerHTML = `<p class="mute" style="font-size:13px;margin-bottom:10px">Module B subscribes to the signal bus and logs every event that crosses its policy: <b>impact ≥ 6</b> or <b>P95 loss ≥ 0.5% of book</b>. The stress itself is attached to each signal.</p>` + (log.length ? `<table class="t"><tr><th>time</th><th>event</th><th>class</th><th>impact</th><th>status</th><th>P50 / P95 loss</th><th>trigger</th></tr>${log.slice().reverse().slice(0, 18).map((t) => `<tr><td>${t.as_of.slice(5, 16).replace("T", " ")}</td><td>${t.signal_id}</td><td>${CLS[t.event_class]}</td><td>${t.impact.toFixed(1)}</td><td>${t.status}</td><td>${V.fmtM(t.loss_p50, 0)} / ${V.fmtM(t.loss_p95, 0)}</td><td class="faint">${t.reason}</td></tr>`).join("")}</table>` : `<div class="empty">No stress triggers yet in this replay.</div>`);
      return;
    }
    if (!s) { d.innerHTML = `<div class="empty">Select a tremor to inspect its evidence and lineage.</div>`; return; }
    if (S.dock === "evidence") {
      d.innerHTML = `<div class="cols2"><div><div class="sub" style="margin-top:0">Source documents <em>${s.evidence.length} of ${s.n_docs} · ${s.independent_sources} independent source${s.independent_sources === 1 ? "" : "s"} · social capped at 0.5 corroboration</em></div>
        ${s.evidence.map((e) => { const [bg, fg] = sentCol(e.sentiment); return `<div class="ev ${e.echo_of ? "echo" : ""}"><div class="ic ${e.source === "news" ? "n" : "s"}">${e.source === "news" ? "N" : "S"}</div><div><div class="t">${esc(e.text.slice(0, 190))}</div><div class="m">${esc(e.publisher)} · ${e.ts.slice(5, 16).replace("T", " ")}${e.echo_of ? " · <b style='color:#ffb02e'>ECHO – syndicated copy, not independent</b>" : ""}</div></div><span class="sent" style="background:${bg};color:${fg}">${e.sentiment >= 0 ? "+" : ""}${e.sentiment.toFixed(2)}<small>${e.tier === "finbert" ? "FIN" : "LEX"}</small></span></div>`; }).join("")}</div>
        <div><div class="sub" style="margin-top:0">Historical analogues <em>posterior weight · text similarity</em></div>
        ${s.analogues.map((a) => `<div class="an"><div><b>${esc(a.title)}</b><div class="d">${a.date} · ${CLS[a.cls] || a.cls}</div></div><div><div class="meter"><i style="width:${Math.min(100, a.weight * 100 * 4)}%"></i></div><div class="d" style="text-align:right">${(a.weight * 100).toFixed(1)}% · sim ${a.similarity.toFixed(2)}</div></div></div>`).join("")}
        <div class="honest" style="margin-top:12px"><b>Honest note.</b> Text conditioning of the analogue weights is kept deliberately soft: in the backtest it did not beat the unconditional historical spread. What text <i>does</i> reliably tell us is systemic vs idiosyncratic, which is applied in the mixture.</div></div></div>`;
    } else if (S.dock === "replay") {
      d.innerHTML = `<div class="cols2"><div><div class="big">Re-derive it. <span style="color:var(--m2)">Byte for byte.</span></div>
        <p class="mute" style="margin:10px 0 14px;max-width:560px">A signal is a pure function of three inputs: the ledger prefix up to <span class="mono">offset ${s.ledger_offset.toLocaleString()}</span>, the book snapshot and the config. This button throws away all engine state, re-reads the ledger from entry 1, re-clusters every document, re-runs the Monte Carlo with the same seed, and compares hashes.</p>
        <div style="display:flex;gap:10px"><button class="btn" id="rp">Replay signal ${s.signal_id}</button></div><div id="rpout"></div></div>
        <div><div class="sub" style="margin-top:0">Provenance stamped on this signal</div>
        <table class="t"><tr><td class="faint">ledger offset</td><td>${s.ledger_offset}</td></tr><tr><td class="faint">book snapshot</td><td>${s.book_hash}</td></tr><tr><td class="faint">config hash</td><td>${s.config_hash}</td></tr><tr><td class="faint">signal hash</td><td>${s.hash}</td></tr><tr><td class="faint">MC seed</td><td>sha256(event | offset | config | book)</td></tr></table></div></div>`;
      $("#rp").onclick = replay;
    } else if (S.dock === "paths") {
      d.innerHTML = `<div class="sub" style="margin-top:0">Obligors the headline never named <em>${s.obligors_hit.filter((h) => !s.entities.includes(h.entity_id)).length} of ${s.n_obligors_hit} hit</em></div>
        <table class="t"><tr><th>obligor</th><th>role</th><th>propagation path</th><th>exposure</th><th>P50 loss</th><th>P95</th></tr>${s.obligors_hit.slice(0, 16).map((h) => `<tr><td><b style="font-family:var(--display)">${esc(h.name)}</b></td><td>${h.edge.replace("_", " ").toLowerCase()}</td><td class="faint">${esc(h.path.join("  →  "))}</td><td>${V.fmtM(h.exposure_usd_m, 0)}</td><td style="color:var(--m1)">${V.fmtM(h.expected_loss_usd_m, 1)}</td><td>${V.fmtM(h.loss_p95_usd_m, 1)}</td></tr>`).join("")}</table>`;
    }
  }
  async function replay() {
    const s = S.detail, out = $("#rpout"); out.innerHTML = `<div class="hashbox" style="margin-top:14px">re-reading ledger entries 1…${s.ledger_offset.toLocaleString()} · re-clustering · re-running Monte Carlo<span class="mono" id="dots">…</span></div>`;
    let k = 0; const iv = setInterval(() => { const d = $("#dots"); if (d) d.textContent = ".".repeat(++k % 4 + 1); }, 300);
    try {
      const r = await post(`/signals/${s.signal_id}/replay`); clearInterval(iv);
      out.innerHTML = `<div class="verify"><div class="hashbox"><span class="faint">original</span><br>${r.original_hash}</div><div class="hashbox"><span class="faint">replayed</span><br><span class="${r.match ? "ok" : "bad"}">${r.replayed_hash}</span></div></div>
        <div class="hashbox" style="margin-top:12px;font-size:13px"><span class="${r.match ? "ok" : "bad"}">${r.match ? "✔ IDENTICAL" : "✘ MISMATCH"}</span> &nbsp; ${r.ledger_entries_replayed.toLocaleString()} ledger entries replayed in ${r.seconds}s · hash chain ${r.ledger_chain_valid ? '<span class="ok">valid</span>' : '<span class="bad">BROKEN</span>'}</div>`;
    } catch (e) { clearInterval(iv); out.innerHTML = `<div class="hashbox bad">${esc(e.message)}</div>`; }
  }

  // ---------------------------------------------------------------------------------------------------------
  // scenarios + playback
  function renderScen() {
    $("#scen").innerHTML = S.scens.map((s) => `<button data-id="${s.id}" class="${S.scen === s.id ? "on" : ""}">${esc(s.title)}<small>${s.start.slice(5)} → ${s.end.slice(5)} · ${s.n_docs.toLocaleString()} docs</small></button>`).join("");
  }
  $("#scen").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) startScenario(b.dataset.id); });
  async function startScenario(id) {
    S.running = false; setPlay(); S.busy = true;
    toast("Warming up from the ledger: re-clustering everything before this window…", 2800);
    try {
      const r = await post("/session/start", { scenario: id });
      S.scen = id; S.events.clear(); S.selected = null; S.detail = null; S.timeline = []; constel.clear(); $("#impact").innerHTML = `<div class="empty">Select a tremor to see the loss range, its drivers and what would break the limit.</div>`; $("#sigid").textContent = "--";
      S.cursor = Date.parse(r.start_ts); S.start = S.cursor; S.end = Date.parse(r.end_ts); $("#clock").textContent = fmtClock(S.cursor);
      const sc = S.scens.find((x) => x.id === id); S.expected = sc.expected_entities || [];
      renderScen(); renderFeed(); renderDock(); S.running = true; setPlay();
    } catch (e) { toast("Could not start: " + esc(e.message)); }
    S.busy = false;
  }
  function setPlay() { $("#play").innerHTML = S.running ? "&#10074;&#10074;" : "&#9654;"; seismo.live = S.running; }
  $("#play").onclick = () => { if (!S.scen) { startScenario(S.scens[0].id); return; } S.running = !S.running; setPlay(); };
  $("#speed").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; S.speedH = +b.dataset.h; $$("#speed button").forEach((x) => x.classList.toggle("on", x === b)); });
  document.addEventListener("keydown", (e) => { if (e.code === "Space" && !/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) { e.preventDefault(); $("#play").click(); } });

  async function tick() {
    if (!S.running || S.busy || !S.scen) return;
    if (S.cursor >= S.end) { S.running = false; setPlay(); toast("Replay window finished. Pick any tremor and press <b>Replay signal</b> to re-derive it from the ledger."); return; }
    S.busy = true;
    try {
      S.cursor = Math.min(S.end, S.cursor + S.speedH * 3600e3 * 0.9);
      const r = await post("/session/advance", { to_ts: new Date(S.cursor).toISOString() });
      $("#clock").textContent = fmtClock(S.cursor);
      let selUpdated = false, best = null;
      for (const s of r.new_signals) {
        const prev = S.events.get(s.event_id); S.events.set(s.event_id, s);
        if (!prev || prev.status !== s.status || s.impact >= 6) seismo.pulse(s.impact / 10);
        if (s.event_id === S.selected) selUpdated = true;
        const score = s.impact + (s.entities.length ? 1.2 : 0) + (s.status === "confirmed" ? .8 : 0) + Math.min(1.5, s.n_docs / 20);
        if (!best || score > best.score) best = { ...s, score };
      }
      renderFeed();
      if (selUpdated) await select(S.selected, true).then(() => constel.ping());
      else if (!S.selected && best && best.impact >= 5.5) await select(best.event_id);
      if (S.dock === "trigger") renderDock();
      refreshMeta();
    } catch (e) { console.error(e); }
    S.busy = false;
  }
  setInterval(tick, 900);

  // ---------------------------------------------------------------------------------------------------------
  async function refreshMeta() {
    try {
      const st = await api("/state"); S.state = st;
      $("#chain").innerHTML = `ledger&nbsp;${st.ledger_head.toLocaleString()}&nbsp;·&nbsp;${st.ledger_hash.slice(0, 8)}…`;
      $("#meta").textContent = `${st.docs_processed.toLocaleString()} docs processed · ${st.n_events.toLocaleString()} events · ${st.n_signals.toLocaleString()} signals · book ${st.book_hash.slice(0, 8)} · cfg ${st.config_hash.slice(0, 8)}`;
    } catch (e) { /* ignore */ }
  }

  // ---------------------------------------------------------------------------------------------------------
  // tabs
  $("#tabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; $$("#tabs button").forEach((x) => x.classList.toggle("on", x === b)); $$(".view").forEach((v) => v.classList.toggle("on", v.id === "v-" + b.dataset.v)); if (b.dataset.v === "rebal") loadA(); if (b.dataset.v === "proof") loadProof(); if (b.dataset.v === "arch") drawArch(); });

  async function loadA() {
    try {
      const a = await api("/module-a/weights"), base = 1 / a.names.length;
      $("#a-chart").innerHTML = V.weightsChart(a.history, a.names, base);
      $("#a-turn").textContent = `cumulative turnover ${(a.turnover * 100).toFixed(1)}%`;
      const last = a.history[a.history.length - 1]; $("#a-why").textContent = last ? `last rebalance ${last.ts.slice(5, 16).replace("T", " ")} · ${last.why}` : "";
      const ws = Object.entries(a.weights).sort((x, y) => y[1] - x[1]), mx = Math.max(...ws.map((w) => w[1]), .1);
      $("#a-bars").innerHTML = ws.map(([k, w]) => `<div class="b"><span>${k.toUpperCase()}</span><div class="track"><div class="fill" style="left:0;width:${w / mx * 100}%;background:${w > base * 1.02 ? "linear-gradient(90deg,#ff6a1f,#ffc23d)" : w < base * .98 ? "linear-gradient(90deg,#8a45ff,#ff2e7e)" : "#655d78"}"></div></div><span style="text-align:right">${(w * 100).toFixed(2)}%</span></div>`).join("");
    } catch (e) { /* ignore */ }
  }
  setInterval(() => { if ($("#v-rebal").classList.contains("on")) loadA(); }, 3000);

  // ---------------------------------------------------------------------------------------------------------
  // live wire
  async function wireSend() {
    const text = $("#w-text").value.trim(); if (!text) return;
    $("#w-tag").textContent = "scoring…";
    try {
      const r = await post("/ingest", { text, source: $("#w-src").value, publisher: $("#w-pub").value || "api.local" });
      $("#w-tag").textContent = `ledger head ${r.ledger_head.toLocaleString()}`;
      $("#w-out").innerHTML = r.new_signals.length ? r.new_signals.map((s) => `<div class="card" style="animation:none;margin-bottom:10px;cursor:default">${ring(s.impact)}<div class="cls"><span>${CLS[s.event_class]}</span><span class="faint">${s.signal_id}</span></div><div class="hl">${esc(s.headline)}</div><div class="row"><span class="pill ${s.status === "confirmed" ? "conf" : "prov"}">${s.status}</span>${s.entities.map((e) => `<span class="chip ent">${esc(e)}</span>`).join("")}<span class="chip">${s.n_docs} docs · ${s.independent_sources} src</span><span class="loss">P50 ${V.fmtM(s.loss.p50, 0)} · P95 ${V.fmtM(s.loss.p95, 0)}</span></div></div>`).join("") + `<div class="faint mono" style="font-size:11px">Appended to the ledger and processed by the same code path as replayed data. Hash ${r.new_signals[0].hash}.</div>` : `<div class="empty">Appended to the ledger (entry ${r.ledger_head.toLocaleString()}). It did not produce a new signal: either it looks routine, or it joined an event whose state has not changed materially.</div>`;
      refreshMeta();
    } catch (e) { $("#w-out").innerHTML = `<div class="hashbox bad">${esc(e.message)}</div>`; }
  }
  $("#w-send").onclick = wireSend;
  $("#w-ex1").onclick = () => { $("#w-text").value = "Hertz Global Holdings to file for Chapter 11 bankruptcy protection after talks with lenders fail, Reuters reports"; $("#w-src").value = "news"; $("#w-pub").value = "reuters.com"; };
  $("#w-ex2").onclick = () => { $("#w-text").value = "OPEC+ emergency meeting fails as crude oil collapses and Exxon and Chevron cut spending plans"; $("#w-src").value = "news"; $("#w-pub").value = "bloomberg.com"; };

  // ---------------------------------------------------------------------------------------------------------
  // tooltip
  const tip = $("#tip");
  function onHover(n, m) {
    if (!n || !m) { tip.style.display = "none"; return; }
    const h = S.detail && S.detail.obligors_hit.find((x) => x.entity_id === n.id);
    tip.innerHTML = `<b>${esc(n.name)}</b> <span class="faint mono" style="font-size:10px">${n.sector}</span><div class="p">exposure ${V.fmtM(n.exp, 1)}${n.loss ? ` · P50 loss ${V.fmtM(n.loss, 2)}` : ""}${h ? `<br>${h.edge.replace("_", " ").toLowerCase()} · ${esc(h.path.join(" → "))}` : n.role ? `<br>${n.role.toLowerCase()}` : ""}</div>`;
    tip.style.display = "block"; tip.style.left = Math.min(innerWidth - 310, m.cx + 16) + "px"; tip.style.top = m.cy + 16 + "px";
  }

  // ---------------------------------------------------------------------------------------------------------
  // proof + architecture (defined in proof.js to keep this file focused)
  async function loadProof() { if (window.renderProof) window.renderProof($("#proof"), api); }
  function drawArch() { if (window.renderArch) window.renderArch($("#arch")); }

  // ---------------------------------------------------------------------------------------------------------
  async function init() {
    seismo = new V.Seismo($("#seismo"));
    constel = new V.Constel($("#constel"), onHover, (n) => toast(`<b>${esc(n.name)}</b><br><span class="faint mono" style="font-size:11px">${n.sector} · exposure ${V.fmtM(n.exp, 1)}</span>`, 2600));
    try {
      const [obl, scens] = await Promise.all([api("/obligors"), api("/scenarios")]);
      S.scens = scens; constel.setObligors(obl); $("#stagetag").textContent = `${obl.length} obligors`; renderScen(); await refreshMeta(); renderDock();
    } catch (e) { toast("Backend not reachable: " + esc(e.message), 9000); }
  }
  init();
})();
