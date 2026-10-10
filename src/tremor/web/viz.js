/* TREMOR visual primitives: seismograph, obligor constellation, loss fan, range timeline, weight lines.
   Hand-built canvas/SVG, no dependencies. */
(function () {
  "use strict";
  const V = (window.V = {});
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  V.clamp = clamp; V.lerp = lerp;

  // ---- magma ramp ---------------------------------------------------------------------------------------
  const RAMP = [[58, 74, 99], [138, 69, 255], [255, 46, 126], [255, 106, 31], [255, 194, 61]];
  V.magma = function (t, a = 1) {
    t = clamp(t, 0, 1) * (RAMP.length - 1);
    const i = Math.min(RAMP.length - 2, Math.floor(t)), f = t - i;
    const c = RAMP[i].map((v, k) => Math.round(lerp(v, RAMP[i + 1][k], f)));
    return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  };
  V.fmtM = (x, d = 1) => (x < 0 ? "-" : "") + "$" + Math.abs(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }) + "m";
  V.pct = (x, d = 2) => (x * 100).toFixed(d) + "%";
  V.hash = (s) => { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return (h >>> 0) / 4294967295; };

  V.countTo = function (el, to, fmt, ms = 700) {
    const from = parseFloat(el.dataset.v || "0") || 0, t0 = performance.now();
    el.dataset.v = to;
    const step = (t) => {
      const k = clamp((t - t0) / ms, 0, 1), e = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(lerp(from, to, e));
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };

  // =============================================================================================================
  // Seismograph (header)
  // =============================================================================================================
  V.Seismo = class {
    constructor(canvas) {
      this.c = canvas; this.x = canvas.getContext("2d"); this.pulses = []; this.buf = []; this.t = 0; this.live = false;
      new ResizeObserver(() => this.resize()).observe(canvas.parentElement); this.resize();
      requestAnimationFrame((t) => this.loop(t));
    }
    resize() {
      const p = this.c.parentElement, d = devicePixelRatio || 1;
      this.w = p.clientWidth; this.h = p.clientHeight; this.c.width = this.w * d; this.c.height = this.h * d;
      this.x.setTransform(d, 0, 0, d, 0, 0); this.buf = new Array(Math.ceil(this.w / 2)).fill(0);
    }
    pulse(amp) { this.pulses.push({ t0: this.t, amp: clamp(amp, .08, 1) }); }
    loop(now) {
      const dt = 1 / 60; this.t += dt;
      let v = (Math.sin(this.t * 3.1) * .5 + Math.sin(this.t * 7.3) * .3 + (Math.random() - .5)) * (this.live ? 1.6 : .7) / this.h * 10;
      for (const p of this.pulses) {
        const a = this.t - p.t0; v += p.amp * Math.exp(-a * 1.7) * Math.sin(a * 24) * (this.h * 0.0155) ;
      }
      this.pulses = this.pulses.filter((p) => this.t - p.t0 < 3.4);
      this.buf.push(v); this.buf.shift();
      const x = this.x, w = this.w, h = this.h; x.clearRect(0, 0, w, h);
      const mid = h * 0.56;
      x.strokeStyle = "rgba(255,255,255,.05)"; x.lineWidth = 1; x.setLineDash([2, 6]);
      x.beginPath(); x.moveTo(0, mid); x.lineTo(w, mid); x.stroke(); x.setLineDash([]);
      const g = x.createLinearGradient(0, 0, w, 0);
      g.addColorStop(0, "rgba(138,69,255,0)"); g.addColorStop(.35, "rgba(255,46,126,.55)"); g.addColorStop(.8, "rgba(255,106,31,.95)"); g.addColorStop(1, "rgba(255,194,61,1)");
      x.strokeStyle = g; x.lineWidth = 1.6; x.shadowColor = "rgba(255,106,31,.6)"; x.shadowBlur = 8; x.beginPath();
      this.buf.forEach((b, i) => { const px = i * 2, py = mid + clamp(b, -h * .5, h * .5) * 2.4; i ? x.lineTo(px, py) : x.moveTo(px, py); });
      x.stroke(); x.shadowBlur = 0;
      requestAnimationFrame((t) => this.loop(t));
    }
  };

  // =============================================================================================================
  // Constellation
  // =============================================================================================================
  V.Constel = class {
    constructor(canvas, onHover, onPick) {
      this.c = canvas; this.x = canvas.getContext("2d"); this.nodes = []; this.byId = {}; this.rings = []; this.sig = null;
      this.focus = null; this.mouse = null; this.hover = null; this.pin = null; this.onHover = onHover; this.onPick = onPick; this.t0 = performance.now();
      new ResizeObserver(() => this.resize()).observe(canvas); this.resize();
      canvas.addEventListener("mousemove", (e) => { const r = canvas.getBoundingClientRect(); this.mouse = { x: e.clientX - r.left, y: e.clientY - r.top, cx: e.clientX, cy: e.clientY }; });
      canvas.addEventListener("mouseleave", () => { this.mouse = null; this.hover = null; onHover(null); });
      canvas.addEventListener("click", () => { this.pin = this.hover ? this.hover.id : null; if (this.hover) onPick(this.hover); });
      requestAnimationFrame((t) => this.loop(t));
    }
    resize() { const d = devicePixelRatio || 1; this.w = this.c.clientWidth; this.h = this.c.clientHeight; this.c.width = this.w * d; this.c.height = this.h * d; this.x.setTransform(d, 0, 0, d, 0, 0); this.layout(); }
    setObligors(list) {
      const maxE = Math.max(...list.map((o) => o.exposure_usd_m), 1);
      this.nodes = list.map((o) => ({ id: o.entity_id, name: o.name, sector: o.sector, region: o.region, exp: o.exposure_usd_m, r: 3 + 8.5 * Math.sqrt(o.exposure_usd_m / maxE),
        x: this.w / 2, y: this.h / 2, tx: 0, ty: 0, a: 1, ta: 1, role: null, loss: 0, col: 0, ph: V.hash(o.entity_id) * 6.28 }));
      this.byId = Object.fromEntries(this.nodes.map((n) => [n.id, n])); this.layout();
    }
    layout() {
      if (!this.nodes.length) return;
      const secs = [...new Set(this.nodes.map((n) => n.sector))].sort(), cx = this.w / 2, cy = this.h / 2, rx = this.w * .4, ry = this.h * .38;
      const centers = {}; secs.forEach((s, i) => { const a = (i / secs.length) * Math.PI * 2 - Math.PI / 2; centers[s] = { x: cx + Math.cos(a) * rx, y: cy + Math.sin(a) * ry, a }; });
      this.centers = centers; const cnt = {};
      this.nodes.forEach((n) => {
        const k = (cnt[n.sector] = (cnt[n.sector] || 0) + 1) - 1, c = centers[n.sector], rr = 9 * Math.sqrt(k + .6), ang = k * 2.39996;
        n.hx = c.x + Math.cos(ang) * rr; n.hy = c.y + Math.sin(ang) * rr;
        if (!this.focus) { n.tx = n.hx; n.ty = n.hy; }
      });
      if (this.focus) this.setFocus(this.sig);
      if (this.nodes[0] && this.nodes[0].x === this.w / 2 && this.nodes[0].y === this.h / 2) this.nodes.forEach((n) => { n.x = n.hx; n.y = n.hy; });
    }
    ping() { this.rings.push({ t0: performance.now() }); this.rings.push({ t0: performance.now() + 260 }); }
    clear() { this.sig = null; this.focus = null; this.nodes.forEach((n) => { n.role = null; n.tx = n.hx; n.ty = n.hy; n.ta = 1; n.loss = 0; n.col = 0; }); }
    setFocus(sig) {
      if (!sig) return this.clear();
      this.sig = sig; this.focus = true;
      const cx = this.w / 2, cy = this.h / 2, R = Math.min(this.w, this.h) * .5;
      const rings = { SELF: .17, GROUP: .31, PEER: .45, MARKET: .42 }, groups = { SELF: [], GROUP: [], PEER: [], MARKET: [] };
      this.nodes.forEach((n) => { n.role = null; });
      sig.obligor_map.forEach((o) => { const n = this.byId[o.entity_id]; if (!n) return; n.loss = o.loss_p50_usd_m; n.col = clamp((o.loss_p50_usd_m / Math.max(o.exposure_usd_m, 1)) / .1, 0, 1); n.role = o.role; if (o.role) groups[o.role].push(n); });
      const hitById = Object.fromEntries(sig.obligors_hit.map((h) => [h.entity_id, h])); this.hitById = hitById;
      for (const role of Object.keys(groups)) {
        const g = groups[role].sort((a, b) => (b.exp - a.exp)), k = g.length;
        g.forEach((n, i) => { const a = (i / Math.max(k, 1)) * Math.PI * 2 + (role === "GROUP" ? .5 : role === "PEER" ? .9 : 0) - Math.PI / 2; const rr = R * rings[role] * (role === "SELF" && k === 1 ? 0 : 1);
          n.tx = cx + Math.cos(a) * rr; n.ty = cy + Math.sin(a) * rr; n.ta = 1; });
      }
      this.nodes.forEach((n) => { if (!n.role) { n.tx = n.hx; n.ty = n.hy; n.ta = .2; n.loss = (sig.obligor_map.find((o) => o.entity_id === n.id) || {}).loss_p50_usd_m || 0; n.col = clamp((n.loss / Math.max(n.exp, 1)) / .1, 0, 1); } });
      this.centerLabel = sig.entities.length ? sig.entities.slice(0, 2).join(" · ").toUpperCase() : "MARKET-WIDE";
    }
    loop(now) {
      const x = this.x, w = this.w, h = this.h, cx = w / 2, cy = h / 2; x.clearRect(0, 0, w, h); const t = (now - this.t0) / 1000;
      // sector halos + labels
      if (this.centers) for (const [s, c] of Object.entries(this.centers)) {
        const dim = this.focus ? .35 : 1; x.fillStyle = `rgba(255,255,255,${.035 * dim})`; x.beginPath(); x.arc(c.x, c.y, 34, 0, 7); x.fill();
        x.fillStyle = `rgba(163,156,179,${.65 * dim})`; x.font = "500 9.5px 'JetBrains Mono', monospace"; x.textAlign = "center";
        const lx = c.x + Math.cos(c.a) * 48, ly = c.y + Math.sin(c.a) * 46; x.fillText(s.toUpperCase(), lx, ly);
      }
      // focus rings
      if (this.focus) {
        const R = Math.min(w, h) * .5;
        const present = new Set(this.nodes.filter((n) => n.role).map((n) => n.role));
        [["SELF / CONTROL", .17, "SELF"], ["PROMOTER GROUP", .31, "GROUP"], ["INDUSTRY PEERS", .45, "PEER"], ["SECTOR / MARKET-WIDE", .42, "MARKET"]].filter((r) => present.has(r[2])).forEach(([l, f], i) => {
          x.strokeStyle = "rgba(255,255,255,.07)"; x.setLineDash([3, 7]); x.beginPath(); x.arc(cx, cy, R * f, 0, 7); x.stroke(); x.setLineDash([]);
          x.fillStyle = "rgba(163,156,179,.55)"; x.font = "500 9px 'JetBrains Mono', monospace"; x.textAlign = "left"; x.fillText(l, cx + R * f * .72 + 4, cy - R * f * .72 - 4);
        });
        // edges
        for (const n of this.nodes) { if (!n.role) continue;
          const h_ = this.hitById && this.hitById[n.id], edge = h_ ? h_.edge : n.role === "PEER" ? "PEER" : n.role === "GROUP" ? "PROMOTER_GROUP" : "CONTROL";
          const col = edge === "CONTROL" || edge === "SELF" ? "255,106,31" : edge === "PROMOTER_GROUP" ? "255,46,126" : edge === "MARKET" ? "138,69,255" : "127,212,255";
          x.strokeStyle = `rgba(${col},${edge === "PEER" || edge === "MARKET" ? .2 : .55})`; x.lineWidth = edge === "CONTROL" ? 2 : 1.2; x.setLineDash(edge === "PROMOTER_GROUP" ? [2, 4] : edge === "PEER" || edge === "MARKET" ? [5, 5] : []);
          const mx = (cx + n.x) / 2 + (n.y - cy) * .08, my = (cy + n.y) / 2 - (n.x - cx) * .08; x.beginPath(); x.moveTo(cx, cy); x.quadraticCurveTo(mx, my, n.x, n.y); x.stroke(); x.setLineDash([]); }
        // epicentre
        const pr = 7 + Math.sin(t * 3) * 1.6; const g = x.createRadialGradient(cx, cy, 0, cx, cy, 48); g.addColorStop(0, "rgba(255,106,31,.55)"); g.addColorStop(1, "rgba(255,106,31,0)");
        x.fillStyle = g; x.beginPath(); x.arc(cx, cy, 48, 0, 7); x.fill(); x.fillStyle = "#ffc23d"; x.beginPath(); x.arc(cx, cy, pr, 0, 7); x.fill();
        x.fillStyle = "rgba(243,238,228,.9)"; x.font = "700 10px 'JetBrains Mono', monospace"; x.textAlign = "center"; x.fillText(this.centerLabel || "", cx, cy + 26);
      }
      // shockwaves
      this.rings = this.rings.filter((r) => performance.now() - r.t0 < 2200);
      for (const r of this.rings) { const k = (performance.now() - r.t0) / 2200; if (k < 0) continue; const R = Math.min(w, h) * .55 * (1 - Math.pow(1 - k, 2.2));
        x.strokeStyle = `rgba(255,106,31,${.7 * (1 - k)})`; x.lineWidth = 2.4 * (1 - k) + .4; x.beginPath(); x.arc(cx, cy, R, 0, 7); x.stroke(); }
      // nodes
      let best = null, bd = 1e9;
      for (const n of this.nodes) {
        n.x = lerp(n.x, n.tx, .09); n.y = lerp(n.y, n.ty, .09); n.a = lerp(n.a, n.ta, .12);
        const breathe = n.role ? 1 + Math.sin(t * 2.2 + n.ph) * .08 : 1, r = n.r * breathe;
        const heat = n.col; const fill = this.focus || n.loss ? V.magma(.18 + heat * .82, n.a) : `rgba(163,156,179,${.55 * n.a})`;
        if (n.role && heat > .15) { const g = x.createRadialGradient(n.x, n.y, 0, n.x, n.y, r * 3.2); g.addColorStop(0, V.magma(.4 + heat * .6, .35 * n.a)); g.addColorStop(1, "rgba(0,0,0,0)"); x.fillStyle = g; x.beginPath(); x.arc(n.x, n.y, r * 3.2, 0, 7); x.fill(); }
        x.fillStyle = fill; x.beginPath(); x.arc(n.x, n.y, r, 0, 7); x.fill();
        if (n.role) { x.strokeStyle = `rgba(243,238,228,${.5 * n.a})`; x.lineWidth = 1; x.stroke(); }
        if (this.mouse) { const d = Math.hypot(n.x - this.mouse.x, n.y - this.mouse.y); if (d < r + 7 && d < bd) { bd = d; best = n; } }
      }
      if (this.focus) { const top = this.nodes.filter((n) => n.role).sort((a, b) => b.loss - a.loss).slice(0, 7);
        x.font = "600 10px 'Bricolage Grotesque', sans-serif"; x.textAlign = "center";
        top.forEach((n) => { x.fillStyle = "rgba(243,238,228,.92)"; x.fillText(n.name.replace(/ (Corporation|Corp\.?|Inc\.?|Holdings|Group|plc|Company|Limited|and SEZ)$/i, ""), n.x, n.y - n.r - 6); }); }
      if (best !== this.hover) { this.hover = best; this.onHover(best, this.mouse); } else if (best && this.mouse) this.onHover(best, this.mouse);
      if (this.pin && this.byId[this.pin]) { const n = this.byId[this.pin]; x.strokeStyle = "#33f0c0"; x.lineWidth = 1.6; x.beginPath(); x.arc(n.x, n.y, n.r + 5, 0, 7); x.stroke(); }
      requestAnimationFrame((tt) => this.loop(tt));
    }
  };

  // =============================================================================================================
  // SVG helpers
  // =============================================================================================================
  function smooth(pts) {
    if (pts.length < 3) return pts.map((p, i) => (i ? "L" : "M") + p[0] + " " + p[1]).join("");
    let d = `M${pts[0][0]} ${pts[0][1]}`;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
      d += `C${p1[0] + (p2[0] - p0[0]) / 6} ${p1[1] + (p2[1] - p0[1]) / 6},${p2[0] - (p3[0] - p1[0]) / 6} ${p2[1] - (p3[1] - p1[1]) / 6},${p2[0]} ${p2[1]}`;
    }
    return d;
  }

  V.lossFan = function (sig) {
    const W = 388, H = 168, L = 10, R = 10, T = 22, B = 34, edges = sig.hist_edges, c = sig.hist_counts.slice();
    const sm = c.map((_, i) => ((c[i - 1] ?? c[i]) + c[i] * 2 + (c[i + 1] ?? c[i])) / 4), mx = Math.max(...sm, 1);
    const lo = edges[0], hi = edges[edges.length - 1], X = (v) => L + ((v - lo) / (hi - lo || 1)) * (W - L - R), Y = (v) => T + (1 - v / mx) * (H - T - B);
    const pts = sm.map((v, i) => [X((edges[i] + edges[i + 1]) / 2), Y(v)]);
    const area = `M${X(lo)} ${Y(0)} L${pts[0][0]} ${pts[0][1]} ` + smooth(pts).slice(smooth(pts).indexOf("C")) + ` L${X(hi)} ${Y(0)}Z`;
    const mk = (v, label, col, dy = 0) => `<line x1="${X(v)}" x2="${X(v)}" y1="${T - 4}" y2="${H - B}" stroke="${col}" stroke-width="1.4" stroke-dasharray="${label === "P50" ? "0" : "3 3"}"/><text x="${X(v)}" y="${T - 8 + dy}" fill="${col}" font-size="9.5" text-anchor="middle" font-weight="600">${label}</text><text x="${X(v)}" y="${H - B + 14}" fill="${col}" font-size="10" text-anchor="middle">${V.fmtM(v, v >= 100 || v <= -100 ? 0 : 1)}</text>`;
    const zero = lo < 0 && hi > 0 ? `<line x1="${X(0)}" x2="${X(0)}" y1="${T}" y2="${H - B}" stroke="rgba(255,255,255,.28)"/><text x="${X(0)}" y="${H - 6}" fill="#655d78" font-size="9" text-anchor="middle">no loss</text>` : "";
    return `<svg viewBox="0 0 ${W} ${H}"><defs><linearGradient id="fan" x1="0" x2="1"><stop offset="0" stop-color="#8a45ff" stop-opacity=".75"/><stop offset=".5" stop-color="#ff2e7e" stop-opacity=".75"/><stop offset="1" stop-color="#ffc23d" stop-opacity=".85"/></linearGradient></defs>
      <rect x="${X(sig.loss.p5)}" y="${T}" width="${Math.max(2, X(sig.loss.p95) - X(sig.loss.p5))}" height="${H - T - B}" fill="rgba(255,255,255,.045)" rx="4"/>
      <path d="${area}" fill="url(#fan)"/><path d="${smooth(pts)}" fill="none" stroke="#ffe3a0" stroke-width="1.5" opacity=".9"/>${zero}
      ${mk(sig.loss.p5, "P5", "#7fd4ff")}${mk(sig.loss.p50, "P50", "#f3eee4")}${mk(sig.loss.p95, "P95", "#ffc23d")}
      <path d="M${X(sig.loss.var99)} ${H - B - 1} l-4.5 -9 h9z" fill="#ff2e7e"/></svg>`;
  };

  V.rangeTimeline = function (tl) {
    const W = 388, H = 132, L = 40, R = 8, T = 12, B = 22, n = tl.length;
    if (n < 2) return `<div class="faint mono" style="font-size:11px;padding:8px 0">One version so far. The band re-draws when new evidence arrives.</div>`;
    const ys = tl.flatMap((s) => [s.loss.p5, s.loss.p95]), lo = Math.min(...ys, 0), hi = Math.max(...ys) * 1.05 || 1;
    const X = (i) => L + (i / (n - 1)) * (W - L - R), Y = (v) => T + (1 - (v - lo) / (hi - lo || 1)) * (H - T - B);
    const up = tl.map((s, i) => [X(i), Y(s.loss.p95)]), dn = tl.map((s, i) => [X(i), Y(s.loss.p5)]).reverse(), mid = tl.map((s, i) => [X(i), Y(s.loss.p50)]);
    const band = "M" + up.map((p) => p.join(" ")).join("L") + "L" + dn.map((p) => p.join(" ")).join("L") + "Z";
    let flips = ""; tl.forEach((s, i) => { if (i && s.status !== tl[i - 1].status) flips += `<line x1="${X(i)}" x2="${X(i)}" y1="${T}" y2="${H - B}" stroke="#33f0c0" stroke-dasharray="2 3"/><text x="${X(i)}" y="${T - 2}" fill="#33f0c0" font-size="8.5" text-anchor="middle">${s.status.toUpperCase()}</text>`; });
    const dots = tl.map((s, i) => `<circle cx="${X(i)}" cy="${Y(s.loss.p50)}" r="3.6" fill="${s.status === "confirmed" ? "#33f0c0" : "#0a0810"}" stroke="${s.status === "confirmed" ? "#33f0c0" : "#ffb02e"}" stroke-width="1.6"/>`).join("");
    const ax = [lo, (lo + hi) / 2, hi].map((v) => `<text x="${L - 6}" y="${Y(v) + 3}" fill="#655d78" font-size="9" text-anchor="end">${v.toFixed(0)}</text><line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="rgba(255,255,255,.05)"/>`).join("");
    const t0 = tl[0].as_of.slice(5, 16).replace("T", " "), t1 = tl[n - 1].as_of.slice(5, 16).replace("T", " ");
    return `<svg viewBox="0 0 ${W} ${H}">${ax}<path d="${band}" fill="rgba(255,106,31,.17)" stroke="rgba(255,106,31,.5)" stroke-width="1"/><path d="${smooth(mid)}" fill="none" stroke="#f3eee4" stroke-width="1.6"/>${flips}${dots}
      <text x="${L}" y="${H - 6}" fill="#655d78" font-size="9">${t0}</text><text x="${W - R}" y="${H - 6}" fill="#655d78" font-size="9" text-anchor="end">${t1}</text></svg>`;
  };

  V.weightsChart = function (hist, names, base) {
    const W = 700, H = 360, L = 44, R = 70, T = 14, B = 26;
    if (hist.length < 2) return `<text x="${W / 2}" y="${H / 2}" fill="#655d78" font-size="12" text-anchor="middle">Waiting for signals that name an index stock...</text>`;
    const all = hist.flatMap((h) => Object.values(h.weights)), lo = Math.min(...all, base) * .9, hi = Math.max(...all, base) * 1.08, n = hist.length;
    const X = (i) => L + (i / (n - 1)) * (W - L - R), Y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
    const move = names.map((k) => [k, Math.abs(hist[n - 1].weights[k] - hist[0].weights[k])]).sort((a, b) => b[1] - a[1]).map((d) => d[0]), top = new Set(move.slice(0, 6));
    const pal = ["#ffc23d", "#ff6a1f", "#ff2e7e", "#8a45ff", "#7fd4ff", "#33f0c0"]; let out = "", lab = [];
    [lo, base, hi].forEach((v) => (out += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="rgba(255,255,255,${v === base ? .2 : .05})" stroke-dasharray="${v === base ? "4 4" : "0"}"/><text x="${L - 6}" y="${Y(v) + 3}" fill="#655d78" font-size="9" text-anchor="end">${(v * 100).toFixed(1)}%</text>`));
    names.filter((k) => !top.has(k)).forEach((k) => { out += `<path d="${smooth(hist.map((h, i) => [X(i), Y(h.weights[k])]))}" fill="none" stroke="rgba(163,156,179,.22)" stroke-width="1"/>`; });
    [...top].forEach((k, j) => { const c = pal[j % pal.length]; out += `<path d="${smooth(hist.map((h, i) => [X(i), Y(h.weights[k])]))}" fill="none" stroke="${c}" stroke-width="2"/>`; lab.push([k, Y(hist[n - 1].weights[k]), c]); });
    lab.sort((a, b) => a[1] - b[1]); for (let i = 1; i < lab.length; i++) if (lab[i][1] - lab[i - 1][1] < 11) lab[i][1] = lab[i - 1][1] + 11;
    lab.forEach(([k, y, c]) => (out += `<text x="${W - R + 6}" y="${y + 3}" fill="${c}" font-size="10" font-weight="700">${k.toUpperCase()}</text>`));
    out += `<text x="${L}" y="${H - 6}" fill="#655d78" font-size="9">${hist[0].ts.slice(5, 16).replace("T", " ")}</text><text x="${W - R}" y="${H - 6}" fill="#655d78" font-size="9" text-anchor="end">${hist[n - 1].ts.slice(5, 16).replace("T", " ")}</text>`;
    return out;
  };
})();
