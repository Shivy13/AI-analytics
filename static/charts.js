/* Minimal dependency-free SVG chart library.
   Renders the chart specs produced by the backend: bar, hbar, line, area,
   donut, scatter, histogram, heatmap + trend/threshold annotations. */
const NS = "http://www.w3.org/2000/svg";

const PALETTE = ["#6366f1", "#22d3ee", "#f59e0b", "#ec4899", "#10b981",
                 "#a78bfa", "#fb7185", "#38bdf8", "#facc15", "#34d399"];

function mk(tag, attrs = {}, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  if (parent) parent.appendChild(n);
  return n;
}

function fmtVal(v, format) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const n = Number(v);
  if (format === "currency") return "₹" + compactNum(n);
  if (format === "percent") return (n * 100).toFixed(1) + "%";
  if (format === "percent_raw") return n.toFixed(1) + "%";
  return compactNum(n);
}

function compactNum(n) {
  const a = Math.abs(n);
  if (a >= 1e12) return (n / 1e12).toFixed(1).replace(/\.0$/, "") + "T";
  if (a >= 1e9) return (n / 1e9).toFixed(1).replace(/\.0$/, "") + "B";
  if (a >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, "") + "M";
  if (a >= 1e4) return (n / 1e3).toFixed(1).replace(/\.0$/, "") + "K";
  if (a >= 100 || Number.isInteger(n)) return n.toLocaleString("en-IN");
  return n.toLocaleString("en-IN", { maximumFractionDigits: 2 });
}

function niceScale(min, max, target = 5) {
  if (min === max) { min = min - 1; max = max + 1; }
  const span = max - min;
  const raw = span / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
  const lo = Math.floor(min / step) * step;
  const hi = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = lo; v <= hi + step * 1e-6; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { lo, hi, step, ticks };
}

function truncate(s, n) {
  s = String(s);
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

class Tip {
  constructor(root) {
    this.root = root;
    this.node = root.querySelector(".chart-tip");
    if (!this.node) {
      this.node = document.createElement("div");
      this.node.className = "chart-tip";
      root.appendChild(this.node);
    }
  }
  show(html, x, y) {
    this.node.innerHTML = html;
    this.node.style.opacity = "1";
    const r = this.root.getBoundingClientRect();
    const w = this.node.offsetWidth;
    let left = x - w / 2;
    left = Math.max(4, Math.min(left, r.width - w - 4));
    this.node.style.left = left + "px";
    this.node.style.top = Math.max(2, y - 12 - this.node.offsetHeight) + "px";
  }
  hide() { this.node.style.opacity = "0"; }
}

function frame(container, chart, height) {
  container.innerHTML = "";
  container.classList.add("chart-wrap");
  const head = document.createElement("div");
  head.className = "chart-head";
  head.innerHTML = `<div class="chart-title">${escapeHtml(chart.title || "")}</div>` +
    (chart.subtitle ? `<div class="chart-sub">${escapeHtml(chart.subtitle)}</div>` : "");
  container.appendChild(head);
  const holder = document.createElement("div");
  holder.className = "chart-holder";
  holder.style.height = height + "px";
  container.appendChild(holder);
  const W = Math.max(260, holder.clientWidth || 520);
  const H = height;
  const svg = mk("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "chart-svg" });
  holder.appendChild(svg);
  const tip = new Tip(holder);
  return { svg, tip, W, H, holder };
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function legend(container, names) {
  if (names.length < 2) return;
  const d = document.createElement("div");
  d.className = "chart-legend";
  d.innerHTML = names.map((n, i) =>
    `<span><i style="background:${PALETTE[i % PALETTE.length]}"></i>${escapeHtml(n)}</span>`).join("");
  container.appendChild(d);
}

function axes(svg, pad, W, H, yTicks, fmt, xLabels, xTickIdx, rotate) {
  mk("line", { x1: pad.l, y1: H - pad.b, x2: W - pad.r, y2: H - pad.b, class: "axis" }, svg);
  yTicks.forEach(t => {
    const y = t.pos;
    mk("line", { x1: pad.l, y1: y, x2: W - pad.r, y2: y, class: "grid" }, svg);
    const tx = mk("text", { x: pad.l - 8, y: y + 4, class: "tick y-tick", "text-anchor": "end" }, svg);
    tx.textContent = t.label;
  });
  xLabels.forEach((lab, i) => {
    if (xTickIdx && !xTickIdx.includes(i)) return;
    const x = lab.x;
    const tx = mk("text", { x, y: H - pad.b + 16, class: "tick x-tick", "text-anchor": rotate ? "end" : "middle" }, svg);
    if (rotate) tx.setAttribute("transform", `rotate(-32 ${x} ${H - pad.b + 16})`);
    tx.textContent = lab.text;
  });
}

/* -------------------------------------------------------------- bar / hbar */
function renderBars(container, chart, horizontal) {
  const values = chart.x.values;
  const series = chart.y.series;
  const H = Math.max(240, Math.min(520, horizontal ? 42 + values.length * 30 : 300));
  const { svg, tip, W } = frame(container, chart, H);
  const pad = { l: horizontal ? Math.min(150, Math.max(70, ...values.map(v => String(v).length * 7))) : 62,
                r: 16, t: 14, b: horizontal ? 26 : 62 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const fmt = chart.format || "number";
  const all = series.flatMap(s => s.values.filter(v => v !== null && !Number.isNaN(v)));
  const lo = Math.min(0, ...all), hi = Math.max(0, ...all);
  const sc = niceScale(lo, hi || 1);
  const valScale = v => pad.l + ((v - sc.lo) / (sc.hi - sc.lo)) * iw;
  const catScale = i => pad.t + (i + 0.5) * (ih / values.length);

  if (horizontal) {
    sc.ticks.forEach(t => {
      const x = valScale(t);
      mk("line", { x1: x, y1: pad.t, x2: x, y2: H - pad.b, class: "grid" }, svg);
      const tx = mk("text", { x, y: H - pad.b + 16, class: "tick", "text-anchor": "middle" }, svg);
      tx.textContent = fmtVal(t, fmt);
    });
    const band = ih / values.length;
    const bh = Math.min(22, (band * 0.72) / series.length);
    values.forEach((lab, i) => {
      const tx = mk("text", { x: pad.l - 8, y: catScale(i) + 4, class: "tick", "text-anchor": "end" }, svg);
      tx.textContent = truncate(lab, 20);
      series.forEach((s, si) => {
        const v = s.values[i];
        if (v === null || Number.isNaN(v)) return;
        const x0 = valScale(0), x1 = valScale(v);
        const rect = mk("rect", {
          x: Math.min(x0, x1), y: catScale(i) - (series.length * bh) / 2 + si * bh,
          width: Math.abs(x1 - x0), height: bh - 2, rx: 3,
          fill: PALETTE[si % PALETTE.length], class: "mark"
        }, svg);
        rect.addEventListener("mousemove", e => {
          const r = container.getBoundingClientRect();
          tip.show(`<b>${escapeHtml(lab)}</b><br>${escapeHtml(s.name)}: ${fmtVal(v, fmt)}`,
                   e.clientX - r.left, e.clientY - r.top);
        });
        rect.addEventListener("mouseleave", () => tip.hide());
      });
    });
  } else {
    const yTicks = sc.ticks.map(t => ({ pos: H - pad.b - ((t - sc.lo) / (sc.hi - sc.lo)) * ih, label: fmtVal(t, fmt) }));
    const rotate = values.some(v => String(v).length > 9) || values.length > 7;
    axes(svg, pad, W, H, yTicks, fmt,
         values.map((v, i) => ({ x: pad.l + (i + 0.5) * (iw / values.length), text: truncate(v, 16) })),
         null, rotate);
    const band = iw / values.length;
    const bw = Math.min(46, (band * 0.68) / series.length);
    values.forEach((lab, i) => {
      series.forEach((s, si) => {
        const v = s.values[i];
        if (v === null || Number.isNaN(v)) return;
        const y0 = H - pad.b - ((0 - sc.lo) / (sc.hi - sc.lo)) * ih;
        const y1 = H - pad.b - ((v - sc.lo) / (sc.hi - sc.lo)) * ih;
        const x = pad.l + i * band + band / 2 - (series.length * bw) / 2 + si * bw;
        const rect = mk("rect", {
          x, y: Math.min(y0, y1), width: bw - 2, height: Math.max(1, Math.abs(y1 - y0)),
          rx: 3, fill: PALETTE[si % PALETTE.length], class: "mark"
        }, svg);
        rect.addEventListener("mousemove", e => {
          const r = container.getBoundingClientRect();
          tip.show(`<b>${escapeHtml(lab)}</b><br>${escapeHtml(s.name)}: ${fmtVal(v, fmt)}`,
                   e.clientX - r.left, e.clientY - r.top);
        });
        rect.addEventListener("mouseleave", () => tip.hide());
      });
    });
  }
  drawAnnotations(svg, chart, { pad, W, H, sc, valScale, catScale, horizontal, values, fmt });
  legend(container, series.map(s => s.name));
}

/* ------------------------------------------------------------- line / area */
function renderLine(container, chart, area) {
  const values = chart.x.values;
  const series = chart.y.series;
  const H = 300;
  const { svg, tip, W } = frame(container, chart, H);
  const pad = { l: 62, r: 18, t: 14, b: 48 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const fmt = chart.format || "number";
  const all = series.flatMap(s => s.values.filter(v => v !== null && !Number.isNaN(v)));
  const sc = niceScale(Math.min(0, ...all), Math.max(0, ...all) || 1);
  const X = i => pad.l + (values.length === 1 ? iw / 2 : (i / (values.length - 1)) * iw);
  const Y = v => H - pad.b - ((v - sc.lo) / (sc.hi - sc.lo)) * ih;
  const yTicks = sc.ticks.map(t => ({ pos: Y(t), label: fmtVal(t, fmt) }));
  const step = Math.max(1, Math.ceil(values.length / 8));
  axes(svg, pad, W, H, yTicks, fmt,
       values.map((v, i) => ({ x: X(i), text: truncate(v, 12) })),
       values.map((_, i) => i).filter(i => i % step === 0), false);

  if (values.length > 1) {
    mk("line", { x1: X(0), y1: pad.t, x2: X(0), y2: H - pad.b, class: "grid" }, svg);
    mk("line", { x1: X(values.length - 1), y1: pad.t, x2: X(values.length - 1), y2: H - pad.b, class: "grid" }, svg);
  }
  series.forEach((s, si) => {
    const col = PALETTE[si % PALETTE.length];
    const pts = s.values.map((v, i) => (v === null || Number.isNaN(v)) ? null : [X(i), Y(v)]);
    if (area) {
      const d = pts.reduce((acc, p, i) => {
        if (!p) return acc;
        return acc + (acc ? " L" : "M") + p[0] + " " + p[1];
      }, "");
      if (d) {
        const first = pts.findIndex(p => p), last = pts.map(p => !!p).lastIndexOf(true);
        mk("path", { d: `${d} L${X(last)} ${H - pad.b} L${X(first)} ${H - pad.b} Z`, fill: col, class: "area-fill" }, svg);
      }
    }
    let d = "", open = false;
    pts.forEach(p => {
      if (!p) { open = false; return; }
      d += (open ? " L" : "M") + p[0] + " " + p[1];
      open = true;
    });
    mk("path", { d, fill: "none", stroke: col, "stroke-width": 2.2, class: "line-path" }, svg);
    const dotR = values.length > 60 ? 0 : 3.2;
    pts.forEach((p, i) => {
      if (!p) return;
      const c = mk("circle", { cx: p[0], cy: p[1], r: dotR || 7, fill: col,
                               class: dotR ? "dot" : "hit", "fill-opacity": dotR ? 1 : 0 }, svg);
      c.addEventListener("mousemove", e => {
        const r = container.getBoundingClientRect();
        tip.show(`<b>${escapeHtml(values[i])}</b><br>${escapeHtml(s.name)}: ${fmtVal(s.values[i], fmt)}`,
                 e.clientX - r.left, e.clientY - r.top);
      });
      c.addEventListener("mouseleave", () => tip.hide());
    });
  });
  drawAnnotations(svg, chart, { pad, W, H, sc, X, Y, horizontal: false, values, fmt, linear: true });
  legend(container, series.map(s => s.name));
}

/* -------------------------------------------------------------- histogram */
function renderHistogram(container, chart) {
  const counts = chart.y.series[0].values;
  const edges = chart.x.edges;
  const H = 300;
  const { svg, tip, W } = frame(container, chart, H);
  const pad = { l: 62, r: 18, t: 14, b: 48 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const sc = niceScale(0, Math.max(...counts) || 1);
  const lo = edges[0], hi = edges[edges.length - 1];
  const X = v => pad.l + ((v - lo) / (hi - lo)) * iw;
  const Y = v => H - pad.b - ((v - sc.lo) / (sc.hi - sc.lo)) * ih;
  const yTicks = sc.ticks.map(t => ({ pos: Y(t), label: fmtVal(t, "number") }));
  axes(svg, pad, W, H, yTicks, "number",
       sc && [0, 0.25, 0.5, 0.75, 1].map(f => ({ x: X(lo + f * (hi - lo)), text: compactNum(lo + f * (hi - lo)) })),
       null, false);
  counts.forEach((c, i) => {
    const x0 = X(edges[i]), x1 = X(edges[i + 1]);
    const rect = mk("rect", { x: x0 + 0.5, y: Y(c), width: Math.max(1, x1 - x0 - 1),
                              height: Math.max(0, H - pad.b - Y(c)), fill: PALETTE[0], class: "mark" }, svg);
    rect.addEventListener("mousemove", e => {
      const r = container.getBoundingClientRect();
      tip.show(`<b>${compactNum(edges[i])} – ${compactNum(edges[i + 1])}</b><br>${c.toLocaleString("en-IN")} rows`,
               e.clientX - r.left, e.clientY - r.top);
    });
    rect.addEventListener("mouseleave", () => tip.hide());
  });
  drawAnnotations(svg, chart, { pad, W, H, sc, X, Y, horizontal: false, values: [], fmt: "number", linear: true });
}

/* ---------------------------------------------------------------- scatter */
function renderScatter(container, chart) {
  const xs = chart.x.values, ys = chart.y.series[0].values;
  const H = 320;
  const { svg, tip, W } = frame(container, chart, H);
  const pad = { l: 66, r: 18, t: 14, b: 48 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const xsc = niceScale(Math.min(...xs), Math.max(...xs));
  const ysc = niceScale(Math.min(...ys), Math.max(...ys));
  const X = v => pad.l + ((v - xsc.lo) / (xsc.hi - xsc.lo)) * iw;
  const Y = v => H - pad.b - ((v - ysc.lo) / (ysc.hi - ysc.lo)) * ih;
  axes(svg, pad, W, H, ysc.ticks.map(t => ({ pos: Y(t), label: fmtVal(t, chart.format) })),
       chart.format,
       xsc.ticks.map(t => ({ x: X(t), text: fmtVal(t, chart.x_format || "number") })), null, false);
  xs.forEach((x, i) => {
    const c = mk("circle", { cx: X(x), cy: Y(ys[i]), r: 4, class: "mark scatter-dot", fill: PALETTE[0] }, svg);
    c.addEventListener("mousemove", e => {
      const r = container.getBoundingClientRect();
      tip.show(`${escapeHtml(chart.x.label)}: <b>${fmtVal(x, chart.x_format)}</b><br>` +
               `${escapeHtml(chart.y.series[0].name)}: <b>${fmtVal(ys[i], chart.format)}</b>`,
               e.clientX - r.left, e.clientY - r.top);
    });
    c.addEventListener("mouseleave", () => tip.hide());
  });
  drawAnnotations(svg, chart, { pad, W, H, sc: ysc, X, Y, horizontal: false, values: [], linear: true });
}

/* ------------------------------------------------------------------ donut */
function renderDonut(container, chart) {
  const values = chart.x.values;
  const data = chart.y.series[0].values;
  const H = 300;
  const { svg, tip, W } = frame(container, chart, H);
  const total = data.reduce((a, b) => a + (b || 0), 0) || 1;
  const cx = W / 2, cy = H / 2, R = Math.min(W, H) / 2 - 14, r = R * 0.58;
  let a0 = -Math.PI / 2;
  data.forEach((v, i) => {
    if (!v) return;
    const a1 = a0 + (v / total) * Math.PI * 2;
    const large = a1 - a0 > Math.PI ? 1 : 0;
    const p = mk("path", {
      d: `M${cx + R * Math.cos(a0)} ${cy + R * Math.sin(a0)} A${R} ${R} 0 ${large} 1 ` +
         `${cx + R * Math.cos(a1)} ${cy + R * Math.sin(a1)} L${cx + r * Math.cos(a1)} ${cy + r * Math.sin(a1)} ` +
         `A${r} ${r} 0 ${large} 0 ${cx + r * Math.cos(a0)} ${cy + r * Math.sin(a0)} Z`,
      fill: PALETTE[i % PALETTE.length], class: "mark"
    }, svg);
    p.addEventListener("mousemove", e => {
      const b = container.getBoundingClientRect();
      tip.show(`<b>${escapeHtml(values[i])}</b><br>${fmtVal(v, chart.format)} (${(v / total * 100).toFixed(1)}%)`,
               e.clientX - b.left, e.clientY - b.top);
    });
    p.addEventListener("mouseleave", () => tip.hide());
    a0 = a1;
  });
  const t1 = mk("text", { x: cx, y: cy - 2, class: "donut-total", "text-anchor": "middle" }, svg);
  t1.textContent = fmtVal(total, chart.format);
  const t2 = mk("text", { x: cx, y: cy + 16, class: "donut-label", "text-anchor": "middle" }, svg);
  t2.textContent = "total";
  legend(container, values.map((v, i) => `${v} · ${(data[i] / total * 100).toFixed(0)}%`));
  container.querySelectorAll(".chart-legend i").forEach((n, i) =>
    n.style.background = PALETTE[i % PALETTE.length]);
}

/* ---------------------------------------------------------------- heatmap */
function renderHeatmap(container, chart) {
  const cols = chart.x.values, M = chart.matrix;
  const cell = 44, padL = 130, padT = 96;
  const H = padT + cols.length * cell + 12;
  const W = Math.max(320, padL + cols.length * cell + 12);
  container.innerHTML = "";
  const head = document.createElement("div");
  head.className = "chart-head";
  head.innerHTML = `<div class="chart-title">${escapeHtml(chart.title)}</div>
                    <div class="chart-sub">${escapeHtml(chart.subtitle || "")}</div>`;
  container.appendChild(head);
  const holder = document.createElement("div");
  holder.className = "chart-holder scroll-x";
  holder.style.height = H + "px";
  container.appendChild(holder);
  const svg = mk("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "chart-svg" });
  holder.appendChild(svg);
  const tip = new Tip(holder);
  cols.forEach((c, j) => {
    const t = mk("text", { x: padL + j * cell + cell / 2, y: padT - 8, class: "tick",
                           "text-anchor": "start",
                           transform: `rotate(-45 ${padL + j * cell + cell / 2} ${padT - 8})` }, svg);
    t.textContent = truncate(c, 18);
  });
  M.forEach((row, i) => {
    const t = mk("text", { x: padL - 8, y: padT + i * cell + cell / 2 + 4, class: "tick", "text-anchor": "end" }, svg);
    t.textContent = truncate(cols[i], 20);
    row.forEach((v, j) => {
      const a = Math.abs(v);
      const col = v >= 0 ? `rgba(99,102,241,${0.12 + a * 0.85})` : `rgba(244,63,94,${0.12 + a * 0.85})`;
      const rect = mk("rect", { x: padL + j * cell + 1, y: padT + i * cell + 1,
                                width: cell - 2, height: cell - 2, rx: 4, fill: col, class: "mark" }, svg);
      const tx = mk("text", { x: padL + j * cell + cell / 2, y: padT + i * cell + cell / 2 + 4,
                              class: "cell-text", "text-anchor": "middle" }, svg);
      tx.textContent = v.toFixed(2);
      rect.addEventListener("mousemove", e => {
        const b = container.getBoundingClientRect();
        tip.show(`<b>${escapeHtml(cols[i])}</b> × <b>${escapeHtml(cols[j])}</b><br>r = ${v.toFixed(3)}`,
                 e.clientX - b.left, e.clientY - b.top);
      });
      rect.addEventListener("mouseleave", () => tip.hide());
    });
  });
}

/* ----------------------------------------------------------- annotations */
function drawAnnotations(svg, chart, ctx) {
  const { pad, W, H, X, Y, horizontal, fmt } = ctx;
  let vlineIdx = 0;   // stagger stacked mean/median labels so they don't overlap
  (chart.annotations || []).forEach(a => {
    if (a.type === "trendline" && X && !horizontal) {
      const pts = a.values.map((v, i) => (v === null || Number.isNaN(v)) ? null : [X(i), Y(v)])
                          .filter(Boolean);
      if (pts.length > 1) {
        mk("path", { d: pts.map((p, i) => (i ? "L" : "M") + p[0] + " " + p[1]).join(" "),
                     fill: "none", stroke: a.color || "#f59e0b", "stroke-width": 2,
                     "stroke-dasharray": "6 5", class: "annotation" }, svg);
        const t = mk("text", { x: pts[pts.length - 1][0] - 4, y: pts[pts.length - 1][1] - 8,
                               class: "ann-label", "text-anchor": "end" }, svg);
        t.textContent = a.label || "trend";
      }
    } else if (a.type === "hline") {
      const y = Y(a.y);
      mk("line", { x1: pad.l, y1: y, x2: W - pad.r, y2: y, stroke: a.color || "#f59e0b",
                   "stroke-width": 1.6, "stroke-dasharray": "7 5", class: "annotation" }, svg);
      const t = mk("text", { x: W - pad.r - 4, y: y - 6, class: "ann-label", "text-anchor": "end" }, svg);
      t.textContent = `${a.label || ""} ${fmtVal(a.y, fmt)}`;
    } else if (a.type === "vline" && ctx.linear) {
      const x = X(a.x);
      if (x >= pad.l && x <= W - pad.r) {
        mk("line", { x1: x, y1: pad.t, x2: x, y2: H - pad.b, stroke: a.color || "#f59e0b",
                     "stroke-width": 1.6, "stroke-dasharray": "7 5", class: "annotation" }, svg);
        const t = mk("text", { x: x + 5, y: pad.t + 12 + (vlineIdx++ * 13), class: "ann-label" }, svg);
        t.textContent = `${a.label || ""} ${fmtVal(a.x, fmt)}`;
      }
    } else if (a.type === "line" && a.x1 !== undefined) {
      mk("line", { x1: X(a.x1), y1: Y(a.y1), x2: X(a.x2), y2: Y(a.y2),
                   stroke: a.color || "#f59e0b", "stroke-width": 2, "stroke-dasharray": "6 5" }, svg);
    }
  });
}

/* ------------------------------------------------------------------ API */
const Charts = {
  render(container, chart) {
    if (!chart) { container.innerHTML = '<div class="empty">No chart</div>'; return; }
    container.dataset.chartType = chart.type;
    switch (chart.type) {
      case "bar": return renderBars(container, chart, false);
      case "hbar": return renderBars(container, chart, true);
      case "line": return renderLine(container, chart, false);
      case "area": return renderLine(container, chart, true);
      case "histogram": return renderHistogram(container, chart);
      case "scatter": return renderScatter(container, chart);
      case "donut": return renderDonut(container, chart);
      case "heatmap": return renderHeatmap(container, chart);
      default: container.innerHTML = `<div class="empty">Unsupported chart type: ${escapeHtml(chart.type)}</div>`;
    }
  },
  svgMarkup(container) {
    const s = container.querySelector("svg");
    if (!s) return null;
    const clone = s.cloneNode(true);
    clone.setAttribute("xmlns", NS);
    const style = document.createElementNS(NS, "style");
    style.textContent = CSS_TEXT;
    clone.insertBefore(style, clone.firstChild);
    return '<?xml version="1.0"?>\n' + clone.outerHTML;
  },
  toPNG(container, scale = 2) {
    const markup = Charts.svgMarkup(container);
    if (!markup) return null;
    const svg = container.querySelector("svg");
    return new Promise(resolve => {
      const img = new Image();
      const url = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(markup)));
      img.onload = () => {
        const c = document.createElement("canvas");
        c.width = svg.clientWidth * scale; c.height = svg.clientHeight * scale;
        const g = c.getContext("2d");
        g.fillStyle = "#0f1220"; g.fillRect(0, 0, c.width, c.height);
        g.drawImage(img, 0, 0, c.width, c.height);
        resolve(c.toDataURL("image/png"));
      };
      img.onerror = () => resolve(null);
      img.src = url;
    });
  }
};

const CSS_TEXT = `
svg { background: transparent; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.axis { stroke: #2c3352; stroke-width: 1; }
.grid { stroke: #1d2340; stroke-width: 1; }
.tick { fill: #e8ebff; font-size: 11px; }
.donut-total { fill: #e8ebff; font-size: 20px; font-weight: 600; }
.donut-label { fill: #e8ebff; font-size: 11px; opacity: .85; }
.cell-text { fill: #e8ebff; font-size: 10px; }
.ann-label { fill: #f59e0b; font-size: 10px; }
.mark { transition: opacity .12s ease; }
.mark:hover { opacity: .78; }
.area-fill { opacity: .16; }
.line-path { stroke-linejoin: round; stroke-linecap: round; }
.scatter-dot { opacity: .62; }
`;
window.Charts = Charts;
