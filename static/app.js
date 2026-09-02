/* AI Analytics Dashboard — front end.
   No framework, no CDN: everything is plain DOM + the SVG chart lib. */
"use strict";

const S = {
  dataset: null, profile: null, report: null, columns: [],
  tab: "overview", llm: { enabled: false, has_key: false },
  explore: { spec: null, result: null },
  table: { offset: 0, limit: 50, search: "", sortCol: null, sortDir: "desc", total: 0 },
  charts: new Map(),
};

const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
};
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function toast(msg, isErr) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast show" + (isErr ? " err" : "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.className = "toast"), 3600);
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {},
    ...opts,
    body: opts.body && !(opts.body instanceof FormData) && typeof opts.body !== "string"
      ? JSON.stringify(opts.body) : opts.body,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* keep statusText */ }
    if (res.status === 402) openPaywall(detail);
    throw new Error(detail);
  }
  return res.json();
}

function fmt(v, format) {
  if (v === null || v === undefined || v === "" || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  if (format === "currency") return "₹" + compact(n);
  if (format === "percent") return (n * 100).toFixed(1) + "%";
  if (format === "percent_raw") return n.toFixed(1) + "%";
  if (format === "text") return String(v);
  return compact(n);
}
function compact(n) {
  const a = Math.abs(n);
  if (a >= 1e12) return (n / 1e12).toFixed(1) + "T";
  if (a >= 1e9) return (n / 1e9).toFixed(1) + "B";
  if (a >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (a >= 1e4) return (n / 1e3).toFixed(1) + "K";
  if (Number.isInteger(n)) return n.toLocaleString("en-IN");
  return n.toLocaleString("en-IN", { maximumFractionDigits: 2 });
}

/* ============================================================ boot */
async function boot() {
  wireEvents();
  refreshAuth();
  if (location.hash === "#upgraded") { setTimeout(refreshAuth, 1500); toast("Welcome to Pro!"); }
  try {
    const h = await api("/api/health");
    S.llm = h.llm;
    renderEngineChip();
    const d = await api("/api/datasets");
    const s = await api("/api/samples");
    renderLanding(s.samples, d.datasets);
  } catch (e) {
    toast("Could not reach the server: " + e.message, true);
  }
}

function renderEngineChip() {
  const on = S.llm.enabled && S.llm.has_key;
  $("engineDot").className = "dot" + (on ? " on" : "");
  $("engineText").textContent = on ? `Model: ${S.llm.model || "connected"}` : "Local engine · offline";
  $("askBadge").textContent = on ? "LLM" : "AI";
  $("askEngineNote").textContent = on
    ? "Questions are translated by your connected model; the numbers always come from local computation."
    : "Built-in parser is answering. Add a model key in ⚙ Model for harder questions — everything works without it.";
}

function renderLanding(samples, recent) {
  const host = $("sampleList");
  host.innerHTML = "";
  (recent || []).slice(0, 3).forEach((d) => {
    const c = el("div", "sample",
      `<div class="t">↺ ${esc(d.name)}</div>
       <div class="d">${d.rows.toLocaleString("en-IN")} rows · ${d.columns} columns</div>
       <span class="chip sm">Reopen</span>`);
    c.onclick = () => loadExisting(d.id);
    host.appendChild(c);
  });
  (samples || []).forEach((s) => {
    const c = el("div", "sample",
      `<div class="t">${esc(s.label)}</div>
       <div class="d">${esc(s.description)}</div>
       <span class="chip sm">${s.rows.toLocaleString("en-IN")} rows · ${s.columns} cols</span>`);
    c.onclick = () => loadSample(s.key, c);
    host.appendChild(c);
  });
}

/* ============================================================ loading */
function busy(node, on, label) {
  if (on) {
    node.dataset.prev = node.innerHTML;
    node.innerHTML = `<span class="loader"></span> ${label || "Analysing…"}`;
    node.style.pointerEvents = "none";
  } else {
    node.innerHTML = node.dataset.prev || node.innerHTML;
    node.style.pointerEvents = "";
  }
}

async function loadSample(key, node) {
  busy(node, true);
  try {
    const r = await api(`/api/samples/${key}/load`, { method: "POST" });
    busy(node, false);
    adopt(r);
  } catch (e) { busy(node, false); toast(e.message, true); }
}

async function loadExisting(id) {
  try {
    const r = await api(`/api/datasets/${id}`);
    adopt(r);
  } catch (e) { toast(e.message, true); }
}

async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const dz = $("dropzone");
  busy(dz, true, "Reading " + file.name + "…");
  try {
    const r = await api("/api/upload", { method: "POST", body: fd });
    busy(dz, false);
    adopt(r);
  } catch (e) {
    busy(dz, false);
    toast(e.message, true);
  }
}

function adopt(r) {
  S.dataset = r.dataset;
  S.profile = r.profile;
  S.report = r.report;
  S.columns = r.profile.map((c) => c.name);
  S.table = { offset: 0, limit: 50, search: "", sortCol: null, sortDir: "desc", total: 0 };
  $("viewLanding").classList.add("hidden");
  $("viewApp").classList.remove("hidden");
  $("btnReset").classList.remove("hidden");
  $("datasetChip").classList.remove("hidden");
  $("dsName").textContent = S.dataset.name;
  $("dsShape").textContent = `· ${S.dataset.rows.toLocaleString("en-IN")} × ${S.dataset.columns}`;
  switchTab("overview");
  renderOverview();
  buildControls();
  renderSchema();
  $("askLog").innerHTML = "";
  renderSuggestions(S.report.examples || []);
  toast(`Analysed ${S.dataset.rows.toLocaleString("en-IN")} rows · ${S.report.insights.length} findings`);
}

/* ============================================================ tabs */
function switchTab(name) {
  S.tab = name;
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === name));
  ["overview", "explore", "ask", "data"].forEach((t) => {
    const n = $("tab" + t[0].toUpperCase() + t.slice(1));
    n.classList.toggle("hidden", t !== name);
  });
  if (name === "explore" && !S.explore.result) runExplore();
  if (name === "data" && !S.table.total) loadRows();
  if (name === "explore") requestAnimationFrame(() => redraw("exploreChart"));
}

/* ============================================================ overview */
function renderOverview() {
  const dash = S.report.dashboard || { kpis: [], charts: [] };
  const narr = $("narrativeBox");
  if (S.report.narrative) {
    narr.classList.remove("hidden");
    narr.className = "narrative";
    narr.textContent = S.report.narrative;
  } else {
    narr.classList.add("hidden");
  }

  const k = $("kpiRow");
  k.innerHTML = "";
  dash.kpis.forEach((kpi) => {
    k.appendChild(el("div", "kpi",
      `<div class="k">${esc(kpi.label)}</div>
       <div class="v">${fmt(kpi.value, kpi.format)}</div>
       <div class="h">${esc(kpi.hint || "")}</div>`));
  });

  const list = $("insightList");
  list.innerHTML = "";
  const counts = S.report.severity_counts || {};
  const head = el("div", "card-sub",
    `${S.report.total_found} checks fired · ${counts.critical || 0} critical · ${counts.warning || 0} warnings · ` +
    `${counts.info || 0} observations · ${counts.positive || 0} positives`);
  head.style.gridColumn = "1 / -1";
  list.appendChild(head);

  S.report.insights.forEach((ins) => list.appendChild(insightCard(ins)));

  const grid = $("autoCharts");
  grid.innerHTML = "";
  (dash.charts || []).forEach((ch, i) => {
    const card = el("div", "chart-card");
    const body = el("div", "chart-wrap");
    card.appendChild(body);
    const acts = el("div", "chart-actions");
    const b = el("button", "btn sm ghost", "Open in Explore →");
    b.onclick = () => openInExplore(ch.spec);
    card.insertBefore(acts, body);
    acts.appendChild(b);
    grid.appendChild(card);
    drawChart(body, ch.chart, "auto" + i);
  });
}

function insightCard(ins) {
  const card = el("div", "insight " + ins.severity);
  const metric = ins.metric
    ? `<div class="metric">${fmt(ins.metric.value, ins.metric.format)}<small>${esc(ins.metric.label)}</small></div>` : "";
  const ev = (ins.evidence || []).slice(0, 4)
    .map((e) => `<li>${esc(e)}</li>`).join("");
  card.innerHTML = `
    <div class="row">
      <span class="sev ${ins.severity}">${ins.severity}</span>
      <div style="flex:1">
        <h4>${esc(ins.title)}</h4>
        <p>${esc(ins.summary)}</p>
        ${metric}
        ${ev ? `<ul class="evidence">${ev}</ul>` : ""}
        <div class="acts">
          ${ins.question ? `<button class="btn sm" data-act="ask">Ask: ${esc(ins.question)}</button>` : ""}
          ${ins.chart ? `<button class="btn sm ghost" data-act="chart">Chart it</button>` : ""}
        </div>
      </div>
    </div>`;
  card.querySelector('[data-act="ask"]')?.addEventListener("click", () => askQuestion(ins.question));
  card.querySelector('[data-act="chart"]')?.addEventListener("click", () => openInExplore(ins.chart.spec));
  return card;
}

/* ============================================================ chart host */
function drawChart(host, chart, key) {
  Charts.render(host, chart);
  S.charts.set(key, { host, chart });
  if (!drawChart._ro) {
    drawChart._ro = new ResizeObserver(() => {
      clearTimeout(drawChart._t);
      drawChart._t = setTimeout(() => {
        S.charts.forEach(({ host: h, chart: c }) => {
          if (h.offsetParent !== null) Charts.render(h, c);
        });
      }, 160);
    });
  }
  drawChart._ro.observe(host);
}
function redraw(key) {
  const c = S.charts.get(key);
  if (c && c.host.offsetParent !== null) Charts.render(c.host, c.chart);
}

/* ============================================================ explore */
function fillSelect(sel, items, current) {
  sel.innerHTML = "";
  items.forEach(([v, label]) => {
    const o = document.createElement("option");
    o.value = v; o.textContent = label;
    if (v === current) o.selected = true;
    sel.appendChild(o);
  });
}

function colType(name) {
  const c = S.profile.find((p) => p.name === name);
  return c ? c.type : "text";
}

function buildControls() {
  const numeric = S.profile.filter((c) => c.type === "numeric").map((c) => c.name);
  const dims = S.profile.filter((c) => ["categorical", "boolean", "id", "datetime"].includes(c.type)).map((c) => c.name);
  fillSelect($("ctlMeasure"), [["", "— row count —"], ...numeric.map((n) => [n, n])]);
  fillSelect($("ctlDimension"), [["", "— none —"], ...dims.map((n) => [n, n])]);
  fillSelect($("ctlY"), numeric.map((n) => [n, n]));
  if (numeric[0]) {
    $("ctlMeasure").value = numeric[0];
    $("ctlAgg").value = /pct|rate|ratio|csat|rating|score|price|hours|mins|duration/.test(numeric[0]) ? "avg" : "sum";
  }
  if (dims[0]) $("ctlDimension").value = dims[0];
  $("filterList").innerHTML = "";
  runExplore();
}

function specFromControls() {
  const type = $("ctlType").value;
  const measure = $("ctlMeasure").value || null;
  const agg = $("ctlAgg").value;
  const dim = $("ctlDimension").value || null;
  const gran = $("ctlGran").value;
  const [sortBy, sortDir] = $("ctlSort").value.split(":");
  const limitRaw = $("ctlLimit").value;
  const limit = limitRaw ? parseInt(limitRaw, 10) : null;
  const filters = readFilters();
  const dimType = dim ? colType(dim) : null;

  if (type === "heatmap") return { kind: "matrix", filters };
  if (type === "scatter") {
    const y = $("ctlY").value;
    return { kind: "correlation", x: measure || y, y, filters };
  }
  if (type === "histogram") {
    return { kind: "distribution", column: measure || numeric0(), filters };
  }
  if (type === "segments") {
    return { kind: "segments", dimension: dim, measure, agg: agg === "count" ? "avg" : agg, filters };
  }
  if (type === "table") {
    return { kind: "table", limit: 100, filters, sort_col: dim || undefined, sort_dir: sortDir };
  }
  if (dimType === "datetime" || (dim && gran !== "auto")) {
    return { kind: "timeseries", time_dimension: dim, measure, agg, granularity: gran, filters };
  }
  return { kind: "aggregate", dimension: dim, measure, agg, limit,
           other_bucket: !!limit && sortDir === "desc",
           sort: { by: sortBy, dir: sortDir }, filters };
}
function numeric0() {
  const n = S.profile.find((c) => c.type === "numeric");
  return n ? n.name : S.columns[0];
}

function applyTypeOverride(chart, type) {
  if (!chart) return chart;
  if (type === "auto") return chart;
  if (type === "donut" && ["bar", "hbar"].includes(chart.type)) {
    return { ...chart, type: "donut" };
  }
  if (["bar", "hbar"].includes(type) && ["bar", "hbar"].includes(chart.type)) {
    return { ...chart, type };
  }
  if ((type === "line" || type === "area") && ["line", "area"].includes(chart.type)) {
    return { ...chart, type };
  }
  return chart;
}

async function runExplore(overrideSpec) {
  const spec = overrideSpec || specFromControls();
  S.explore.spec = spec;
  const host = $("exploreChart");
  const tableHost = $("exploreTable");
  $("exploreMeta").innerHTML = '<span class="loader"></span> running…';
  try {
    const res = await api(`/api/datasets/${S.dataset.id}/query`, { method: "POST", body: { spec } });
    S.explore.result = res;
    if (res.error) {
      host.innerHTML = `<div class="empty">${esc(res.error)}</div>`;
      $("exploreMeta").textContent = "Could not run";
      tableHost.innerHTML = "";
      return;
    }
    const chart = applyTypeOverride(res.chart, $("ctlType").value);
    if (chart) drawChart(host, chart, "explore");
    else host.innerHTML = '<div class="empty">No chart for this combination</div>';
    $("exploreMeta").textContent =
      `${res.rows_used.toLocaleString("en-IN")} rows used` +
      (res.filtered_out ? ` · ${res.filtered_out.toLocaleString("en-IN")} filtered out` : "") +
      (res.groups ? ` · ${res.groups} groups` : "") +
      (res.trend ? ` · trend ${res.trend.pct_change >= 0 ? "+" : ""}${(res.trend.pct_change * 100).toFixed(0)}% (R² ${res.trend.r2.toFixed(2)})` : "");
    renderResultTable(tableHost, res, 100);
  } catch (e) {
    host.innerHTML = `<div class="empty">${esc(e.message)}</div>`;
    $("exploreMeta").textContent = "Error";
  }
}

function renderResultTable(host, res, maxRows) {
  const cols = res.columns || [];
  const rows = (res.rows || []).slice(0, maxRows || 100);
  if (!rows.length) { host.innerHTML = '<div class="empty">No rows</div>'; return; }
  const isDict = !Array.isArray(rows[0]);
  const head = cols.map((c) =>
    `<th class="${c.type === "numeric" ? "num" : ""}">${esc(c.label)}</th>`).join("");
  const body = rows.map((r) => {
    const cells = cols.map((c) => {
      const v = isDict ? r[c.key] : r[cols.indexOf(c)];
      const num = c.type === "numeric";
      return `<td class="${num ? "num" : ""}">${esc(num && v !== null && v !== undefined ? fmt(v, c.format) : v)}</td>`;
    }).join("");
    return `<tr>${cells}</tr>`;
  }).join("");
  host.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function openInExplore(spec) {
  switchTab("explore");
  applySpecToControls(spec || {});
  runExplore(spec);
}

function applySpecToControls(spec) {
  const kindToType = { aggregate: "auto", timeseries: "line", distribution: "histogram",
                       correlation: "scatter", matrix: "heatmap", segments: "segments", table: "table" };
  $("ctlType").value = kindToType[spec.kind] || "auto";
  if (spec.measure !== undefined) $("ctlMeasure").value = spec.measure || "";
  if (spec.agg) $("ctlAgg").value = spec.agg;
  if (spec.dimension !== undefined) $("ctlDimension").value = spec.dimension || spec.time_dimension || "";
  if (spec.time_dimension) $("ctlDimension").value = spec.time_dimension;
  if (spec.granularity) $("ctlGran").value = spec.granularity;
  if (spec.y) $("ctlY").value = spec.y;
  if (spec.limit) $("ctlLimit").value = String(spec.limit);
  else if (!["5", "10", "15", "25"].includes($("ctlLimit").value)) $("ctlLimit").value = "";
  if (spec.sort) $("ctlSort").value = `${spec.sort.by || "measure"}:${spec.sort.dir || "desc"}`;
  renderFilterRows(spec.filters || []);
  syncControlVisibility();
}

/* ---- filters ---- */
function filterRowHTML(f = {}) {
  // default to the column the user is actually working with, not the first one
  const wanted = f.col
    || ($("ctlMeasure") && $("ctlMeasure").value && colType($("ctlMeasure").value) === "numeric"
        ? $("ctlMeasure").value
        : ($("ctlDimension") && $("ctlDimension").value) || S.columns[0]);
  const cols = S.columns.map((c) => `<option value="${esc(c)}" ${c === wanted ? "selected" : ""}>${esc(c)}</option>`).join("");
  const ops = [["eq", "="], ["ne", "≠"], ["gt", ">"], ["gte", "≥"], ["lt", "<"], ["lte", "≤"],
               ["contains", "contains"], ["in", "in"], ["is_null", "is empty"], ["not_null", "not empty"]]
    .map(([v, l]) => `<option value="${v}" ${v === (f.op || "eq") ? "selected" : ""}>${l}</option>`).join("");
  const val = Array.isArray(f.value) ? f.value.join(",") : (f.value ?? "");
  return `<div class="filter-row">
      <select data-k="col">${cols}</select>
      <select data-k="op">${ops}</select>
      <input data-k="value" type="text" value="${esc(val)}" placeholder="value" />
      <button class="icon-btn" data-k="del" title="Remove">✕</button>
    </div>`;
}
function renderFilterRows(filters) {
  const host = $("filterList");
  host.innerHTML = filters.length ? filters.map(filterRowHTML).join("") : "";
  host.querySelectorAll(".filter-row").forEach(wireFilterRow);
}
function wireFilterRow(row) {
  row.querySelector('[data-k="del"]').onclick = () => { row.remove(); runExplore(); };
  row.querySelectorAll("select,input").forEach((n) => {
    n.onchange = () => runExplore();
    if (n.tagName === "INPUT") n.onkeydown = (e) => { if (e.key === "Enter") runExplore(); };
  });
}
function readFilters() {
  return [...$("filterList").querySelectorAll(".filter-row")].map((row) => {
    const col = row.querySelector('[data-k="col"]').value;
    const op = row.querySelector('[data-k="op"]').value;
    let value = row.querySelector('[data-k="value"]').value.trim();
    if (op === "in") value = value.split(",").map((s) => s.trim()).filter(Boolean);
    if (["is_null", "not_null"].includes(op)) value = null;
    if (colType(col) === "numeric" && value !== null && !Array.isArray(value)) {
      const n = parseFloat(String(value).replace(/,/g, ""));
      if (!Number.isNaN(n)) value = n;
    }
    return { col, op, value };
  });
}
function syncControlVisibility() {
  const t = $("ctlType").value;
  $("scatterWrap").classList.toggle("hidden", t !== "scatter");
  $("granWrap").classList.toggle("hidden", !["auto", "line", "area"].includes(t));
}

/* ---- exports ---- */
function download(name, text, mime) {
  const blob = new Blob([text], { type: mime || "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}
async function exportTable(kind) {
  if (!S.explore.spec) return toast("Run a query first", true);
  try {
    const r = await api(`/api/datasets/${S.dataset.id}/export`, { method: "POST", body: { spec: S.explore.spec } });
    const stem = (S.dataset.name || "export").replace(/\.\w+$/, "");
    if (kind === "csv") download(`${stem}-query.csv`, r.csv, "text/csv");
    else download(`${stem}-query.md`, `# Query result\n\n${r.markdown}\n`, "text/markdown");
    toast(`${r.rows} rows exported`);
  } catch (e) { toast(e.message, true); }
}

/* ============================================================ ask */
function renderSuggestions(list) {
  const host = $("askSuggestions");
  host.innerHTML = "";
  (list || []).forEach((q) => {
    const b = el("button", "", esc(q));
    b.onclick = () => askQuestion(q);
    host.appendChild(b);
  });
}

async function askQuestion(q) {
  q = (q || $("askInput").value || "").trim();
  if (!q) return;
  switchTab("ask");
  $("askInput").value = "";
  const log = $("askLog");
  const msg = el("div", "msg");
  msg.appendChild(el("div", "q", esc(q)));
  const answer = el("div", "a", '<span class="loader"></span> thinking…');
  msg.appendChild(answer);
  log.prepend(msg);

  try {
    const r = await api(`/api/datasets/${S.dataset.id}/ask`, { method: "POST", body: { question: q } });
    answer.innerHTML = "";
    answer.appendChild(el("div", "answer-text", esc(r.answer)));

    const chips = el("div", "parse-chips");
    const sp = r.parsed.spec;
    const add = (k, v) => { if (v !== undefined && v !== null && v !== "") chips.appendChild(el("span", "chip", `${k} <b>${esc(v)}</b>`)); };
    add("intent", r.parsed.intent);
    add("measure", sp.measure);
    add("agg", sp.agg);
    add("by", sp.dimension || sp.time_dimension);
    add("x", sp.x); add("y", sp.y);
    add("column", sp.column);
    add("bucket", sp.granularity === "auto" ? null : sp.granularity);
    add("limit", sp.limit);
    (sp.filters || []).forEach((f) => add("filter", `${f.col} ${f.op} ${Array.isArray(f.value) ? f.value.join("|") : f.value}`));
    chips.appendChild(el("span", "chip", `engine <b>${r.engine}</b>`));
    chips.appendChild(el("span", "chip", `confidence <b>${Math.round((r.parsed.confidence || 0) * 100)}%</b>`));
    answer.appendChild(chips);

    if ((r.parsed.notes || []).length) {
      const d = el("details", "raw");
      d.innerHTML = `<summary>How this was parsed</summary><pre>${esc(r.parsed.notes.join("\n"))}</pre>`;
      answer.appendChild(d);
    }

    if (r.result.chart) {
      const holder = el("div", "chart-wrap");
      answer.appendChild(holder);
      drawChart(holder, r.result.chart, "ask" + Date.now() + Math.random());
    }
    if (r.result.rows && r.result.rows.length && r.result.columns) {
      const d = el("details", "raw");
      d.innerHTML = "<summary>Show data table</summary><div class='table-scroll' style='max-height:280px;margin-top:8px'></div>";
      answer.appendChild(d);
      renderResultTable(d.querySelector(".table-scroll"), r.result, 50);
    }
    if (r.follow_ups && r.follow_ups.length) {
      const s = el("div", "suggest");
      r.follow_ups.forEach((f) => {
        const b = el("button", "", esc(f));
        b.onclick = () => askQuestion(f);
        s.appendChild(b);
      });
      answer.appendChild(s);
    }
    if (r.llm_error) {
      answer.appendChild(el("div", "conf", "Model call failed, fell back to the local parser: " + esc(r.llm_error)));
    }
  } catch (e) {
    answer.innerHTML = `<div class="answer-text">Sorry — ${esc(e.message)}</div>`;
  }
}

/* ============================================================ data tab */
function renderSchema() {
  const host = $("schemaTable");
  const head = `<tr><th>Column</th><th>Type</th><th>Missing</th><th>Distinct</th><th>Detail</th></tr>`;
  const body = S.profile.map((c) => {
    let detail = "";
    if (c.type === "numeric") {
      detail = `min ${fmt(c.min)} · median ${fmt(c.median)} · mean ${fmt(c.mean)} · max ${fmt(c.max)}` +
        (c.outlier_count ? ` · <span style="color:var(--warn)">${c.outlier_count} outliers</span>` : "");
    } else if (c.type === "datetime") {
      detail = `${String(c.min).slice(0, 10)} → ${String(c.max).slice(0, 10)} · ${esc(c.granularity || "?")} granularity`;
    } else if (c.top) {
      detail = c.top.slice(0, 3).map((t) => `${esc(t.value)} (${Math.round(t.pct * 100)}%)`).join(", ");
    }
    const pct = (c.missing_pct || 0) * 100;
    return `<tr>
      <td><b>${esc(c.name)}</b></td>
      <td><span class="type-pill ${esc(c.type)}">${esc(c.type)}</span></td>
      <td>${pct ? `<div style="display:flex;gap:6px;align-items:center">${pct.toFixed(1)}%
        <span class="mini-bar"><i style="width:${Math.min(100, pct)}%"></i></span></div>` : "—"}</td>
      <td class="num">${c.distinct ? c.distinct.toLocaleString("en-IN") : "—"}</td>
      <td style="white-space:normal;max-width:420px;color:var(--muted)">${detail}</td>
    </tr>`;
  }).join("");
  host.innerHTML = `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

async function loadRows() {
  const t = S.table;
  const params = new URLSearchParams({ limit: t.limit, offset: t.offset });
  if (t.search) params.set("search", t.search);
  if (t.sortCol) { params.set("sort_col", t.sortCol); params.set("sort_dir", t.sortDir); }
  try {
    const r = await api(`/api/datasets/${S.dataset.id}/table?` + params);
    t.total = r.total;
    $("rowCount").textContent = `${r.total.toLocaleString("en-IN")} matching rows`;
    $("pageInfo").textContent = `${t.offset + 1}–${t.offset + r.rows.length} of ${r.total.toLocaleString("en-IN")}`;
    const host = $("rowTable");
    const head = r.columns.map((c) =>
      `<th data-col="${esc(c.key)}" class="${c.type === "numeric" ? "num" : ""}">${esc(c.label)}${t.sortCol === c.key ? (t.sortDir === "asc" ? " ↑" : " ↓") : ""}</th>`).join("");
    const body = r.rows.map((row) =>
      "<tr>" + r.columns.map((c) => {
        const v = row[c.key];
        return `<td class="${c.type === "numeric" ? "num" : ""}">${esc(v === null || v === undefined ? "" : v)}</td>`;
      }).join("") + "</tr>").join("");
    host.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    host.querySelectorAll("th").forEach((th) => {
      th.onclick = () => {
        const c = th.dataset.col;
        if (t.sortCol === c) t.sortDir = t.sortDir === "asc" ? "desc" : "asc";
        else { t.sortCol = c; t.sortDir = "desc"; }
        loadRows();
      };
    });
  } catch (e) { toast(e.message, true); }
}

/* ============================================================ settings */
function openSettings() {
  const host = $("modalHost");
  host.innerHTML = `
    <div class="modal-bg">
      <div class="modal">
        <h3>Model connection <span style="font-size:11px;color:var(--dim)">(optional)</span></h3>
        <div class="sub">The dashboard analyses your data locally. Connect any OpenAI-compatible endpoint
          and the model will translate harder questions and write the executive summary.
          Only the schema and aggregate statistics are sent — never your rows.</div>
        <label class="fld">Base URL</label>
        <input type="text" id="mBase" value="${esc(S.llm.base_url || "https://api.openai.com/v1")}" />
        <label class="fld" style="margin-top:10px">Model</label>
        <input type="text" id="mModel" value="${esc(S.llm.model || "gpt-4o-mini")}" />
        <label class="fld" style="margin-top:10px">API key</label>
        <input type="password" id="mKey" placeholder="${S.llm.has_key ? S.llm.api_key_masked : "sk-…"}" />
        <label style="display:flex;gap:8px;align-items:center;margin-top:12px;font-size:13px">
          <input type="checkbox" id="mEnabled" ${S.llm.enabled ? "checked" : ""} style="width:auto" /> Use the model for questions
        </label>
        <div class="row">
          <button class="btn" id="mTest">Test</button>
          <button class="btn" id="mCancel">Cancel</button>
          <button class="btn primary" id="mSave">Save</button>
        </div>
      </div>
    </div>`;
  $("mCancel").onclick = () => (host.innerHTML = "");
  $("mSave").onclick = async () => {
    const body = { base_url: $("mBase").value.trim(), model: $("mModel").value.trim(),
                   enabled: $("mEnabled").checked };
    const k = $("mKey").value.trim();
    if (k) body.api_key = k;
    try {
      S.llm = await api("/api/llm/settings", { method: "POST", body });
      renderEngineChip();
      host.innerHTML = "";
      toast("Model settings saved");
    } catch (e) { toast(e.message, true); }
  };
  $("mTest").onclick = async () => {
    const body = { base_url: $("mBase").value.trim(), model: $("mModel").value.trim(), enabled: true };
    const k = $("mKey").value.trim();
    if (k) body.api_key = k;
    await api("/api/llm/settings", { method: "POST", body });
    const r = await api("/api/llm/test", { method: "POST" });
    toast(r.ok ? "Connected: " + r.reply : "Failed: " + r.error, !r.ok);
  };
}

/* ============================================================ auth + billing */
async function refreshAuth() {
  try { S.auth = await api("/api/auth/me"); }
  catch (e) { S.auth = { guest: true, plan: "free", usage: {}, limits: {} }; }
  const b = $("btnAccount");
  b.textContent = S.auth.guest ? "Sign in"
    : `${(S.auth.email || "").split("@")[0]} · ${S.auth.plan.toUpperCase()}`;
  b.classList.toggle("primary", !S.auth.guest && S.auth.plan === "pro");
  const adm = $("btnAdmin");
  if (adm) adm.classList.toggle("hidden", !(S.auth && S.auth.is_admin));
}

function openAccount() {
  const host = $("modalHost");
  if (S.auth.guest) {
    host.innerHTML = `<div class="modal-bg"><div class="modal">
      <h3>Sign in / create account</h3>
      <div class="sub">Save your dashboards and unlock higher limits. Free plan:
        ${(S.auth.limits.questions ?? 100)} questions & ${(S.auth.limits.max_rows ?? 50000).toLocaleString()} rows /mo.</div>
      <label class="fld">Email</label><input type="text" id="aEmail" autocomplete="email" />
      <label class="fld" style="margin-top:10px">Password (8+ chars)</label><input type="password" id="aPass" />
      <div class="row">
        <button class="btn" id="aLogin">Sign in</button>
        <button class="btn primary" id="aSignup">Create free account</button>
      </div></div></div>`;
    const doIt = async (p) => {
      try {
        const r = await api(p, { method: "POST",
          body: { email: $("aEmail").value, password: $("aPass").value } });
        host.innerHTML = ""; await refreshAuth(); toast("Signed in as " + r.email);
      } catch (e) { toast(e.message, true); }
    };
    $("aLogin").onclick = () => doIt("/api/auth/login");
    $("aSignup").onclick = () => doIt("/api/auth/signup");
  } else {
    const u = S.auth.usage || {}, L = S.auth.limits || {};
    host.innerHTML = `<div class="modal-bg"><div class="modal">
      <h3>${esc(S.auth.email)}</h3>
      <div class="sub">Plan: <b style="color:var(--accent-2)">${S.auth.plan.toUpperCase()}</b></div>
      <div class="sub">This month: ${u.questions || 0} questions · ${u.exports || 0} exports · ${u.datasets || 0} datasets</div>
      <div class="sub">Free limits: ${(L.questions ?? 100)} questions, ${(L.max_rows ?? 50000).toLocaleString()} rows, ${L.exports ?? 20} exports /mo.</div>
      <div class="row">
        <button class="btn" id="aLogout">Sign out</button>
        ${S.auth.plan === "pro" ? '<button class="btn primary" id="aManage">Manage subscription</button>'
                                : '<button class="btn primary" id="aUpgrade">Upgrade to Pro</button>'}
      </div></div></div>`;
    $("aLogout").onclick = async () => {
      await api("/api/auth/logout", { method: "POST" });
      host.innerHTML = ""; await refreshAuth();
    };
    const up = $("aUpgrade"); if (up) up.onclick = openPricing;
    const mg = $("aManage"); if (mg) mg.onclick = manageSubscription;
  }
}

async function startCheckout() {
  try {
    const r = await api("/api/billing/checkout", { method: "POST" });
    if (r.mode === "stripe" && r.url) { window.open(r.url, "_blank"); toast("Finish payment in the new tab, then come back."); return; }
    showDryCheckout(r.price || 29);
  } catch (e) { toast(e.message, true); }
}

function showDryCheckout(price) {
  const host = $("modalHost");
  host.innerHTML = `<div class="modal-bg"><div class="modal">
    <h3>Stripe · test mode</h3>
    <div class="sub">No live keys on this server, so this is a simulated Checkout that
      runs the real upgrade path. Test card 4242 4242 4242 4242.</div>
    <div class="drycard"><div class="dc-top">AutoAnalytics <b>Pro</b></div>
      <div class="dc-amt">$${price}.00 <span>/ month</span></div>
      <div class="dc-cc">4242 4242 4242 4242 &nbsp;·&nbsp; 12/34 &nbsp;·&nbsp; 123</div></div>
    <div class="row"><button class="btn" id="dCancel">Cancel</button>
      <button class="btn primary" id="dPay">Pay $${price}.00</button></div></div></div>`;
  $("dCancel").onclick = () => (host.innerHTML = "");
  $("dPay").onclick = async () => {
    busy($("dPay"), true);
    try { await api("/api/billing/dryrun", { method: "POST" });
      host.innerHTML = ""; await refreshAuth(); toast("Welcome to Pro — limits lifted!"); }
    catch (e) { toast(e.message, true); }
    busy($("dPay"), false);
  };
}

function openPricing() {
  const host = $("modalHost"), auth = S.auth || {};
  const isPro = !auth.guest && auth.plan === "pro";
  const free = ["50,000 rows per file", "5 MB upload", "100 questions / mo", "20 exports / mo", "10 saved datasets"];
  const pro = ["5,000,000 rows per file", "40 MB upload", "Unlimited questions", "Unlimited exports", "2x-resolution PNG exports", "Priority support"];
  const li = (arr) => arr.map((f) => `<li>${f}</li>`).join("");
  host.innerHTML = `<div class="modal-bg"><div class="modal wide">
    <h3>Simple, transparent pricing</h3>
    <div class="sub">Start free. Upgrade the moment you need more.</div>
    <div class="plans">
      <div class="plan"><div class="pname">Free</div><div class="pprice">$0</div>
        <ul>${li(free)}</ul>
        <button class="btn" id="prFree">${auth.guest ? "Start free" : "Your plan"}</button></div>
      <div class="plan hl"><div class="tag">Popular</div><div class="pname">Pro</div>
        <div class="pprice">$29<span>/mo</span></div>
        <ul>${li(pro)}</ul>
        <button class="btn primary" id="prPro">${isPro ? "You're on Pro" : "Upgrade to Pro"}</button></div>
    </div>
    <div class="sub" style="margin-top:12px">${auth.guest
      ? "Create a free account, then upgrade in one click."
      : "Payments are handled by Stripe. Cancel anytime."}</div>
  </div></div>`;
  $("prFree").onclick = () => (auth.guest ? openAccount() : (host.innerHTML = ""));
  const p = $("prPro");
  if (isPro) p.onclick = () => (host.innerHTML = ""); else p.onclick = startCheckout;
}

async function manageSubscription() {
  const host = $("modalHost");
  try {
    const r = await api("/api/billing/portal", { method: "POST" });
    if (r.mode === "stripe" && r.url) { window.open(r.url, "_blank"); return; }
    host.innerHTML = `<div class="modal-bg"><div class="modal">
      <h3>Manage subscription</h3>
      <div class="sub">No live Stripe on this server, so this demo subscription can be
        cancelled right here. With live Stripe you'd be taken to the billing portal.</div>
      <div class="row"><button class="btn" id="mKeep">Keep Pro</button>
        <button class="btn danger" id="mCancel">Cancel Pro &rarr; Free</button></div></div></div>`;
    $("mKeep").onclick = () => (host.innerHTML = "");
    $("mCancel").onclick = async () => {
      try { await api("/api/billing/cancel", { method: "POST" });
        host.innerHTML = ""; await refreshAuth(); toast("Subscription cancelled — back on Free."); }
      catch (e) { toast(e.message, true); }
    };
  } catch (e) { toast(e.message, true); }
}

async function openAdmin() {
  const host = $("modalHost");
  host.innerHTML = `<div class="modal-bg"><div class="modal wide"><h3>Admin · usage</h3><div class="sub">Loading…</div></div></div>`;
  try {
    const r = await api("/api/admin/usage");
    const rows = r.accounts.map((x) =>
      `<tr><td>${esc(x.email)}</td><td>${x.plan.toUpperCase()}${x.is_admin ? " · admin" : ""}</td>` +
      `<td>${x.questions}</td><td>${x.exports}</td><td>${x.datasets}</td></tr>`).join("");
    host.innerHTML = `<div class="modal-bg"><div class="modal wide">
      <h3>Admin · usage — ${r.month}</h3>
      <div class="sub">${r.totals.accounts} accounts · ${r.totals.pro} on Pro ·
        ${r.totals.questions} questions · ${r.totals.exports} exports this month</div>
      <table class="utable"><thead><tr><th>Account</th><th>Plan</th><th>Questions</th>
        <th>Exports</th><th>Datasets</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="5">No accounts yet.</td></tr>'}</tbody></table>
    </div></div>`;
  } catch (e) {
    host.innerHTML = `<div class="modal-bg"><div class="modal"><h3>Admin</h3><div class="sub">${esc(e.message)}</div></div></div>`;
  }
}

function openPaywall(msg) {
  const host = $("modalHost");
  host.innerHTML = `<div class="modal-bg"><div class="modal">
    <h3>You've hit the free limit</h3>
    <div class="sub">${esc(msg || "Upgrade to Pro for higher limits.")}</div>
    <div class="row">
      <button class="btn" id="pClose">Not now</button>
      ${S.auth.guest ? '<button class="btn primary" id="pGo">Create free account</button>'
                     : '<button class="btn primary" id="pGo">Upgrade to Pro</button>'}
    </div></div></div>`;
  $("pClose").onclick = () => (host.innerHTML = "");
  $("pGo").onclick = () => (S.auth.guest ? openAccount() : openPricing());
}

/* ============================================================ events */
function wireEvents() {
  const dz = $("dropzone");
  dz.onclick = () => $("fileInput").click();
  $("fileInput").onchange = (e) => { if (e.target.files[0]) uploadFile(e.target.files[0]); };
  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) uploadFile(f);
  });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  $("btnTogglePaste").onclick = () => {
    $("pasteArea").classList.toggle("hidden");
    $("pasteText").focus();
  };
  $("btnPasteAnalyze").onclick = async () => {
    const csv = $("pasteText").value;
    if (!csv.trim()) return toast("Paste some CSV first", true);
    busy($("btnPasteAnalyze"), true);
    try {
      const r = await api("/api/paste", { method: "POST",
        body: { csv, name: ($("pasteName").value.trim() || "pasted") + ".csv" } });
      adopt(r);
    } catch (e) { toast(e.message, true); }
    busy($("btnPasteAnalyze"), false);
  };

  $("btnSamples").onclick = () => {
    $("viewApp").classList.add("hidden");
    $("viewLanding").classList.remove("hidden");
  };
  $("btnReset").onclick = () => {
    $("viewApp").classList.add("hidden");
    $("viewLanding").classList.remove("hidden");
    busy($("dropzone"), false);
  };
  $("btnSettings").onclick = openSettings;
  $("btnAccount").onclick = openAccount;
  $("btnPricing").onclick = openPricing;
  $("btnAdmin").onclick = openAdmin;

  document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => switchTab(t.dataset.tab)));

  ["ctlType", "ctlMeasure", "ctlAgg", "ctlDimension", "ctlGran", "ctlY", "ctlSort", "ctlLimit"]
    .forEach((id) => {
      $(id).onchange = () => { syncControlVisibility(); runExplore(); };
    });
  $("btnRun").onclick = () => runExplore();
  $("btnAddFilter").onclick = () => {
    $("filterList").insertAdjacentHTML("beforeend", filterRowHTML());
    wireFilterRow($("filterList").lastElementChild);
  };
  $("btnExportCsv").onclick = () => exportTable("csv");
  $("btnExportMd").onclick = () => exportTable("md");
  $("btnExportPng").onclick = async () => {
    const isPro = !!(S.auth && S.auth.plan === "pro");
    const scale = isPro ? 4 : 2;                    // Pro exports at 2x resolution
    const url = await Charts.toPNG($("exploreChart"), scale);
    if (!url) return toast("Nothing to export", true);
    const a = document.createElement("a");
    a.href = url; a.download = isPro ? "chart@2x.png" : "chart.png"; a.click();
    toast(isPro ? "Exported high-res (2x) PNG." : "Exported PNG. Pro unlocks 2x resolution.");
  };

  $("btnAsk").onclick = () => askQuestion($("askInput").value);
  $("askInput").addEventListener("keydown", (e) => { if (e.key === "Enter") askQuestion(e.target.value); });

  let searchTimer;
  $("rowSearch").addEventListener("input", (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { S.table.search = e.target.value.trim(); S.table.offset = 0; loadRows(); }, 250);
  });
  $("btnPrev").onclick = () => { S.table.offset = Math.max(0, S.table.offset - S.table.limit); loadRows(); };
  $("btnNext").onclick = () => {
    if (S.table.offset + S.table.limit < S.table.total) { S.table.offset += S.table.limit; loadRows(); }
  };
}

boot();
