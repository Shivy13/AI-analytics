# AI Analytics Dashboard

**Upload CSV → AI analyzes it → generates insights → interactive charts → natural-language questions.**

A self-contained analytics workbench. It profiles every column, writes the
interesting findings out in plain English, auto-builds a dashboard, and then
lets you ask questions about your data in natural language.

It runs **fully offline by default** — the "AI" is a transparent, deterministic
engine (type inference, statistics, an NL→query parser, and a prose generator).
Optionally you can connect any OpenAI-compatible endpoint to translate harder
questions and write the executive summary; only the schema + aggregate
statistics are ever sent, never your rows.

## Run

```bash
pip install fastapi "uvicorn[standard]" python-multipart pandas numpy scipy httpx
cd ai-analytics
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## Test

```bash
pip install pytest
python3 -m pytest tests/test_engine.py -q        # 52 checks, numbers verified vs independent pandas
```

Browser smoke test (optional, needs a browser):

```bash
pip install playwright && python3 -m playwright install chromium
python3 tests/ui_smoke.py                        # drives the real UI, fails on any console error
```

## What it does

- **Profile** — semantic type inference (numeric / categorical / datetime /
  boolean / id / text), missing %, distinct, quantiles, skew, outlier fences,
  date span & granularity.
- **Insights** — ~20 detectors: data-quality (duplicates, blanks, mixed types),
  outliers & skew, correlations, trends (with direction-aware severity),
  year-on-year (like-for-like months, so a partial current year can't fake a
  collapse), seasonality, weekday rhythm, concentration (Pareto), and
  strongest/weakest segments. Each finding carries its evidence, a metric,
  a chart and a follow-up question.
- **Ask** — a schema-aware parser maps questions to a structured query and the
  answer text is generated from the same numbers the chart is drawn from, so
  prose can't drift from the data. It shows its working: matched slots,
  filters, engine and confidence.
- **Explore** — a chart builder (bar/hbar/line/area/donut/histogram/scatter/
  correlation-matrix/segments/table) with filters, sorting, limits and
  CSV / Markdown / PNG export.
- **Data** — detected schema and a searchable, sortable, paginated row browser.

## Honesty by design

- Sum vs average is chosen per measure (a CSAT score or a resolution time is
  never *summed*).
- Share-of-total is only computed for additive aggregations.
- Trends on "lower is better" columns (cost, resolution time) are flagged as
  warnings, not wins.
- Year-on-year compares only the months present in both years.
- A comparison like "North vs South" becomes one `in` filter, not two
  contradictory ones.

## Layout

```
server.py            FastAPI app (upload, analyze, ask, query, export, LLM settings)
engine/
  loader.py          CSV parsing + type inference + store (survives restarts)
  profile.py         column profiling + measure/column selection
  semantics.py       which direction is "good", sum vs average
  insights.py        the finding detectors + dashboard builder
  query.py           structured query engine (one path for charts, tables, NL)
  nlq.py             natural-language parser + answer composition
  llm.py             optional OpenAI-compatible layer (validate + narrative)
  samples.py         bundled, deliberately-imperfect sample datasets
static/              dependency-free SPA (vanilla JS + hand-rolled SVG charts)
tests/               engine unit tests + Playwright UI smoke test
```
