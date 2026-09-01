"""Optional LLM layer.

Everything works without it (the local engine in nlq.py does the parsing).
When the user supplies an OpenAI-compatible endpoint, we use the model to
(a) translate harder questions into a query spec and (b) write the
executive narrative. We only ever send schema + aggregate statistics,
never raw row data.
"""
from __future__ import annotations

import json
import os
import re

import httpx

from .loader import DATA_DIR, Dataset
from .query import VALID_AGGS
from .utils import compact

SETTINGS_PATH = os.path.join(DATA_DIR, "llm.json")
ALLOWED_KINDS = {"aggregate", "timeseries", "distribution", "correlation", "matrix", "table", "segments"}
OPS = {"eq", "ne", "gt", "gte", "lt", "lte", "between", "in", "not_in", "contains",
       "not_contains", "startswith", "is_null", "not_null", "year", "month"}


class LLMError(RuntimeError):
    pass


def get_settings() -> dict:
    if os.path.exists(SETTINGS_PATH):
        try:
            return json.load(open(SETTINGS_PATH))
        except Exception:
            pass
    return {"enabled": False, "base_url": "https://api.openai.com/v1", "api_key": "", "model": "gpt-4o-mini"}


def save_settings(patch: dict) -> dict:
    cur = get_settings()
    for k in ("enabled", "base_url", "api_key", "model"):
        if k in patch and patch[k] is not None:
            cur[k] = patch[k]
    if not cur["base_url"]:
        cur["base_url"] = "https://api.openai.com/v1"
    json.dump(cur, open(SETTINGS_PATH, "w"), indent=2)
    return cur


def available() -> bool:
    s = get_settings()
    return bool(s.get("enabled") and s.get("api_key"))


def masked() -> dict:
    s = get_settings()
    k = s.get("api_key") or ""
    return {"enabled": bool(s.get("enabled")), "base_url": s.get("base_url"),
            "model": s.get("model"), "has_key": bool(k),
            "api_key_masked": (k[:6] + "…" + k[-4:]) if len(k) > 12 else ("set" if k else "")}


def _chat(messages: list[dict], json_mode: bool = True, max_tokens: int = 1200) -> str:
    s = get_settings()
    if not s.get("api_key"):
        raise LLMError("No API key configured.")
    url = s.get("base_url", "").rstrip("/") + "/chat/completions"
    payload = {"model": s.get("model") or "gpt-4o-mini", "messages": messages,
               "temperature": 0.1, "max_tokens": max_tokens}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    try:
        r = httpx.post(url, json=payload, timeout=75.0,
                       headers={"Authorization": f"Bearer {s['api_key']}"})
    except httpx.HTTPError as e:
        raise LLMError(f"Network error reaching the model endpoint: {e}") from e
    if r.status_code != 200:
        raise LLMError(f"Model endpoint returned {r.status_code}: {r.text[:300]}")
    try:
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise LLMError(f"Unexpected response shape: {e}") from e


def schema_prompt(ds: Dataset) -> str:
    by = (ds.profile or {}).get("by_name", {})
    lines = []
    for c in ds.df.columns:
        p = by.get(c, {})
        t = ds.types.get(c, "text")
        extra = ""
        if t == "numeric":
            extra = f" (min {compact(p.get('min'))}, max {compact(p.get('max'))}, mean {compact(p.get('mean'))})"
        elif t in ("categorical", "boolean"):
            vals = ", ".join(v["value"] for v in (p.get("top") or [])[:6])
            extra = f" ({p.get('distinct')} values, e.g. {vals})"
        elif t == "datetime":
            extra = f" ({p.get('min', '')[:10]} → {p.get('max', '')[:10]}, {p.get('granularity')} granularity)"
        lines.append(f"- {c}: {t}{extra}")
    return (f"Dataset '{ds.name}': {ds.rows:,} rows.\n" + "\n".join(lines))


SPEC_RULES = """Return ONLY a JSON object describing an analytics query. Schema:
{
 "kind": "aggregate" | "timeseries" | "distribution" | "correlation" | "matrix" | "segments" | "table",
 "measure": "<numeric column or null>",
 "agg": "sum"|"avg"|"count"|"count_distinct"|"median"|"min"|"max"|"std"|"p90",
 "dimension": "<categorical column or null>",
 "time_dimension": "<datetime column or null>",
 "granularity": "auto"|"day"|"week"|"month"|"quarter"|"year",
 "filters": [{"col": "<column>", "op": "eq|ne|gt|gte|lt|lte|between|in|contains|is_null|not_null|year|month", "value": <scalar or array>}],
 "x": "<numeric column, correlation only>", "y": "<numeric column, correlation only>",
 "column": "<column, distribution only>",
 "sort": {"by": "measure"|"dimension", "dir": "desc"|"asc"},
 "limit": <int or null>
}
Rules: only use column names that exist. Use agg "count" with measure null for row counts.
For "top N" set limit=N and sort desc. Dates in filters use "YYYY-MM-DD"."""


def nl_to_spec(ds: Dataset, question: str) -> dict:
    content = _chat([
        {"role": "system", "content": "You convert analytics questions into query specs.\n" + SPEC_RULES},
        {"role": "user", "content": schema_prompt(ds) + f"\n\nQuestion: {question}"},
    ])
    try:
        raw = re.search(r"\{.*\}", content, re.S)
        spec = json.loads(raw.group(0) if raw else content)
    except Exception as e:
        raise LLMError(f"Model did not return valid JSON: {e}") from e
    return validate_spec(ds, spec)


def validate_spec(ds: Dataset, spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise LLMError("Spec is not an object")
    cols = set(ds.df.columns)
    kind = spec.get("kind", "aggregate")
    if kind not in ALLOWED_KINDS:
        kind = "aggregate"
    out = {"kind": kind, "filters": []}
    for key in ("measure", "dimension", "time_dimension", "x", "y", "column"):
        v = spec.get(key)
        if v:
            if v not in cols:
                raise LLMError(f"Unknown column '{v}' for {key}")
            out[key] = v
    agg = (spec.get("agg") or ("count" if not out.get("measure") else "sum")).lower()
    out["agg"] = agg if agg in VALID_AGGS else "sum"
    out["granularity"] = spec.get("granularity") or "auto"
    if spec.get("limit") is not None:
        try:
            out["limit"] = max(1, min(500, int(spec["limit"])))
        except (TypeError, ValueError):
            pass
    if isinstance(spec.get("sort"), dict):
        out["sort"] = {"by": spec["sort"].get("by", "measure"),
                       "dir": spec["sort"].get("dir", "desc")}
    for f in spec.get("filters") or []:
        if isinstance(f, dict) and f.get("col") in cols and (f.get("op") or "eq") in OPS:
            out["filters"].append({"col": f["col"], "op": f.get("op") or "eq", "value": f.get("value")})
    if out["kind"] == "correlation" and not (out.get("x") and out.get("y")):
        raise LLMError("Correlation needs both x and y")
    return out


def narrative(ds: Dataset, insights: list[dict], kpis: list[dict]) -> str:
    bullets = []
    for i in insights[:8]:
        bullets.append(f"- [{i['severity']}] {i['title']}: {i['summary']}")
    kpi_txt = ", ".join(f"{k['label']}={k['value']}" for k in kpis[:6])
    try:
        return _chat([
            {"role": "system", "content":
                "You are a terse senior data analyst. Write a 120-180 word executive summary in plain English. "
                "Use ONLY the facts given. No invented numbers, no hedging filler, no markdown headings. "
                "End with one sentence naming the single most important thing to investigate."},
            {"role": "user", "content": f"Dataset: {ds.name} ({ds.rows:,} rows, {ds.profile['columns']} columns)\n"
                                        f"KPIs: {kpi_txt}\n\nFindings:\n" + "\n".join(bullets)},
        ], json_mode=False, max_tokens=500).strip()
    except LLMError as e:
        return f"(LLM narrative unavailable: {e})"


def test_connection() -> dict:
    try:
        out = _chat([{"role": "user", "content": "Reply with the single word: ready"}],
                    json_mode=False, max_tokens=10)
        return {"ok": True, "reply": out.strip()[:120]}
    except LLMError as e:
        return {"ok": False, "error": str(e)}
