"""Structured query engine. Every chart, table and natural-language answer
is produced by executing one of these specs, which keeps the "AI" honest:
the prose is generated from the same numbers the chart is drawn from."""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from .loader import Dataset
from .profile import auto_granularity, period_code, resample_key
from .semantics import default_agg, higher_is_better
from .utils import js

CURRENCY_HINTS = ("price", "revenue", "cost", "profit", "amount", "sales", "spend",
                  "salary", "gmv", "value", "fee", "budget", "margin_amt", "total", "income", "bill")
PERCENT_HINTS = ("pct", "percent", "percentage", "rate", "ratio", "share", "margin", "discount", "ctr")
COUNT_HINTS = ("count", "qty", "quantity", "units", "orders", "tickets", "visits", "sessions", "clicks", "impressions")

from .semantics import default_agg, higher_is_better  # re-exported for insights


AGG_LABELS = {
    "sum": "Total", "avg": "Average", "mean": "Average", "count": "Count",
    "count_distinct": "Distinct count", "median": "Median", "min": "Minimum",
    "max": "Maximum", "std": "Std. dev", "p90": "90th pct", "p10": "10th pct",
}
VALID_AGGS = set(AGG_LABELS)


def guess_format(col: str | None, values=None) -> str:
    if not col:
        return "number"
    low = col.lower()
    if any(h in low for h in CURRENCY_HINTS):
        return "currency"
    if any(h in low for h in PERCENT_HINTS):
        if values is not None and len(values):
            m = np.nanmax(np.abs(np.asarray(values, dtype=float)))
            if m is not None and not np.isnan(m) and m <= 100:
                return "percent_raw"
        return "percent_raw"
    if any(h in low for h in COUNT_HINTS):
        return "number"
    return "number"


def _num_label(m: dict) -> str:
    col = m.get("col")
    if col is None:
        return "Rows"
    return m.get("label") or f"{AGG_LABELS.get(m.get('agg', 'sum'), 'Value')} {col}"


def normalize_measures(spec: dict) -> list[dict]:
    ms = spec.get("measures")
    if ms:
        out = []
        for m in ms:
            if isinstance(m, str):
                m = {"col": m}
            out.append({"col": m.get("col"), "agg": (m.get("agg") or "sum").lower(),
                        "label": m.get("label")})
        return [m for m in out if m["agg"] in VALID_AGGS]
    agg = (spec.get("agg") or ("count" if not spec.get("measure") else "sum")).lower()
    if agg not in VALID_AGGS:
        agg = "sum"
    return [{"col": spec.get("measure"), "agg": agg, "label": spec.get("label")}]


# ------------------------------------------------------------------
# filters
# ------------------------------------------------------------------
def _coerce_value(ds: Dataset, col: str, value):
    sem = ds.types.get(col)
    if sem == "numeric":
        try:
            return float(str(value).replace(",", "").replace("%", ""))
        except (TypeError, ValueError):
            return None
    if sem == "datetime":
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y", "%d/%m/%Y", "%m/%d/%Y", "%b %Y", "%B %Y"):
            try:
                return pd.Timestamp(pd.to_datetime(str(value), format=fmt))
            except Exception:
                continue
        try:
            return pd.Timestamp(pd.to_datetime(str(value)))
        except Exception:
            return None
    return str(value).strip().lower()


def apply_filters(ds: Dataset, filters: list[dict] | None) -> pd.Series:
    mask = pd.Series(True, index=ds.df.index)
    for f in filters or []:
        col, op = f.get("col"), (f.get("op") or "eq").lower()
        if col not in ds.df.columns:
            continue
        s = ds.series(col)
        val = f.get("value")
        if op == "is_null":
            mask &= s.isna()
            continue
        if op == "not_null":
            mask &= s.notna()
            continue
        if ds.types.get(col) == "numeric":
            v = _coerce_value(ds, col, val)
            if v is None:
                continue
            if op in ("eq", "="):
                mask &= s == v
            elif op in ("ne", "!="):
                mask &= ~(s == v)
            elif op in ("gt", ">"):
                mask &= s > v
            elif op in ("gte", ">="):
                mask &= s >= v
            elif op in ("lt", "<"):
                mask &= s < v
            elif op in ("lte", "<="):
                mask &= s <= v
            elif op == "between":
                lo, hi = (list(val) + [None, None])[:2] if isinstance(val, (list, tuple)) else (val, None)
                lo = _coerce_value(ds, col, lo)
                hi = _coerce_value(ds, col, hi)
                if lo is not None and hi is not None:
                    mask &= s.between(lo, hi)
            continue
        if ds.types.get(col) == "datetime":
            v = _coerce_value(ds, col, val)
            if v is None:
                continue
            if op in ("eq", "="):
                mask &= s.dt.normalize() == v.normalize()
            elif op in ("gt", ">", "after", "since"):
                mask &= s > v
            elif op in ("lt", "<", "before"):
                mask &= s < v
            elif op == "between":
                lo, hi = (list(val) + [None, None])[:2] if isinstance(val, (list, tuple)) else (val, None)
                lo, hi = _coerce_value(ds, col, lo), _coerce_value(ds, col, hi)
                if lo is not None and hi is not None:
                    mask &= s.between(lo, hi)
            elif op in ("year",):
                mask &= s.dt.year == int(v.year)
            elif op in ("month",):
                mask &= s.dt.month == int(v.month)
            continue
        # categorical / text / boolean / id
        sval = s.astype("string").str.strip().str.lower()
        v = str(val).strip().lower() if not isinstance(val, (list, tuple)) else [str(x).strip().lower() for x in val]
        if op in ("gt", ">", "gte", ">=", "lt", "<", "lte", "<="):
            # a comparison on a text column: numeric when the column parses as
            # numbers, otherwise lexicographic - never silently ignored
            num = pd.to_numeric(sval, errors="coerce")
            try:
                nv = float(str(val).replace(",", ""))
            except (TypeError, ValueError):
                nv = None
            if nv is not None and num.notna().mean() > 0.5:
                cmp_series, cmp_val = num, nv
            else:
                cmp_series, cmp_val = sval, v
            if op in ("gt", ">"):
                mask &= cmp_series > cmp_val
            elif op in ("gte", ">="):
                mask &= cmp_series >= cmp_val
            elif op in ("lt", "<"):
                mask &= cmp_series < cmp_val
            else:
                mask &= cmp_series <= cmp_val
            continue
        if op in ("eq", "="):
            mask &= sval == v
        elif op in ("ne", "!="):
            mask &= ~(sval == v)
        elif op in ("in",):
            vals = v if isinstance(v, list) else [v]
            mask &= sval.isin(vals)
        elif op in ("not_in",):
            vals = v if isinstance(v, list) else [v]
            mask &= ~sval.isin(vals)
        elif op in ("contains", "like"):
            mask &= sval.str.contains(re.escape(v), na=False, regex=True)
        elif op in ("not_contains",):
            mask &= ~sval.str.contains(re.escape(v), na=False, regex=True)
        elif op == "startswith":
            mask &= sval.str.startswith(v, na=False)
    return mask


def _agg_series(s: pd.Series, agg: str) -> float:
    if agg == "count":
        return float(len(s))
    if agg == "count_distinct":
        return float(s.nunique(dropna=True))
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return float("nan")
    return {
        "sum": lambda: float(v.sum()), "avg": lambda: float(v.mean()), "mean": lambda: float(v.mean()),
        "median": lambda: float(v.median()), "min": lambda: float(v.min()), "max": lambda: float(v.max()),
        "std": lambda: float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "p90": lambda: float(v.quantile(0.9)), "p10": lambda: float(v.quantile(0.1)),
    }[agg]()


def _agg_group(df: pd.DataFrame, col: str | None, agg: str) -> float:
    if agg == "count":
        return float(len(df))
    if col is None:
        return float(len(df))
    if agg == "count_distinct":
        return float(df[col].nunique(dropna=True))
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    if v.empty:
        return float("nan")
    return _agg_series(v, agg)


def describe_filters(ds: Dataset, filters: list[dict]) -> list[str]:
    out = []
    sym = {"eq": "=", "ne": "≠", "gt": ">", "gte": "≥", "lt": "<", "lte": "≤",
           "contains": "contains", "not_contains": "excludes", "in": "in", "between": "between",
           "is_null": "is empty", "not_null": "is not empty", "startswith": "starts with"}
    for f in filters or []:
        col = f.get("col")
        if col not in ds.df.columns:
            continue
        op = (f.get("op") or "eq").lower()
        val = f.get("value")
        if isinstance(val, (list, tuple)):
            val = " and ".join(str(x) for x in val)
        if op in ("is_null", "not_null"):
            out.append(f"{col} {sym[op]}")
        elif ds.types.get(col) == "datetime" and op in ("year", "month"):
            out.append(f"{col} in {pd.Timestamp(val).year if op == 'year' else pd.Timestamp(val).strftime('%B %Y')}")
        else:
            out.append(f"{col} {sym.get(op, op)} {val}")
    return out


# ------------------------------------------------------------------
# main execution
# ------------------------------------------------------------------
def run_query(ds: Dataset, spec: dict) -> dict:
    kind = (spec.get("kind") or "aggregate").lower()
    filters = spec.get("filters") or []
    mask = apply_filters(ds, filters)
    df = ds.df.loc[mask]
    n_used = int(len(df))
    filtered_out = int(len(ds.df) - n_used)
    base = {
        "kind": kind,
        "rows_used": n_used,
        "rows_total": int(len(ds.df)),
        "filtered_out": filtered_out,
        "filter_descriptions": describe_filters(ds, filters),
        "spec": js(spec),
    }
    if kind == "correlation":
        return {**base, **_correlation(ds, df, spec)}
    if kind == "distribution":
        return {**base, **_distribution(ds, df, spec)}
    if kind == "matrix":
        return {**base, **_matrix(ds, df, spec)}
    if kind == "table":
        return {**base, **_table(ds, df, spec)}
    if kind == "segments":
        return {**base, **_segments(ds, df, spec)}
    if spec.get("time_dimension") or kind == "timeseries":
        return {**base, **_timeseries(ds, df, spec)}
    return {**base, **_aggregate(ds, df, spec)}


def _common(ds: Dataset, df: pd.DataFrame, spec: dict):
    measures = normalize_measures(spec)
    for m in measures:
        if m["col"] and m["col"] not in ds.df.columns:
            m["col"] = None
    return measures


_PANDAS_AGG = {"sum": "sum", "avg": "mean", "mean": "mean", "median": "median",
               "min": "min", "max": "max", "std": "std", "p90": None, "p10": None}


def _agg_frame(g, col: str, agg: str) -> pd.Series:
    if agg == "p90":
        return g[col].quantile(0.9)
    if agg == "p10":
        return g[col].quantile(0.1)
    return getattr(g[col], _PANDAS_AGG[agg])()


def _aggregate(ds: Dataset, df: pd.DataFrame, spec: dict) -> dict:
    measures = _common(ds, df, spec)
    dim = spec.get("dimension")
    if dim not in ds.df.columns:
        dim = None
    limit = None
    if spec.get("limit") is not None:
        try:
            limit = int(spec.get("limit"))
        except (TypeError, ValueError):
            limit = None
    other_bucket = bool(spec.get("other_bucket", False))
    y_labels = [_num_label(m) for m in measures]

    if dim is None:
        rows = [{"label": "All rows", "count": int(len(df)),
                 **{lab: _agg_group(df, m["col"], m["agg"]) for lab, m in zip(y_labels, measures)}}]
        groups, total_rows = 1, rows
    else:
        key = ds.series(dim).loc[df.index]
        if ds.types.get(dim) == "datetime":
            key = key.dt.strftime("%Y-%m-%d")
        work = pd.DataFrame({"__k__": key.astype("string").fillna("(blank)").values}, index=df.index)
        for j, m in enumerate(measures):
            if m["col"] is None or m["agg"] == "count":
                continue
            work[f"__v{j}__"] = pd.to_numeric(ds.series(m["col"]).loc[df.index],
                                              errors="coerce").values
        g = work.groupby("__k__", sort=False)
        size = g.size()
        agg_frames = {}
        for j, m in enumerate(measures):
            if m["agg"] == "count" or m["col"] is None:
                agg_frames[j] = size            # matches _agg_group: row count of the group
            elif m["agg"] == "count_distinct":
                agg_frames[j] = g[f"__v{j}__"].nunique()
            else:
                agg_frames[j] = _agg_frame(g, f"__v{j}__", m["agg"])
        recs = []
        for k in size.index:
            rec = {"label": str(k), "count": int(size[k])}
            for j, lab in enumerate(y_labels):
                v = agg_frames[j].get(k)
                rec[lab] = float(v) if v is not None and pd.notna(v) else None
            recs.append(rec)
        groups = len(recs)

        primary = y_labels[0] if y_labels else "count"
        reverse = (spec.get("sort") or {}).get("dir", "desc") != "asc"
        if (spec.get("sort") or {}).get("by", "measure") == "dimension":
            recs.sort(key=lambda r: str(r["label"]).lower(), reverse=reverse)
        else:
            def _key(r):
                v = r.get(primary)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    return float("-inf") if reverse else float("inf")
                return float(v)
            recs.sort(key=_key, reverse=reverse)
        total_rows = recs
        if limit and limit > 0 and len(recs) > limit:
            head, rest = recs[:limit], recs[limit:]
            if other_bucket:
                other = {"label": f"Other ({len(rest)} groups)",
                         "count": sum(r["count"] for r in rest)}
                for j, lab in enumerate(y_labels):
                    vals = [r[lab] for r in rest if r.get(lab) is not None]
                    if not vals:
                        other[lab] = None
                    elif measures[j]["agg"] in ("sum", "count"):
                        other[lab] = float(sum(vals))
                    else:
                        other[lab] = float(np.mean(vals))
                rows = head + [other]
            else:
                rows = head
        else:
            rows = recs

    columns = [{"key": "label", "label": dim or "Group",
                "type": ds.types.get(dim, "text") if dim else "text"},
               {"key": "count", "label": "Rows", "type": "numeric"}]
    columns += [{"key": y, "label": y, "type": "numeric",
                 "format": guess_format(m["col"], [r[y] for r in rows])}
                for y, m in zip(y_labels, measures)]
    totals = {"count": int(len(df)),
              **{lab: _agg_group(df, m["col"], m["agg"]) for lab, m in zip(y_labels, measures)}}

    prim = y_labels[0] if y_labels else None
    tot = totals.get(prim)
    additive = (measures[0]["agg"] in ("sum", "count")) if measures else False
    if prim and tot and additive:
        for r in rows:
            v = r.get(prim)
            r["pct_of_total"] = (v / tot) if (v is not None and tot) else None

    chart = None
    if dim is not None:
        vals = [r["label"] for r in rows]
        long_labels = any(len(str(v)) > 14 for v in vals)
        ctype = "hbar" if (len(vals) > 8 or long_labels) else "bar"
        if len(vals) == 2:
            ctype = "bar"
        chart = {
            "type": ctype, "title": f"{y_labels[0]} by {dim}" if y_labels else f"Rows by {dim}",
            "subtitle": None,
            "x": {"label": dim, "values": vals, "type": "category"},
            "y": {"label": y_labels[0] if y_labels else "Rows",
                  "series": [{"name": y, "values": [r.get(y) for r in rows]} for y in y_labels]},
            "format": guess_format(measures[0]["col"], [r.get(y_labels[0]) for r in rows]) if y_labels else "number",
            "show_pct": bool(tot and additive),
        }

    return {"columns": columns, "rows": js(rows), "all_rows": js(total_rows[:200]),
            "totals": js(totals), "groups": groups, "groups_shown": len(rows),
            "chart": chart, "measure_labels": y_labels, "dimension": dim}


def _timeseries(ds: Dataset, df: pd.DataFrame, spec: dict) -> dict:
    measures = _common(ds, df, spec)
    tcol = spec.get("time_dimension")
    if tcol not in ds.df.columns or ds.types.get(tcol) != "datetime":
        cands = ds.datetime_cols
        if not cands:
            return {"error": "No datetime column available for a time series", "columns": [], "rows": []}
        tcol = cands[0]
    ts = ds.series(tcol).loc[df.index]
    gran = (spec.get("granularity") or "auto").lower()
    if gran == "auto":
        gran = auto_granularity(ts)
    code = period_code(gran)
    key = resample_key(ts, gran)
    work = pd.DataFrame({"__k__": key.values, "__i__": df.index.values}).dropna(subset=["__k__"])
    labels, rows = [], []
    for k, grp in work.groupby("__k__", sort=True):
        sub = df.loc[grp["__i__"].values]
        rec = {"label": _period_label(k, code), "period": k.isoformat(), "count": int(len(sub))}
        for m in measures:
            rec[_num_label(m)] = _agg_group(sub, m["col"], m["agg"])
        rows.append(rec)
        labels.append(_period_label(k, code))
    y_labels = [_num_label(m) for m in measures]
    totals = {"count": int(len(df))}
    for m in measures:
        totals[_num_label(m)] = _agg_group(df, m["col"], m["agg"])

    # trend line on the primary measure (OLS over the bucket index)
    annotations = []
    trend = None
    if y_labels:
        ys = np.array([r.get(y_labels[0]) for r in rows], dtype=float)
        xs = np.arange(len(ys), dtype=float)
        ok = ~np.isnan(ys)
        if ok.sum() >= 3:
            slope, intercept = np.polyfit(xs[ok], ys[ok], 1)
            fitted = slope * xs + intercept
            ss_res = float(np.nansum((ys[ok] - fitted[ok]) ** 2))
            ss_tot = float(np.nansum((ys[ok] - np.nanmean(ys[ok])) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
            mean_y = float(np.nanmean(ys[ok]))
            total_change = float(slope * (len(ys) - 1))
            trend = {"slope": float(slope), "r2": float(r2), "total_change": total_change,
                     "pct_change": float(total_change / mean_y) if mean_y else None,
                     "granularity": gran}
            annotations.append({"type": "trendline", "values": js(fitted),
                                "label": "Trend", "color": "#f59e0b"})
    chart = {
        "type": "area" if len(y_labels) == 1 else "line",
        "title": f"{y_labels[0]} over time" if y_labels else f"Rows over time",
        "subtitle": f"by {gran}" if gran else None,
        "x": {"label": tcol, "values": labels, "type": "time"},
        "y": {"label": y_labels[0] if y_labels else "Rows",
              "series": [{"name": y, "values": [r.get(y) for r in rows]} for y in y_labels]},
        "format": guess_format(measures[0]["col"], [r.get(y_labels[0]) for r in rows]) if y_labels else "number",
        "annotations": annotations,
    }
    columns = [{"key": "label", "label": tcol, "type": "text"},
               {"key": "count", "label": "Rows", "type": "numeric"}]
    columns += [{"key": y, "label": y, "type": "numeric"} for y in y_labels]
    return {"columns": columns, "rows": js(rows), "all_rows": js(rows[:400]), "totals": js(totals),
            "groups": len(rows), "groups_shown": len(rows), "chart": chart,
            "measure_labels": y_labels, "dimension": tcol, "granularity": gran, "trend": js(trend)}


def _period_label(k, code: str) -> str:
    k = pd.Timestamp(k)
    if code == "Q":
        return f"Q{((k.month - 1) // 3) + 1} {k.year}"
    return k.strftime({"D": "%d %b", "W": "w/c %d %b", "M": "%b %Y", "Y": "%Y"}.get(code, "%d %b %Y"))


def _correlation(ds: Dataset, df: pd.DataFrame, spec: dict) -> dict:
    x, y = spec.get("x"), spec.get("y")
    nums = ds.numeric_cols
    if x not in nums or y not in nums:
        return {"error": "Correlation needs two numeric columns", "columns": [], "rows": [],
                "available": nums}
    a = pd.to_numeric(ds.series(x).loc[df.index], errors="coerce")
    b = pd.to_numeric(ds.series(y).loc[df.index], errors="coerce")
    both = a.notna() & b.notna()
    a, b = a[both], b[both]
    if len(a) < 3:
        return {"error": "Not enough overlapping values", "columns": [], "rows": []}
    r = float(np.corrcoef(a, b)[0, 1])
    rho = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    slope, intercept = np.polyfit(a, b, 1)
    points = list(zip(a.tolist(), b.tolist()))
    if len(points) > 1500:
        step = max(1, len(points) // 1500)
        points = points[::step]
    lo, hi = float(min(a)), float(max(a))
    chart = {
        "type": "scatter", "title": f"{y} vs {x}", "subtitle": f"Pearson r = {r:+.2f}",
        "x": {"label": x, "values": js([p[0] for p in points]), "type": "number"},
        "y": {"label": y, "series": [{"name": y, "values": js([p[1] for p in points])}]},
        "format": guess_format(y),
        "annotations": [{"type": "line", "x1": lo, "y1": intercept + slope * lo,
                         "x2": hi, "y2": intercept + slope * hi, "label": "fit", "color": "#f59e0b"}],
        "x_format": guess_format(x),
    }
    return {"columns": [{"key": "x", "label": x, "type": "numeric"}, {"key": "y", "label": y, "type": "numeric"}],
            "rows": js([{"x": p[0], "y": p[1]} for p in points[:500]]),
            "correlation": js({"pearson": r, "spearman": rho, "r2": r * r, "slope": float(slope),
                               "intercept": float(intercept), "n": int(len(a))}),
            "chart": chart, "groups": int(len(a))}


def _distribution(ds: Dataset, df: pd.DataFrame, spec: dict) -> dict:
    col = spec.get("column") or spec.get("measure")
    sem = ds.types.get(col)
    if sem not in ("numeric", "datetime", "categorical", "text", "id", "boolean"):
        return {"error": "Pick a column to profile", "columns": [], "rows": []}
    s = ds.series(col).loc[df.index].dropna()
    if sem in ("categorical", "text", "id", "boolean"):
        vc = s.astype("string").value_counts()
        head = vc.head(15)
        chart = {"type": "hbar" if len(head) > 6 else "bar", "title": f"Distribution of {col}",
                 "x": {"label": col, "values": js(list(head.index)), "type": "category"},
                 "y": {"label": "Rows", "series": [{"name": "Rows", "values": js(head.values)}]},
                 "format": "number"}
        rows = [{"label": str(k), "count": int(v), "pct_of_total": float(v / len(s))} for k, v in vc.items()]
        return {"columns": [{"key": "label", "label": col, "type": "text"},
                            {"key": "count", "label": "Rows", "type": "numeric"},
                            {"key": "pct_of_total", "label": "Share", "type": "numeric", "format": "percent"}],
                "rows": js(rows[:500]), "totals": {"count": int(len(s))}, "chart": chart,
                "groups": int(vc.size), "stats": js({"n": int(len(s)), "distinct": int(vc.size)})}
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return {"error": "No numeric values", "columns": [], "rows": []}
    bins = min(30, max(6, int(np.sqrt(len(v)))))
    counts, edges = np.histogram(v, bins=bins)
    labels = [f"{edges[i]:g}" for i in range(len(edges) - 1)]
    chart = {"type": "histogram", "title": f"Distribution of {col}",
             "subtitle": f"n = {len(v):,} · {bins} bins",
             "x": {"label": col, "values": labels, "type": "number",
                   "edges": js(edges.tolist())},
             "y": {"label": "Rows", "series": [{"name": "Rows", "values": js(counts.tolist())}]},
             "format": "number",
             "annotations": [{"type": "vline", "x": float(v.mean()), "label": "mean", "color": "#f59e0b"},
                             {"type": "vline", "x": float(v.median()), "label": "median", "color": "#22d3ee"}]}
    q = v.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    stats = {"n": int(len(v)), "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
             "median": float(v.median()), "std": float(v.std(ddof=1)), "p05": float(q[0.05]),
             "p25": float(q[0.25]), "p75": float(q[0.75]), "p95": float(q[0.95]),
             "skew": float(v.skew()), "kurtosis": float(v.kurt())}
    rows = [{"label": f"{edges[i]:g} – {edges[i + 1]:g}", "count": int(counts[i]),
             "pct_of_total": float(counts[i] / len(v))} for i in range(len(counts))]
    return {"columns": [{"key": "label", "label": "Bin", "type": "text"},
                        {"key": "count", "label": "Rows", "type": "numeric"},
                        {"key": "pct_of_total", "label": "Share", "type": "numeric", "format": "percent"}],
            "rows": js(rows), "totals": {"count": int(len(v))}, "chart": chart,
            "groups": len(rows), "stats": js(stats)}


def _matrix(ds: Dataset, df: pd.DataFrame, spec: dict) -> dict:
    cols = [c for c in (spec.get("columns") or ds.numeric_cols)[:12] if c in ds.df.columns]
    if len(cols) < 2:
        return {"error": "Need at least two numeric columns", "columns": [], "rows": []}
    mat = pd.DataFrame({c: pd.to_numeric(ds.series(c).loc[df.index], errors="coerce") for c in cols})
    corr = mat.corr(min_periods=10)
    chart = {"type": "heatmap", "title": "Correlation matrix",
             "subtitle": "Pearson r, numeric columns",
             "x": {"label": "", "values": list(corr.columns), "type": "category"},
             "matrix": js(corr.round(3).values.tolist()), "format": "number",
             "color_scale": "diverging"}
    rows = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r):
                rows.append({"x": a, "y": b, "r": float(r), "abs_r": abs(float(r))})
    rows.sort(key=lambda r: -r["abs_r"])
    return {"columns": [{"key": "x", "label": "Column A", "type": "text"},
                        {"key": "y", "label": "Column B", "type": "text"},
                        {"key": "r", "label": "Pearson r", "type": "numeric"}],
            "rows": js(rows), "chart": chart, "groups": len(rows)}


def _segments(ds: Dataset, df: pd.DataFrame, spec: dict) -> dict:
    """Compare a metric across the values of a dimension, with the overall baseline."""
    dim, m = spec.get("dimension"), normalize_measures(spec)[0]
    if dim not in ds.df.columns:
        return {"error": "Pick a dimension", "columns": [], "rows": []}
    lab = _num_label(m)
    key = ds.series(dim).loc[df.index].astype("string").fillna("(blank)")
    work = pd.DataFrame({"__k__": key.values, "__i__": df.index.values})
    baseline = _agg_group(df, m["col"], m["agg"])
    recs = []
    for k, grp in work.groupby("__k__"):
        sub = df.loc[grp["__i__"].values]
        if len(sub) < 5:
            continue
        val = _agg_group(sub, m["col"], m["agg"])
        recs.append({"label": str(k), "count": int(len(sub)), lab: val,
                     "index_vs_baseline": (val / baseline) if baseline else None})
    recs.sort(key=lambda r: -(r[lab] if r[lab] is not None else float("-inf")))
    chart = {"type": "bar", "title": f"{lab} by {dim}", "subtitle": f"dashed line = overall {lab}",
             "x": {"label": dim, "values": [r["label"] for r in recs[:20]], "type": "category"},
             "y": {"label": lab, "series": [{"name": lab, "values": [r[lab] for r in recs[:20]]}]},
             "format": guess_format(m["col"], [r[lab] for r in recs[:20]]),
             "annotations": [{"type": "hline", "y": baseline, "label": "overall", "color": "#f59e0b"}]}
    return {"columns": [{"key": "label", "label": dim, "type": "text"},
                        {"key": "count", "label": "Rows", "type": "numeric"},
                        {"key": lab, "label": lab, "type": "numeric"},
                        {"key": "index_vs_baseline", "label": "vs overall", "type": "numeric", "format": "percent"}],
            "rows": js(recs), "totals": js({"count": int(len(df)), lab: baseline}),
            "chart": chart, "groups": len(recs), "baseline": js(baseline), "measure_labels": [lab]}


def _table(ds: Dataset, df: pd.DataFrame, spec: dict) -> dict:
    cols = [c for c in (spec.get("columns") or ds.df.columns) if c in ds.df.columns]
    sort_col, sort_dir = spec.get("sort_col"), (spec.get("sort_dir") or "desc")
    view = df[cols]
    if sort_col in cols:
        s = ds.series(sort_col).loc[df.index]
        view = view.assign(__s__=s.values).sort_values("__s__", ascending=(sort_dir == "asc"),
                                                      na_position="last").drop(columns="__s__")
    limit = int(spec.get("limit") or 100)
    offset = int(spec.get("offset") or 0)
    out = view.iloc[offset:offset + limit]
    records = []
    for _, row in out.iterrows():
        records.append({c: js(row[c]) for c in cols})
    return {"columns": [{"key": c, "label": c, "type": ds.types.get(c, "text")} for c in cols],
            "rows": js(records), "totals": {"count": int(len(df))},
            "groups": int(len(df)), "offset": offset, "limit": limit}
