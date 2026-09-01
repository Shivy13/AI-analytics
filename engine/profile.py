"""Column + dataset profiling: the statistical backbone of every insight."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import Dataset
from .semantics import measure_priority
from .utils import js

TOP_VALUES = 12
GRANULARITY_STEPS = [
    ("second", 1.5),
    ("minute", 60 * 1.5),
    ("hour", 3600 * 1.5),
    ("day", 86400 * 1.5),
    ("week", 86400 * 7 * 1.5),
    ("month", 86400 * 31 * 1.5),
    ("quarter", 86400 * 92 * 1.5),
    ("year", 86400 * 366 * 1.5),
]


def _granularity(ts: pd.Series) -> str | None:
    u = ts.dropna().drop_duplicates().sort_values()
    if len(u) < 2:
        return None
    diffs = u.diff().dropna().dt.total_seconds()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return None
    med = float(diffs.median())
    for name, limit in GRANULARITY_STEPS:
        if med <= limit:
            return name
    return "year"


def profile_numeric(name: str, s: pd.Series, n_rows: int) -> dict:
    v = s.dropna()
    out = {
        "count": int(len(v)),
        "missing": int(n_rows - len(v)),
        "missing_pct": float((n_rows - len(v)) / n_rows) if n_rows else 0.0,
    }
    if len(v) == 0:
        return out
    q = v.quantile([0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    iqr = float(q[0.75] - q[0.25])
    lo, hi = float(q[0.25]) - 1.5 * iqr, float(q[0.75]) + 1.5 * iqr
    extreme_lo, extreme_hi = float(q[0.25]) - 3 * iqr, float(q[0.75]) + 3 * iqr
    outliers = v[(v < lo) | (v > hi)]
    extreme = v[(v < extreme_lo) | (v > extreme_hi)]
    out.update(
        min=float(q[0.0]), max=float(q[1.0]), mean=mean, median=float(q[0.5]),
        std=std, cv=float(std / mean) if mean else None,
        q1=float(q[0.25]), q3=float(q[0.75]), p90=float(q[0.9]), iqr=iqr,
        sum=float(v.sum()),
        skew=float(v.skew()) if len(v) > 2 else 0.0,
        kurtosis=float(v.kurt()) if len(v) > 3 else 0.0,
        zeros=int((v == 0).sum()), negatives=int((v < 0).sum()),
        zero_pct=float((v == 0).sum() / len(v)),
        outlier_count=int(len(outliers)), outlier_pct=float(len(outliers) / len(v)),
        extreme_count=int(len(extreme)),
        outlier_max=float(outliers.max()) if len(outliers) else None,
        outlier_min=float(outliers.min()) if len(outliers) else None,
        fence_lo=lo, fence_hi=hi,
        distinct=int(v.nunique()),
    )
    out["range"] = out["max"] - out["min"]
    return out


def profile_categorical(name: str, s: pd.Series, n_rows: int) -> dict:
    v = s.dropna().astype("string")
    out = {
        "count": int(len(v)),
        "missing": int(n_rows - len(v)),
        "missing_pct": float((n_rows - len(v)) / n_rows) if n_rows else 0.0,
    }
    if len(v) == 0:
        return out
    vc = v.value_counts(dropna=True)
    distinct = int(len(vc))
    probs = vc.values / vc.values.sum()
    entropy = float(-(probs * np.log(np.where(probs > 0, probs, 1))).sum())
    max_entropy = float(np.log(distinct)) if distinct > 1 else 1.0
    out.update(
        distinct=distinct,
        distinct_pct=float(distinct / len(v)),
        top=[{"value": str(k), "count": int(c), "pct": float(c / len(v))}
             for k, c in vc.head(TOP_VALUES).items()],
        top1_pct=float(vc.iloc[0] / len(v)),
        top5_pct=float(vc.head(5).sum() / len(v)),
        entropy=entropy,
        norm_entropy=float(entropy / max_entropy) if max_entropy > 0 else 0.0,
        singletons=int((vc == 1).sum()),
        long_tail=int((vc / len(v) < 0.01).sum()),
        max_label_len=int(v.astype(str).str.len().max()),
    )
    return out


def profile_datetime(name: str, s: pd.Series, n_rows: int) -> dict:
    v = s.dropna().sort_values()
    out = {
        "count": int(len(v)),
        "missing": int(n_rows - len(v)),
        "missing_pct": float((n_rows - len(v)) / n_rows) if n_rows else 0.0,
    }
    if len(v) == 0:
        return out
    span = v.iloc[-1] - v.iloc[0]
    out.update(
        min=v.iloc[0].isoformat(), max=v.iloc[-1].isoformat(),
        span_days=float(span.total_seconds() / 86400),
        distinct=int(v.nunique()),
        granularity=_granularity(v),
        year_min=int(v.dt.year.min()), year_max=int(v.dt.year.max()),
        n_years=int(v.dt.year.nunique()),
    )
    # gap detection at day granularity
    if out["granularity"] in ("day", "week", "month", "quarter"):
        days = pd.Series(v.dt.normalize().unique()).sort_values()
        if len(days) > 2:
            gaps = days.diff().dropna().dt.days
            out["max_gap_days"] = int(gaps.max())
            out["median_gap_days"] = float(gaps.median())
            out["gap_count"] = int((gaps > out["median_gap_days"] * 2.5).sum())
    return out


def build_profile(ds: Dataset) -> dict:
    df, n = ds.df, ds.rows
    cols = []
    for c in df.columns:
        sem = ds.types.get(c, "text")
        s = ds.series(c)
        base = {"name": c, "type": sem, "raw_dtype": str(df[c].dtype)}
        if sem == "numeric":
            base.update(profile_numeric(c, s, n))
        elif sem == "datetime":
            base.update(profile_datetime(c, s, n))
        elif sem in ("categorical", "boolean", "id", "text"):
            if sem == "boolean":
                # profile the raw labels ("Yes"/"No"), not the 1.0/0.0 coercion
                base.update(profile_categorical(c, df[c].astype("string").str.strip(), n))
                coerced = pd.to_numeric(s, errors="coerce")
                base["true_pct"] = float(coerced.mean()) if coerced.notna().any() else 0.0
            else:
                base.update(profile_categorical(c, s, n))
        cols.append(base)

    dup_rows = int(df.duplicated().sum())
    total_cells = n * len(df.columns)
    missing_cells = int(df.isna().sum().sum())
    prof = {
        "rows": n,
        "columns": len(df.columns),
        "duplicate_rows": dup_rows,
        "duplicate_row_pct": float(dup_rows / n) if n else 0.0,
        "missing_cells": missing_cells,
        "missing_cell_pct": float(missing_cells / total_cells) if total_cells else 0.0,
        "columns_detail": js(cols),
        "by_name": {c["name"]: c for c in cols},
    }
    ds.profile = prof
    return prof


# ------------------------------------------------------------------
# time helpers shared by insights + NLQ
# ------------------------------------------------------------------
GRAN_ALIASES = {
    "day": "D", "daily": "D", "date": "D", "d": "D",
    "week": "W", "weekly": "W", "w": "W",
    "month": "M", "monthly": "M", "m": "M", "months": "M",
    "quarter": "Q", "quarterly": "Q", "q": "Q",
    "year": "Y", "yearly": "Y", "annual": "Y", "annualy": "Y", "y": "Y", "years": "Y",
}
GRAN_LABELS = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}


def period_code(granularity: str) -> str:
    return GRAN_ALIASES.get((granularity or "").lower(), "M")


def resample_key(ts: pd.Series, granularity: str) -> pd.Series:
    """Bucket timestamps into period start-times (works with to_period codes)."""
    code = period_code(granularity)
    return ts.dt.to_period(code).dt.start_time


def auto_granularity(ts: pd.Series) -> str:
    v = ts.dropna()
    if v.empty:
        return "month"
    span = (v.max() - v.min()).days
    if span <= 45:
        return "day"
    if span <= 400:
        return "month"
    return "quarter" if span <= 2000 else "year"


# ------------------------------------------------------------------
# column selection helpers (shared by insights and the NL layer)
# ------------------------------------------------------------------
def pick_measures(ds: "Dataset", limit: int = 4) -> list[str]:
    """Numeric columns that behave like measures, most report-worthy first."""
    import re as _re
    by = (ds.profile or {}).get("by_name", {})
    out = []
    for c in ds.numeric_cols:
        meta = by.get(c, {})
        if (meta.get("distinct") or 0) < 3 or (meta.get("missing_pct") or 0) > 0.6:
            continue
        if _re.search(r"(^|_)(id|ids|no|num|number|code|zip|pin|year)$", c.lower()):
            continue
        out.append(c)
    out.sort(key=lambda c: (-measure_priority(c), -(by.get(c, {}).get("distinct") or 0)))
    return out[:limit]


def pick_time_col(ds: "Dataset") -> str | None:
    cands = list(ds.datetime_cols)
    if not cands:
        return None
    by = (ds.profile or {}).get("by_name", {})
    cands.sort(key=lambda c: -(by.get(c, {}).get("distinct") or 0))
    return cands[0]
