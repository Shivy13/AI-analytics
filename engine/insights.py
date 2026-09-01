"""Insight engine.

Each detector inspects the profile / data and emits a structured finding:
severity, a plain-English summary, hard evidence, a chart spec and a
follow-up question. Nothing here is generated text that could drift away
from the numbers - every claim carries the statistic it came from.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from .loader import Dataset
from .profile import auto_granularity, period_code, pick_measures, pick_time_col, resample_key
from .query import AGG_LABELS, default_agg, guess_format, higher_is_better, run_query
from .semantics import agg_word
from .utils import compact, fmt_num, fmt_pct

SEVERITY_WEIGHT = {"critical": 100, "warning": 60, "info": 30, "positive": 20}
def _money_like(col: str) -> bool:
    low = str(col).lower()
    return any(h in low for h in ("revenue", "sales", "price", "cost", "profit", "amount",
                                  "spend", "salary", "income", "gmv", "fee", "budget", "value"))


def _insight(id_: str, type_: str, title: str, severity: str, summary: str, *,
             evidence=None, metric=None, chart=None, question=None, impact=0.0,
             columns=None) -> dict:
    return {
        "id": id_, "type": type_, "title": title, "severity": severity,
        "summary": summary, "evidence": evidence or [], "metric": metric,
        "chart": chart, "question": question,
        "impact": impact or SEVERITY_WEIGHT.get(severity, 10),
        "columns": columns or [],
    }


# ------------------------------------------------------------------
# data quality
# ------------------------------------------------------------------
def dq_insights(ds: Dataset) -> list[dict]:
    out = []
    p = ds.profile
    by = p["by_name"]

    if p["duplicate_rows"] > 0:
        n = p["duplicate_rows"]
        out.append(_insight(
            "dup-rows", "duplicate_rows", "Duplicate rows detected", "warning",
            f"{n:,} of {p['rows']:,} rows ({fmt_pct(p['duplicate_row_pct'])}) are exact duplicates of another row. "
            f"Deduplicating would change every aggregate you compute on this file.",
            evidence=[f"{n:,} duplicate rows", f"{p['rows']:,} rows in total",
                      f"{len(ds.df.columns)} columns compared"],
            metric={"value": p["duplicate_row_pct"], "label": "rows duplicated", "format": "percent"},
            impact=70 + min(25, p["duplicate_row_pct"] * 100),
            question="Show me the duplicate rows"))

    missing = sorted([c for c in p["columns_detail"] if c.get("missing_pct", 0) >= 0.02],
                     key=lambda c: -c["missing_pct"])
    for c in missing[:3]:
        sev = "critical" if c["missing_pct"] >= 0.25 else "warning"
        out.append(_insight(
            f"missing-{c['name']}", "missing_data", f"Missing values in {c['name']}", sev,
            f"{c['missing']:,} of {p['rows']:,} values ({fmt_pct(c['missing_pct'])}) are empty in "
            f"{c['name']} ({c['type']}). Rows with blanks are dropped from any aggregate on this column, "
            f"so totals here are understated unless you impute.",
            evidence=[f"{c['missing']:,} empty cells", f"{fmt_pct(c['missing_pct'])} of the column",
                      f"detected type: {c['type']}"],
            metric={"value": c["missing_pct"], "label": "missing", "format": "percent"},
            impact=SEVERITY_WEIGHT[sev] + c["missing_pct"] * 60, columns=[c["name"]],
            question=f"How many rows are missing {c['name']}?"))

    for c in p["columns_detail"]:
        if c["type"] == "numeric" and c.get("count"):
            raw = ds.df[c["name"]]
            nonnum = int((raw.notna() & pd.to_numeric(raw, errors="coerce").isna()).sum())
            if nonnum >= max(3, 0.01 * c["count"]):
                out.append(_insight(
                    f"mixed-{c['name']}", "mixed_types", f"Mixed content in {c['name']}", "warning",
                    f"{nonnum:,} values in {c['name']} are text where the rest of the column is numeric. "
                    f"They are excluded from every calculation, which silently lowers sums and averages.",
                    evidence=[f"{nonnum:,} non-numeric values", f"{c['count']:,} numeric values kept"],
                    metric={"value": nonnum, "label": "non-numeric cells", "format": "number"},
                    impact=65, columns=[c["name"]],
                    question=f"What values in {c['name']} are not numbers?"))

    for c in p["columns_detail"]:
        if c["type"] in ("categorical", "boolean", "numeric") and c.get("distinct") == 1 and c.get("count"):
            out.append(_insight(
                f"const-{c['name']}", "constant", f"{c['name']} never changes", "info",
                f"Every one of the {c['count']:,} populated values in {c['name']} is identical, so the column "
                f"carries no information for grouping, filtering or modelling. Safe to drop.",
                evidence=[f"1 distinct value", f"{c['count']:,} populated rows"],
                impact=25, columns=[c["name"]]))

    ids = [c for c in p["columns_detail"] if c["type"] == "id"]
    for c in ids[:2]:
        vc = ds.series(c["name"]).astype("string").value_counts()
        dups = int((vc > 1).sum())
        if dups:
            out.append(_insight(
                f"id-dup-{c['name']}", "duplicate_id", f"{c['name']} looks like an ID but repeats", "warning",
                f"{c['name']} is unique for most rows but {dups:,} values appear more than once. If you treat it "
                f"as a primary key, joins and per-record counts will be wrong.",
                evidence=[f"{dups:,} repeated values", f"{int(vc.max())} times at most"],
                impact=55, columns=[c["name"]]))

    if p["missing_cell_pct"] < 0.005 and p["duplicate_rows"] == 0:
        out.append(_insight(
            "dq-clean", "data_quality", "Data quality looks clean", "positive",
            f"No duplicate rows and only {fmt_pct(p['missing_cell_pct'])} of all cells are empty across "
            f"{p['columns']} columns and {p['rows']:,} rows. Aggregates can be trusted as-is.",
            evidence=[f"{p['rows']:,} rows", f"{p['columns']} columns",
                      f"{fmt_pct(p['missing_cell_pct'])} empty cells"],
            impact=18))
    return out


# ------------------------------------------------------------------
# distributions / outliers
# ------------------------------------------------------------------
def distribution_insights(ds: Dataset) -> list[dict]:
    out = []
    by = ds.profile["by_name"]
    measures = pick_measures(ds, limit=6)

    scored = []
    for c in measures:
        p = by.get(c, {})
        if not p.get("count"):
            continue
        score = (p.get("outlier_pct") or 0) * 100 + (p.get("extreme_count") or 0) * 0.5
        if score > 0:
            scored.append((score, c, p))
    scored.sort(reverse=True)
    for score, c, p in scored[:3]:
        lo, hi = p.get("fence_lo"), p.get("fence_hi")
        out.append(_insight(
            f"outlier-{c}", "outliers", f"Outliers in {c}", "warning",
            f"{p['outlier_count']:,} values ({fmt_pct(p['outlier_pct'])}) in {c} sit outside the 1.5×IQR fence "
            f"[{fmt_num(lo)} – {fmt_num(hi)}], reaching {fmt_num(p.get('outlier_max'))}. "
            f"With a mean of {fmt_num(p['mean'])} and a median of {fmt_num(p['median'])}, a handful of rows are "
            f"pulling the average around.",
            evidence=[f"{p['outlier_count']:,} of {p['count']:,} values flagged",
                      f"largest value {fmt_num(p.get('outlier_max'))} vs median {fmt_num(p['median'])}",
                      f"1.5×IQR fence {fmt_num(lo)} – {fmt_num(hi)}",
                      f"{p.get('extreme_count', 0):,} beyond 3×IQR"],
            metric={"value": p["outlier_pct"], "label": "outliers", "format": "percent"},
            chart={"type": "distribution", "spec": {"kind": "distribution", "column": c}},
            question=f"Show me the distribution of {c}",
            impact=45 + min(30, p["outlier_pct"] * 200), columns=[c]))

    for c in measures:
        p = by.get(c, {})
        skew = p.get("skew")
        if skew is not None and abs(skew) >= 1.5 and (p.get("count") or 0) >= 50:
            side = "right" if skew > 0 else "left"
            out.append(_insight(
                f"skew-{c}", "skew", f"{c} is heavily skewed to the {side}", "info",
                f"{c} has a skewness of {skew:+.2f}: the mean ({fmt_num(p['mean'])}) is far from the median "
                f"({fmt_num(p['median'])}). Use the median when comparing typical values, and a log scale if you plot it.",
                evidence=[f"skewness {skew:+.2f}", f"mean {fmt_num(p['mean'])}",
                          f"median {fmt_num(p['median'])}", f"max {fmt_num(p['max'])}"],
                impact=32, columns=[c],
                question=f"What is the distribution of {c}?"))

    for c in measures:
        p = by.get(c, {})
        if p.get("zero_pct") and p["zero_pct"] >= 0.3 and (p.get("count") or 0) >= 50:
            out.append(_insight(
                f"zeros-{c}", "zero_inflation", f"{fmt_pct(p['zero_pct'])} of {c} is zero", "info",
                f"{p['zeros']:,} of {p['count']:,} values in {c} are exactly zero. Averages that include them "
                f"({fmt_num(p['mean'])}) understate the typical non-zero value "
                f"({fmt_num(p['sum'] / max(1, p['count'] - p['zeros']))}).",
                evidence=[f"{p['zeros']:,} zeros ({fmt_pct(p['zero_pct'])})",
                          f"mean incl. zeros {fmt_num(p['mean'])}",
                          f"mean excl. zeros {fmt_num(p['sum'] / max(1, p['count'] - p['zeros']))}"],
                impact=34, columns=[c],
                question=f"Average {c} excluding zeros"))

    for c in measures:
        p = by.get(c, {})
        if _money_like(c) and (p.get("negatives") or 0) > 0:
            out.append(_insight(
                f"neg-{c}", "negatives", f"Negative values in {c}", "warning",
                f"{p['negatives']:,} values in {c} are negative (lowest {fmt_num(p['min'])}). In a money column these "
                f"are usually refunds, reversals or sign errors — worth checking before you net them against revenue.",
                evidence=[f"{p['negatives']:,} negative values", f"minimum {fmt_num(p['min'])}",
                          f"sum {fmt_num(p['sum'])}"],
                impact=48, columns=[c],
                question=f"How many rows have negative {c}?"))
    return out


def categorical_insights(ds: Dataset) -> list[dict]:
    out = []
    for c in ds.profile["columns_detail"]:
        if c["type"] not in ("categorical", "boolean") or not c.get("count"):
            continue
        top = c.get("top") or []
        if not top:
            continue
        t = top[0]
        if c["type"] == "boolean":
            share = c.get("true_pct")
            if share is None:
                share = sum(x["pct"] for x in top if x["value"].lower() in ("true", "yes", "y", "t", "1"))
            out.append(_insight(
                f"bool-{c['name']}", "boolean_split",
                f"{c['name']} splits {fmt_pct(share, 0)} / {fmt_pct(1 - share, 0)}", "info",
                f"{fmt_pct(share)} of the {c['count']:,} rows in {c['name']} are true. "
                f"That is a usable segment split for comparison.",
                evidence=[f"{fmt_pct(share)} true", f"{fmt_pct(1 - share)} false", f"{c['count']:,} populated"],
                impact=22, columns=[c["name"]],
                question=f"Compare rows where {c['name']} is true vs false"))
            continue
        if t["pct"] >= 0.6:
            out.append(_insight(
                f"imbalance-{c['name']}", "imbalance",
                f"{c['name']} is dominated by {t['value']}", "info",
                f"'{t['value']}' accounts for {fmt_pct(t['pct'])} of {c['name']} "
                f"({t['count']:,} of {c['count']:,} rows). Any average over this column is mostly a statement "
                f"about that one value; the remaining {c['distinct'] - 1} values are a thin tail.",
                evidence=[f"top value '{t['value']}' = {fmt_pct(t['pct'])}",
                          f"{c['distinct']} distinct values",
                          f"{c.get('singletons', 0):,} values appear only once"],
                impact=30, columns=[c["name"]]))
        if c["distinct"] >= 20 and (c.get("long_tail") or 0) / max(1, c["distinct"]) > 0.5:
            out.append(_insight(
                f"longtail-{c['name']}", "long_tail",
                f"{c['name']} has a long tail", "info",
                f"{c['long_tail']:,} of the {c['distinct']} values in {c['name']} cover under 1% of rows each, while "
                f"the top 5 cover {fmt_pct(c['top5_pct'])}. Group the tail into an 'Other' bucket before charting.",
                evidence=[f"{c['distinct']} distinct values",
                          f"{c['long_tail']:,} below 1% share",
                          f"top 5 = {fmt_pct(c['top5_pct'])}"],
                impact=26, columns=[c["name"]]))
    return out


# ------------------------------------------------------------------
# correlation
# ------------------------------------------------------------------
def correlation_insights(ds: Dataset) -> list[dict]:
    nums = [c for c in pick_measures(ds, limit=12)]
    if len(nums) < 2:
        return []
    sample = ds.df
    if len(sample) > 20000:
        sample = sample.sample(20000, random_state=0)
    mat = pd.DataFrame({c: pd.to_numeric(ds.series(c).loc[sample.index], errors="coerce") for c in nums})
    corr = mat.corr(min_periods=30)
    pairs = []
    for i, a in enumerate(nums):
        for b in nums[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= 0.55:
                pairs.append((abs(r), r, a, b))
    pairs.sort(reverse=True)
    out = []
    for _, r, a, b in pairs[:3]:
        strength = "very strong" if abs(r) >= 0.85 else "strong" if abs(r) >= 0.7 else "moderate"
        direction = "positive" if r > 0 else "negative"
        out.append(_insight(
            f"corr-{a}-{b}", "correlation", f"{strength} {direction} link: {a} ↔ {b}",
            "positive" if r > 0 else "warning",
            f"{a} and {b} move together with r = {r:+.2f} ({strength} {direction}); {abs(r) ** 2 * 100:.0f}% of the "
            f"variation in one is shared with the other. This is association, not proof of causation.",
            evidence=[f"Pearson r = {r:+.2f}", f"R² = {r * r:.2f}", f"n = {int(len(sample)):,} rows"],
            metric={"value": r, "label": "Pearson r", "format": "number"},
            chart={"type": "correlation", "spec": {"kind": "correlation", "x": a, "y": b}},
            question=f"Plot {b} against {a}",
            impact=40 + abs(r) * 40, columns=[a, b]))
    return out


# ------------------------------------------------------------------
# time series
# ------------------------------------------------------------------
def time_insights(ds: Dataset) -> list[dict]:
    tcol = pick_time_col(ds)
    out = []
    if not tcol:
        return out
    tp = ds.profile["by_name"][tcol]
    span = tp.get("span_days") or 0
    out.append(_insight(
        "coverage", "coverage", f"Data spans {fmt_num(span / 365.25, 1)} years", "info",
        f"{tcol} runs from {pd.Timestamp(tp['min']).strftime('%d %b %Y')} to "
        f"{pd.Timestamp(tp['max']).strftime('%d %b %Y')} — {fmt_num(span)} days across "
        f"{tp.get('n_years', 1)} calendar years, recorded at {tp.get('granularity') or 'mixed'} granularity.",
        evidence=[f"{tp['count']:,} timestamps", f"{fmt_num(span)} days",
                  f"{tp.get('n_years', 1)} calendar years"],
        impact=20, columns=[tcol]))

    if tp.get("gap_count"):
        out.append(_insight(
            "gaps", "gaps", f"{tp['gap_count']} gaps in {tcol}", "warning",
            f"The typical gap between records is {fmt_num(tp.get('median_gap_days'), 1)} days but the largest is "
            f"{tp['max_gap_days']} days, and {tp['gap_count']} gaps are more than 2.5× the norm. Trend lines drawn "
            f"across those holes are interpolating, not measuring.",
            evidence=[f"median gap {fmt_num(tp.get('median_gap_days'), 1)} days",
                      f"largest gap {tp['max_gap_days']} days",
                      f"{tp['gap_count']} suspicious gaps"],
            impact=42, columns=[tcol]))

    measures = pick_measures(ds, limit=3)
    gran = auto_granularity(ds.series(tcol))
    code = period_code(gran)
    ts = ds.series(tcol)

    for m in measures:
        dagg = default_agg(m)
        good_up = higher_is_better(m)
        word = "Total" if dagg == "sum" else "Average"
        work = pd.DataFrame({"k": resample_key(ts, gran).values,
                             "v": pd.to_numeric(ds.series(m), errors="coerce").values}).dropna()
        if work["k"].nunique() < 4:
            continue
        agg = work.groupby("k")["v"].sum() if dagg == "sum" else work.groupby("k")["v"].mean()
        xs = np.arange(len(agg), dtype=float)
        ys = agg.values.astype(float)
        slope, intercept = np.polyfit(xs, ys, 1)
        fitted = slope * xs + intercept
        ss_res = float(np.sum((ys - fitted) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
        mean_y = float(ys.mean())
        total_change = float(slope * (len(ys) - 1))
        pct = total_change / mean_y if mean_y else 0.0
        first, last = float(ys[0]), float(ys[-1])
        if abs(pct) >= 0.12 and len(ys) >= 4:
            direction = "rising" if slope > 0 else "falling"
            conf = "steadily " if r2 >= 0.5 else "noisily "
            sev = "positive" if (slope > 0) == good_up else "warning"
            out.append(_insight(
                f"trend-{m}", "trend", f"{m} is {direction}", sev,
                f"Aggregated by {gran} ({word.lower()}), {m} is {conf}{direction}: the fitted trend covers "
                f"{total_change:+,.0f} across the period ({pct:+.0%} of the mean {fmt_num(mean_y)}), going from "
                f"{fmt_num(first)} in the first bucket to {fmt_num(last)} in the last. The line explains "
                f"{r2 * 100:.0f}% of the movement.",
                evidence=[f"per-{gran} slope {slope:+,.1f}", f"R² = {r2:.2f}",
                          f"first bucket {fmt_num(first)} → last {fmt_num(last)}",
                          f"{len(ys)} {gran} buckets"],
                metric={"value": pct, "label": "trend change", "format": "percent"},
                chart={"type": "timeseries",
                       "spec": {"kind": "timeseries", "time_dimension": tcol, "measure": m,
                                "agg": dagg, "granularity": gran}},
                question=f"Show the {gran}ly trend of {m}",
                impact=50 + min(40, abs(pct) * 100), columns=[tcol, m]))

        # spikes / anomalies
        if len(ys) >= 8:
            z = (ys - ys.mean()) / (ys.std(ddof=1) or 1)
            idx = np.argsort(-np.abs(z))[:2]
            strong = [i for i in idx if abs(z[i]) >= 2.5]
            if strong:
                i = int(strong[0])
                when = agg.index[i]
                lbl = pd.Timestamp(when).strftime("%b %Y" if code in ("M", "Q", "Y") else "%d %b %Y")
                out.append(_insight(
                    f"spike-{m}", "anomaly", f"Spike in {m} ({lbl})", "warning",
                    f"{m} hit {fmt_num(ys[i])} in {lbl}, which is {z[i]:+.1f} standard deviations from the typical "
                    f"{gran} of {fmt_num(ys.mean())}. That single bucket is "
                    f"{(ys[i] / ys.mean()):.1f}× the norm and will dominate any total.",
                    evidence=[f"{lbl}: {fmt_num(ys[i])}", f"{z[i]:+.1f}σ from the mean",
                              f"typical {gran} {fmt_num(ys.mean())}"],
                    impact=44, columns=[tcol, m],
                    chart={"type": "timeseries",
                           "spec": {"kind": "timeseries", "time_dimension": tcol, "measure": m,
                                    "agg": dagg, "granularity": gran}},
                    question=f"Which rows fall in {lbl}?"))

        # seasonality on monthly data spanning > 1 year
        if span > 400 and len(ys) >= 12:
            monthly = work.copy()
            monthly["mo"] = pd.to_datetime(monthly["k"]).dt.month
            prof = monthly.groupby("mo")["v"].mean()
            peak, trough = int(prof.idxmax()), int(prof.idxmin())
            ratio = prof.max() / prof.min() if prof.min() else float("nan")
            if ratio and ratio >= 1.4:
                names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                out.append(_insight(
                    f"season-{m}", "seasonality", f"{m} peaks in {names[peak - 1]}", "info",
                    f"Averaging {m} by calendar month, {names[peak - 1]} is the strongest month "
                    f"({fmt_num(prof.max())}) and {names[trough - 1]} the weakest ({fmt_num(prof.min())}) — "
                    f"{ratio:.1f}× apart. Year-on-year comparisons should be made month-to-month, not overall.",
                    evidence=[f"peak {names[peak - 1]}: {fmt_num(prof.max())}",
                              f"trough {names[trough - 1]}: {fmt_num(prof.min())}",
                              f"{ratio:.1f}× peak-to-trough"],
                    impact=38, columns=[tcol, m],
                    question=f"Average {m} by month"))

        # year-on-year
        if tp.get("n_years", 1) >= 2 and m:
            yearly = work.copy()
            yearly["dt"] = pd.to_datetime(yearly["k"])
            yearly["yr"] = yearly["dt"].dt.year
            yearly["mo"] = yearly["dt"].dt.month
            yrs = sorted(int(y) for y in yearly["yr"].unique())
            ys_y = pd.Series(dtype=float)
            y1 = y2 = None
            common = set()
            if len(yrs) >= 2:
                y1, y2 = yrs[-2], yrs[-1]
                # compare only months present in BOTH years, so a partial
                # current year cannot masquerade as a collapse
                common = set(yearly.loc[yearly["yr"] == y1, "mo"]) & set(yearly.loc[yearly["yr"] == y2, "mo"])
                cmp_df = yearly[yearly["yr"].isin([y1, y2]) & yearly["mo"].isin(common)]
                ys_y = cmp_df.groupby("yr")["v"].sum() if dagg == "sum" else cmp_df.groupby("yr")["v"].mean()
            if len(ys_y) >= 2:
                a, b = float(ys_y.loc[y1]), float(ys_y.loc[y2])
                yoy = (b - a) / a if a else None
                if yoy is not None and abs(yoy) >= 0.05:
                    out.append(_insight(
                        f"yoy-{m}", "yoy", f"{m} {'grew' if yoy > 0 else 'fell'} {abs(yoy):.0%} year-on-year",
                        "positive" if (yoy > 0) == good_up else "warning",
                        f"{m} went from {fmt_num(a)} in {int(ys_y.index[-2])} to {fmt_num(b)} in "
                        f"{int(ys_y.index[-1])}, a change of {yoy:+.0%}.",
                        evidence=[f"{y1}: {fmt_num(a)}", f"{y2}: {fmt_num(b)}",
                                  f"{yoy:+.0%} year-on-year, like-for-like over {len(common)} shared months"],
                        metric={"value": yoy, "label": "YoY", "format": "percent"},
                        impact=36, columns=[tcol, m],
                        question=f"Compare {m} by year"))
    return out


# ------------------------------------------------------------------
# segments & concentration
# ------------------------------------------------------------------
def segment_insights(ds: Dataset) -> list[dict]:
    out = []
    measures = pick_measures(ds, limit=2)
    dims = [c["name"] for c in ds.profile["columns_detail"]
            if c["type"] == "categorical" and 2 <= (c.get("distinct") or 0) <= 40
            and (c.get("count") or 0) >= 50]
    dims.sort(key=lambda c: ds.df[c].nunique())
    if not measures or not dims:
        return out

    for m in measures[:1]:
        dagg = default_agg(m)
        col = pd.to_numeric(ds.series(m), errors="coerce")
        overall_sum = float(col.sum())
        for d in dims[:6]:
            key = ds.series(d).astype("string").fillna("(blank)")
            grp = pd.DataFrame({"k": key.values, "v": col.values}).dropna()
            if grp["k"].nunique() < 2:
                continue
            agg = grp.groupby("k")["v"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
            agg = agg[agg["count"] >= max(5, 0.01 * len(grp))]
            if len(agg) < 2:
                continue

            # concentration only makes sense for additive measures
            if dagg == "sum" and overall_sum:
                share = float(agg["sum"].iloc[0] / overall_sum)
                top = agg.index[0]
                top3 = float(agg["sum"].head(3).sum() / overall_sum)
                if share >= 0.3:
                    out.append(_insight(
                        f"conc-{d}-{m}", "concentration",
                        f"{top} drives {fmt_pct(share, 0)} of {m}", "info",
                        f"Of the {fmt_num(overall_sum)} total {m}, {top} contributes "
                        f"{fmt_num(agg['sum'].iloc[0])} ({fmt_pct(share)}) across "
                        f"{int(agg['count'].iloc[0]):,} rows. The top {min(3, len(agg))} of {len(agg)} {d} "
                        f"values hold {fmt_pct(top3)} between them — concentration risk if that segment slips.",
                        evidence=[f"{top}: {fmt_num(agg['sum'].iloc[0])} ({fmt_pct(share)})",
                                  f"{len(agg)} {d} values in play",
                                  f"top 3 hold {fmt_pct(top3)}"],
                        metric={"value": share, "label": f"share of {m}", "format": "percent"},
                        chart={"type": "aggregate",
                               "spec": {"kind": "aggregate", "dimension": d, "measure": m, "agg": "sum",
                                        "limit": 10, "other_bucket": True}},
                        question=f"Total {m} by {d}",
                        impact=40 + share * 40, columns=[d, m]))

            # strongest / weakest segment on the average
            base = float(grp["v"].mean())
            if not base or len(agg) < 3:
                continue
            ratio = agg["mean"] / base
            hi, lo = str(ratio.idxmax()), str(ratio.idxmin())
            min_n = max(10, 0.02 * len(grp))
            if ratio.max() >= 1.6 and agg.loc[hi, "count"] >= min_n:
                out.append(_insight(
                    f"seg-hi-{d}-{m}", "segment",
                    f"{hi} has the highest average {m}", "positive",
                    f"Average {m} for {hi} is {fmt_num(agg.loc[hi, 'mean'])} against an overall "
                    f"{fmt_num(base)} — {ratio.max():.1f}× the norm across {int(agg.loc[hi, 'count']):,} rows. "
                    f"Worth understanding what is different there.",
                    evidence=[f"{hi}: {fmt_num(agg.loc[hi, 'mean'])} avg",
                              f"overall avg: {fmt_num(base)}",
                              f"{int(agg.loc[hi, 'count']):,} rows behind it"],
                    metric={"value": float(ratio.max()), "label": "× overall avg", "format": "number"},
                    chart={"type": "segments",
                           "spec": {"kind": "segments", "dimension": d, "measure": m, "agg": "avg"}},
                    question=f"Average {m} by {d}",
                    impact=36 + min(20, float(ratio.max()) * 5), columns=[d, m]))
            if ratio.min() <= 0.6 and agg.loc[lo, "count"] >= min_n:
                out.append(_insight(
                    f"seg-lo-{d}-{m}", "segment",
                    f"{lo} is the weakest segment for {m}", "warning",
                    f"Average {m} for {lo} is only {fmt_num(agg.loc[lo, 'mean'])} versus an overall "
                    f"{fmt_num(base)} ({ratio.min():.2f}× the norm) across {int(agg.loc[lo, 'count']):,} rows — "
                    f"the clearest place to look for a problem.",
                    evidence=[f"{lo}: {fmt_num(agg.loc[lo, 'mean'])} avg",
                              f"overall avg: {fmt_num(base)}",
                              f"{int(agg.loc[lo, 'count']):,} rows behind it"],
                    impact=34, columns=[d, m],
                    question=f"Average {m} by {d}"))
    return out


def weekday_insights(ds: Dataset) -> list[dict]:
    tcol = pick_time_col(ds)
    if not tcol:
        return []
    tp = ds.profile["by_name"][tcol]
    if tp.get("granularity") not in ("day", "hour", "minute"):
        return []
    measures = pick_measures(ds, limit=1)
    ts = ds.series(tcol)
    out = []
    if tp.get("granularity") == "day" and measures:
        m = measures[0]
        work = pd.DataFrame({"d": ts.dt.dayofweek.values,
                             "v": pd.to_numeric(ds.series(m), errors="coerce").values}).dropna()
        prof = work.groupby("d")["v"].mean()
        names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        peak, trough = int(prof.idxmax()), int(prof.idxmin())
        if prof.max() and prof.max() / prof.min() >= 1.25:
            out.append(_insight(
                "weekday", "seasonality", f"{m} is strongest on {names[peak]}s", "info",
                f"Grouping by weekday, the average {m} is {fmt_num(prof.max())} on {names[peak]} and "
                f"{fmt_num(prof.min())} on {names[trough]} — a {prof.max() / prof.min():.1f}× weekly rhythm. "
                f"Week-over-week comparisons should be weekday-aligned.",
                evidence=[f"{names[peak]}: {fmt_num(prof.max())} avg",
                          f"{names[trough]}: {fmt_num(prof.min())} avg"],
                impact=28, columns=[tcol, m],
                question=f"Average {m} by {tcol}"))
    return out


# ------------------------------------------------------------------
# orchestration
# ------------------------------------------------------------------
def generate_insights(ds: Dataset, limit: int = 14) -> dict:
    from .profile import build_profile
    if ds.profile is None:
        build_profile(ds)
    detectors = [dq_insights, distribution_insights, categorical_insights,
                 correlation_insights, time_insights, segment_insights, weekday_insights]
    found: list[dict] = []
    for fn in detectors:
        try:
            found.extend(fn(ds))
        except Exception as e:  # a detector must never kill the report
            found.append(_insight(f"err-{fn.__name__}", "error",
                                  f"{fn.__name__} failed", "info",
                                  f"This check raised {type(e).__name__}: {e}. The rest of the report is unaffected.",
                                  impact=1))
    seen, dedup = set(), []
    for i in found:
        if i["id"] in seen:
            continue
        seen.add(i["id"])
        dedup.append(i)
    dedup.sort(key=lambda i: (-i["impact"], i["title"]))
    keep = dedup[:limit]
    counts = {"critical": 0, "warning": 0, "info": 0, "positive": 0}
    for i in dedup:
        counts[i["severity"]] = counts.get(i["severity"], 0) + 1
    return {"insights": keep, "total_found": len(dedup), "severity_counts": counts,
            "all_insights": dedup}


def build_dashboard(ds: Dataset) -> dict:
    """KPI cards + a default chart grid, derived from the profile."""
    from .profile import build_profile
    if ds.profile is None:
        build_profile(ds)
    measures = pick_measures(ds, limit=4)
    tcol = pick_time_col(ds)
    kpis = [{"label": "Rows", "value": ds.rows, "format": "number", "hint": f"{ds.profile['columns']} columns"}]
    for m in measures[:3]:
        p = ds.profile["by_name"][m]
        agg = default_agg(m)
        kpis.append({
            "label": f"{agg_word(m)} {m}",
            "value": p.get("mean") if agg == "avg" else p.get("sum"),
            "format": guess_format(m),
            "hint": (f"median {fmt_num(p.get('median'))} · max {fmt_num(p.get('max'))}" if agg == "avg"
                     else f"avg {fmt_num(p.get('mean'))} · median {fmt_num(p.get('median'))}")})
    if tcol:
        tp = ds.profile["by_name"][tcol]
        kpis.append({"label": "Period covered",
                     "value": f"{pd.Timestamp(tp['min']).strftime('%b %Y')} – {pd.Timestamp(tp['max']).strftime('%b %Y')}",
                     "format": "text", "hint": f"{fmt_num(tp.get('span_days'))} days"})
    if ds.profile["duplicate_rows"]:
        kpis.append({"label": "Duplicate rows", "value": ds.profile["duplicate_rows"], "format": "number",
                     "hint": "worth removing"})

    charts = []
    if tcol and measures:
        charts.append({"title": f"{measures[0]} over time",
                       "spec": {"kind": "timeseries", "time_dimension": tcol, "measure": measures[0],
                                "agg": "sum", "granularity": "auto"}})
    dims = [c for c in ds.profile["columns_detail"]
            if c["type"] == "categorical" and 2 <= (c.get("distinct") or 0) <= 15]
    dims.sort(key=lambda c: c["distinct"])
    for d in dims[:3]:
        name = d["name"]
        if measures:
            charts.append({"title": f"{agg_word(measures[0])} {measures[0]} by {name}",
                           "spec": {"kind": "aggregate", "dimension": name, "measure": measures[0],
                                    "agg": default_agg(measures[0]), "limit": 10, "other_bucket": True,
                                    "sort": {"by": "measure", "dir": "desc"}}})
        else:
            charts.append({"title": f"Rows by {name}",
                           "spec": {"kind": "aggregate", "dimension": name, "limit": 10}})
    if len(measures) >= 2:
        charts.append({"title": f"{measures[1]} distribution",
                       "spec": {"kind": "distribution", "column": measures[1]}})
    if len(ds.numeric_cols) >= 3:
        charts.append({"title": "Correlation matrix", "spec": {"kind": "matrix"}})

    resolved = []
    for ch in charts:
        try:
            res = run_query(ds, ch["spec"])
            if res.get("chart") and not res.get("error"):
                resolved.append({"title": ch["title"], "chart": res["chart"],
                                 "spec": ch["spec"], "rows": res.get("groups")})
        except Exception:
            continue
    return {"kpis": kpis, "charts": resolved, "measure_candidates": measures,
            "time_column": tcol,
            "dimension_candidates": [c["name"] for c in dims]}
