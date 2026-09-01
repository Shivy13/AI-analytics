"""Natural-language questions -> structured query -> prose answer.

The parser is deliberately transparent: it reports which column it matched
for each slot, a confidence score, and the alternatives it rejected, so the
UI can show its working instead of pretending to be magic.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from .loader import Dataset
from .profile import pick_measures
from .query import AGG_LABELS, guess_format, run_query
from .semantics import default_agg
from .utils import compact, fmt_num, fmt_pct, norm_text

STOPWORDS = set("""the a an of for to in on by with and or is are was were be been what which how many much
show me give tell please can you i want would like there this that it its from as at over under into out
me my we our your their his her top bottom best worst list display chart plot graph analyze analyse
data dataset question about between per each vs versus compare comparison""".split())

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
                "twenty five": 25, "thirty": 30, "fifty": 50, "hundred": 100}

AGG_PATTERNS = [
    (r"\b(count distinct|distinct count|unique count|number of distinct|unique)\b", "count_distinct"),
    (r"\b(how many|count of|number of|count|no of|total number)\b", "count"),
    (r"\b(average|avg|mean of|mean)\b", "avg"),
    (r"\b(median)\b", "median"),
    (r"\b(total|sum of|sum|grand total|combined|overall)\b", "sum"),
    (r"\b(maximum|max value|highest value|largest value|peak)\b", "max"),
    (r"\b(minimum|min value|lowest value|smallest value)\b", "min"),
    (r"\b(standard deviation|std dev|std|volatility|variation)\b", "std"),
    (r"\b(90th percentile|p90)\b", "p90"),
    (r"\b(10th percentile|p10)\b", "p10"),
]

GRAN_PATTERNS = [
    (r"\b(daily|by day|per day|day wise|daywise|each day)\b", "day"),
    (r"\b(weekly|by week|per week|week wise|weekwise|each week|week over week|wow)\b", "week"),
    (r"\b(monthly|by month|per month|month wise|monthwise|each month|month over month|mom)\b", "month"),
    (r"\b(quarterly|by quarter|per quarter|quarter wise|qoq)\b", "quarter"),
    (r"\b(yearly|annual|by year|per year|year wise|yearwise|each year|year over year|yoy)\b", "year"),
]

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
MONTHS.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
               "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12})


def _singular(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ses") or w.endswith("xes") or w.endswith("zes"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _plural(w: str) -> str:
    if w.endswith(("s", "x", "z", "ch", "sh")):
        return w + "es"
    if w.endswith("y") and len(w) > 2 and w[-2] not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


class Schema:
    """Column + value index with fuzzy matching."""

    def __init__(self, ds: Dataset):
        self.ds = ds
        self.cols = list(ds.df.columns)
        self.aliases: dict[str, set[str]] = {}
        for c in self.cols:
            base = norm_text(c)
            variants = {base, norm_text(str(c).replace("_", " ").replace("-", " "))}
            words = base.split()
            if words:
                head = words[-1]
                variants.add(" ".join(words[:-1] + [_singular(head)]))
                variants.add(_singular(head))
                variants.add(_plural(_singular(head)))
                variants.add(head)
                if len(words) > 1:
                    variants.add(" ".join(words))
            for v in variants:
                v = v.strip()
                if v:
                    self.aliases.setdefault(v, set()).add(c)
        # value index for categorical-ish columns
        self.values: dict[str, tuple[str, str, int]] = {}
        for c in self.cols:
            sem = ds.types.get(c)
            if sem not in ("categorical", "boolean") or c not in ds.df.columns:
                continue
            s = ds.series(c).astype("string").dropna()
            if s.nunique() > 600:
                continue
            vc = s.value_counts()
            for val, cnt in vc.items():
                key = norm_text(val)
                if not key or len(key) < 2 or key in STOPWORDS:
                    continue
                prev = self.values.get(key)
                if prev is None or cnt > prev[2]:
                    self.values[key] = (c, str(val), int(cnt))
        self.value_grams = sorted({k for k in self.values if len(k.split()) >= 2},
                                  key=lambda k: -len(k))

    # ---------------- matching ----------------
    def match_column(self, phrase: str) -> tuple[str | None, float]:
        """Fuzzy column resolution. Deliberately strict: a wrong column match
        produces a confident wrong answer, which is worse than 'not recognised'."""
        p = norm_text(phrase).strip()
        if not p:
            return None, 0.0
        if p in self.aliases:
            return sorted(self.aliases[p])[0], 1.0
        words = p.split()
        best, score = None, 0.0
        # "discount" -> "discount pct", "resolution" -> "resolution hours"
        if len(p) >= 4:
            for alias, cols in self.aliases.items():
                if alias.startswith(p + " ") or alias.endswith(" " + p):
                    cand = 0.9 if alias.startswith(p + " ") else 0.84
                    if cand > score:
                        score, best = cand, sorted(cols)[0]
        floor = 0.82 if len(words) == 1 else 0.78
        for alias, cols in self.aliases.items():
            if abs(len(alias) - len(p)) > max(6, 0.5 * len(p)):
                continue
            s = SequenceMatcher(None, p, alias).ratio()
            if len(words) == 1 and len(words) != len(alias.split()):
                # single word vs multi-word alias: only the prefix rule may match
                if not (alias.startswith(p + " ") or alias.endswith(" " + p)):
                    continue
            if s >= floor and s > score:
                score, best = s, sorted(cols)[0]
        # singular/plural tolerance ("order" -> "orders")
        if score < 0.85 and words:
            for alias, cols in self.aliases.items():
                aw = alias.split()
                if len(aw) == len(words) and _singular(aw[-1]) == _singular(words[-1]):
                    cand = SequenceMatcher(None, p, alias).ratio()
                    if cand >= 0.7 and cand > score:
                        score, best = cand, sorted(cols)[0]
        return (best, score) if score >= floor else (None, score)

    def find_columns(self, tokens: list[str]) -> list[dict]:
        """Non-overlapping best column matches over 1..4-grams."""
        found = []
        n = len(tokens)
        for size in (4, 3, 2, 1):
            for i in range(n - size + 1):
                gram = " ".join(tokens[i:i + size])
                if size == 1 and (gram in STOPWORDS or gram.isdigit()):
                    continue
                col, score = self.match_column(gram)
                if not col:
                    continue
                if any(not (i + size <= f["start"] or i >= f["end"]) for f in found):
                    continue
                bonus = 0.06 * size
                found.append({"col": col, "start": i, "end": i + size, "text": gram,
                              "score": min(1.0, score + bonus)})
        found.sort(key=lambda f: f["start"])
        return found

    def find_values(self, tokens: list[str], used: list[dict]) -> list[dict]:
        out = []
        text = " ".join(tokens)
        taken = [(u["start"], u["end"]) for u in used]
        for key in self.value_grams:
            for m in re.finditer(r"(?<!\w)" + re.escape(key) + r"(?!\w)", text):
                s = text[:m.start()].count(" ")
                e = s + len(key.split())
                if any(not (e <= t[0] or s >= t[1]) for t in taken):
                    continue
                col, val, cnt = self.values[key]
                out.append({"col": col, "value": val, "start": s, "end": e, "text": key})
                taken.append((s, e))
        for i, t in enumerate(tokens):
            if t in self.values and not any(not (i + 1 <= x[0] or i >= x[1]) for x in taken):
                col, val, cnt = self.values[t]
                out.append({"col": col, "value": val, "start": i, "end": i + 1, "text": t})
                taken.append((i, i + 1))
        return out


# ----------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------
def parse(ds: Dataset, question: str) -> dict:
    schema = Schema(ds)
    raw = (question or "").strip()
    text = norm_text(raw)
    text = re.sub(r"[?？]", " ", text)
    tokens = text.split()
    notes: list[str] = []
    spec: dict = {"kind": "aggregate", "filters": [], "sort": {"by": "measure", "dir": "desc"}}

    # --- intent flags -------------------------------------------------
    corr_hint = bool(re.search(r"\b(correlat\w*|relationship between|associat\w+|scatter)\b", text)
                     or re.search(r"\b(against|vs|versus)\b", text))
    dist_hint = bool(re.search(r"\b(distribution of|histogram|how is \w+ distributed|spread of|percentiles)\b", text))
    outlier_hint = bool(re.search(r"\b(outlier\w*|anomal\w*|unusual|spike\w*|weird|abnormal)\b", text))
    intent = "aggregate"   # finalised below, once the columns are known

    # --- slot matching ------------------------------------------------
    col_matches = schema.find_columns(tokens)
    matched_cols = [m["col"] for m in col_matches]
    value_matches = schema.find_values(tokens, col_matches)

    numeric = [m["col"] for m in col_matches if ds.types.get(m["col"]) == "numeric"]
    cats = [m["col"] for m in col_matches if ds.types.get(m["col"]) in ("categorical", "boolean", "id")]
    times = [m["col"] for m in col_matches if ds.types.get(m["col"]) == "datetime"]

    # "plot cost against revenue" only means correlation if two numeric columns
    # were actually found; "compare North vs South" is a comparison, not one
    if corr_hint and len(numeric) >= 2:
        intent = "correlation"
    elif dist_hint:
        intent = "distribution"
    elif outlier_hint:
        intent = "outliers"
    elif corr_hint or re.search(r"\b(compare|comparison|difference between)\b", text):
        intent = "compare"

    # --- aggregation --------------------------------------------------
    agg = None
    for pat, a in AGG_PATTERNS:
        if re.search(pat, text):
            agg = a
            break

    # --- time granularity / trend -------------------------------------
    gran = None
    for pat, g in GRAN_PATTERNS:
        if re.search(pat, text):
            gran = g
            break
    is_time = bool(gran) or bool(re.search(r"\b(trend|over time|time series|timeline|by date|growth over|across time)\b", text))

    # --- "by / per <column>" ------------------------------------------
    dim = None
    measure = None
    by_dim_cols: list[str] = []
    for m in re.finditer(r"\b(?:by|per|for each|across|grouped by|breakdown by|split by|according to)\s+([a-z0-9 ]{2,40})", text):
        phrase = m.group(1).strip()
        col, score = schema.match_column(phrase.split(" and ")[0])
        if col:
            if ds.types.get(col) == "numeric":
                if measure is None:
                    measure = col
                    notes.append(f"measure from '{phrase.split(' and ')[0].strip()}' → {col}")
            elif ds.types.get(col) == "datetime":
                is_time = True
                times = times or [col]
            else:
                by_dim_cols.append(col)

    # --- filters from values & comparisons ----------------------------
    filters = []
    negated = {"cols": set()}
    for vm in value_matches:
        start = vm["start"]
        preceding = " ".join(tokens[max(0, start - 3):start])
        op = "ne" if re.search(r"\b(not|except|excluding|without|other than|besides)\b", preceding) else "eq"
        # a categorical column used as the group-by dimension is not a filter
        if op == "eq" and vm["col"] in by_dim_cols:
            continue
        filters.append({"col": vm["col"], "op": op, "value": vm["value"]})
        notes.append(f"filter {vm['col']} {'≠' if op == 'ne' else '='} {vm['value']} (matched value '{vm['text']}')")

    # explicit "<col> is/= <value>"
    for m in re.finditer(r"\b([a-z0-9_ ]{2,25}?)\s+(?:is|are|=|equals?)\s+([a-z0-9 ]{2,25})", text):
        col, _ = schema.match_column(m.group(1))
        vkey = norm_text(m.group(2).strip())
        if col and vkey in schema.values and schema.values[vkey][0] == col:
            f = {"col": col, "op": "eq", "value": schema.values[vkey][1]}
            if f not in filters:
                filters.append(f)
                notes.append(f"filter {col} = {f['value']} (explicit)")

    # numeric comparisons
    for m in re.finditer(r"\b([a-z0-9_ ]{2,25}?)\s+(?:greater than|more than|above|over|exceeding|>|bigger than)\s+([\d,\.]+)", text):
        col, _ = schema.match_column(m.group(1))
        if col and ds.types.get(col) == "numeric":
            filters.append({"col": col, "op": "gt", "value": float(m.group(2).replace(",", ""))})
            notes.append(f"filter {col} > {m.group(2)}")
    for m in re.finditer(r"\b([a-z0-9_ ]{2,25}?)\s+(?:less than|below|under|fewer than|<|at most)\s+([\d,\.]+)", text):
        col, _ = schema.match_column(m.group(1))
        if col and ds.types.get(col) == "numeric":
            filters.append({"col": col, "op": "lt", "value": float(m.group(2).replace(",", ""))})
            notes.append(f"filter {col} < {m.group(2)}")
    for m in re.finditer(r"\b([a-z0-9_ ]{2,25}?)\s+between\s+([\d,\.]+)\s+and\s+([\d,\.]+)", text):
        col, _ = schema.match_column(m.group(1))
        if col and ds.types.get(col) == "numeric":
            filters.append({"col": col, "op": "between",
                            "value": [float(m.group(2).replace(",", "")), float(m.group(3).replace(",", ""))]})
            notes.append(f"filter {col} between {m.group(2)} and {m.group(3)}")

    # date filters: year / month / relative window
    tcol = times[0] if times else (ds.datetime_cols[0] if ds.datetime_cols else None)
    if tcol:
        if re.search(r"\b(last|previous|past)\s+(year)\b", text):
            yr = pd.Timestamp(ds.series(tcol).max()).year - 1
            filters.append({"col": tcol, "op": "year", "value": f"{yr}-01-01"})
            notes.append(f"filter {tcol} year = {yr}")
        elif re.search(r"\b(this|current)\s+(year)\b", text):
            yr = pd.Timestamp(ds.series(tcol).max()).year
            filters.append({"col": tcol, "op": "year", "value": f"{yr}-01-01"})
            notes.append(f"filter {tcol} year = {yr}")
        else:
            ym = re.search(r"\b(?:(in|for|during|of)\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+(20\d{2})\b", text)
            yonly = re.search(r"\b(?:in|for|during|of)\s+(20\d{2})\b", text)
            mrel = re.search(r"\b(last|previous|past)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s+(day|days|week|weeks|month|months|year|years)\b", text)
            if ym:
                month = MONTHS.get(ym.group(2)[:3], None) or MONTHS.get(ym.group(2), None)
                year = int(ym.group(3))
                if month:
                    filters.append({"col": tcol, "op": "between",
                                    "value": [f"{year}-{month:02d}-01", f"{year}-{month:02d}-28"]})
                    notes.append(f"filter {tcol} in {ym.group(2).title()} {year}")
            elif yonly:
                filters.append({"col": tcol, "op": "year", "value": f"{yonly.group(1)}-01-01"})
                notes.append(f"filter {tcol} year = {yonly.group(1)}")
            elif mrel:
                n = int(mrel.group(2)) if mrel.group(2).isdigit() else NUMBER_WORDS.get(mrel.group(2), 3)
                unit = {"day": "D", "days": "D", "week": "W", "weeks": "W",
                        "month": "M", "months": "M", "year": "Y", "years": "Y"}[mrel.group(3)]
                cutoff = pd.Timestamp(ds.series(tcol).max()) - pd.Timedelta(days=n * {"D": 1, "W": 7, "M": 30, "Y": 365}[unit])
                filters.append({"col": tcol, "op": "gt", "value": cutoff.strftime("%Y-%m-%d")})
                notes.append(f"filter {tcol} after {cutoff.date()} (last {n} {mrel.group(3)})")

    # "compare North vs South" yields two eq filters on one column, which would
    # contradict each other; merge them into a single `in` filter instead
    merged: list[dict] = []
    eq_by_col: dict[str, list[str]] = {}
    for f in filters:
        if f.get("op") == "eq" and isinstance(f.get("value"), str):
            eq_by_col.setdefault(f["col"], []).append(f["value"])
        else:
            merged.append(f)
    for col, vals in eq_by_col.items():
        if len(vals) > 1:
            merged.append({"col": col, "op": "in", "value": vals})
            notes.append(f"merged {len(vals)} values of {col} into one filter")
        else:
            merged.append({"col": col, "op": "eq", "value": vals[0]})
    filters = merged
    spec["filters"] = filters

    # --- top / bottom N -----------------------------------------------
    limit = None
    order = "desc"
    mtop = re.search(r"\b(?:top|best|highest|largest|biggest|most|leading|leading)\s+(\d+|ten|five|twenty|three|two)\b", text)
    mbot = re.search(r"\b(?:bottom|worst|lowest|smallest|least|fewest)\s+(\d+|ten|five|twenty|three|two)\b", text)
    if mtop:
        limit = int(mtop.group(1)) if mtop.group(1).isdigit() else NUMBER_WORDS.get(mtop.group(1), 5)
        order = "desc"
    elif mbot:
        limit = int(mbot.group(1)) if mbot.group(1).isdigit() else NUMBER_WORDS.get(mbot.group(1), 5)
        order = "asc"
    elif re.search(r"\b(top|best|highest|largest|most)\b", text):
        limit = 5
    elif re.search(r"\b(bottom|worst|lowest|least|fewest)\b", text):
        limit, order = 5, "asc"

    # --- final slot resolution ----------------------------------------
    if measure is None:
        non_measure = [c for c in numeric if c not in by_dim_cols]
        if non_measure:
            measure = non_measure[0]
    dim = by_dim_cols[0] if by_dim_cols else (cats[0] if cats else None)
    if intent == "compare" and dim is None:
        for f in filters:
            if f.get("op") == "in" or (f.get("op") == "eq" and f.get("col") in ds.categorical_cols):
                dim = f["col"]
                notes.append(f"comparison grouped by {dim}")
                break

    confidence = 0.35
    if measure or agg == "count":
        confidence += 0.2
    if dim or is_time:
        confidence += 0.2
    if agg:
        confidence += 0.1
    if not col_matches:
        confidence = 0.12
    confidence = round(min(0.98, confidence), 2)

    # --- build spec ---------------------------------------------------
    if intent == "compare" and measure is None:
        prim = pick_measures(ds, 1)
        if prim:
            measure = prim[0]
            notes.append(f"comparison uses the primary measure {measure}")
    alternates: list[str] = []
    if intent == "correlation":
        if len(numeric) >= 2:
            spec = {"kind": "correlation", "x": numeric[0], "y": numeric[1], "filters": filters}
            notes.append(f"correlation of {numeric[0]} and {numeric[1]}")
            measure = None
            confidence = round(max(confidence, 0.75), 2)
        else:
            spec["kind"] = "matrix"
            notes.append("no two numeric columns named — showing the full correlation matrix")
    elif intent == "distribution":
        col = measure or (numeric[0] if numeric else (dim or ds.df.columns[0]))
        spec = {"kind": "distribution", "column": col, "filters": filters}
        notes.append(f"distribution of {col}")
        measure = None
        confidence = round(max(confidence, 0.7), 2)
    elif intent == "outliers":
        col = measure or (numeric[0] if numeric else None)
        spec = {"kind": "distribution", "column": col, "filters": filters} if col else {"kind": "table", "limit": 20}
        notes.append(f"outlier view of {col}")
    elif is_time and (measure or agg == "count"):
        spec = {"kind": "timeseries", "time_dimension": tcol, "measure": measure,
                "agg": agg or default_agg(measure), "granularity": gran or "auto",
                "filters": filters}
        notes.append(f"time series of {measure or 'row count'} by {gran or 'auto-detected'} buckets")
        confidence = round(max(confidence, 0.7), 2)
    elif dim and (measure or agg == "count"):
        spec = {"kind": "aggregate", "dimension": dim, "measure": measure,
                "agg": agg or ("count" if not measure else default_agg(measure)),
                "limit": limit, "other_bucket": bool(limit) and order == "desc",
                "sort": {"by": "measure", "dir": order}, "filters": filters}
        notes.append(f"{AGG_LABELS.get(spec['agg'], spec['agg'])} {measure or 'of rows'} by {dim}")
    elif agg == "count" and not measure and not dim:
        spec = {"kind": "aggregate", "measure": None, "agg": "count", "filters": filters}
        notes.append("plain row count")
    elif re.search(r"\b(list|show (me )?the rows|raw data|records|sample rows)\b", text):
        spec = {"kind": "table", "limit": limit or 20, "filters": filters}
        notes.append("raw rows")
        intent = "table"
    elif measure:
        spec = {"kind": "aggregate", "measure": measure, "agg": agg or default_agg(measure),
                "filters": filters}
        notes.append(f"{AGG_LABELS.get(agg or 'sum')} {measure} over all rows")
    else:
        spec = {"kind": "table", "limit": limit or 20, "filters": filters}
        intent = "table"
        notes.append("no measure or dimension recognised — showing raw rows")

    # ambiguity reporting
    if len(numeric) > 1:
        alternates.append("measure could also be " + ", ".join(numeric[1:4]))
    if len(cats) > 1:
        alternates.append("group-by could also be " + ", ".join(cats[1:4]))

    return {"intent": intent, "spec": spec, "confidence": confidence, "notes": notes,
            "matched_columns": matched_cols, "alternates": alternates,
            "slots": {"measure": measure, "agg": agg or ("count" if not measure else "sum"),
                      "dimension": dim, "time": tcol, "granularity": gran,
                      "limit": limit, "order": order, "filters": filters,
                      "x": spec.get("x"), "y": spec.get("y"), "column": spec.get("column")}}


# ----------------------------------------------------------------------
# prose
# ----------------------------------------------------------------------
def _fmt(value, fmt: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if fmt == "currency":
        return "₹" + compact(value) if abs(value) >= 1000 else "₹" + fmt_num(value)
    if fmt == "percent":
        return fmt_pct(value)
    if fmt == "percent_raw":
        return f"{value:.1f}%"
    return fmt_num(value)


def compose_answer(ds: Dataset, question: str, parsed: dict, result: dict) -> str:
    spec = parsed["spec"]
    fmt = result.get("chart", {}).get("format", "number") if result.get("chart") else "number"
    if fmt == "number":
        fmt = guess_format(spec.get("measure") or spec.get("column") or spec.get("y"))
    rows = result.get("rows") or []
    totals = result.get("totals") or {}
    parts: list[str] = []

    if result.get("error"):
        parts.append(f"I couldn't run that: {result['error']}.")
    kind = spec.get("kind")

    if kind == "correlation" and result.get("correlation"):
        c = result["correlation"]
        r = c["pearson"]
        strength = "very strong" if abs(r) >= 0.85 else "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
        parts.append(f"{spec['x']} and {spec['y']} have a {strength} "
                     f"{'positive' if r > 0 else 'negative'} relationship: Pearson r = {r:+.2f} over "
                     f"{c['n']:,} rows (R² = {c['r2']:.2f}, Spearman {c['spearman']:+.2f}).")
        parts.append(f"On the fitted line, one extra unit of {spec['x']} goes with "
                     f"{c['slope']:+.3g} of {spec['y']}. Association only - this does not prove causation.")
    elif kind == "distribution":
        s = result.get("stats") or {}
        col = spec.get("column")
        if s.get("n") and "mean" in s:
            parts.append(f"{col} across {s['n']:,} values: mean {_fmt(s['mean'], fmt)}, "
                         f"median {_fmt(s['median'], fmt)}, std dev {_fmt(s['std'], fmt)}, "
                         f"range {_fmt(s['min'], fmt)} - {_fmt(s['max'], fmt)}.")
            parts.append(f"Skewness {s['skew']:+.2f} "
                         f"({'right-skewed: a few high values' if s['skew'] > 0.5 else 'left-skewed: a few low values' if s['skew'] < -0.5 else 'roughly symmetric'}); "
                         f"the middle 90% sits between {_fmt(s['p05'], fmt)} and {_fmt(s['p95'], fmt)}.")
        elif rows:
            top = rows[0]
            parts.append(f"{col} takes {result.get('groups', len(rows)):,} distinct values. "
                         f"The most common is '{top['label']}' with {top['count']:,} rows "
                         f"({fmt_pct(top.get('pct_of_total'))}).")
    elif kind == "timeseries":
        lab = (result.get("measure_labels") or ["Rows"])[0]
        vals = [r.get(lab) for r in rows if r.get(lab) is not None]
        if vals:
            total = totals.get(lab)
            i_max = int(np.argmax(vals)); i_min = int(np.argmin(vals))
            head = f"{lab} across {len(vals)} {result.get('granularity', 'period')} buckets"
            if spec.get("agg") == "sum" and total is not None:
                head += f" totals {_fmt(total, fmt)}"
            elif total is not None:
                head += f" averages {_fmt(total, fmt)}"
            parts.append(head + ".")
            parts.append(f"It peaks in {rows[i_max]['label']} at {_fmt(vals[i_max], fmt)} and bottoms out in "
                         f"{rows[i_min]['label']} at {_fmt(vals[i_min], fmt)}.")
            t = result.get("trend")
            if t:
                parts.append(f"The fitted trend moves {t['total_change']:+,.0f} across the window "
                             f"({t['pct_change']:+.0%} of the mean, R² = {t['r2']:.2f}) - "
                             f"{'a real trend' if t['r2'] >= 0.5 else 'directionally suggestive but noisy'}.")
    elif kind == "matrix":
        top = rows[0] if rows else None
        if top:
            parts.append(f"Strongest pair: {top['x']} and {top['y']} at r = {top['r']:+.2f}. "
                         f"{len(rows)} numeric pairs compared; the matrix shows Pearson r for each.")
    elif kind == "table":
        parts.append(f"Showing {len(rows):,} of {totals.get('count', 0):,} matching rows"
                     + (f" after filtering ({', '.join(result.get('filter_descriptions') or [])})"
                        if result.get("filtered_out") else "") + ".")
    elif kind == "segments":
        lab = (result.get("measure_labels") or ["value"])[0]
        base = result.get("baseline")
        if rows and base:
            best, worst = rows[0], rows[-1]
            parts.append(f"{lab} averages {_fmt(base, fmt)} overall. "
                         f"{best['label']} is highest at {_fmt(best[lab], fmt)} "
                         f"({best['index_vs_baseline']:.2f}x the overall figure) and "
                         f"{worst['label']} lowest at {_fmt(worst[lab], fmt)} "
                         f"({worst['index_vs_baseline']:.2f}x), across {len(rows)} segments.")
    else:  # aggregate
        dim = result.get("dimension")
        lab = (result.get("measure_labels") or ["Rows"])[0]
        ascending = (spec.get("sort") or {}).get("dir") == "asc"
        n_groups = result.get("groups", len(rows))
        if dim and rows:
            top = rows[0]
            total = totals.get(lab)
            share = top.get("pct_of_total")
            if spec.get("agg") == "count" and result.get("filtered_out") and len(rows) == 1:
                parts.append(f"There are {int(top.get(lab) or 0):,} rows matching the filters "
                             f"({', '.join(result.get('filter_descriptions') or [])}).")
            else:
                if spec.get("agg") == "count":
                    lead = (f"{top['label']} has the most rows - {int(top.get(lab) or 0):,}"
                            + (f" ({fmt_pct(share)})" if share else "")
                            + f" - across {n_groups} {dim} values.")
                else:
                    word = "lowest" if ascending else "highest"
                    lead = (f"{lab} is {word} for {top['label']} at {_fmt(top.get(lab), fmt)}"
                            + (f" ({fmt_pct(share)} of the total)" if share else "")
                            + f", across {n_groups} {dim} values.")
                parts.append(lead)
                if len(rows) > 1:
                    bits = [f"{r['label']} {_fmt(r.get(lab), fmt)}" for r in rows[1:4]]
                    parts.append(("Next lowest: " if ascending else "Then ") + ", ".join(bits) + ".")
                if total is not None and spec.get("agg") == "sum":
                    shown = sum(r.get(lab) or 0 for r in rows)
                    if total and len(rows) < n_groups:
                        parts.append(f"The {len(rows)} groups shown cover {fmt_pct(shown / total)} "
                                     f"of the {_fmt(total, fmt)} total.")
                    else:
                        parts.append(f"Total across all groups: {_fmt(total, fmt)}.")
                if result.get("filtered_out"):
                    parts.append(f"Based on {result['rows_used']:,} of {result['rows_total']:,} rows after "
                                 f"filtering ({', '.join(result.get('filter_descriptions') or [])}).")
        elif rows:
            val = rows[0].get(lab)
            if spec.get("agg") == "count":
                parts.append(f"There are {int(val or 0):,} rows"
                             + (f" matching the filters ({', '.join(result.get('filter_descriptions') or [])})"
                                if result.get("filtered_out") else "."))
            else:
                extra = f" over {totals['count']:,} rows" if totals.get("count") else ""
                parts.append(f"{lab} is {_fmt(val, fmt)}{extra}.")
                if result.get("filtered_out"):
                    parts.append(f"That excludes {result['filtered_out']:,} of {result['rows_total']:,} rows "
                                 f"(filters applied: {', '.join(result.get('filter_descriptions') or [])}).")

    return " ".join(parts).strip()


def follow_ups(ds: Dataset, parsed: dict) -> list[str]:
    spec = parsed["spec"]
    out = []
    dim, meas = spec.get("dimension"), spec.get("measure")
    times = ds.datetime_cols
    if meas and dim:
        out.append(f"Average {meas} by {dim}")
        out.append(f"Which {dim} has the lowest {meas}?")
    if meas and times:
        out.append(f"Show the monthly trend of {meas}")
    if len(ds.numeric_cols) >= 2 and not meas:
        out.append(f"Correlation between {ds.numeric_cols[0]} and {ds.numeric_cols[1]}")
    dims = [c for c in ds.categorical_cols if 2 <= ds.df[c].nunique() <= 20]
    if dims:
        out.append(f"Count rows by {dims[0]}")
    if meas:
        out.append(f"Distribution of {meas}")
    seen, uniq = set(), []
    for q in out:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq[:4]


def answer_question(ds: Dataset, question: str) -> dict:
    parsed = parse(ds, question)
    result = run_query(ds, parsed["spec"])
    prose = compose_answer(ds, question, parsed, result)
    return {
        "question": question,
        "answer": prose,
        "parsed": parsed,
        "result": result,
        "follow_ups": follow_ups(ds, parsed),
        "engine": "local",
    }


def example_questions(ds: Dataset) -> list[str]:
    """Suggested questions, ordered so the first one is the most useful."""
    from .profile import pick_measures, pick_time_col
    from .semantics import agg_word, default_agg

    measures = pick_measures(ds, limit=3)
    m = measures[0] if measures else None
    word = agg_word(m)
    dims = sorted([c for c in ds.categorical_cols if 2 <= ds.df[c].nunique() <= 25],
                  key=lambda c: ds.df[c].nunique())
    d = dims[0] if dims else None
    t = pick_time_col(ds)
    qs = []
    if d:
        qs.append(f"How many rows per {d}?")           # always true, always useful first
    if m and d:
        qs.append(f"{word} {m} by {d}")
        qs.append(f"Top 5 {d} by {m}")
    if m and t:
        qs.append(f"Monthly trend of {m}")
        qs.append(f"Show {m} over time")
    if m:
        qs.append(f"Distribution of {m}")
    if len(measures) >= 2:
        qs.append(f"Correlation between {measures[0]} and {measures[1]}")
    if len(dims) >= 2 and m:
        vc = ds.series(dims[1]).astype("string").value_counts()
        if len(vc):
            qs.append(f"{m} for {vc.index[0]}")
    if m and dims:
        qs.append(f"Which {dims[-1]} has the lowest {m}?")
    if t and not m:
        qs.append(f"Rows by {t}")
    seen, out = set(), []
    for q in qs:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out[:8]
