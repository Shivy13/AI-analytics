"""Column semantics: which direction is 'good', and how a measure should be
aggregated. Kept dependency-free so profile, query and insights can all use it.
"""
from __future__ import annotations

LOWER_IS_BETTER = ("cost", "refund", "return", "complaint", "resolution", "response", "delay",
                   "wait", "bounce", "churn", "error", "defect", "loss", "discount", "cpc", "cpa",
                   "downtime", "backlog", "aging", "hours", "mins", "minutes", "duration", "latency")

AVG_LIKE = ("rate", "pct", "percent", "ratio", "csat", "rating", "score", "satisfaction", "cpc",
            "ctr", "roas", "price", "margin", "avg", "average", "share", "hours", "mins",
            "minutes", "duration", "latency", "age", "time")

# outcome measures businesses actually report on
PRIORITY_HINTS = ("revenue", "sales", "profit", "amount", "orders", "conversions", "tickets",
                  "spend", "gmv", "income", "quantity", "units", "clicks", "sessions", "salary")


def higher_is_better(col: str) -> bool:
    """Direction of 'good' - decides whether a trend is welcome or worrying."""
    return not any(h in str(col).lower() for h in LOWER_IS_BETTER)


def default_agg(col: str | None) -> str:
    """Sum additive measures; average rates, ratings and durations.
    Summing a CSAT score or a unit price gives a meaningless number."""
    if not col:
        return "count"
    return "avg" if any(h in col.lower() for h in AVG_LIKE) else "sum"


def agg_word(col: str | None) -> str:
    return "Average" if default_agg(col) == "avg" else "Total"


def measure_priority(col: str) -> int:
    """Ranking score for 'which column is the headline measure'."""
    low = str(col).lower()
    score = 3 if any(h in low for h in PRIORITY_HINTS) else 0
    if default_agg(col) == "avg":
        score -= 2          # prices, rates and ratings are drivers, not outcomes
    return score
