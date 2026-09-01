"""Small shared helpers: JSON-safe conversion + human number formatting."""
from __future__ import annotations

import math
import re
from datetime import date, datetime

import numpy as np
import pandas as pd


def js(o):
    """Make any python/numpy object JSON-serialisable."""
    if o is None or isinstance(o, (str, bool)):
        return o
    if isinstance(o, (int,)):
        return o
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
        return o
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return [js(x) for x in o.tolist()]
    if isinstance(o, (pd.Series,)):
        return [js(x) for x in o.tolist()]
    if isinstance(o, dict):
        return {str(k): js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [js(x) for x in o]
    if isinstance(o, pd.DataFrame):
        return js(o.to_dict(orient="list"))
    return str(o)


def compact(x, digits: int = 1) -> str:
    if x is None:
        return "—"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(x):
        return "—"
    a = abs(x)
    for lim, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= lim:
            v = x / lim
            return f"{v:.{digits}f}".rstrip("0").rstrip(".") + suf
    if a >= 100 or x == int(x):
        return f"{x:,.0f}"
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def fmt_num(x, digits: int = 2) -> str:
    if x is None:
        return "—"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(x):
        return "—"
    if abs(x) >= 1e6:
        return compact(x)
    if x == int(x):
        return f"{x:,.0f}"
    s = f"{x:,.{digits}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def fmt_pct(x, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:.{digits}f}%".replace(".0%", "%") if digits else f"{x * 100:.0f}%"


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def titleize(s: str) -> str:
    out = re.sub(r"[_\-]+", " ", str(s)).strip()
    return out[0].upper() + out[1:] if out else out
