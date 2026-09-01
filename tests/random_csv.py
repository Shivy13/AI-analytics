"""Random CSV generator for fuzz-testing the analytics pipeline.

Produces several plausible schemas with *injected* defects (missing values,
outliers, duplicate rows, mixed-case categories, high-cardinality ids, free
text, alternate delimiters/encodings) so every detector and the NL layer get
exercised on data they have never seen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHAPES = ["sales", "sensors", "hr"]


def make_random_csv(seed: int, shape: str | None = None, rows: int | None = None,
                    delim: str = ",", encoding: str = "utf-8") -> tuple[bytes, str]:
    rng = np.random.default_rng(seed)
    shape = shape or SHAPES[seed % len(SHAPES)]
    n = rows or int(rng.integers(150, 900))
    start = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 400)))

    if shape == "sales":
        ts = start + pd.to_timedelta(np.sort(rng.integers(0, 500, n)), unit="D")
        region = rng.choice(["North", "South", "East", "West"], n)
        sku = rng.choice([f"SKU-{i:03d}" for i in range(int(rng.integers(6, 30)))], n)
        qty = rng.integers(1, 12, n)
        price = np.round(rng.uniform(40, 900, n), 2)
        amount = np.round(qty * price * rng.uniform(0.85, 1.0, n), 2)
        returned = rng.random(n) < 0.06
        df = pd.DataFrame({"order_date": ts.strftime("%Y-%m-%d"), "region": region,
                           "sku": sku, "qty": qty, "unit_price": price,
                           "amount": amount, "returned": np.where(returned, "yes", "no")})
        amount_col, date_col = "amount", "order_date"
        out = rng.random(n) < 0.03
        df.loc[out, "amount"] = np.round(rng.uniform(20000, 60000, int(out.sum())), 2)
    elif shape == "sensors":
        ts = start + pd.to_timedelta(np.sort(rng.integers(0, 60 * 24 * 30, n)), unit="m")
        device = rng.choice([f"DEV-{i:02d}" for i in range(int(rng.integers(4, 14)))], n)
        temp = np.round(rng.normal(22, 3, n) + 4 * np.sin(np.arange(n) / 40), 2)
        hum = np.round(np.clip(rng.normal(55, 12, n), 5, 98), 1)
        batt = np.round(np.clip(100 - np.arange(n) * rng.uniform(0.01, 0.05) + rng.normal(0, 3, n), 0, 100), 1)
        status = rng.choice(["ok", "ok", "ok", "warn", "fault"], n)
        df = pd.DataFrame({"ts": ts.strftime("%Y-%m-%d %H:%M"), "device_id": device,
                           "temperature_c": temp, "humidity_pct": hum,
                           "battery": batt, "status": status})
        amount_col, date_col = "temperature_c", "ts"
        hot = rng.random(n) < 0.02
        df.loc[hot, "temperature_c"] = np.round(rng.uniform(60, 90, int(hot.sum())), 2)
    else:  # hr
        hire = start - pd.to_timedelta(np.sort(rng.integers(0, 2000, n))[::-1], unit="D")
        dept_raw = rng.choice(["engineering", "Engineering", "sales", "SALES", "hr", "HR", "ops"], n)
        salary = np.round(rng.lognormal(10.8, 0.4, n), 0)
        tenure = ((pd.Timestamp("2025-01-01") - hire).days / 365.25).round(1)
        rating = np.round(np.clip(rng.normal(3.6, 0.7, n), 1, 5), 1)
        attr = rng.random(n) < 0.14
        notes = rng.choice(["needs ramp-up", "strong q", "on leave", "mentor", "remote", ""], n)
        df = pd.DataFrame({"emp_id": [f"E{1000 + i}" for i in range(n)],
                           "hire_date": hire.strftime("%Y-%m-%d"), "department": dept_raw,
                           "salary": salary, "tenure_years": tenure,
                           "rating": rating, "attrited": np.where(attr, "Y", "N"),
                           "notes": notes})
        amount_col, date_col = "salary", "hire_date"
        rich = rng.random(n) < 0.02
        df.loc[rich, "salary"] = np.round(rng.uniform(400000, 900000, int(rich.sum())), 0)

    # ---- inject generic defects --------------------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in rng.choice(num_cols, size=min(2, len(num_cols)), replace=False):
        df.loc[rng.random(n) < rng.uniform(0.01, 0.07), c] = np.nan
    cat_cols = [c for c in df.columns if df[c].dtype == object and c not in
                (date_col, "notes")]
    if cat_cols:
        c = str(rng.choice(cat_cols))
        df.loc[rng.random(n) < 0.02, c] = np.nan
    dups = int(rng.integers(0, max(2, n // 60)))
    if dups:
        df = pd.concat([df, df.sample(dups, random_state=seed)], ignore_index=True)

    text = df.to_csv(index=False, sep=delim)
    return text.encode(encoding), f"random_{shape}_{seed}.csv", amount_col, date_col


def gen(seed: int, **kw):
    raw, name, amount_col, date_col = make_random_csv(seed, **kw)
    return raw, name, amount_col, date_col
