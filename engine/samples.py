"""Bundled sample datasets, generated deterministically on first run.

They are deliberately imperfect: missing values, mixed-case categories,
duplicate rows, negative profits, outliers and seasonality, so the insight
engine has something real to find.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .loader import SAMPLES_DIR

SAMPLES = {
    "retail_sales": {
        "file": "retail_sales.csv", "label": "Retail sales orders",
        "description": "2,400 orders over 2.5 years: region, product, channel, revenue, profit, returns.",
    },
    "marketing": {
        "file": "marketing_campaigns.csv", "label": "Marketing campaigns",
        "description": "Weekly channel spend vs impressions, clicks, conversions and revenue.",
    },
    "support_tickets": {
        "file": "support_tickets.csv", "label": "Support tickets (messy)",
        "description": "Ticket log with mixed-case categories, duplicate rows, blanks and outliers.",
    },
}


def _retail(rng: np.random.Generator) -> pd.DataFrame:
    n = 2400
    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        np.sort(rng.integers(0, 912, n)), unit="D")
    # growth + seasonality
    doy = dates.dayofyear.to_numpy()
    month = dates.month.to_numpy()
    season = 1 + 0.35 * np.sin((month - 3) / 12 * 2 * np.pi) + 0.28 * (month == 11) + 0.22 * (month == 12)
    growth = 1 + 0.55 * (np.arange(n) / n)

    regions = rng.choice(["North", "South", "East", "West"], n, p=[0.34, 0.26, 0.22, 0.18])
    cities = {"North": ["Delhi", "Jaipur", "Chandigarh", "Lucknow"],
              "South": ["Bengaluru", "Chennai", "Hyderabad", "Kochi"],
              "East": ["Kolkata", "Guwahati", "Bhubaneswar", "Patna"],
              "West": ["Mumbai", "Pune", "Ahmedabad", "Surat"]}
    city = np.array([rng.choice(cities[r]) for r in regions])
    category = rng.choice(["Electronics", "Apparel", "Home & Kitchen", "Beauty", "Sports"],
                          n, p=[0.3, 0.24, 0.2, 0.15, 0.11])
    products = {"Electronics": ["Wireless Earbuds", "Smart Watch", "Bluetooth Speaker", "Power Bank", "4K Monitor"],
                "Apparel": ["Cotton T-Shirt", "Denim Jeans", "Rain Jacket", "Wool Sweater"],
                "Home & Kitchen": ["Chef Knife", "Air Fryer", "Cookware Set", "Espresso Machine"],
                "Beauty": ["Vitamin C Serum", "Sunscreen SPF50", "Hair Oil", "Face Wash"],
                "Sports": ["Yoga Mat", "Running Shoes", "Dumbbell Set", "Cricket Bat"]}
    product = np.array([rng.choice(products[c]) for c in category])
    base_price = {"Electronics": 4200, "Apparel": 1400, "Home & Kitchen": 3100, "Beauty": 700, "Sports": 1900}
    unit_price = np.round(np.array([base_price[c] for c in category]) * rng.normal(1, 0.22, n)
                          * season * growth, 2)
    unit_price = np.clip(unit_price, 149, None)
    units = rng.integers(1, 9, n)
    discount = np.round(rng.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], n), 2)
    revenue = np.round(unit_price * units * (1 - discount), 2)
    cost = np.round(revenue * rng.uniform(0.45, 0.78, n), 2)
    profit = np.round(revenue - cost, 2)
    returned = rng.random(n) < 0.07
    profit[returned] = -np.round(revenue[returned] * rng.uniform(0.1, 0.5, returned.sum()), 2)
    channel = rng.choice(["Online", "Retail Store", "Partner"], n, p=[0.58, 0.3, 0.12])
    segment = rng.choice(["Consumer", "Corporate", "Home Office"], n, p=[0.62, 0.26, 0.12])
    # outliers: a handful of bulk orders
    idx = rng.choice(n, 9, replace=False)
    units[idx] = rng.integers(120, 400, len(idx))
    revenue[idx] = np.round(unit_price[idx] * units[idx] * (1 - discount[idx]), 2)
    cost[idx] = np.round(revenue[idx] * 0.62, 2)
    profit[idx] = np.round(revenue[idx] - cost[idx], 2)

    df = pd.DataFrame({
        "order_id": [f"ORD-{100000 + i}" for i in range(n)],
        "order_date": dates.strftime("%Y-%m-%d"),
        "region": regions, "city": city, "category": category, "product": product,
        "channel": channel, "segment": segment, "units": units,
        "unit_price": unit_price, "discount_pct": discount,
        "revenue": revenue, "cost": cost, "profit": profit,
        "returned": np.where(returned, "Yes", "No"),
        "customer_id": [f"CUST-{rng.integers(1000, 1900)}" for _ in range(n)],
    })
    # realistic messiness
    blank_city = rng.choice(n, 42, replace=False)
    df.loc[blank_city, "city"] = np.nan
    blank_disc = rng.choice(n, 118, replace=False)
    df.loc[blank_disc, "discount_pct"] = np.nan
    df.loc[rng.choice(n, 14, replace=False), "segment"] = np.nan
    return df


def _marketing(rng: np.random.Generator) -> pd.DataFrame:
    weeks = pd.date_range("2023-01-02", "2025-06-23", freq="W-MON")
    channels = ["Paid Search", "Social", "Email", "Display", "Affiliate"]
    # click-through rates scale with cost-per-click so that conversions-per-rupee
    # is comparable across channels; otherwise pooling channels hides the funnel
    eff = {"Paid Search": 0.030, "Social": 0.021, "Email": 0.0095, "Display": 0.0135, "Affiliate": 0.019}
    cpc = {"Paid Search": 38, "Social": 26, "Email": 9, "Display": 15, "Affiliate": 22}
    scale = {"Paid Search": 1.25, "Social": 1.0, "Email": 0.8, "Display": 0.9, "Affiliate": 0.7}
    aov = {"Paid Search": 3200, "Social": 2100, "Email": 2600, "Display": 1500, "Affiliate": 2900}
    rows = []
    for i, w in enumerate(weeks):
        for ch in channels:
            spend = float(rng.gamma(6, 320) * rng.uniform(0.6, 1.7) * scale[ch])
            if ch == "Display" and w.month == 11:
                spend *= 2.2  # festive push
            impressions = spend / cpc[ch] * rng.uniform(35, 48)
            clicks = impressions * eff[ch] * rng.uniform(0.8, 1.25)
            conversions = clicks * rng.uniform(0.03, 0.06)
            revenue = conversions * aov[ch] * rng.uniform(0.85, 1.2)
            rows.append({"week_start": w.strftime("%Y-%m-%d"), "channel": ch,
                         "campaign": f"{ch.split()[0]}-{w.strftime('%y%m')}",
                         "spend": round(spend, 2), "impressions": int(impressions),
                         "clicks": int(clicks), "conversions": int(conversions),
                         "revenue": round(revenue, 2),
                         "ctr_pct": round(clicks / impressions * 100, 3) if impressions else 0,
                         "cpc": round(spend / clicks, 2) if clicks else 0,
                         "roas": round(revenue / spend, 2) if spend else 0})
    df = pd.DataFrame(rows)
    df.loc[rng.choice(len(df), 26, replace=False), "revenue"] = np.nan
    return df.sample(frac=1.0, random_state=7).reset_index(drop=True)


def _tickets(rng: np.random.Generator) -> pd.DataFrame:
    n = 1600
    created = pd.Timestamp("2024-01-01 00:00") + pd.to_timedelta(rng.integers(0, 545 * 24, n), unit="h")
    cats = ["billing", "Billing", "BILLING", "login issue", "Login Issue", "feature request",
            "Feature Request", "bug", "Bug", "shipping", "Shipping"]
    category = rng.choice(cats, n, p=[.12, .1, .03, .14, .11, .07, .05, .13, .09, .09, .07])
    priority = rng.choice(["low", "medium", "high", "urgent"], n, p=[.35, .35, .22, .08])
    channel = rng.choice(["email", "chat", "phone", "portal"], n, p=[.34, .3, .12, .24])
    agent = rng.choice(["Ananya", "Rahul", "Meera", "Vikram", "Sara", "Imran"], n,
                       p=[.22, .2, .18, .16, .13, .11])
    base = {"low": 9, "medium": 6, "high": 3.5, "urgent": 2.2}
    resolution = np.round(np.array([base[p] for p in priority]) * rng.lognormal(0, 0.55, n), 2)
    first_resp = np.round(resolution * rng.uniform(0.05, 0.4, n), 2)
    csat = np.clip(np.round(rng.normal(4.1, 0.7, n) - np.where(resolution > 24, 0.6, 0)), 1, 5)
    reopened = rng.random(n) < 0.11
    df = pd.DataFrame({
        "ticket_id": [f"TKT-{5000 + i}" for i in range(n)],
        "created_at": created.strftime("%Y-%m-%d %H:%M"),
        "category": category, "priority": priority, "channel": channel, "agent": agent,
        "first_response_mins": first_resp, "resolution_hours": resolution,
        "csat": csat, "reopened": np.where(reopened, "Yes", "No"),
        "summary": [rng.choice(["Customer cannot reset password", "Refund not received after 14 days",
                                "Wants dark mode in the mobile app", "App crashes on checkout",
                                "Delivery delayed by courier", "Invoice shows wrong GST number",
                                "Unable to apply coupon code", "Request to merge duplicate accounts"])
                    for _ in range(n)],
    })
    # data-quality problems on purpose
    df.loc[rng.choice(n, 96, replace=False), "csat"] = np.nan
    df.loc[rng.choice(n, 40, replace=False), "agent"] = np.nan
    dup = df.sample(18, random_state=3)          # exact duplicates
    stuck = df.sample(6, random_state=4).copy()  # absurd resolution times
    stuck["resolution_hours"] = np.round(rng.uniform(400, 900, len(stuck)), 2)
    out = pd.concat([df, dup, stuck], ignore_index=True)
    return out.sample(frac=1.0, random_state=11).reset_index(drop=True)


def ensure_samples() -> dict[str, str]:
    paths = {}
    builders = {"retail_sales": _retail, "marketing": _marketing, "support_tickets": _tickets}
    for key, meta in SAMPLES.items():
        path = os.path.join(SAMPLES_DIR, meta["file"])
        if not os.path.exists(path):
            rng = np.random.default_rng(42)
            builders[key](rng).to_csv(path, index=False)
        paths[key] = path
    return paths


def sample_meta() -> list[dict]:
    ensure_samples()
    out = []
    for key, meta in SAMPLES.items():
        path = os.path.join(SAMPLES_DIR, meta["file"])
        try:
            df = pd.read_csv(path, nrows=5)
            n = sum(1 for _ in open(path)) - 1
        except Exception:
            n, df = 0, None
        out.append({"key": key, "label": meta["label"], "description": meta["description"],
                    "rows": n, "columns": len(df.columns) if df is not None else 0,
                    "column_names": list(df.columns) if df is not None else []})
    return out
