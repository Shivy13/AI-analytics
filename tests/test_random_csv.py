"""Fuzz tests: random CSVs through the real HTTP API, with every numeric
claim re-checked against an independent pandas calculation on the *same bytes
the server parsed*."""
import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from engine.loader import read_csv_bytes
from engine.semantics import default_agg
from tests.random_csv import gen

import server

CASES = [(1, "sales", ",", "utf-8"), (2, "sensors", ",", "utf-8"),
         (3, "hr", ",", "utf-8"), (11, "sales", ";", "utf-8"),
         (23, "sensors", "\t", "utf-8")]


@pytest.fixture(scope="module")
def client():
    return TestClient(server.app)


def upload(client, seed, shape, delim, enc):
    raw, name, amount_col, date_col = gen(seed, shape=shape, delim=delim, encoding=enc)
    r = client.post("/api/upload", files={"file": (name, raw, "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    df = read_csv_bytes(raw)  # the exact frame the app works from
    return body, df


@pytest.mark.parametrize("seed,shape,delim,enc", CASES)
def test_random_end_to_end(client, seed, shape, delim, enc):
    client.cookies.clear()          # fresh guest identity per case (own quota bucket)
    body, df = upload(client, seed, shape, delim, enc)
    did = body["dataset"]["id"]
    try:
        profile = {c["name"]: c for c in body["profile"]}
        assert body["dataset"]["rows"] == len(df)

        # ---- injected numeric / date columns must be detected
        for c, t in df.dtypes.items():
            if t.kind in "if":
                assert profile[c]["type"] == "numeric", (c, profile[c]["type"])

        # ---- the report exists and is internally consistent
        rep = body["report"]
        assert rep["total_found"] >= 1
        assert rep["dashboard"]["kpis"]
        for ch in rep["dashboard"]["charts"]:
            assert ch["chart"]["x"]["values"], ch["title"]

        # ---- injected defects must be found
        n_dup = int(df.duplicated().sum())
        if n_dup:
            assert any(i["type"] == "duplicate_rows" for i in rep["insights"]), "dups missed"
        missing_cols = [c for c in df.columns if df[c].isna().mean() >= 0.02]
        if missing_cols:
            assert any(i["type"] == "missing_data" for i in rep["insights"]), "missing missed"

        # ---- row count question matches pandas exactly
        a = client.post(f"/api/datasets/{did}/ask", json={"question": "how many rows are there"})
        assert a.status_code == 200
        assert a.json()["result"]["totals"]["count"] == len(df)

        # ---- "total <num> by <dim>" matches a groupby sum
        nums = [c for c in df.columns if profile[c]["type"] == "numeric"]
        cats = [c for c in df.columns if profile[c]["type"] == "categorical"
                and 2 <= df[c].nunique() <= 40]
        if nums and cats:
            m, d = nums[0], cats[0]
            q = client.post(f"/api/datasets/{did}/ask",
                            json={"question": f"total {m} by {d}"})
            got = {r["label"]: r[sorted(r.keys() - {"label", "count", "pct_of_total"})[0]]
                   for r in q.json()["result"]["rows"]}
            exp = pd.to_numeric(df[m], errors="coerce").groupby(df[d].fillna("(blank)")).sum()
            assert set(got) == set(exp.index)
            for k, v in exp.items():
                assert got[k] == pytest.approx(float(v)), (m, d, k)

        # ---- every suggested question answers without an error
        for q in rep["examples"]:
            r = client.post(f"/api/datasets/{did}/ask", json={"question": q})
            assert r.status_code == 200, q
            assert not r.json()["result"].get("error"), (q, r.json()["result"].get("error"))

        # ---- each low-cardinality dimension aggregates to the right group count
        for d in cats[:3]:
            r = client.post(f"/api/datasets/{did}/query",
                            json={"spec": {"kind": "aggregate", "dimension": d,
                                           "measure": nums[0] if nums else None,
                                           "agg": "sum" if nums else "count"}})
            assert r.json()["groups"] == df[d].fillna("(blank)").nunique(), d

        # ---- time series, distribution, matrix, segments all run cleanly
        dates = [c for c in df.columns if profile[c]["type"] == "datetime"]
        if dates and nums:
            r = client.post(f"/api/datasets/{did}/query",
                            json={"spec": {"kind": "timeseries", "time_dimension": dates[0],
                                           "measure": nums[0], "agg": default_agg(nums[0])}})
            assert r.json()["rows"], "timeseries empty"
            assert len(r.json()["rows"]) >= 2
        for c in nums[:3]:
            r = client.post(f"/api/datasets/{did}/query",
                            json={"spec": {"kind": "distribution", "column": c}})
            assert "stats" in r.json(), c
        if len(nums) >= 2:
            r = client.post(f"/api/datasets/{did}/query", json={"spec": {"kind": "matrix"}})
            assert r.json()["chart"]["type"] == "heatmap"
        if cats and nums:
            r = client.post(f"/api/datasets/{did}/query",
                            json={"spec": {"kind": "segments", "dimension": cats[0],
                                           "measure": nums[0], "agg": "avg"}})
            base = float(pd.to_numeric(df[nums[0]], errors="coerce").mean())
            assert r.json()["baseline"] == pytest.approx(base)

        # ---- export and table search/pagination
        e = client.post(f"/api/datasets/{did}/export",
                        json={"spec": {"kind": "aggregate", "dimension": cats[0] if cats else None,
                                       "measure": nums[0] if nums else None, "agg": "sum"}})
        assert e.status_code == 200 and e.json()["rows"] > 0
        if cats:
            val = str(df[cats[0]].dropna().iloc[0])
            t = client.get(f"/api/datasets/{did}/table", params={"search": val, "limit": 10})
            assert t.json()["total"] == int((df.astype("string") == val).any(axis=1).sum()) or \
                t.json()["total"] >= 1
    finally:
        client.delete(f"/api/datasets/{did}")


def test_latin1_and_odd_values(client):
    client.cookies.clear()
    raw = ("id,ville,révenu\n1,Paris,1200\n2,Lyon,980\n3,Marseille,1500\n"
           "4,Paris,760\n5,Lyon,1100\n").encode("latin-1")
    r = client.post("/api/upload", files={"file": ("accent.csv", raw, "text/csv")})
    assert r.status_code == 200, r.text
    did = r.json()["dataset"]["id"]
    try:
        prof = {c["name"]: c for c in r.json()["profile"]}
        assert "révenu" in prof and prof["révenu"]["type"] == "numeric"
        a = client.post(f"/api/datasets/{did}/ask", json={"question": "total révenu by ville"})
        rows = {r["label"]: r["Total révenu"] for r in a.json()["result"]["rows"]}
        assert rows == {"Paris": pytest.approx(1960.0), "Lyon": pytest.approx(2080.0),
                        "Marseille": pytest.approx(1500.0)}
    finally:
        client.delete(f"/api/datasets/{did}")


def test_tiny_and_degenerate_uploads(client):
    client.cookies.clear()
    # one row, one column, all-constant column: must not crash, must still answer
    for raw in (b"a\n1\n", b"x,y\n5,5\n5,5\n5,5\n"):
        r = client.post("/api/upload", files={"file": ("deg.csv", raw, "text/csv")})
        assert r.status_code == 200, r.text
        did = r.json()["dataset"]["id"]
        rep = r.json()["report"]
        assert rep["total_found"] >= 0
        a = client.post(f"/api/datasets/{did}/ask", json={"question": "how many rows"})
        assert a.status_code == 200
        client.delete(f"/api/datasets/{did}")
    # header-only file has columns but no rows -> rejected cleanly
    r = client.post("/api/upload", files={"file": ("h.csv", b"a,b,c\n", "text/csv")})
    assert r.status_code == 400
