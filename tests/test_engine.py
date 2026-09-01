"""Engine tests: every assertion is checked against an independent pandas
calculation, so a wrong number in the app fails here rather than in the UI."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import llm, nlq, samples  # noqa: E402
from engine.insights import build_dashboard, generate_insights  # noqa: E402
from engine.loader import Dataset, read_csv_bytes  # noqa: E402
from engine.profile import build_profile  # noqa: E402
from engine.query import run_query  # noqa: E402


@pytest.fixture(scope="module")
def retail():
    path = samples.ensure_samples()["retail_sales"]
    ds = Dataset("t-retail", "retail", pd.read_csv(path, dtype=str, na_values=[""]))
    build_profile(ds)
    return ds


@pytest.fixture(scope="module")
def marketing():
    path = samples.ensure_samples()["marketing"]
    ds = Dataset("t-mkt", "marketing", pd.read_csv(path, dtype=str, na_values=[""]))
    build_profile(ds)
    return ds


@pytest.fixture(scope="module")
def tickets():
    path = samples.ensure_samples()["support_tickets"]
    ds = Dataset("t-tk", "tickets", pd.read_csv(path, dtype=str, na_values=[""]))
    build_profile(ds)
    return ds


# --------------------------------------------------------------- profiling
def test_type_inference(retail):
    t = retail.types
    assert t["order_date"] == "datetime"
    assert t["revenue"] == "numeric"
    assert t["region"] == "categorical"
    assert t["returned"] == "boolean"
    assert t["order_id"] == "id"
    assert t["customer_id"] == "id"


def test_profile_matches_pandas(retail):
    df = retail.df
    p = retail.profile["by_name"]["revenue"]
    v = pd.to_numeric(df["revenue"], errors="coerce")
    assert p["count"] == int(v.notna().sum())
    assert p["missing"] == int(v.isna().sum())
    assert p["sum"] == pytest.approx(float(v.sum()))
    assert p["mean"] == pytest.approx(float(v.mean()))
    assert p["median"] == pytest.approx(float(v.median()))
    assert p["negatives"] == int((v < 0).sum())
    # IQR outlier count reproduced by hand
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr = q3 - q1
    manual = int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum())
    assert p["outlier_count"] == manual


def test_missing_counts(retail):
    df = retail.df
    assert retail.profile["by_name"]["city"]["missing"] == int(df["city"].isna().sum())
    assert retail.profile["by_name"]["city"]["missing"] > 0


# --------------------------------------------------------------- queries
def test_aggregate_matches_pandas(retail):
    res = run_query(retail, {"kind": "aggregate", "dimension": "region", "measure": "revenue",
                             "agg": "sum", "sort": {"by": "measure", "dir": "desc"}})
    df = retail.df.assign(revenue=pd.to_numeric(retail.df["revenue"], errors="coerce"))
    expected = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    got = {r["label"]: r["Total revenue"] for r in res["rows"]}
    assert list(got) == list(expected.index)
    for k, v in expected.items():
        assert got[k] == pytest.approx(float(v))
    assert res["totals"]["Total revenue"] == pytest.approx(float(df["revenue"].sum()))
    assert res["chart"]["type"] in ("bar", "hbar")


def test_top_n_limit_and_other_bucket(retail):
    res = run_query(retail, {"kind": "aggregate", "dimension": "product", "measure": "revenue",
                             "agg": "sum", "limit": 5, "other_bucket": True})
    df = retail.df.assign(revenue=pd.to_numeric(retail.df["revenue"], errors="coerce"))
    top5 = df.groupby("product")["revenue"].sum().sort_values(ascending=False).head(5)
    head = res["rows"][:5]
    assert [r["label"] for r in head] == list(top5.index)
    other = res["rows"][-1]
    assert other["label"].startswith("Other")
    assert sum(r["Total revenue"] for r in res["rows"]) == pytest.approx(float(df["revenue"].sum()))


def test_average_aggregation(retail):
    res = run_query(retail, {"kind": "aggregate", "dimension": "channel", "measure": "profit",
                             "agg": "avg"})
    df = retail.df.assign(profit=pd.to_numeric(retail.df["profit"], errors="coerce"))
    exp = df.groupby("channel")["profit"].mean()
    for r in res["rows"]:
        assert r["Average profit"] == pytest.approx(float(exp[r["label"]]))


def test_filter_eq(retail):
    res = run_query(retail, {"kind": "aggregate", "measure": "revenue", "agg": "sum",
                             "filters": [{"col": "region", "op": "eq", "value": "North"}]})
    df = retail.df.assign(revenue=pd.to_numeric(retail.df["revenue"], errors="coerce"))
    expected = float(df.loc[df["region"] == "North", "revenue"].sum())
    assert res["rows"][0]["Total revenue"] == pytest.approx(expected)
    assert res["filtered_out"] == int((retail.df["region"] != "North").sum())


def test_filter_numeric_gt(retail):
    res = run_query(retail, {"kind": "table", "limit": 5,
                             "filters": [{"col": "revenue", "op": "gt", "value": 50000}]})
    df = retail.df
    manual = int((pd.to_numeric(df["revenue"], errors="coerce") > 50000).sum())
    assert res["totals"]["count"] == manual
    assert all(float(r["revenue"]) > 50000 for r in res["rows"])


def test_timeseries_matches_pandas(retail):
    res = run_query(retail, {"kind": "timeseries", "time_dimension": "order_date",
                             "measure": "revenue", "agg": "sum", "granularity": "month"})
    df = retail.df.assign(revenue=pd.to_numeric(retail.df["revenue"], errors="coerce")).copy()
    df["m"] = pd.to_datetime(df["order_date"]).dt.to_period("M").dt.start_time
    exp = df.groupby("m")["revenue"].sum()
    assert len(res["rows"]) == len(exp)
    for row, (k, v) in zip(res["rows"], exp.items()):
        assert pd.Timestamp(row["label"]) == k
        assert row["Total revenue"] == pytest.approx(float(v))
    assert res["trend"] is not None
    assert res["chart"]["type"] in ("area", "line")
    assert len(res["chart"]["annotations"][0]["values"]) == len(res["rows"])


def test_correlation_matches_numpy(marketing):
    res = run_query(marketing, {"kind": "correlation", "x": "spend", "y": "conversions"})
    a = pd.to_numeric(marketing.df["spend"], errors="coerce")
    b = pd.to_numeric(marketing.df["conversions"], errors="coerce")
    both = a.notna() & b.notna()
    r = float(np.corrcoef(a[both], b[both])[0, 1])
    assert res["correlation"]["pearson"] == pytest.approx(r, abs=1e-9)
    assert res["correlation"]["n"] == int(both.sum())


def test_distribution_histogram(tickets):
    res = run_query(tickets, {"kind": "distribution", "column": "resolution_hours"})
    v = pd.to_numeric(tickets.df["resolution_hours"], errors="coerce").dropna()
    assert sum(r["count"] for r in res["rows"]) == int(len(v))
    assert res["stats"]["median"] == pytest.approx(float(v.median()))
    assert res["chart"]["type"] == "histogram"


def test_segments_baseline(retail):
    res = run_query(retail, {"kind": "segments", "dimension": "segment", "measure": "revenue",
                             "agg": "avg"})
    v = pd.to_numeric(retail.df["revenue"], errors="coerce")
    assert res["baseline"] == pytest.approx(float(v.mean()))
    for r in res["rows"]:
        assert r["index_vs_baseline"] == pytest.approx(r["Average revenue"] / float(v.mean()))


def test_matrix(marketing):
    res = run_query(marketing, {"kind": "matrix"})
    assert res["chart"]["type"] == "heatmap"
    n = len(res["chart"]["x"]["values"])
    assert len(res["chart"]["matrix"]) == n
    assert res["rows"][0]["abs_r"] == max(r["abs_r"] for r in res["rows"])


def test_unknown_column_is_safe(retail):
    res = run_query(retail, {"kind": "aggregate", "dimension": "nope", "measure": "revenue",
                             "agg": "sum"})
    assert res["rows"][0]["label"] == "All rows"


# --------------------------------------------------------------- NLQ
@pytest.mark.parametrize("q,expect", [
    ("total revenue by region", {"measure": "revenue", "agg": "sum", "dimension": "region"}),
    ("What is the average profit by channel?", {"measure": "profit", "agg": "avg", "dimension": "channel"}),
    ("top 5 products by revenue", {"measure": "revenue", "dimension": "product"}),
    ("How many orders per city?", {"agg": "count", "dimension": "city"}),
    ("monthly trend of revenue", {"measure": "revenue", "granularity": "month"}),
    ("total revenue for North", {"measure": "revenue"}),
    ("revenue in 2024", {"measure": "revenue"}),
    ("distribution of unit_price", {"measure": None}),
    ("Correlation between units and revenue", {"measure": None}),
])
def test_nlq_slots(retail, q, expect):
    parsed = nlq.parse(retail, q)
    slots = parsed["slots"]
    for k, v in expect.items():
        assert slots.get(k) == v, f"{q}: {k} = {slots.get(k)}"


def test_nlq_filter_value_detected(retail):
    parsed = nlq.parse(retail, "total revenue for North region")
    assert {"col": "region", "op": "eq", "value": "North"} in parsed["spec"]["filters"]
    res = run_query(retail, parsed["spec"])
    df = retail.df.assign(revenue=pd.to_numeric(retail.df["revenue"], errors="coerce"))
    assert res["rows"][0]["Total revenue"] == pytest.approx(float(df.loc[df["region"] == "North", "revenue"].sum()))


def test_nlq_year_filter(retail):
    parsed = nlq.parse(retail, "revenue in 2024")
    assert any(f.get("op") == "year" for f in parsed["spec"]["filters"])
    res = run_query(retail, parsed["spec"])
    df = retail.df.assign(revenue=pd.to_numeric(retail.df["revenue"], errors="coerce"),
                          y=pd.to_datetime(retail.df["order_date"]).dt.year)
    assert res["rows"][0]["Total revenue"] == pytest.approx(float(df.loc[df["y"] == 2024, "revenue"].sum()))


def test_nlq_top_n_limit(retail):
    parsed = nlq.parse(retail, "top 3 categories by revenue")
    assert parsed["spec"]["limit"] == 3
    res = run_query(retail, parsed["spec"])
    assert len(res["rows"]) <= 4


def test_nlq_answer_contains_true_number(retail):
    out = nlq.answer_question(retail, "total revenue by region")
    df = retail.df.assign(revenue=pd.to_numeric(retail.df["revenue"], errors="coerce"))
    top = df.groupby("region")["revenue"].sum().idxmax()
    assert top in out["answer"]
    assert out["result"]["chart"] is not None
    assert out["follow_ups"]


def test_nlq_graceful_on_nonsense(retail):
    out = nlq.answer_question(retail, "please tell me about the weather in paris")
    assert out["answer"]
    assert out["parsed"]["confidence"] < 0.5


def test_example_questions_answer(retail):
    for q in nlq.example_questions(retail):
        out = nlq.answer_question(retail, q)
        assert out["answer"], q
        assert not out["result"].get("error"), (q, out["result"].get("error"))


# --------------------------------------------------------------- insights
def test_insights_find_known_flaws(tickets):
    found = generate_insights(tickets)
    kinds = {i["type"] for i in found["insights"]}
    assert "duplicate_rows" in kinds                      # the file has 18 exact dups
    assert any(i["type"] == "missing_data" for i in found["insights"])
    for i in found["insights"]:
        assert i["summary"] and i["severity"] in ("critical", "warning", "info", "positive")


def test_insights_duplicate_count(tickets):
    found = generate_insights(tickets)
    dup = next(i for i in found["insights"] if i["type"] == "duplicate_rows")
    manual = int(tickets.df.duplicated().sum())
    assert f"{manual:,}" in dup["summary"]
    assert manual == 18


def test_insights_outliers_on_bulk_orders(retail):
    found = generate_insights(retail, limit=25)
    outs = [i for i in found["insights"] if i["type"] == "outliers"]
    assert outs, "bulk orders should be flagged"
    assert any("revenue" in i["columns"] or "units" in i["columns"] for i in outs)


def test_insights_trend_direction(retail):
    found = generate_insights(retail, limit=25)
    trend = [i for i in found["insights"] if i["type"] == "trend" and "revenue" in i["columns"]]
    assert trend and "rising" in trend[0]["title"].lower()   # generator adds growth


def test_dashboard_kpis(retail):
    dash = build_dashboard(retail)
    assert dash["kpis"][0]["value"] == retail.rows
    assert dash["charts"], "expected auto-generated charts"
    for ch in dash["charts"]:
        assert ch["chart"]["x"]["values"]
        if ch["chart"]["type"] == "heatmap":
            assert len(ch["chart"]["matrix"]) == len(ch["chart"]["x"]["values"])
        else:
            assert ch["chart"]["y"]["series"][0]["values"]


def test_all_insights_render_specs(retail):
    found = generate_insights(retail, limit=25)
    for i in found["insights"]:
        if i.get("chart"):
            res = run_query(retail, i["chart"]["spec"])
            assert res.get("chart"), i["id"]


# --------------------------------------------------------------- LLM layer
def test_validate_spec_rejects_unknown_column(retail):
    with pytest.raises(llm.LLMError):
        llm.validate_spec(retail, {"kind": "aggregate", "measure": "nope", "agg": "sum"})


def test_validate_spec_normalises(retail):
    spec = llm.validate_spec(retail, {"kind": "aggregate", "measure": "revenue", "agg": "TOTAL",
                                      "dimension": "region", "limit": 5,
                                      "filters": [{"col": "region", "op": "eq", "value": "North"},
                                                  {"col": "ghost", "op": "eq", "value": 1}]})
    assert spec["agg"] == "sum"
    assert len(spec["filters"]) == 1
    res = run_query(retail, spec)
    assert res["rows"]


def test_upload_parser_handles_semicolons():
    df = read_csv_bytes(b"a;b;c\n1;2;3\n4;5;6\n")
    assert list(df.columns) == ["a", "b", "c"]
    assert len(df) == 2


# ------------------------------------------------------- regression guards
def test_dashboard_dimension_charts_use_real_columns(retail):
    """A profile dict was once passed where a column name belongs, which
    silently dropped every 'by dimension' chart."""
    dash = build_dashboard(retail)
    dim_charts = [c for c in dash["charts"] if c["spec"].get("dimension")]
    assert len(dim_charts) >= 2
    for c in dim_charts:
        assert c["spec"]["dimension"] in retail.df.columns
        assert c["spec"]["dimension"] in c["title"]


def test_kpis_do_not_sum_ratings(tickets):
    dash = build_dashboard(tickets)
    labels = [k["label"] for k in dash["kpis"]]
    assert "Average csat" in labels
    assert not any(l.startswith("Total csat") for l in labels)


def test_example_questions_are_answerable_and_sane(retail, marketing, tickets):
    for ds in (retail, marketing, tickets):
        qs = nlq.example_questions(ds)
        assert qs, "no suggestions"
        for q in qs:
            out = nlq.answer_question(ds, q)
            assert out["answer"], q
            assert not out["result"].get("error"), (q, out["result"].get("error"))


# ------------------------------------------------------------- HTTP API
from fastapi.testclient import TestClient  # noqa: E402
import server  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(server.app)


def test_api_upload_and_ask(client):
    csv = ("order_id,region,revenue,order_date\n"
           "1,North,100,2024-01-05\n2,South,250,2024-01-06\n3,North,400,2024-02-01\n"
           "4,East,50,2024-02-02\n5,South,600,2024-03-03\n")
    r = client.post("/api/upload", files={"file": ("tiny.csv", csv, "text/csv")})
    assert r.status_code == 200, r.text
    did = r.json()["dataset"]["id"]
    assert r.json()["dataset"]["rows"] == 5
    assert {c["name"]: c["type"] for c in r.json()["profile"]}["revenue"] == "numeric"

    a = client.post(f"/api/datasets/{did}/ask", json={"question": "total revenue by region"})
    assert a.status_code == 200, a.text
    body = a.json()
    assert body["result"]["rows"][0]["label"] == "South"
    assert body["result"]["rows"][0]["Total revenue"] == pytest.approx(850.0)

    t = client.get(f"/api/datasets/{did}/table", params={"search": "North"})
    assert t.status_code == 200 and t.json()["total"] == 2

    assert client.get("/api/datasets/nope").status_code == 404
    assert client.post(f"/api/datasets/{did}/ask", json={"question": ""}).status_code == 400
    client.delete(f"/api/datasets/{did}")
    assert client.get(f"/api/datasets/{did}").status_code == 404


def test_api_rejects_garbage_upload(client):
    r = client.post("/api/upload", files={"file": ("empty.csv", b"", "text/csv")})
    assert r.status_code == 400


def test_static_assets_served(client):
    for path in ("/", "/static/app.js", "/static/charts.js", "/static/styles.css", "/favicon.svg"):
        assert client.get(path).status_code == 200, path


# ------------------------------------------------- fixes found via the browser
def test_clean_dataset_still_reports(retail):
    """missing_cell_pct == 0.0 is falsy; spotless data used to report nothing."""
    import io
    df = pd.read_csv(io.StringIO(
        "id,team,score\n" + "\n".join(f"{i},{'alpha' if i % 2 else 'beta'},{i * 7 % 53}"
                                      for i in range(60))), dtype=str)
    from engine.loader import Dataset as DS
    ds = DS("t-clean", "clean", df)
    build_profile(ds)
    assert ds.profile["missing_cell_pct"] == 0.0
    found = generate_insights(ds)
    assert found["total_found"] >= 1
    assert any(i["type"] == "data_quality" for i in found["insights"])


@pytest.mark.parametrize("q,kind", [
    ("Plot cost against revenue", "correlation"),
    ("revenue vs profit", "correlation"),
    ("correlation between units and revenue", "correlation"),
])
def test_correlation_phrasings(retail, q, kind):
    assert nlq.parse(retail, q)["spec"]["kind"] == kind


def test_scatter_wording_on_marketing(marketing):
    assert nlq.parse(marketing, "scatter spend and conversions")["spec"]["kind"] == "correlation"


def test_compare_two_values_merges_filters(retail):
    """'North vs South' must not become two contradictory eq filters."""
    parsed = nlq.parse(retail, "compare North vs South")
    fs = parsed["spec"]["filters"]
    assert len(fs) == 1 and fs[0]["op"] == "in"
    assert sorted(fs[0]["value"]) == ["North", "South"]
    assert parsed["spec"]["dimension"] == "region"
    res = run_query(retail, parsed["spec"])
    assert len(res["rows"]) == 2


def test_comparison_filter_on_text_column_is_not_dropped(retail):
    """A > filter on a non-numeric column used to be silently ignored."""
    res = run_query(retail, {"kind": "table", "limit": 5,
                             "filters": [{"col": "order_id", "op": "gt", "value": "ORD-102390"}]})
    assert res["totals"]["count"] < retail.rows
    assert all(r["order_id"] > "ORD-102390" for r in res["rows"])


def test_boolean_insight_reports_true_share(tickets):
    found = generate_insights(tickets, limit=30)
    b = [i for i in found["insights"] if i["type"] == "boolean_split"]
    assert b and "11%" in b[0]["title"].replace(" ", "")
