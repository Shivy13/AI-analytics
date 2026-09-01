"""AI Analytics Dashboard - FastAPI backend.

Serves the SPA and the analysis API. Every endpoint is thin: the work
happens in engine/ so it can be tested without a server.
"""
from __future__ import annotations

import os

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import llm, nlq, samples
from engine.insights import build_dashboard, generate_insights
from engine.loader import DATA_DIR, STORE, read_csv_bytes
from engine.profile import build_profile
from engine.query import run_query
from engine.utils import js

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")
MAX_UPLOAD_BYTES = 40 * 1024 * 1024

app = FastAPI(title="AI Analytics Dashboard", version="1.0")


def _ds(dataset_id: str):
    ds = STORE.get(dataset_id)
    if ds is None:
        raise HTTPException(404, f"Unknown dataset '{dataset_id}'")
    if ds.profile is None:
        build_profile(ds)
    return ds


def _report(ds) -> dict:
    if ds.insights is None:
        found = generate_insights(ds)
        dash = build_dashboard(ds)
        narrative = None
        if llm.available():
            try:
                narrative = llm.narrative(ds, found["insights"], dash["kpis"])
            except Exception as e:
                narrative = f"(LLM narrative unavailable: {e})"
        ds.insights = {"insights": js(found["insights"]),
                       "total_found": found["total_found"],
                       "severity_counts": js(found["severity_counts"]),
                       "dashboard": js(dash), "narrative": narrative,
                       "profile_summary": js({
                           "rows": ds.profile["rows"], "columns": ds.profile["columns"],
                           "duplicate_rows": ds.profile["duplicate_rows"],
                           "missing_cell_pct": ds.profile["missing_cell_pct"]}),
                       "examples": js(nlq.example_questions(ds)),
                       "engine": "llm+local" if narrative else "local"}
    return ds.insights


# ------------------------------------------------------------------
# datasets
# ------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"ok": True, "datasets": len(STORE.list()), "llm": llm.masked()}


@app.get("/api/samples")
def list_samples():
    return {"samples": samples.sample_meta()}


@app.post("/api/samples/{key}/load")
def load_sample(key: str):
    if key not in samples.SAMPLES:
        raise HTTPException(404, "Unknown sample")
    path = samples.ensure_samples()[key]
    df = pd.read_csv(path, dtype=str, keep_default_na=True,
                     na_values=["", "NA", "N/A", "null", "None"], low_memory=False)
    ds = STORE.add(samples.SAMPLES[key]["label"] + ".csv", df)
    _ds(ds.id)
    return {"dataset": ds.meta(), "profile": js(ds.profile["columns_detail"]),
            "report": _report(ds)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File larger than 40 MB")
    try:
        df = read_csv_bytes(raw, file.filename or "upload.csv")
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")
    if df.empty or len(df.columns) < 1:
        raise HTTPException(400, "No columns found in that file")
    ds = STORE.add(file.filename or "upload.csv", df)
    _ds(ds.id)
    return {"dataset": ds.meta(), "profile": js(ds.profile["columns_detail"]), "report": _report(ds)}


class PasteIn(BaseModel):
    csv: str
    name: str = "pasted.csv"


@app.post("/api/paste")
def paste(body: PasteIn):
    if not body.csv.strip():
        raise HTTPException(400, "Nothing pasted")
    try:
        df = read_csv_bytes(body.csv.encode("utf-8"), body.name)
    except Exception as e:
        raise HTTPException(400, f"Could not parse that text: {e}")
    ds = STORE.add(body.name, df)
    _ds(ds.id)
    return {"dataset": ds.meta(), "profile": js(ds.profile["columns_detail"]), "report": _report(ds)}


@app.get("/api/datasets")
def datasets():
    return {"datasets": STORE.list()}


@app.get("/api/datasets/{dataset_id}")
def dataset(dataset_id: str):
    ds = _ds(dataset_id)
    return {"dataset": ds.meta(), "profile": js(ds.profile["columns_detail"]), "report": _report(ds)}


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    if not STORE.remove(dataset_id):
        raise HTTPException(404, "Unknown dataset")
    return {"ok": True}


@app.get("/api/datasets/{dataset_id}/download.csv")
def download(dataset_id: str):
    path = os.path.join(DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(path):
        raise HTTPException(404, "Unknown dataset")
    return FileResponse(path, filename=f"{dataset_id}.csv", media_type="text/csv")


@app.get("/api/datasets/{dataset_id}/table")
def table(dataset_id: str, limit: int = 50, offset: int = 0, search: str | None = None,
          sort_col: str | None = None, sort_dir: str = "desc"):
    ds = _ds(dataset_id)
    view = ds.df
    if search:
        mask = pd.Series(False, index=view.index)
        for c in view.columns:
            mask |= view[c].astype("string").str.contains(search, case=False, na=False, regex=False)
        view = view.loc[mask]
    total = int(len(view))
    if sort_col and sort_col in view.columns:
        s = ds.series(sort_col).loc[view.index]
        order = s.sort_values(ascending=(sort_dir == "asc"), na_position="last").index
        view = view.loc[order]
    page = view.iloc[max(0, offset):max(0, offset) + max(1, min(500, limit))]
    rows = [{c: js(v) for c, v in rec.items()} for rec in page.to_dict("records")]
    columns = [{"key": c, "label": c, "type": ds.types.get(c, "text")} for c in page.columns]
    return {"columns": columns, "rows": rows, "total": total, "offset": offset, "limit": len(rows)}


# ------------------------------------------------------------------
# analysis
# ------------------------------------------------------------------
@app.post("/api/datasets/{dataset_id}/analyze")
def analyze(dataset_id: str, refresh: bool = False):
    ds = _ds(dataset_id)
    if refresh:
        ds.insights = None
        ds.profile = None
        build_profile(ds)
    return _report(ds)


class QueryIn(BaseModel):
    spec: dict


@app.post("/api/datasets/{dataset_id}/query")
def query(dataset_id: str, body: QueryIn):
    ds = _ds(dataset_id)
    try:
        return js(run_query(ds, body.spec))
    except Exception as e:
        raise HTTPException(400, f"Query failed: {e}")


class AskIn(BaseModel):
    question: str
    use_llm: bool = True


@app.post("/api/datasets/{dataset_id}/ask")
def ask(dataset_id: str, body: AskIn):
    ds = _ds(dataset_id)
    q = (body.question or "").strip()
    if not q:
        raise HTTPException(400, "Empty question")
    used_llm, llm_error, parsed = False, None, None
    if body.use_llm and llm.available():
        try:
            spec = llm.nl_to_spec(ds, q)
            used_llm = True
            parsed = {"intent": spec.get("kind", "aggregate"), "spec": spec, "confidence": 0.9,
                      "notes": ["translated by the connected model"],
                      "matched_columns": [spec.get(k) for k in ("measure", "dimension", "x", "y") if spec.get(k)],
                      "alternates": [], "slots": {"measure": spec.get("measure"),
                                                  "agg": spec.get("agg"), "dimension": spec.get("dimension"),
                                                  "filters": spec.get("filters", [])}}
        except Exception as e:
            llm_error = str(e)
    if parsed is None:
        parsed = nlq.parse(ds, q)
    result = run_query(ds, parsed["spec"])
    answer = nlq.compose_answer(ds, q, parsed, result)
    return {"question": q, "answer": answer, "parsed": js(parsed), "result": js(result),
            "follow_ups": js(nlq.follow_ups(ds, parsed)),
            "engine": "llm" if used_llm else "local", "llm_error": llm_error}


@app.post("/api/datasets/{dataset_id}/export")
def export(dataset_id: str, body: QueryIn):
    ds = _ds(dataset_id)
    res = run_query(ds, body.spec)
    rows = res.get("rows") or []
    cols = [c["key"] for c in res.get("columns", [])]
    if not rows:
        return PlainTextResponse("(no rows)")
    if isinstance(rows[0], dict):
        df = pd.DataFrame(rows)[[c for c in cols if c in rows[0]]]
    else:
        df = pd.DataFrame(rows, columns=cols)
    csv = df.to_csv(index=False)
    md = "| " + " | ".join(df.columns) + " |\n|" + "|".join(["---"] * len(df.columns)) + "|\n"
    md += "\n".join("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
                    for r in df.head(200).values)
    return JSONResponse({"csv": csv, "markdown": md, "rows": len(df)})


# ------------------------------------------------------------------
# LLM settings
# ------------------------------------------------------------------
class LLMSettings(BaseModel):
    enabled: bool | None = None
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None


@app.get("/api/llm/settings")
def llm_settings():
    return llm.masked()


@app.post("/api/llm/settings")
def llm_settings_save(body: LLMSettings):
    llm.save_settings(body.model_dump(exclude_none=True))
    return llm.masked()


@app.post("/api/llm/test")
def llm_test():
    return llm.test_connection()


# ------------------------------------------------------------------
# static
# ------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/favicon.svg")
def favicon():
    return FileResponse(os.path.join(STATIC, "favicon.svg"))
