"""AI Analytics Dashboard - FastAPI backend.

Serves the SPA and the analysis API. Every endpoint is thin: the work
happens in engine/ so it can be tested without a server.
"""
from __future__ import annotations

import os
import secrets
import time as _time
from collections import defaultdict, deque

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import db, llm, mailer, nlq, samples

try:
    import stripe
except Exception:
    stripe = None
from engine.insights import build_dashboard, generate_insights
from engine.loader import DATA_DIR, STORE, read_csv_bytes
from engine.profile import build_profile
from engine.query import run_query
from engine.utils import js

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
SESSION_COOKIE = "aa_session"
GUEST_COOKIE = "aa_guest"

app = FastAPI(title="AI Analytics Dashboard", version="1.0")


PRO_PRICE_USD = 29

# Per-plan rate limiting (asks per minute). Free/anonymous users are throttled
# harder than Pro. In-memory sliding window; 429 when exceeded.
RATE_PER_MIN = {"free": 30, "pro": 120}
RATE_WINDOW = 60.0
_RATE = defaultdict(deque)


def _rate_limit(request: Request) -> None:
    plan = request.state.plan
    limit = RATE_PER_MIN.get(plan, RATE_PER_MIN["free"])
    now = _time.time()
    dq = _RATE[request.state.identity]
    while dq and now - dq[0] > RATE_WINDOW:
        dq.popleft()
    if len(dq) >= limit:
        raise HTTPException(
            429, f"Rate limit exceeded ({limit} asks/min on {plan}). "
                 f"Upgrade to Pro for more.")
    dq.append(now)


def _notify_upgrade(uid: str) -> None:
    email = db.email_for(uid)
    if email:
        mailer.send_receipt(email, "pro", PRO_PRICE_USD)


def stripe_ready() -> bool:
    return stripe is not None and bool(os.environ.get("STRIPE_SECRET_KEY"))


@app.middleware("http")
async def identity_middleware(request: Request, call_next):
    token = request.cookies.get(SESSION_COOKIE)
    auth = request.headers.get("authorization", "")
    if not token and auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    user = db.session_user(token)
    if user:
        request.state.identity, request.state.user = user["id"], user
        request.state.plan = db.get_plan(user["id"])
        request.state.is_guest = False
    else:
        gid = request.cookies.get(GUEST_COOKIE) or ("g_" + secrets.token_hex(8))
        request.state.identity, request.state.user = gid, None
        request.state.plan, request.state.is_guest = "free", True
        request.state.guest_token = gid
    response = await call_next(request)
    if getattr(request.state, "is_guest", False) and GUEST_COOKIE not in request.cookies:
        response.set_cookie(GUEST_COOKIE, request.state.guest_token,
                            max_age=60 * 60 * 24 * 365, httponly=True, samesite="lax")
    return response


def _quota_exceeded(request: Request, field: str, need: int = 1) -> bool:
    lim = db.limits_for(request.state.plan)
    return db.usage_for(request.state.identity).get(field, 0) + need > lim[field]


def _require(request: Request, field: str, need: int = 1):
    if _quota_exceeded(request, field, need):
        lim = db.limits_for(request.state.plan)
        raise HTTPException(402, f"Free plan limit reached: {lim[field]:,} {field}/month. "
                                 f"Upgrade to Pro for more.")


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
def load_sample(request: Request, key: str):
    if key not in samples.SAMPLES:
        raise HTTPException(404, "Unknown sample")
    path = samples.ensure_samples()[key]
    _require(request, "datasets")
    df = pd.read_csv(path, dtype=str, keep_default_na=True,
                     na_values=["", "NA", "N/A", "null", "None"], low_memory=False)
    ds = STORE.add(samples.SAMPLES[key]["label"] + ".csv", df, owner=request.state.identity)
    db.bump(request.state.identity, "datasets")
    _ds(ds.id)
    return {"dataset": ds.meta(), "profile": js(ds.profile["columns_detail"]),
            "report": _report(ds)}


@app.post("/api/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    lim = db.limits_for(request.state.plan)
    if len(raw) > min(MAX_UPLOAD_BYTES, lim["max_file_mb"] * 1024 * 1024):
        raise HTTPException(413, f"File exceeds your plan limit of {lim['max_file_mb']} MB.")
    _require(request, "datasets")
    try:
        df = read_csv_bytes(raw, file.filename or "upload.csv")
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")
    if df.empty or len(df.columns) < 1:
        raise HTTPException(400, "No columns found in that file")
    if len(df) > lim["max_rows"]:
        raise HTTPException(402, f"Free plan supports up to {lim['max_rows']:,} rows; this file has "
                                 f"{len(df):,}. Upgrade to Pro.")
    ds = STORE.add(file.filename or "upload.csv", df, owner=request.state.identity)
    db.bump(request.state.identity, "datasets")
    _ds(ds.id)
    return {"dataset": ds.meta(), "profile": js(ds.profile["columns_detail"]), "report": _report(ds)}


class PasteIn(BaseModel):
    csv: str
    name: str = "pasted.csv"


@app.post("/api/paste")
def paste(request: Request, body: PasteIn):
    if not body.csv.strip():
        raise HTTPException(400, "Nothing pasted")
    _require(request, "datasets")
    try:
        df = read_csv_bytes(body.csv.encode("utf-8"), body.name)
    except Exception as e:
        raise HTTPException(400, f"Could not parse that text: {e}")
    if len(df) > db.limits_for(request.state.plan)["max_rows"]:
        raise HTTPException(402, "Free plan row limit exceeded. Upgrade to Pro.")
    ds = STORE.add(body.name, df, owner=request.state.identity)
    db.bump(request.state.identity, "datasets")
    _ds(ds.id)
    return {"dataset": ds.meta(), "profile": js(ds.profile["columns_detail"]), "report": _report(ds)}


@app.get("/api/datasets")
def datasets(request: Request):
    return {"datasets": STORE.list(owner=request.state.identity)}


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
def ask(request: Request, dataset_id: str, body: AskIn):
    ds = _ds(dataset_id)
    q = (body.question or "").strip()
    if not q:
        raise HTTPException(400, "Empty question")
    _rate_limit(request)
    _require(request, "questions")
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
    db.bump(request.state.identity, "questions")
    return {"question": q, "answer": answer, "parsed": js(parsed), "result": js(result),
            "follow_ups": js(nlq.follow_ups(ds, parsed)),
            "engine": "llm" if used_llm else "local", "llm_error": llm_error}


@app.post("/api/datasets/{dataset_id}/export")
def export(request: Request, dataset_id: str, body: QueryIn):
    ds = _ds(dataset_id)
    _require(request, "exports")
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
    db.bump(request.state.identity, "exports")
    return JSONResponse({"csv": csv, "markdown": md, "rows": len(df)})


# ------------------------------------------------------------------
# auth + billing
# ------------------------------------------------------------------
class AuthIn(BaseModel):
    email: str
    password: str


@app.post("/api/auth/signup")
def signup(body: AuthIn, response: Response):
    try:
        user = db.create_user(body.email, body.password)
    except ValueError as e:
        raise HTTPException(400, str(e))
    response.set_cookie(SESSION_COOKIE, db.new_session(user["id"]),
                        max_age=db.SESSION_DAYS * 86400, httponly=True, samesite="lax")
    return {"email": user["email"], "plan": "free"}


@app.post("/api/auth/login")
def login(body: AuthIn, response: Response):
    user = db.authenticate(body.email, body.password)
    if not user:
        raise HTTPException(401, "Invalid email or password.")
    response.set_cookie(SESSION_COOKIE, db.new_session(user["id"]),
                        max_age=db.SESSION_DAYS * 86400, httponly=True, samesite="lax")
    return {"email": user["email"], "plan": db.get_plan(user["id"])}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    tok = request.cookies.get(SESSION_COOKIE)
    if tok:
        db.end_session(tok)
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


@app.get("/api/auth/me")
def me(request: Request):
    return {"email": request.state.user["email"] if request.state.user else None,
            "guest": request.state.is_guest, "plan": request.state.plan,
            "is_admin": (not request.state.is_guest) and db.is_admin(request.state.identity),
            "usage": db.usage_for(request.state.identity),
            "limits": db.limits_for(request.state.plan)}


@app.get("/api/billing")
def billing(request: Request):
    return {"plan": request.state.plan, "guest": request.state.is_guest,
            "usage": db.usage_for(request.state.identity),
            "limits": db.limits_for(request.state.plan), "stripe": stripe_ready(),
            "smtp": mailer.smtp_ready(), "rate_per_min": RATE_PER_MIN.get(request.state.plan)}


@app.post("/api/billing/checkout")
def checkout(request: Request):
    if request.state.is_guest:
        raise HTTPException(401, "Create an account first.")
    if not stripe_ready():
        # No live keys: the UI runs the upgrade as a local test-mode dry run.
        return {"mode": "dryrun", "url": None, "plan": "pro", "price": PRO_PRICE_USD}
    base = str(request.base_url).rstrip("/")
    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": os.environ.get("STRIPE_PRICE_ID"), "quantity": 1}],
        client_reference_id=request.state.identity,
        success_url=base + "/#upgraded", cancel_url=base + "/#cancelled")
    return {"mode": "stripe", "url": session.url, "plan": "pro"}


def _apply_event(event: dict) -> None:
    """Apply a Stripe event. Shared by the real webhook and the test-mode dry run."""
    t, obj = event.get("type"), event.get("data", {}).get("object", {})
    if t == "checkout.session.completed":
        uid = obj.get("client_reference_id")
        if uid:
            db.set_plan(uid, "pro", obj.get("customer"), obj.get("subscription"))
            _notify_upgrade(uid)
    elif t == "customer.subscription.deleted":
        db.set_plan_by_sub(obj.get("id"), "free")


@app.post("/api/billing/webhook")
async def billing_webhook(request: Request):
    raw = await request.body()
    if stripe_ready() and os.environ.get("STRIPE_WEBHOOK_SECRET"):
        try:
            event = stripe.Webhook.construct_event(
                raw, request.headers.get("stripe-signature", ""),
                os.environ["STRIPE_WEBHOOK_SECRET"])
        except Exception as e:
            raise HTTPException(400, f"Bad webhook signature: {e}")
    else:
        import json as _json
        event = _json.loads(raw)
    _apply_event(event)
    return {"ok": True}


@app.post("/api/billing/dryrun")
def billing_dryrun(request: Request):
    """Test-mode dry run: simulate a completed Stripe Checkout for this user,
    through the exact same event handler the live webhook uses."""
    if request.state.is_guest:
        raise HTTPException(401, "Sign in first.")
    _apply_event({"type": "checkout.session.completed",
                  "data": {"object": {"client_reference_id": request.state.identity,
                                      "customer": "cus_dryrun", "subscription": "sub_dryrun"}}})
    return {"plan": db.get_plan(request.state.identity), "simulated": True, "price": PRO_PRICE_USD}


@app.get("/api/admin/usage")
def admin_usage_api(request: Request):
    if request.state.is_guest or not db.is_admin(request.state.identity):
        raise HTTPException(403, "Admins only.")
    accounts = db.admin_usage()
    return {"month": db._period(), "accounts": accounts,
            "totals": {"accounts": len(accounts),
                       "pro": sum(1 for a in accounts if a["plan"] == "pro"),
                       "questions": sum(a["questions"] for a in accounts),
                       "exports": sum(a["exports"] for a in accounts)}}


@app.post("/api/billing/portal")
def billing_portal(request: Request):
    """Stripe customer portal (manage / cancel). Falls back to demo mode offline."""
    if request.state.is_guest:
        raise HTTPException(401, "Sign in first.")
    customer, _ = db.billing_for(request.state.identity)
    if stripe_ready() and customer:
        sess = stripe.billing_portal.Session.create(
            customer=customer, return_url=str(request.base_url).rstrip("/") + "/")
        return {"mode": "stripe", "url": sess.url}
    return {"mode": "demo"}


@app.post("/api/billing/cancel")
def billing_cancel(request: Request):
    """Demo cancel (no live Stripe). With live Stripe, users cancel via the portal."""
    if request.state.is_guest:
        raise HTTPException(401, "Sign in first.")
    if stripe_ready():
        raise HTTPException(400, "With live Stripe, cancel from the billing portal instead.")
    db.set_plan(request.state.identity, "free")
    return {"plan": "free"}


class GrantIn(BaseModel):
    plan: str


@app.post("/api/billing/grant")
def grant(request: Request, body: GrantIn):
    if request.state.is_guest:
        raise HTTPException(401, "Sign in first.")
    if stripe_ready():
        raise HTTPException(400, "Manual grants are disabled while Stripe is live.")
    if body.plan not in ("free", "pro"):
        raise HTTPException(400, "plan must be 'free' or 'pro'")
    db.set_plan(request.state.identity, body.plan)
    if body.plan == "pro":
        _notify_upgrade(request.state.identity)
    return {"plan": body.plan}


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
