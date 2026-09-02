"""Tests for accounts, plans, quotas and Stripe webhook handling."""
import uuid
import pytest
from fastapi.testclient import TestClient

from engine import db
import server

TINY = b"a,region,v\n1,North,10\n2,South,20\n3,North,30\n"


def em():
    return f"{uuid.uuid4().hex[:10]}@test.local"


@pytest.fixture()
def client():
    return TestClient(server.app)


def _bearer(uid):
    return {"Authorization": "Bearer " + db.new_session(uid)}


def test_signup_login_me_logout(client):
    email = em()
    r = client.post("/api/auth/signup", json={"email": email, "password": "supersecret1"})
    assert r.status_code == 200, r.text
    me = client.get("/api/auth/me").json()
    assert me["email"] == email and me["plan"] == "free" and not me["guest"]

    # wrong password rejected
    bad = TestClient(server.app).post("/api/auth/login",
                                      json={"email": email, "password": "wrongpass1"})
    assert bad.status_code == 401

    # duplicate email rejected
    dup = client.post("/api/auth/signup", json={"email": email, "password": "supersecret1"})
    assert dup.status_code == 400

    # weak password rejected
    weak = client.post("/api/auth/signup", json={"email": "w@b.com", "password": "short"})
    assert weak.status_code == 400

    client.post("/api/auth/logout")
    me2 = client.get("/api/auth/me").json()
    assert me2["guest"] is True


def test_guest_is_anonymous_and_metered(client):
    me = client.get("/api/auth/me").json()
    assert me["guest"] is True and me["plan"] == "free"
    b = client.get("/api/billing").json()
    assert b["limits"]["questions"] == db.FREE["questions"]


def test_question_quota_enforced_and_upgraded(client):
    up = client.post("/api/upload", files={"file": ("t.csv", TINY, "text/csv")})
    did = up.json()["dataset"]["id"]
    uid = db.create_user(em(), "supersecret1")["id"]
    h = _bearer(uid)

    # exhaust the free question quota, then expect 402
    db.bump(uid, "questions", db.FREE["questions"])
    r = client.post(f"/api/datasets/{did}/ask", json={"question": "how many rows"}, headers=h)
    assert r.status_code == 402

    # upgrade via webhook (Stripe not configured -> plain JSON), then ask succeeds
    wh = client.post("/api/billing/webhook", json={
        "type": "checkout.session.completed",
        "data": {"object": {"client_reference_id": uid, "customer": "cus_x", "subscription": "sub_x"}}})
    assert wh.status_code == 200
    assert db.get_plan(uid) == "pro"
    r = client.post(f"/api/datasets/{did}/ask", json={"question": "how many rows"}, headers=h)
    assert r.status_code == 200

    # subscription deleted downgrades
    client.post("/api/billing/webhook", json={
        "type": "customer.subscription.deleted", "data": {"object": {"id": "sub_x"}}})
    assert db.get_plan(uid) == "free"


def test_row_limit_on_free(client):
    big = ("x\n" + "\n".join(str(i) for i in range(db.FREE["max_rows"] + 50))).encode()
    r = client.post("/api/upload", files={"file": ("big.csv", big, "text/csv")})
    assert r.status_code == 402


def test_export_quota(client):
    up = client.post("/api/upload", files={"file": ("t.csv", TINY, "text/csv")})
    did = up.json()["dataset"]["id"]
    uid = db.create_user(em(), "supersecret1")["id"]
    h = _bearer(uid)
    db.bump(uid, "exports", db.FREE["exports"])
    r = client.post(f"/api/datasets/{did}/export",
                    json={"spec": {"kind": "aggregate", "dimension": "region", "measure": "v", "agg": "sum"}},
                    headers=h)
    assert r.status_code == 402


def test_grant_disabled_with_stripe_and_for_guests(client):
    g = client.post("/api/billing/grant", json={"plan": "pro"})
    assert g.status_code == 401  # guest


def test_datasets_scoped_to_identity(client):
    a = TestClient(server.app)
    b = TestClient(server.app)
    da = a.post("/api/upload", files={"file": ("a.csv", TINY, "text/csv")}).json()["dataset"]["id"]
    db_ = b.post("/api/upload", files={"file": ("b.csv", TINY, "text/csv")}).json()["dataset"]["id"]
    a_ids = {d["id"] for d in a.get("/api/datasets").json()["datasets"]}
    b_ids = {d["id"] for d in b.get("/api/datasets").json()["datasets"]}
    assert da in a_ids and da not in b_ids
    assert db_ in b_ids and db_ not in a_ids


def test_dryrun_grants_pro(client):
    uid = db.create_user(em(), "supersecret1")["id"]
    r = client.post("/api/billing/dryrun", headers=_bearer(uid))
    assert r.status_code == 200 and r.json()["plan"] == "pro"
    assert db.get_plan(uid) == "pro"


def test_admin_usage_report(client):
    admin = db.create_user(em(), "supersecret1")["id"]
    db.set_admin(admin, True)
    other = db.create_user(em(), "supersecret1")["id"]
    db.bump(other, "questions", 7)
    db.bump(other, "exports", 2)

    r = client.get("/api/admin/usage", headers=_bearer(admin))
    assert r.status_code == 200
    data = r.json()
    row = next(a for a in data["accounts"] if a["id"] == other)
    assert row["questions"] == 7 and row["exports"] == 2 and row["plan"] == "free"
    assert data["totals"]["accounts"] >= 2

    # non-admin and guest are forbidden
    assert client.get("/api/admin/usage", headers=_bearer(other)).status_code == 403
    assert client.get("/api/admin/usage").status_code == 403


def test_portal_demo_and_cancel(client):
    uid = db.create_user(em(), "supersecret1")["id"]
    h = _bearer(uid)
    client.post("/api/billing/dryrun", headers=h)
    assert db.get_plan(uid) == "pro"
    # offline, the portal reports demo mode
    r = client.post("/api/billing/portal", headers=h)
    assert r.status_code == 200 and r.json()["mode"] == "demo"
    # demo cancel downgrades to free
    r = client.post("/api/billing/cancel", headers=h)
    assert r.status_code == 200 and db.get_plan(uid) == "free"
    # guest is blocked
    assert client.post("/api/billing/portal").status_code == 401


def test_rate_limit_throttles_free_then_pro_higher(client):
    orig = dict(server.RATE_PER_MIN)
    server._RATE.clear()
    server.RATE_PER_MIN["free"] = 2
    server.RATE_PER_MIN["pro"] = 5
    try:
        up = client.post("/api/upload", files={"file": ("t.csv", TINY, "text/csv")})
        did = up.json()["dataset"]["id"]
        # free/guest: 2 allowed, 3rd throttled with 429
        for _ in range(2):
            assert client.post(f"/api/datasets/{did}/ask",
                               json={"question": "rows"}).status_code == 200
        assert client.post(f"/api/datasets/{did}/ask",
                           json={"question": "rows"}).status_code == 429
        # pro gets the higher limit (not throttled at free's 2)
        uid = db.create_user(em(), "supersecret1")["id"]
        db.set_plan(uid, "pro")
        h = _bearer(uid)
        for _ in range(5):
            assert client.post(f"/api/datasets/{did}/ask",
                               json={"question": "rows"}, headers=h).status_code == 200
        assert client.post(f"/api/datasets/{did}/ask",
                           json={"question": "rows"}, headers=h).status_code == 429
    finally:
        server.RATE_PER_MIN.update(orig)
        server._RATE.clear()


def test_receipt_noop_without_smtp():
    from engine import mailer
    assert mailer.smtp_ready() is False
    assert mailer.send_receipt("x@y.com", "pro", 29) is False
