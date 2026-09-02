"""SQLite-backed accounts, sessions, plans and usage metering.

Self-contained: no external service needed to run. Stripe is layered on top
in server.py and only activates when keys are configured. Passwords are
hashed with scrypt; sessions are opaque tokens in an HttpOnly cookie.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import time
from datetime import datetime, timezone

from .loader import DATA_DIR

DB_PATH = os.path.join(DATA_DIR, "app.db")
SESSION_DAYS = 30

# ------------------------------------------------------------------ quotas
FREE = {"max_rows": 50_000, "max_file_mb": 5, "questions": 100, "exports": 20, "datasets": 10}
PRO = {"max_rows": 5_000_000, "max_file_mb": 40, "questions": 10_000_000,
       "exports": 10_000_000, "datasets": 10_000_000}


def limits_for(plan: str) -> dict:
    return PRO if plan == "pro" else FREE


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
                pass_hash TEXT NOT NULL, created TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                expires INTEGER NOT NULL, created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS plans (
                user_id TEXT PRIMARY KEY, plan TEXT NOT NULL DEFAULT 'free',
                stripe_customer TEXT, stripe_sub TEXT, updated TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS usage (
                identity TEXT NOT NULL, period TEXT NOT NULL,
                questions INTEGER NOT NULL DEFAULT 0,
                exports INTEGER NOT NULL DEFAULT 0,
                datasets INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (identity, period));
            """
        )
        # lightweight migration for databases created before is_admin existed
        cols = [r["name"] for r in c.execute("PRAGMA table_info(users)").fetchall()]
        if "is_admin" not in cols:
            c.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")


def _period() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


# ------------------------------------------------------------------ auth
def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1)
    return salt.hex() + "$" + h.hex()


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, h_hex = stored.split("$", 1)
        h = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex),
                           n=2 ** 14, r=8, p=1)
        return secrets.compare_digest(h.hex(), h_hex)
    except Exception:
        return False


def create_user(email: str, password: str) -> dict:
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        raise ValueError("A valid email is required.")
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    with _conn() as c:
        if c.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone():
            raise ValueError("An account with that email already exists.")
        uid = secrets.token_hex(8)
        now = datetime.now(timezone.utc).isoformat()
        first = c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
        c.execute("INSERT INTO users (id,email,pass_hash,created,is_admin) VALUES (?,?,?,?,?)",
                  (uid, email, hash_password(password), now, 1 if first else 0))
        c.execute("INSERT INTO plans (user_id,plan,updated) VALUES (?,?,?)",
                  (uid, "free", now))
    return {"id": uid, "email": email, "plan": "free", "is_admin": first}


def authenticate(email: str, password: str) -> dict | None:
    email = (email or "").strip().lower()
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not row or not verify_password(password, row["pass_hash"]):
        return None
    return {"id": row["id"], "email": row["email"], "is_admin": bool(row["is_admin"])}


def new_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    exp = int(time.time()) + SESSION_DAYS * 86400
    with _conn() as c:
        c.execute("INSERT INTO sessions (token,user_id,expires,created) VALUES (?,?,?,?)",
                  (token, user_id, exp, datetime.now(timezone.utc).isoformat()))
    return token


def session_user(token: str | None) -> dict | None:
    if not token:
        return None
    with _conn() as c:
        row = c.execute(
            "SELECT s.user_id, u.email, u.is_admin FROM sessions s JOIN users u ON u.id=s.user_id "
            "WHERE s.token=? AND s.expires>?", (token, int(time.time()))).fetchone()
    if not row:
        return None
    return {"id": row["user_id"], "email": row["email"], "is_admin": bool(row["is_admin"])}


def end_session(token: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE token=?", (token,))


# ------------------------------------------------------------------ plans
def get_plan(user_id: str) -> str:
    with _conn() as c:
        row = c.execute("SELECT plan FROM plans WHERE user_id=?", (user_id,)).fetchone()
    return row["plan"] if row else "free"


def set_plan(user_id: str, plan: str, stripe_customer: str | None = None,
             stripe_sub: str | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute("INSERT INTO plans (user_id,plan,stripe_customer,stripe_sub,updated) "
                  "VALUES (?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET plan=excluded.plan, "
                  "stripe_customer=COALESCE(excluded.stripe_customer,stripe_customer), "
                  "stripe_sub=COALESCE(excluded.stripe_sub,stripe_sub), updated=excluded.updated",
                  (user_id, plan, stripe_customer, stripe_sub, now))


def email_for(user_id: str) -> str | None:
    with _conn() as c:
        row = c.execute("SELECT email FROM users WHERE id=?", (user_id,)).fetchone()
    return row["email"] if row else None


def billing_for(user_id: str) -> tuple:
    with _conn() as c:
        row = c.execute("SELECT stripe_customer, stripe_sub FROM plans WHERE user_id=?",
                        (user_id,)).fetchone()
    return (row["stripe_customer"], row["stripe_sub"]) if row else (None, None)


def set_plan_by_sub(stripe_sub: str, plan: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute("UPDATE plans SET plan=?, updated=? WHERE stripe_sub=?", (plan, now, stripe_sub))


# ------------------------------------------------------------------ usage
def bump(identity: str, field: str, amount: int = 1) -> int:
    if field not in ("questions", "exports", "datasets"):
        raise ValueError("unknown usage field")
    with _conn() as c:
        c.execute("INSERT OR IGNORE INTO usage (identity,period,questions,exports,datasets) "
                  "VALUES (?,?,?,?,0)", (identity, _period(), 0, 0))
        c.execute(f"UPDATE usage SET {field}={field}+? WHERE identity=? AND period=?",
                  (amount, identity, _period()))
        row = c.execute(f"SELECT {field} FROM usage WHERE identity=? AND period=?",
                        (identity, _period())).fetchone()
    return int(row[0])


def usage_for(identity: str) -> dict:
    with _conn() as c:
        row = c.execute("SELECT * FROM usage WHERE identity=? AND period=?",
                        (identity, _period())).fetchone()
    if not row:
        return {"questions": 0, "exports": 0, "datasets": 0}
    return {"questions": row["questions"], "exports": row["exports"],
            "datasets": row["datasets"]}


# ------------------------------------------------------------------ admin
def is_admin(user_id: str) -> bool:
    with _conn() as c:
        row = c.execute("SELECT is_admin FROM users WHERE id=?", (user_id,)).fetchone()
    return bool(row and row["is_admin"])


def set_admin(user_id: str, value: bool = True) -> None:
    with _conn() as c:
        c.execute("UPDATE users SET is_admin=? WHERE id=?", (1 if value else 0, user_id))


def admin_usage() -> list[dict]:
    """Every account with its plan and current-month usage (for the admin report)."""
    period = _period()
    with _conn() as c:
        rows = c.execute(
            """SELECT u.id, u.email, u.created, u.is_admin, p.plan,
                      COALESCE(g.questions,0) AS questions,
                      COALESCE(g.exports,0)  AS exports,
                      COALESCE(g.datasets,0) AS datasets
               FROM users u
               LEFT JOIN plans p ON p.user_id=u.id
               LEFT JOIN usage g ON g.identity=u.id AND g.period=?
               ORDER BY (COALESCE(g.questions,0) + COALESCE(g.exports,0)) DESC, u.created""",
            (period,)).fetchall()
    return [{"id": r["id"], "email": r["email"], "plan": r["plan"] or "free",
             "is_admin": bool(r["is_admin"]), "created": r["created"],
             "questions": r["questions"], "exports": r["exports"], "datasets": r["datasets"]}
            for r in rows]


init_db()
