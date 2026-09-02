"""Optional email receipts over SMTP.

Zero-config by default: if ``SMTP_HOST`` is not set, :func:`send_receipt` is a
no-op and returns ``False``. When configured, receipts are sent on a background
thread so a slow/unreachable mail server never blocks or fails a request.

Environment (all optional):
    SMTP_HOST   e.g. smtp.mailgun.org           (required to enable sending)
    SMTP_PORT   default 587
    SMTP_USER / SMTP_PASS   optional credentials
    SMTP_FROM   sender address (defaults to SMTP_USER)
    SMTP_TLS    set to 0/false to disable STARTTLS
"""
from __future__ import annotations

import os
import smtplib
import threading
from email.mime.text import MIMEText


def smtp_ready() -> bool:
    return bool(os.environ.get("SMTP_HOST"))


def send_receipt(email: str, plan: str, price: float) -> bool:
    """Queue a 'you upgraded' receipt. Returns False when SMTP is not configured."""
    if not smtp_ready() or not email:
        return False
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    pw = os.environ.get("SMTP_PASS")
    frm = os.environ.get("SMTP_FROM", user or "noreply@ai-analytics.local")

    msg = MIMEText(
        "Hi,\n\n"
        "Thanks for upgrading — your receipt is below.\n\n"
        f"  Plan:   {plan.upper()}\n"
        f"  Amount: ${price:.2f} / month\n\n"
        "You can manage or cancel your subscription from the app at any time.\n"
    )
    msg["Subject"] = "Your AutoAnalytics Pro receipt"
    msg["From"] = frm
    msg["To"] = email

    def _send() -> None:
        try:
            with smtplib.SMTP(host, port, timeout=10) as s:
                s.ehlo()
                if os.environ.get("SMTP_TLS", "1") not in ("0", "false"):
                    s.starttls()
                    s.ehlo()
                if user and pw:
                    s.login(user, pw)
                s.sendmail(frm, [email], msg.as_string())
        except Exception:
            # Never let mail failures break the product.
            pass

    threading.Thread(target=_send, daemon=True).start()
    return True
