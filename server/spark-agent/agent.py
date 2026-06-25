#!/usr/bin/env python3
"""spark-agent — provisions capstone Linux accounts on the campus Spark machine.

Runs INSIDE the campus network on the Spark host. Makes only OUTBOUND HTTPS to
Supabase (no inbound exposure needed). Polls for admin-APPROVED account requests,
derives a sanitized username from the learner's profile, generates a one-time
temp password locally, and runs the fixed provision.sh. The web app and the AI
tutor never touch this privileged path — they only write/approve request rows.

Config via environment (see .env.example):
  SUPABASE_URL                  https://<ref>.supabase.co
  SUPABASE_SERVICE_ROLE_KEY     service-role key (bypasses RLS; KEEP SECRET, agent-only)
  SPARK_HOST                    display hostname shown to the learner, e.g. spark.campus.edu
  PROVISION_CMD                 default: sudo /opt/spark-agent/provision.sh
  POLL_SECONDS                  default: 20
"""
import json
import os
import re
import secrets
import string
import subprocess
import time
import urllib.error
import urllib.request

URL = os.environ["SUPABASE_URL"].rstrip("/")
KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
HOST = os.environ.get("SPARK_HOST", "spark")
PROVISION = os.environ.get("PROVISION_CMD", "sudo /opt/spark-agent/provision.sh").split()
POLL = int(os.environ.get("POLL_SECONDS", "20"))

REST = f"{URL}/rest/v1"
HEADERS = {
    "apikey": KEY,
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
}
USERNAME_RE = re.compile(r"^[a-z][a-z0-9_-]{2,30}$")


def _req(method, path, params="", body=None, prefer=None):
    headers = dict(HEADERS)
    if prefer:
        headers["Prefer"] = prefer
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{REST}/{path}{params}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode()
        return json.loads(raw) if raw else None


def sanitize_username(email, display_name):
    """Derive a safe Linux username from verified identity (never from chat)."""
    base = (email or "").split("@")[0].lower()
    base = re.sub(r"[^a-z0-9]", "", base)
    if not base or not base[0].isalpha():
        base = "cap" + base
    base = base[:24] or "capstone"
    return base


def gen_password(n=16):
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))


def claim(user_id):
    """Atomically move approved -> provisioning so two agents can't double-claim."""
    rows = _req(
        "PATCH",
        "spark_accounts",
        f"?user_id=eq.{user_id}&status=eq.approved",
        body={"status": "provisioning"},
        prefer="return=representation",
    )
    return bool(rows)


def finalize(user_id, fields):
    _req("PATCH", "spark_accounts", f"?user_id=eq.{user_id}", body=fields, prefer="return=minimal")


def profile(user_id):
    rows = _req("GET", "profiles", f"?id=eq.{user_id}&select=email,display_name")
    return rows[0] if rows else {}


def existing_usernames():
    rows = _req("GET", "spark_accounts", "?select=ssh_username&ssh_username=not.is.null") or []
    return {r["ssh_username"] for r in rows if r.get("ssh_username")}


def provision_one(user_id):
    taken = existing_usernames()
    p = profile(user_id)
    base = sanitize_username(p.get("email"), p.get("display_name"))
    # ensure uniqueness across the queue; provision.sh also refuses OS-existing names
    username = base
    for i in range(20):
        if username not in taken and USERNAME_RE.match(username):
            break
        username = f"{base[:22]}{secrets.randbelow(90) + 10}"
    if not USERNAME_RE.match(username):
        raise ValueError("could not derive a valid username")
    password = gen_password()
    proc = subprocess.run([*PROVISION, username, password], capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"provision.sh exit {proc.returncode}: {proc.stderr.strip()[:200]}")
    return username, password


def tick():
    pending = _req("GET", "spark_accounts", "?status=eq.approved&select=user_id") or []
    for row in pending:
        uid = row["user_id"]
        if not claim(uid):
            continue  # another agent took it
        try:
            username, password = provision_one(uid)
            finalize(uid, {
                "status": "ready",
                "ssh_username": username,
                "temp_password": password,
                "host": HOST,
                "error": None,
                "provisioned_at": "now()",
            })
            print(f"[ok] provisioned {username} for {uid}", flush=True)
        except Exception as e:  # noqa: BLE001 — log generically, never leak to client
            finalize(uid, {"status": "failed", "error": "开户失败,请联系管理员"})
            print(f"[fail] {uid}: {e}", flush=True)


def main():
    print(f"spark-agent up; polling {REST} every {POLL}s; host={HOST}", flush=True)
    while True:
        try:
            tick()
        except urllib.error.HTTPError as e:
            print(f"[http] {e.code} {e.read().decode()[:200]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[err] {e}", flush=True)
        time.sleep(POLL)


if __name__ == "__main__":
    main()
