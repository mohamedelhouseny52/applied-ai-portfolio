"""Signed webhook receipt and a transactional outbox for replay-safe analytics."""
import hashlib
import hmac
import json
import os
from pathlib import Path
import sqlite3
import time
from decimal import Decimal, InvalidOperation
from flask import Flask, request, jsonify
from .common import ROOT, write_json

SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
 event_id TEXT PRIMARY KEY, payload_sha TEXT NOT NULL, event_type TEXT NOT NULL,
 payload TEXT NOT NULL, received_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS outbox(
 event_id TEXT PRIMARY KEY REFERENCES events(event_id), state TEXT NOT NULL DEFAULT 'pending',
 attempts INTEGER NOT NULL DEFAULT 0, error TEXT);
CREATE TABLE IF NOT EXISTS daily_revenue(
 day TEXT NOT NULL, currency TEXT NOT NULL, amount_minor INTEGER NOT NULL,
 event_count INTEGER NOT NULL, PRIMARY KEY(day,currency));
"""


def connect(path):
    db = sqlite3.connect(path, timeout=10)
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    return db


def sign(secret, timestamp, body):
    return hmac.new(secret.encode(), str(timestamp).encode() + b"." + body, hashlib.sha256).hexdigest()


def validate(payload):
    if not isinstance(payload, dict) or set(payload) != {"event_id", "event_type", "data"}:
        raise ValueError("Expected event_id, event_type and data only")
    if not isinstance(payload["event_id"], str) or not 1 <= len(payload["event_id"]) <= 100:
        raise ValueError("Invalid event_id")
    if payload["event_type"] != "payment.received":
        raise ValueError("Unsupported event_type")
    data = payload["data"]
    if not isinstance(data, dict) or set(data) != {"occurred_at", "currency", "amount"}:
        raise ValueError("Invalid payment fields")
    from datetime import datetime
    if not isinstance(data["occurred_at"], str):
        raise ValueError("occurred_at must be an ISO timestamp")
    dt = datetime.fromisoformat(data["occurred_at"].replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Timestamp requires timezone")
    if data["currency"] not in {"USD", "EUR", "GBP"}:
        raise ValueError("Only two-decimal currencies USD/EUR/GBP supported")
    if not isinstance(data["amount"], str):
        raise ValueError("amount must be a decimal string")
    try:
        amount = Decimal(data["amount"])
        if not amount.is_finite() or amount < 0 or amount > 10000000 or amount * 100 != (amount * 100).to_integral_value():
            raise ValueError("Invalid amount or currency precision")
    except InvalidOperation as exc:
        raise ValueError("Invalid decimal amount") from exc
    return payload


def create_app(db_path, secret, clock=time.time):
    if not secret:
        raise ValueError("WEBHOOK_SECRET is required")
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 65536
    db_path = str(db_path)
    initial = connect(db_path)
    initial.close()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/webhooks/payments")
    def ingest():
        body = request.get_data()
        stamp = request.headers.get("X-Timestamp", "")
        try:
            timestamp = int(stamp)
        except ValueError:
            return jsonify(error="Invalid timestamp"), 401
        if abs(clock() - timestamp) > 300:
            return jsonify(error="Expired signature"), 401
        expected = sign(secret, stamp, body)
        if not hmac.compare_digest(expected, request.headers.get("X-Signature", "")):
            return jsonify(error="Invalid signature"), 401
        try:
            payload = validate(json.loads(body))
        except (ValueError, TypeError, KeyError, UnicodeDecodeError):
            return jsonify(error="Invalid event schema"), 422
        # Canonical payload hashing permits identical events with different JSON whitespace.
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        sha = hashlib.sha256(canonical.encode()).hexdigest()
        db = connect(db_path)
        try:
            with db:
                db.execute("BEGIN IMMEDIATE")
                prior = db.execute("SELECT payload_sha FROM events WHERE event_id=?", (payload["event_id"],)).fetchone()
                if prior:
                    return jsonify(event_id=payload["event_id"], status="duplicate" if prior[0] == sha else "conflict"), 200 if prior[0] == sha else 409
                db.execute("INSERT INTO events VALUES(?,?,?,?,?)", (payload["event_id"], sha, payload["event_type"], canonical, int(clock())))
                db.execute("INSERT INTO outbox(event_id) VALUES(?)", (payload["event_id"],))
            return jsonify(event_id=payload["event_id"], status="accepted"), 202
        finally:
            db.close()

    return app


def process_pending(db_path, max_attempts=3):
    """Atomically update the local aggregate and mark each outbox item done."""
    from datetime import datetime, timezone
    db = connect(db_path)
    try:
        ids = db.execute("SELECT event_id FROM outbox WHERE state='pending' ORDER BY rowid").fetchall()
        for (event_id,) in ids:
            try:
                with db:
                    db.execute("BEGIN IMMEDIATE")
                    item = db.execute("SELECT state FROM outbox WHERE event_id=?", (event_id,)).fetchone()
                    if item[0] != "pending":
                        continue
                    payload = validate(json.loads(db.execute("SELECT payload FROM events WHERE event_id=?", (event_id,)).fetchone()[0]))
                    data = payload["data"]
                    day = datetime.fromisoformat(data["occurred_at"].replace("Z", "+00:00")).astimezone(timezone.utc).date().isoformat()
                    amount = int(Decimal(data["amount"]) * 100)
                    db.execute("INSERT INTO daily_revenue VALUES(?,?,?,1) ON CONFLICT(day,currency) DO UPDATE SET amount_minor=amount_minor+excluded.amount_minor,event_count=event_count+1", (day, data["currency"], amount))
                    db.execute("UPDATE outbox SET state='done', attempts=attempts+1,error=NULL WHERE event_id=?", (event_id,))
            except (ValueError, KeyError, TypeError) as exc:
                with db:
                    db.execute("UPDATE outbox SET attempts=attempts+1,error=?,state=CASE WHEN attempts+1>=? THEN 'dead' ELSE 'pending' END WHERE event_id=?", (type(exc).__name__, max_attempts, event_id))
        return db.execute("SELECT day,currency,amount_minor,event_count FROM daily_revenue ORDER BY day,currency").fetchall()
    finally:
        db.close()


def demo():
    out = ROOT / "artifacts/ingestion"; out.mkdir(parents=True, exist_ok=True)
    db = out / "demo.sqlite"
    # Reset only this generated demonstration database for repeatable output.
    if db.exists():
        db.unlink()
    secret, stamp = "local-example-key", 1700000000
    client = create_app(db, secret, clock=lambda: stamp).test_client()
    results = []
    for payload in [
        {"event_id": "PAY-001", "event_type": "payment.received", "data": {"occurred_at": "2025-01-04T10:00:00Z", "amount": "120.10", "currency": "USD"}},
        {"event_id": "PAY-002", "event_type": "payment.received", "data": {"occurred_at": "2025-01-04T11:00:00Z", "amount": "25.50", "currency": "EUR"}},
    ]:
        body = json.dumps(payload).encode()
        headers = {"X-Timestamp": str(stamp), "X-Signature": sign(secret, stamp, body), "Content-Type": "application/json"}
        response = client.post("/webhooks/payments", data=body, headers=headers)
        duplicate = client.post("/webhooks/payments", data=body, headers=headers)
        results.append({"first": response.status_code, "replay": duplicate.status_code, "response": duplicate.json})
    first = process_pending(db); second = process_pending(db)
    report = {"requests": results, "aggregate": first, "reprocessing_unchanged": first == second, "storage": "SQLite/WAL; local reproducible reference implementation"}
    write_json(out / "audit.json", report)
    import csv
    with (out / "daily_revenue.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["day", "currency", "amount_minor", "event_count"]); writer.writerows(first)
    return report


if __name__ == "__main__":
    create_app(os.environ.get("WEBHOOK_DB", "events.sqlite"), os.environ.get("WEBHOOK_SECRET", "")).run(host="127.0.0.1", port=5050, debug=False)
