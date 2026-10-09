"""Local prediction service with strict request contracts."""
import hmac
import os
import hashlib
import json
import sqlite3
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from .common import ROOT
from .routing import infer


def create_app(artifact_root=None, api_key=None, ticket_db=None):
    root = Path(artifact_root or ROOT / "artifacts")
    api_key = api_key or os.environ.get("MODEL_API_KEY")
    if not api_key:
        raise ValueError("MODEL_API_KEY is required")
    # Only load artifacts generated locally by these training scripts.
    churn = joblib.load(root / "retention/model.joblib")
    routing = joblib.load(root / "routing/model.joblib")
    app = Flask(__name__); app.config["MAX_CONTENT_LENGTH"] = 32768
    ticket_db = str(ticket_db or os.environ.get("TICKET_DB", ROOT / "artifacts/tickets.sqlite"))
    with sqlite3.connect(ticket_db) as db:
        db.execute("CREATE TABLE IF NOT EXISTS tickets(ticket_id TEXT PRIMARY KEY, text_sha TEXT NOT NULL, result TEXT NOT NULL)")

    @app.before_request
    def authenticate():
        if request.path != "/health" and not hmac.compare_digest(request.headers.get("X-API-Key", ""), api_key):
            return jsonify(error="Unauthorized"), 401

    @app.get("/health")
    def health():
        return {"status": "ok", "models": ["retention", "intent_routing"]}

    @app.post("/predict/retention")
    def retention():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or set(payload) != {"records"} or not isinstance(payload["records"], list) or not 1 <= len(payload["records"]) <= 100:
            return jsonify(error="Supply 1–100 records"), 422
        if any(not isinstance(r, dict) or set(r) != set(churn["features"]) for r in payload["records"]):
            return jsonify(error="Fields must match the model feature contract", fields=churn["features"]), 422
        data = pd.DataFrame(payload["records"])
        for c in churn["numeric"]:
            if data[c].map(lambda v: isinstance(v, bool)).any():
                return jsonify(error="Invalid numeric value"), 422
            try:
                data[c] = pd.to_numeric(data[c], errors="raise")
            except (ValueError, TypeError):
                return jsonify(error="Invalid numeric value"), 422
            if not np.isfinite(data[c].to_numpy(dtype=float)).all() or (data[c] < 0).any():
                return jsonify(error="Numeric values must be finite and nonnegative"), 422
        for c in set(churn["features"]) - set(churn["numeric"]):
            if not data[c].map(lambda v: isinstance(v, str) and 0 < len(v) <= 100).all():
                return jsonify(error="Invalid categorical value"), 422
        probabilities = churn["model"].predict_proba(data)[:, 1]
        return {"predictions": [{"risk_score": float(p), "prioritize_review": bool(p >= churn["threshold"])} for p in probabilities], "threshold": churn["threshold"]}

    @app.post("/route")
    def route():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or set(payload) != {"text"}:
            return jsonify(error="Expected text only"), 422
        try:
            return infer(routing, payload["text"])
        except ValueError as exc:
            return jsonify(error=str(exc)), 422

    @app.post("/tickets")
    def ticket():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or set(payload) != {"ticket_id", "text"} or not isinstance(payload["ticket_id"], str) or not 1 <= len(payload["ticket_id"]) <= 100:
            return jsonify(error="Expected ticket_id and text"), 422
        text = payload["text"]
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            return jsonify(error="Invalid text"), 422
        # Persist only a hash and the partially redacted routing result; never raw input.
        sha = hashlib.sha256(text.encode()).hexdigest()
        with sqlite3.connect(ticket_db, timeout=10) as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT text_sha,result FROM tickets WHERE ticket_id=?", (payload["ticket_id"],)).fetchone()
            if prior:
                if prior[0] != sha:
                    return jsonify(error="Ticket ID already used with different text"), 409
                return {"status": "duplicate", "result": json.loads(prior[1])}
            result = infer(routing, text)
            db.execute("INSERT INTO tickets VALUES(?,?,?)", (payload["ticket_id"], sha, json.dumps(result)))
        return {"status": "processed", "result": result}, 201

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5051, debug=False)
