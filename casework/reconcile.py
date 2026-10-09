"""Deterministic multi-source matching with reviewable evidence."""
from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata
import pandas as pd
from .common import ROOT, digest, profile, write_json, bar_chart


@dataclass(frozen=True)
class Policy:
    max_days: int = 90
    auto_score: float = 0.85
    min_margin: float = 0.15


def normalize(value):
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value).strip().casefold())
    return re.sub(r"\s+", " ", "".join(c for c in text if not unicodedata.combining(c)))


def prepare(frame):
    required = {"record_id", "name", "email", "ip", "event_date", "segment", "user_agent"}
    if required - set(frame):
        raise ValueError(f"Missing columns: {sorted(required - set(frame))}")
    if frame.record_id.isna().any() or frame.record_id.duplicated().any():
        raise ValueError("record_id must be present and unique")
    out = frame.copy().reset_index(drop=True)
    out["_row"] = range(len(out))
    for c in ["name", "email", "ip", "segment", "user_agent"]:
        out["_" + c] = out[c].map(normalize)
    out["_date"] = pd.to_datetime(out.event_date, errors="coerce", utc=True)
    # Empty addresses are never treated as a shared identity.
    duplicate = out._email.ne("") & out._email.duplicated(keep="first")
    return out.loc[~duplicate].copy(), out.loc[duplicate, ["record_id", "_row"]]


def reconcile(left, right, policy=Policy()):
    a, dup_a = prepare(left)
    b, dup_b = prepare(right)
    blocks = {}
    for idx, row in a.iterrows():
        for key in ["_email", "_name", "_ip"]:
            if row[key]:
                blocks.setdefault((key, row[key]), set()).add(idx)
    evidence, decisions = [], []
    for _, row in b.iterrows():
        candidates = set()
        for key in ["_email", "_name", "_ip"]:
            if row[key]:
                candidates |= blocks.get((key, row[key]), set())
        ranked = []
        for idx in sorted(candidates):
            other = a.loc[idx]
            exact = bool(row._email and row._email == other._email)
            conflicting = bool(row._email and other._email and row._email != other._email)
            flags = {k: bool(row["_" + k] and row["_" + k] == other["_" + k]) for k in ["name", "ip", "segment", "user_agent"]}
            days = abs((row._date - other._date).total_seconds()) / 86400 if pd.notna(row._date) and pd.notna(other._date) else None
            date_ok = days is not None and days <= policy.max_days
            score = 1.0 if exact else sum(flags[k] * w for k, w in [("name", .35), ("ip", .20), ("segment", .20), ("user_agent", .10)]) + .15 * date_ok
            safe = exact or (flags["name"] and flags["segment"] and date_ok and not conflicting)
            item = {"right_id": row.record_id, "left_id": other.record_id, "score": round(score, 4), "email_exact": exact, "conflicting_emails": conflicting, "days_apart": days, "safe_rule": safe, **flags}
            evidence.append(item)
            ranked.append(item)
        # A unique exact email key takes precedence over weaker composite evidence.
        # The email deduplication step makes this an at-most-one-key match.
        exact_candidates = [item for item in ranked if item["email_exact"]]
        if exact_candidates:
            ranked = exact_candidates
        ranked.sort(key=lambda x: (-x["score"], str(x["left_id"])))
        top = ranked[0] if ranked else None
        margin = top["score"] - ranked[1]["score"] if len(ranked) > 1 else 1.0
        status = "unmatched" if not top else "review"
        if top and top["safe_rule"] and top["score"] >= policy.auto_score and margin >= policy.min_margin:
            status = "matched"
        decisions.append({"right_id": row.record_id, "left_id": top["left_id"] if top else None, "status": status, "score": top["score"] if top else 0, "margin": round(margin, 4), "reason": "no_candidates" if not top else "rule_and_margin_pass" if status == "matched" else "ambiguous_or_insufficient_evidence"})
    decisions = pd.DataFrame(decisions, columns=["right_id", "left_id", "status", "score", "margin", "reason"])
    # Conflicting multiple right records are reviewed rather than overwriting one another.
    collision = decisions.status.eq("matched") & decisions.loc[decisions.status.eq("matched"), "left_id"].duplicated(keep=False).reindex(decisions.index, fill_value=False)
    decisions.loc[collision, ["status", "reason"]] = ["review", "multiple_right_records_for_one_left"]
    accepted = decisions.loc[decisions.status.eq("matched"), ["left_id", "right_id", "score"]]
    payload = [c for c in right if c not in {"name", "email", "ip", "event_date", "segment", "user_agent", "record_id"}]
    result = a[[c for c in left]].merge(accepted, left_on="record_id", right_on="left_id", how="left", sort=False, validate="one_to_one")
    result = result.merge(b[["record_id", *payload]].rename(columns={"record_id": "right_id", **{c: "source_b_" + c for c in payload}}), on="right_id", how="left", sort=False, validate="many_to_one")
    return result, decisions, pd.DataFrame(evidence), pd.concat([dup_a.assign(source="A"), dup_b.assign(source="B")])


def run(left_path, right_path, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    left, right = pd.read_csv(left_path, dtype=str).fillna(""), pd.read_csv(right_path, dtype=str).fillna("")
    merged, decisions, evidence, duplicates = reconcile(left, right)
    for name, frame in [("merged", merged), ("decisions", decisions), ("candidate_evidence", evidence), ("duplicates", duplicates)]:
        frame.to_csv(out / f"{name}.csv", index=False)
    summary = {"source_a": profile(left), "source_b": profile(right), "output_rows": len(merged), "duplicates_removed": len(duplicates), "decisions": decisions.status.value_counts().to_dict(), "policy": Policy().__dict__, "sha256": {"A": digest(left_path), "B": digest(right_path)}, "ordering": "Earliest file occurrence; source A order preserved", "score_meaning": "Evidence weights, not calibrated match probabilities"}
    write_json(out / "audit.json", summary)
    bar_chart(summary["decisions"], "Reconciliation outcomes", out / "outcomes.png")
    return summary


def demo():
    out = ROOT / "artifacts/reconciliation"
    return run(ROOT / "data/examples/source_a.csv", ROOT / "data/examples/source_b.csv", out)

