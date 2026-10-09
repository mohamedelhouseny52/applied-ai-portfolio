"""Controlled synthetic stress fixture for matching decisions, not real-world accuracy."""
from pathlib import Path
import sys
import time
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from casework.reconcile import run
from casework.common import ROOT, write_json


def main():
    source = ROOT / "data/examples/benchmark"; source.mkdir(parents=True, exist_ok=True)
    left = []
    for i in range(2000):
        left.append({"record_id": f"A{i:04d}", "name": f"Account Holder {i}", "email": f"person{i}@example.test", "ip": f"192.0.2.{i % 200 + 1}", "event_date": "2025-01-01", "segment": f"S{i}", "user_agent": "Browser A"})
    # Same names and network metadata, distinct emails: fuzzy identity remains ambiguous.
    for i in range(20):
        left.append({**left[i], "record_id": f"TWIN{i}", "email": f"twin{i}@example.test"})
    for i in range(50):
        left.append({**left[i + 1000], "record_id": f"DUP{i}"})
    right, truth = [], []
    for i in range(800):
        row = {**left[i], "record_id": f"B{i:04d}", "order_value": str(10 + i % 100), "plan": "annual"}
        row["email"] = "" if i % 2 else row["email"].upper()
        row["name"] = "  " + row["name"].upper() + "  "
        row["event_date"] = "2025-01-05"
        status = "matched"
        if i < 20 and i % 2:
            status = "review"
        elif 100 <= i < 140:
            row["email"] = "conflicting" + str(i) + "@example.test"; status = "review"
        elif 200 <= i < 240:
            row["email"] = ""; row["event_date"] = "not-a-date"; status = "review"
        elif 300 <= i < 340:
            row["email"] = ""; row["event_date"] = "2025-12-01"; status = "review"
        right.append(row)
        truth.append({"right_id": row["record_id"], "expected_status": status, "expected_left_id": f"A{i:04d}"})
    for i in range(20):
        right.append({"record_id": f"NEW{i}", "name": f"New Account {i}", "email": "", "ip": "198.51.100.1", "event_date": "2025-01-03", "segment": "NEW", "user_agent": "Browser B", "order_value": "50", "plan": "monthly"})
        truth.append({"right_id": f"NEW{i}", "expected_status": "unmatched", "expected_left_id": None})
    pd.DataFrame(left).to_csv(source / "source_a.csv", index=False)
    pd.DataFrame(right).to_csv(source / "source_b.csv", index=False)
    pd.DataFrame(truth).to_csv(source / "ground_truth.csv", index=False)
    out = ROOT / "artifacts/reconciliation/benchmark"
    start = time.perf_counter()
    report = run(source / "source_a.csv", source / "source_b.csv", out)
    elapsed = time.perf_counter() - start
    evaluated = pd.read_csv(out / "decisions.csv").merge(pd.DataFrame(truth), on="right_id", validate="one_to_one")
    accepted = evaluated.status.eq("matched")
    report["controlled_fixture_checks"] = {"status_agreement": float(evaluated.status.eq(evaluated.expected_status).mean()), "false_auto_links": int((accepted & evaluated.left_id.ne(evaluated.expected_left_id)).sum()), "accepted_links": int(accepted.sum()), "elapsed_seconds_including_outputs": round(elapsed, 3), "note": "Designed synthetic edge-case fixture; not an estimate of client-data matching precision"}
    write_json(out / "audit.json", report)
    print(report["controlled_fixture_checks"])


if __name__ == "__main__":
    main()
