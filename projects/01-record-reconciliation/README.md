# Auditable record reconciliation

Merge imperfect exports without hiding uncertainty.

## Business problem

Two systems export overlapping records, but one source lacks a reliable email key. A naive join drops matches, duplicates records or attaches one person's attributes to another.

## Implementation

Normalize identity fields, block candidates by email/name/IP, apply explicit evidence rules, require a separation margin and preserve a review queue. Conflicting emails, uncertain dates and multiple records claiming the same target never silently overwrite the source.

```text
CSV exports → normalization → candidate blocks → evidence rules → matched / review / unmatched → audit + output
```

## Measured result

**2,070 source rows · 820 enrichment records · 130 review cases.** Results are from the included reference run.

## Deliverables

Order-preserving merged CSV; duplicate ledger; per-candidate evidence; match decisions; source hashes and policy audit.

## Limitations

Synthetic fixtures exercise edge cases. Evidence scores are policy weights, not calibrated identity probabilities. Matching quality on a real dataset requires reviewed ground truth and an agreed matching policy.

## Reproduce and inspect

Run `python scripts/run_all.py` from the repository root. Open `artifacts/reconciliation/` for evidence and `notebooks/01-record-reconciliation.ipynb` for the executed analysis walkthrough. The source module is `casework/reconcile.py`.
