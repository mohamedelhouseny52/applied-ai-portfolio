# Retention outreach prioritization

Turn model scores into a bounded review queue.

## Business problem

A retention team has limited outreach capacity. Accuracy alone does not show whether the first customers reviewed are more likely to churn than a random contact list.

## Implementation

Compare a base-rate predictor, logistic regression and a bounded random forest. Fit preprocessing inside each training pipeline, select on validation PR-AUC, choose the outreach threshold from validation only, then evaluate the untouched test set.

```text
Customer snapshot → train/validation/test → fitted preprocessing → model selection → capacity threshold → holdout scores + API
```

## Measured result

**2.40× lift at 20% contact capacity.** Results are from the included reference run.

## Deliverables

Ranked customer scores; PR-AUC/ROC-AUC; confusion matrix; lift at 20%; calibration bins; permutation importance; split manifest; schema-validated scoring API.

## Limitations

Public cross-sectional demonstration data. The split validates benchmark generalization, not future churn. Scores do not prove outreach reduces churn. Sensitive demographic features are excluded; a live business needs temporal labels, fairness review and threshold revalidation.

## Reproduce and inspect

Run `python scripts/run_all.py` from the repository root. Open `artifacts/retention/` for evidence and `notebooks/03-retention.ipynb` for the executed analysis walkthrough. The source module is `casework/churn.py`.
