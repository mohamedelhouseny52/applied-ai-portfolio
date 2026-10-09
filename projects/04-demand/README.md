# Hourly demand estimation

Plan with a baseline, error slices and uncertainty.

## Business problem

Rental operations need hourly demand estimates for staffing and availability. Random train/test splits can hide time drift and make a model look better than it will on a future period.

## Implementation

Train on 2011, select and estimate residual widths on Jan–Jun 2012, and test on Jul–Dec 2012. Compare Poisson and boosted regressors to a historical hour/weekday baseline, add rolling training checks and report error by hour.

```text
Hourly history → chronological split → baseline + candidate models → validation selection → frozen model → future-period error audit
```

## Measured result

**25.2% lower holdout MAE than the historical baseline.** Results are from the included reference run.

## Deliverables

Future-period predictions; baseline comparison; MAE/RMSE/WAPE; empirical interval coverage; rolling validation; hour-level error CSV and reproducible model artifact.

## Limitations

Uses observed weather: conditional estimation rather than a weather forecast. Temporal drift means the nominal 90% intervals have no coverage guarantee. The measured coverage is reported. No claim of performance in a new city or fleet.

## Reproduce and inspect

Run `python scripts/run_all.py` from the repository root. Open `artifacts/demand/` for evidence and `notebooks/04-demand.ipynb` for the executed analysis walkthrough. The source module is `casework/demand.py`.
