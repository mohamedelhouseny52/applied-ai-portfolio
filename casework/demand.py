"""Time-ordered rental demand estimation with an empirical uncertainty audit."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from .common import ROOT, write_json, digest, versions, bar_chart

FEATURES = ["hr", "weekday", "workingday", "holiday", "month", "year_index", "temp", "hum", "windspeed", "weathersit"]


def prepare(raw):
    frame = raw.copy()
    frame["timestamp"] = pd.to_datetime(frame.dteday) + pd.to_timedelta(frame.hr, unit="h")
    frame["month"] = frame.timestamp.dt.month
    frame["year_index"] = frame.timestamp.dt.year - 2011
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if frame.timestamp.duplicated().any() or (frame.cnt < 0).any():
        raise ValueError("Unique timestamps and nonnegative counts required")
    return frame


def metrics(y, pred):
    return {"mae": mean_absolute_error(y, pred), "rmse": float(np.sqrt(mean_squared_error(y, pred))), "wape": float(np.abs(np.asarray(y) - pred).sum() / np.asarray(y).sum())}


def run(path=None, out=None):
    path = Path(path or ROOT / "data/raw/bike_hour.csv")
    out = Path(out or ROOT / "artifacts/demand"); out.mkdir(parents=True, exist_ok=True)
    frame = prepare(pd.read_csv(path))
    train = frame.timestamp < "2012-01-01"
    val = (frame.timestamp >= "2012-01-01") & (frame.timestamp < "2012-07-01")
    test = frame.timestamp >= "2012-07-01"
    if not train.any() or not val.any() or not test.any():
        raise ValueError("Expected 2011–2012 data for the published split")
    x, y = frame[FEATURES], frame.cnt
    models = {"poisson": Pipeline([("scale", StandardScaler()), ("model", PoissonRegressor(alpha=.1, max_iter=1000))]), "boosted": HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=20, l2_regularization=3, random_state=42)}
    fitted, validation = {}, {}
    for name, model in models.items():
        model.fit(x[train], y[train]); fitted[name] = model
        validation[name] = metrics(y[val], np.maximum(0, model.predict(x[val])))
    best = min(validation, key=lambda n: validation[n]["mae"])
    selected = fitted[best]
    # Model is frozen before calibration; do not refit after computing interval widths.
    residual = np.abs(y[val].values - np.maximum(0, selected.predict(x[val])))
    quantile = min(1, np.ceil((len(residual) + 1) * .9) / len(residual))
    width = float(np.quantile(residual, quantile, method="higher"))
    pred = np.maximum(0, selected.predict(x[test]))
    low, high = np.maximum(0, pred - width), pred + width
    historic = frame[train].groupby(["hr", "weekday"]).cnt.mean()
    baseline = np.array([historic.get((r.hr, r.weekday), y[train].mean()) for r in frame[test].itertuples()])
    result = metrics(y[test], pred)
    result["empirical_interval_coverage"] = float(((y[test].values >= low) & (y[test].values <= high)).mean())
    result["mean_interval_width"] = float((high - low).mean())
    rolling = []
    xt, yt = x[train].reset_index(drop=True), y[train].reset_index(drop=True)
    from sklearn.base import clone
    for tr, va in TimeSeriesSplit(n_splits=3).split(xt):
        m = clone(models[best]); m.fit(xt.iloc[tr], yt.iloc[tr]); rolling.append(metrics(yt.iloc[va], np.maximum(0, m.predict(xt.iloc[va]))))
    scores = frame[test][["timestamp", "cnt", "weathersit", "hr"]].copy()
    scores["predicted"] = pred; scores["lower"] = low; scores["upper"] = high
    scores["absolute_error"] = np.abs(scores.cnt - pred)
    scores.to_csv(out / "holdout_predictions.csv", index=False)
    scores.groupby("hr").absolute_error.agg(["mean", "count"]).to_csv(out / "error_by_hour.csv")
    report = {"selected_model": best, "validation": validation, "test": result, "test_baseline": metrics(y[test], baseline), "rolling_train_folds": rolling, "interval_residual_width": width, "features": FEATURES, "excluded_leakage_columns": ["casual", "registered", "cnt", "instant"], "split": {"train": "2011", "calibration_and_selection": "Jan–Jun 2012", "test": "Jul–Dec 2012"}, "split_sizes": {"train": int(train.sum()), "validation": int(val.sum()), "test": int(test.sum())}, "source_sha256": digest(path), "versions": versions(), "limitations": ["Observed weather is used: conditional demand estimation, not a weather forecast", "Temporal shift breaks exchangeability; intervals report measured coverage, not guaranteed 90% coverage", "Validation selects the model and interval width; a separate calibration period is needed for formal guarantees", "Historical 2011–2012 city data does not validate a new fleet or city"]}
    write_json(out / "metrics.json", report)
    joblib.dump({"model": selected, "features": FEATURES, "interval_width": width}, out / "model.joblib")
    bar_chart({"Historical baseline": report["test_baseline"]["mae"], "Selected model": result["mae"]}, "Hourly demand: future-period holdout", out / "evaluation.png", "Mean absolute error (rentals)")
    return report
