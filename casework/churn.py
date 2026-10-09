"""Retention prioritization with validation-only threshold choice."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss
from sklearn.inspection import permutation_importance
from .common import ROOT, write_json, profile, digest, versions, bar_chart

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["Contract", "InternetService", "PaymentMethod", "TechSupport", "PaperlessBilling"]
FEATURES = NUMERIC + CATEGORICAL


def preprocess():
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])


def score(y, prob, threshold):
    pred = prob >= threshold
    k = max(1, int(np.ceil(len(y) * .2)))
    top = np.argsort(-prob, kind="stable")[:k]
    return {"roc_auc": roc_auc_score(y, prob), "pr_auc": average_precision_score(y, prob), "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0), "brier": brier_score_loss(y, prob), "confusion_matrix": confusion_matrix(y, pred).tolist(), "precision_at_20pct": float(np.asarray(y)[top].mean()), "lift_at_20pct": float(np.asarray(y)[top].mean() / np.asarray(y).mean()), "flagged_share": float(pred.mean())}


def run(path=None, out=None):
    path = Path(path or ROOT / "data/raw/telco.csv")
    out = Path(out or ROOT / "artifacts/retention"); out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(path)
    if raw.customerID.duplicated().any() or raw.Churn.isna().any() or not set(raw.Churn.unique()) <= {"Yes", "No"}:
        raise ValueError("Unique customer IDs and valid target required")
    data = raw[FEATURES].copy()
    for c in NUMERIC:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    y = raw.Churn.eq("Yes").astype(int)
    train, rest = train_test_split(np.arange(len(data)), test_size=.4, stratify=y, random_state=42)
    val, test = train_test_split(rest, test_size=.5, stratify=y.iloc[rest], random_state=42)
    assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)
    models = {
        "base_rate": DummyClassifier(strategy="prior"),
        "logistic": LogisticRegression(C=1, max_iter=1500, random_state=42),
        "forest": RandomForestClassifier(n_estimators=150, max_depth=7, min_samples_leaf=15, random_state=42, n_jobs=2),
    }
    fitted, validation = {}, {}
    for name, estimator in models.items():
        pipeline = Pipeline([("prepare", preprocess()), ("model", estimator)])
        pipeline.fit(data.iloc[train], y.iloc[train])
        fitted[name] = pipeline
        validation[name] = score(y.iloc[val], pipeline.predict_proba(data.iloc[val])[:, 1], .5)
    best = max(["logistic", "forest"], key=lambda n: validation[n]["pr_auc"])
    model = fitted[best]
    val_prob = model.predict_proba(data.iloc[val])[:, 1]
    # Select an operating threshold for a finite outreach budget, without using test labels.
    threshold = float(np.quantile(val_prob, .8))
    probabilities = model.predict_proba(data.iloc[test])[:, 1]
    metrics = score(y.iloc[test], probabilities, threshold)
    baseline = score(y.iloc[test], fitted["base_rate"].predict_proba(data.iloc[test])[:, 1], .5)
    importances = permutation_importance(model, data.iloc[val], y.iloc[val], scoring="average_precision", n_repeats=5, random_state=42, n_jobs=2)
    pd.DataFrame({"feature": FEATURES, "importance_mean": importances.importances_mean, "importance_std": importances.importances_std}).sort_values("importance_mean", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    predictions = pd.DataFrame({"customer_id": raw.customerID.iloc[test].values, "actual_churn": y.iloc[test].values, "risk_score": probabilities, "prioritize_review": probabilities >= threshold})
    predictions.sort_values("risk_score", ascending=False).to_csv(out / "holdout_scores.csv", index=False)
    pd.DataFrame({"row": np.r_[train, val, test], "split": ["train"] * len(train) + ["validation"] * len(val) + ["test"] * len(test)}).to_csv(out / "split_manifest.csv", index=False)
    bins = pd.cut(probabilities, bins=np.linspace(0, 1, 11), include_lowest=True)
    calibration = pd.DataFrame({"bin": bins, "score": probabilities, "actual": y.iloc[test].values}).groupby("bin", observed=True).agg(count=("actual", "size"), mean_score=("score", "mean"), observed_churn=("actual", "mean"))
    calibration.to_csv(out / "calibration.csv")
    report = {"selected_model": best, "threshold": threshold, "threshold_rule": "80th percentile of validation scores; test share can differ", "validation": validation, "test": metrics, "test_baseline": baseline, "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)}, "features": FEATURES, "excluded": [c for c in raw if c not in FEATURES], "data_profile": profile(data), "source_sha256": digest(path), "versions": versions(), "limitations": ["Cross-sectional benchmark: no future observation window or temporal churn validation", "Public demonstration data, not a deployed customer retention system", "No causal claim that contacting high-risk users prevents churn", "Not calibrated to a new business; revalidate label availability and demographics before use"]}
    write_json(out / "metrics.json", report)
    joblib.dump({"model": model, "threshold": threshold, "features": FEATURES, "numeric": NUMERIC}, out / "model.joblib")
    bar_chart({"Baseline PR-AUC": baseline["pr_auc"], "Model PR-AUC": metrics["pr_auc"], "ROC-AUC": metrics["roc_auc"]}, "Retention model: untouched holdout", out / "evaluation.png", "Score")
    return report
