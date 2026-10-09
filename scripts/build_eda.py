"""Additional diagnostics from the published training splits and holdout outputs."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from casework.common import ROOT, bar_chart
import pandas as pd
import matplotlib.pyplot as plt


def main():
    retention = ROOT / "artifacts/retention"
    raw = pd.read_csv(ROOT / "data/raw/telco.csv")
    split = pd.read_csv(retention / "split_manifest.csv")
    train = raw.iloc[split.loc[split.split.eq("train"), "row"]].copy()
    train["churn"] = train.Churn.eq("Yes").astype(int)
    bar_chart(train.Churn.value_counts().to_dict(), "Training target distribution", retention / "target_distribution.png")
    contracts = train.groupby("Contract").agg(customers=("churn", "size"), churn_rate=("churn", "mean"))
    contracts.to_csv(retention / "training_contract_eda.csv")
    bar_chart(contracts.churn_rate.to_dict(), "Training snapshot: churn by contract", retention / "contract_eda.png", "Observed churn fraction")
    demand = ROOT / "artifacts/demand"
    errors = pd.read_csv(demand / "error_by_hour.csv")
    bar_chart({str(int(row.hr)): row["mean"] for _, row in errors.iterrows()}, "Future-period error by hour", demand / "hourly_errors.png", "Mean absolute error")
    predictions = pd.read_csv(demand / "holdout_predictions.csv")
    sample = predictions.iloc[:168]
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    t = pd.to_datetime(sample.timestamp)
    ax.plot(t, sample.cnt, label="Observed", color="#202b31", linewidth=1.4)
    ax.plot(t, sample.predicted, label="Estimate", color="#256d78", linewidth=1.4)
    ax.fill_between(t, sample.lower, sample.upper, color="#256d78", alpha=.12, label="Residual interval")
    ax.set(title="First 168 observed hours in the test period", ylabel="Hourly rentals")
    ax.legend(); fig.savefig(demand / "sample_week.png", dpi=160); plt.close(fig)
    routing = ROOT / "artifacts/routing"
    routes = pd.read_csv(routing / "holdout_predictions.csv")
    confused = routes[routes.actual.ne(routes.predicted)].groupby(["actual", "predicted"]).size().sort_values(ascending=False)
    confused.rename("count").to_csv(routing / "confusion_pairs.csv")
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    pairs = confused.head(8).sort_values()
    ax.barh([a.replace("_", " ") + " → " + b.replace("_", " ") for a, b in pairs.index], pairs.values, color="#256d78")
    ax.set(title="Most frequent held-out intent confusions", xlabel="Query count")
    fig.savefig(routing / "confusions.png", dpi=160); plt.close(fig)
    print("Generated training EDA, temporal diagnostics and NLP confusion analysis.")


if __name__ == "__main__":
    main()
