from pathlib import Path
import hashlib
import json
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions():
    return {p: importlib.metadata.version(p) for p in ["pandas", "numpy", "scikit-learn", "flask", "joblib"]}


def profile(frame):
    return {"rows": len(frame), "columns": len(frame.columns), "duplicate_rows": int(frame.duplicated().sum()),
            "missing": {c: int((frame[c].isna() | frame[c].map(lambda v: isinstance(v, str) and not v.strip())).sum()) for c in frame}, "types": frame.dtypes.astype(str).to_dict()}


def bar_chart(values, title, path, ylabel="Count"):
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.bar(list(values), list(values.values()), color="#256d78")
    ax.set(title=title, ylabel=ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=160)
    plt.close(fig)

