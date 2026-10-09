import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from casework import reconcile, ingestion, churn, demand, routing
from reconciliation_benchmark import main as reconciliation_benchmark
from build_eda import main as build_eda

if __name__ == "__main__":
    for name, action in [("reconciliation", reconcile.demo), ("ingestion", ingestion.demo), ("retention", churn.run), ("demand", demand.run), ("routing", routing.run)]:
        print(f"Running {name}...", flush=True)
        result = action()
        print(result.get("test", result.get("decisions", result.get("aggregate"))), flush=True)
    reconciliation_benchmark()
    build_eda()
