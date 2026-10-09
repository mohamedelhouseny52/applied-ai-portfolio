import argparse
import json
from . import reconcile, ingestion, churn, demand, routing


def main():
    parser = argparse.ArgumentParser(description="Run an auditable data workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    merge = sub.add_parser("reconcile")
    merge.add_argument("--left", required=True)
    merge.add_argument("--right", required=True)
    merge.add_argument("--out", required=True)
    sub.add_parser("demo")
    train = sub.add_parser("train")
    train.add_argument("task", choices=["retention", "demand", "routing"])
    worker = sub.add_parser("worker")
    worker.add_argument("--db", required=True)
    args = parser.parse_args()
    if args.command == "reconcile":
        result = reconcile.run(args.left, args.right, args.out)
    elif args.command == "worker":
        result = ingestion.process_pending(args.db)
    elif args.command == "train":
        result = {"retention": churn.run, "demand": demand.run, "routing": routing.run}[args.task]()
    else:
        result = {"reconciliation": reconcile.demo(), "ingestion": ingestion.demo()}
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
