"""Build measured case summaries and executable analysis walkthroughs."""
from pathlib import Path
import base64
import contextlib
import html
import io
import json
import sys
import importlib.metadata
import platform
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from casework.common import ROOT
import matplotlib.pyplot as plt

CASES = [
    {"id": "01-record-reconciliation", "artifact": "reconciliation", "title": "Auditable record reconciliation", "subtitle": "Merge imperfect exports without hiding uncertainty", "problem": "Two systems export overlapping records, but one source lacks a reliable email key. A naive join drops matches, duplicates records or attaches one person's attributes to another.", "approach": "Normalize identity fields, block candidates by email/name/IP, apply explicit evidence rules, require a separation margin and preserve a review queue. Conflicting emails, uncertain dates and multiple records claiming the same target never silently overwrite the source.", "deliverables": "Order-preserving merged CSV; duplicate ledger; per-candidate evidence; match decisions; source hashes and policy audit.", "limits": "Synthetic fixtures exercise edge cases. Evidence scores are policy weights, not calibrated identity probabilities. Matching quality on a real dataset requires reviewed ground truth and an agreed matching policy.", "chart": "outcomes.png", "flow": "CSV exports → normalization → candidate blocks → evidence rules → matched / review / unmatched → audit + output"},
    {"id": "02-event-ingestion", "artifact": "ingestion", "title": "Replay-safe webhook analytics", "subtitle": "Accept signed events and count each payment once", "problem": "Webhook senders retry events. Duplicate deliveries, conflicting IDs and inconsistent money handling can corrupt dashboard totals even when the endpoint appears to work.", "approach": "Authenticate the raw body with HMAC and a timestamp window, validate the schema, store a canonical hash and event ID, and commit receipt plus outbox together. A separate worker atomically updates daily currency totals and marks work complete.", "deliverables": "Flask endpoint; durable event ledger; transactional outbox; retry/dead-letter state; UTC daily aggregates; duplicate/concurrency tests.", "limits": "Local SQLite/WAL reference implementation with one payment schema and USD/EUR/GBP. PostgreSQL deployment, queue backoff, external delivery, load tests and additional provider contracts are future integration work. Atomicity covers this local database, not external side effects.", "chart": None, "flow": "Signed HTTP event → freshness + schema → event ledger + outbox → transactional worker → daily analytics"},
    {"id": "03-retention", "artifact": "retention", "title": "Retention outreach prioritization", "subtitle": "Turn model scores into a bounded review queue", "problem": "A retention team has limited outreach capacity. Accuracy alone does not show whether the first customers reviewed are more likely to churn than a random contact list.", "approach": "Compare a base-rate predictor, logistic regression and a bounded random forest. Fit preprocessing inside each training pipeline, select on validation PR-AUC, choose the outreach threshold from validation only, then evaluate the untouched test set.", "deliverables": "Ranked customer scores; PR-AUC/ROC-AUC; confusion matrix; lift at 20%; calibration bins; permutation importance; split manifest; schema-validated scoring API.", "limits": "Public cross-sectional demonstration data. The split validates benchmark generalization, not future churn. Scores do not prove outreach reduces churn. Sensitive demographic features are excluded; a live business needs temporal labels, fairness review and threshold revalidation.", "chart": "evaluation.png", "flow": "Customer snapshot → train/validation/test → fitted preprocessing → model selection → capacity threshold → holdout scores + API"},
    {"id": "04-demand", "artifact": "demand", "title": "Hourly demand estimation", "subtitle": "Plan with a baseline, error slices and uncertainty", "problem": "Rental operations need hourly demand estimates for staffing and availability. Random train/test splits can hide time drift and make a model look better than it will on a future period.", "approach": "Train on 2011, select and estimate residual widths on Jan–Jun 2012, and test on Jul–Dec 2012. Compare Poisson and boosted regressors to a historical hour/weekday baseline, add rolling training checks and report error by hour.", "deliverables": "Future-period predictions; baseline comparison; MAE/RMSE/WAPE; empirical interval coverage; rolling validation; hour-level error CSV and reproducible model artifact.", "limits": "Uses observed weather: conditional estimation rather than a weather forecast. Temporal drift means the nominal 90% intervals have no coverage guarantee. The measured coverage is reported. No claim of performance in a new city or fleet.", "chart": "evaluation.png", "flow": "Hourly history → chronological split → baseline + candidate models → validation selection → frozen model → future-period error audit"},
    {"id": "05-request-routing", "artifact": "routing", "title": "Customer request intent routing", "subtitle": "Classify 77 intents and review uncertain requests", "problem": "Support queues receive short, ambiguous customer questions. A classifier that always chooses a category can route vague, sensitive or unrelated requests incorrectly.", "approach": "Combine word and character TF-IDF features with a linear probabilistic classifier. Remove normalized duplicate conflicts and train/test overlap. Select a confidence/margin gate on validation, add lexical similarity and sensitive-request review checks, and return structured routing results with reference IDs.", "deliverables": "77-intent model; macro-F1 and routed accuracy/coverage; per-intent report; out-of-domain probes; partial redaction; ranked alternatives; reference retrieval; authenticated routing API and duplicate-safe ticket receipt.", "limits": "English BANKING77 only. Scores are not independently calibrated. Additional inference checks reduce routing coverage beyond the benchmark score gate. Regex redaction is partial. No generated answers, transformer fine-tuning, multilingual claim or production helpdesk integration.", "chart": "evaluation.png", "flow": "Request text → partial redaction → intent + alternatives → review gates → team recommendation + reference IDs"},
]


def load(case):
    filename = "audit.json" if case["artifact"] in {"reconciliation", "ingestion"} else "metrics.json"
    return json.loads((ROOT / "artifacts" / case["artifact"] / filename).read_text(encoding="utf-8"))


def headline(case, report):
    if case["artifact"] == "reconciliation":
        bench = json.loads((ROOT / "artifacts/reconciliation/benchmark/audit.json").read_text())
        return f'{bench["source_a"]["rows"]:,} source rows · {bench["source_b"]["rows"]:,} enrichment records · {bench["decisions"]["review"]} review cases'
    if case["artifact"] == "ingestion":
        return "Duplicate and worker replay leave totals unchanged"
    if case["artifact"] == "retention":
        return f'{report["test"]["lift_at_20pct"]:.2f}× lift at 20% contact capacity'
    if case["artifact"] == "demand":
        improvement = 1 - report["test"]["mae"] / report["test_baseline"]["mae"]
        return f'{improvement:.1%} lower holdout MAE than the historical baseline'
    return f'{report["test"]["auto_route_accuracy"]:.1%} score-gated accuracy · {report["test"]["auto_route_coverage"]:.1%} coverage'


def notebook(case):
    artifact = case["artifact"]
    code = f'''from pathlib import Path
import json
root = Path.cwd()
if root.name == "notebooks":
    root = root.parent
filename = "{'audit.json' if artifact in {'reconciliation', 'ingestion'} else 'metrics.json'}"
report = json.loads((root / "artifacts/{artifact}" / filename).read_text(encoding="utf-8"))
print(json.dumps(report.get("test", report.get("decisions", report.get("aggregate"))), indent=2))'''
    csv = {"reconciliation": "decisions.csv", "ingestion": "daily_revenue.csv", "retention": "holdout_scores.csv", "demand": "holdout_predictions.csv", "routing": "holdout_predictions.csv"}[artifact]
    code2 = f'''import pandas as pd
result = pd.read_csv(root / "artifacts/{artifact}/{csv}")
print(result.head(8).to_string(index=False))'''
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [f'# {case["title"]}\n\n{case["problem"]}\n\n{case["approach"]}\n\nRebuild source artifacts with `python scripts/run_all.py` from the repository root. This walkthrough inspects the measured run.']}]
    ns = {}
    # Execute only the two authored analysis cells; capture their actual output.
    import os
    prior = Path.cwd()
    try:
        os.chdir(ROOT)
        for i, source in enumerate([code, code2], 1):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(source, "<analysis-cell>", "exec"), ns)
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": i, "source": source.splitlines(keepends=True), "outputs": [{"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(keepends=True)}]})
    finally:
        os.chdir(prior)
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Interpretation and limits\n\n" + case["limits"]]})
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": platform.python_version()}}, "nbformat": 4, "nbformat_minor": 4}


def main():
    reports = ROOT / "reports"; reports.mkdir(exist_ok=True)
    notebooks = ROOT / "notebooks"; notebooks.mkdir(exist_ok=True)
    slides = []
    for case in CASES:
        report = load(case); project = ROOT / "projects" / case["id"]; project.mkdir(parents=True, exist_ok=True)
        result = headline(case, report)
        md = f'# {case["title"]}\n\n{case["subtitle"]}.\n\n## Business problem\n\n{case["problem"]}\n\n## Implementation\n\n{case["approach"]}\n\n```text\n{case["flow"]}\n```\n\n## Measured result\n\n**{result}.** Results are from the included reference run.\n\n## Deliverables\n\n{case["deliverables"]}\n\n## Limitations\n\n{case["limits"]}\n\n## Reproduce and inspect\n\nRun `python scripts/run_all.py` from the repository root. Open `artifacts/{case["artifact"]}/` for evidence and `notebooks/{case["id"]}.ipynb` for the executed analysis walkthrough. The source module is `casework/{ {"reconciliation":"reconcile", "retention":"churn", "routing":"routing", "demand":"demand", "ingestion":"ingestion"}[case["artifact"]]}.py`.\n'
        if case["artifact"] == "ingestion":
            md += '\nThe supported POST endpoint is `/webhooks/payments`. Supply `X-Timestamp` (Unix seconds), `X-Signature` (hex HMAC-SHA256 of timestamp + dot + raw request body), and JSON with `event_id`, `event_type="payment.received"`, and `data={"occurred_at":"2025-01-01T12:00:00Z","currency":"USD","amount":"12.34"}`. Run `process_pending(database_path)` from `casework.ingestion` as a separate scheduled worker. Stale messages must be re-signed by the sender; the event ID protects against recounting. Dead items require operator investigation, not automatic deletion.\n'
        project.joinpath("README.md").write_text(md, encoding="utf-8")
        notebooks.joinpath(case["id"] + ".ipynb").write_text(json.dumps(notebook(case), indent=2), encoding="utf-8")
        chart = ''
        if case["chart"]:
            image = base64.b64encode((ROOT / "artifacts" / case["artifact"] / case["chart"]).read_bytes()).decode()
            chart = f'<img alt="Measured evaluation chart" src="data:image/png;base64,{image}">'
        additional = {"retention": ["target_distribution.png", "contract_eda.png"], "demand": ["sample_week.png", "hourly_errors.png"], "routing": ["confusions.png"]}.get(case["artifact"], [])
        for filename in additional:
            if (ROOT / "artifacts" / case["artifact"] / filename).exists():
                image = base64.b64encode((ROOT / "artifacts" / case["artifact"] / filename).read_bytes()).decode()
                chart += f'<img alt="Additional diagnostic: {filename}" src="data:image/png;base64,{image}">'
        slides.append(f'<section id="{case["id"]}"><div class="eyebrow">{case["artifact"].upper()}</div><h2>{html.escape(case["title"])}</h2><p class="lead">{html.escape(case["subtitle"])}</p><div class="result">{html.escape(result)}</div><h3>Problem</h3><p>{html.escape(case["problem"])}</p><h3>Method</h3><p>{html.escape(case["approach"])}</p><div class="flow">{html.escape(case["flow"])}</div>{chart}<h3>Deliverables</h3><p>{html.escape(case["deliverables"])}</p><details><summary>Evaluation details</summary><pre>{html.escape(json.dumps(report, indent=2))}</pre></details><h3>Scope and limits</h3><p>{html.escape(case["limits"])}</p></section>')
        # Business-focused cover assets; all headline values come from recorded runs.
        fig = plt.figure(figsize=(12, 7), facecolor="#f5f3ec")
        fig.text(.07, .84, "CASEWORK  /  " + case["artifact"].upper(), color="#256d78", fontsize=15, weight="bold")
        import textwrap
        fig.text(.07, .60, textwrap.fill(case["title"], 33), fontsize=34, color="#202b31", weight="bold")
        fig.text(.07, .43, textwrap.fill(case["subtitle"], 55), fontsize=19, color="#48545b")
        fig.text(.07, .22, textwrap.fill(result, 62), fontsize=19, color="#256d78", weight="bold")
        fig.text(.07, .07, "Independent reference implementation · Reproducible evidence · Explicit limitations", fontsize=12, color="#596268")
        fig.savefig(project / "cover.png", dpi=160, facecolor=fig.get_facecolor()); plt.close(fig)
    css = 'body{margin:0;background:#f5f3ec;color:#202b31;font:17px/1.65 system-ui,sans-serif}main{max-width:980px;margin:auto;padding:60px 28px}h1{font-size:48px;line-height:1.1;max-width:750px}h2{font-size:34px;line-height:1.2}h3{font-size:19px;margin-bottom:5px}.lead{font-size:21px;color:#52616a}.eyebrow{letter-spacing:.16em;font-size:13px;color:#256d78;font-weight:700}section{border-top:1px solid #bdc6c3;padding:45px 0;margin-top:35px}.result{font-size:23px;color:#256d78;padding:12px 0;font-weight:700}.flow{background:#e8eeea;padding:20px;margin:25px 0;font-size:15px}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;padding:20px;font-size:13px}nav{display:flex;flex-wrap:wrap;gap:15px}a{color:#256d78}summary{cursor:pointer}footer{border-top:1px solid #bdc6c3;margin-top:40px;padding-top:20px;font-size:14px}@media print{details{display:none}section{break-before:page}}'
    nav = ''.join(f'<a href="#{c["id"]}">{html.escape(c["artifact"].capitalize())}</a>' for c in CASES)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Casework — Applied Data & AI</title><style>' + css + '</style><main><div class="eyebrow">APPLIED DATA & AI</div><h1>Reliable workflows.<br>Evidence you can inspect.</h1><p class="lead">Five focused systems for data operations, predictive analytics and customer-request routing.</p><p>Independent reference implementations using public data and synthetic examples. Results are measured on the included runs; they are not client outcomes.</p><nav>' + nav + '</nav>' + ''.join(slides) + '<footer>Source, datasets, attribution and reproducibility instructions are included in README.md. Models need business-specific validation before deployment.</footer></main></html>'
    reports.joinpath("index.html").write_text(page, encoding="utf-8")
    # Pin the installed dependency closure without unrelated machine packages.
    from packaging.requirements import Requirement
    requested = ["numpy", "pandas", "scikit-learn", "matplotlib", "flask", "joblib", "pytest"]
    found = {}
    while requested:
        name = requested.pop()
        if name.lower() in found:
            continue
        distribution = importlib.metadata.distribution(name)
        found[name.lower()] = f'{distribution.metadata["Name"]}=={distribution.version}'
        for item in distribution.requires or []:
            requirement = Requirement(item)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                requested.append(requirement.name)
    ROOT.joinpath("requirements-lock.txt").write_text('\n'.join(sorted(found.values(), key=str.lower)) + '\n', encoding="utf-8")
    print("Built 5 case summaries, covers, executed notebooks and an offline report.")


if __name__ == "__main__":
    main()
