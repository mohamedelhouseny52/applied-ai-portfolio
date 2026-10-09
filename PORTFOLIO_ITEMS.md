# Project descriptions

These descriptions identify independent reference implementations. Use the associated cover and report with each entry; benchmark values are measured offline results.

## Auditable CSV Reconciliation & Data Quality Pipeline

**Role:** Python data processing and validation

Built a reusable reconciliation pipeline for inconsistent exports with incomplete shared identifiers. The workflow normalizes fields, generates blocked candidates, records matching evidence and routes uncertain records to review. It preserves source order, prevents conflicting enrichments and produces merged data, duplicate logs and an audit summary. A controlled 2,070-record synthetic fixture checks ambiguous names, conflicting emails, invalid dates and one-to-many collisions.

**Skills:** Python, pandas, Data Cleaning, Data Processing, Data Validation

**Assets:** `projects/01-record-reconciliation/cover.png`, case summary, `artifacts/reconciliation/benchmark/` and executed notebook.

## Signed Webhook Ingestion & Reliable Analytics

**Role:** Python API and data pipeline implementation

Implemented signed webhook receipt with timestamp validation, strict event contracts and duplicate/conflict protection. Events and work items commit together through a transactional outbox. A worker builds UTC daily aggregates using integer currency units and records retry/dead-letter state. Replay and concurrent-delivery tests verify that duplicate payments are counted once. The reference implementation runs locally with Flask and SQLite/WAL.

**Skills:** Python, Flask, REST API, SQL, Data Engineering

**Assets:** `projects/02-event-ingestion/cover.png`, case summary, replay audit, daily revenue CSV and tests.

## Customer Retention Scoring & Model Evaluation

**Role:** Classification, feature preprocessing and scoring API

Built a retention-prioritization workflow using public Telco demonstration data. Compared logistic regression and a bounded random forest against a base-rate predictor, with preprocessing fitted inside each training pipeline. Model selection and the contact-capacity threshold use validation data; the untouched test set achieved 2.40× lift at 20% review capacity. Deliverables include ranked scores, PR-AUC/ROC-AUC, calibration bins, feature importance, split provenance and a validated Flask scoring endpoint. Results are benchmark evidence, not measured retention savings.

**Skills:** Machine Learning, scikit-learn, Classification, Feature Engineering, Model Evaluation

**Assets:** `projects/03-retention/cover.png`, evaluation chart, model report, score CSV and executed notebook.

## Hourly Demand Estimation & Uncertainty Analysis

**Role:** Regression and time-ordered model validation

Developed an hourly rental-demand workflow using UCI Bike Sharing data. Chronological training, selection and test periods prevent future observations from entering model fitting. Compared Poisson and boosted regressors with an hour/weekday historical baseline, added rolling checks and audited errors by hour. The selected model achieved 94.1 rentals MAE against 125.7 for the baseline, a 25.2% reduction on the future-period holdout. Observed weather makes this conditional demand estimation; interval coverage and time-drift limitations are documented.

**Skills:** Regression, Python, Data Analysis, scikit-learn, Model Evaluation

**Assets:** `projects/04-demand/cover.png`, future-period chart, error slices, interval report and executed notebook.

## NLP Customer Request Routing & Review Workflow

**Role:** Text classification, reference retrieval and API implementation

Built an English customer-request router for 77 banking intents using public BANKING77 data. Word and character features support short/noisy queries, while validation-selected score and margin gates keep uncertain requests in a review path. On a deduplicated held-out set, the score gate achieved 95.9% accuracy at 85.9% coverage. Additional inference rules check reference similarity, insufficient context and sensitive requests. The API returns structured categories, alternatives and reference IDs, partially redacts common identifiers, and protects against duplicate ticket processing.

**Skills:** Natural Language Processing, Text Classification, Python, scikit-learn, REST API

**Assets:** `projects/05-request-routing/cover.png`, per-intent report, probe examples, holdout chart and executed notebook.
