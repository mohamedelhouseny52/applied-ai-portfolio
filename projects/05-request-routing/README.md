# Customer request intent routing

Classify 77 intents and review uncertain requests.

## Business problem

Support queues receive short, ambiguous customer questions. A classifier that always chooses a category can route vague, sensitive or unrelated requests incorrectly.

## Implementation

Combine word and character TF-IDF features with a linear probabilistic classifier. Remove normalized duplicate conflicts and train/test overlap. Select a confidence/margin gate on validation, add lexical similarity and sensitive-request review checks, and return structured routing results with reference IDs.

```text
Request text → partial redaction → intent + alternatives → review gates → team recommendation + reference IDs
```

## Measured result

**95.9% score-gated accuracy · 85.9% coverage.** Results are from the included reference run.

## Deliverables

77-intent model; macro-F1 and routed accuracy/coverage; per-intent report; out-of-domain probes; partial redaction; ranked alternatives; reference retrieval; authenticated routing API and duplicate-safe ticket receipt.

## Limitations

English BANKING77 only. Scores are not independently calibrated. Additional inference checks reduce routing coverage beyond the benchmark score gate. Regex redaction is partial. No generated answers, transformer fine-tuning, multilingual claim or production helpdesk integration.

## Reproduce and inspect

Run `python scripts/run_all.py` from the repository root. Open `artifacts/routing/` for evidence and `notebooks/05-request-routing.ipynb` for the executed analysis walkthrough. The source module is `casework/routing.py`.
