# Replay-safe webhook analytics

Accept signed events and count each payment once.

## Business problem

Webhook senders retry events. Duplicate deliveries, conflicting IDs and inconsistent money handling can corrupt dashboard totals even when the endpoint appears to work.

## Implementation

Authenticate the raw body with HMAC and a timestamp window, validate the schema, store a canonical hash and event ID, and commit receipt plus outbox together. A separate worker atomically updates daily currency totals and marks work complete.

```text
Signed HTTP event → freshness + schema → event ledger + outbox → transactional worker → daily analytics
```

## Measured result

**Duplicate and worker replay leave totals unchanged.** Results are from the included reference run.

## Deliverables

Flask endpoint; durable event ledger; transactional outbox; retry/dead-letter state; UTC daily aggregates; duplicate/concurrency tests.

## Limitations

Local SQLite/WAL reference implementation with one payment schema and USD/EUR/GBP. PostgreSQL deployment, queue backoff, external delivery, load tests and additional provider contracts are future integration work. Atomicity covers this local database, not external side effects.

## Reproduce and inspect

Run `python scripts/run_all.py` from the repository root. Open `artifacts/ingestion/` for evidence and `notebooks/02-event-ingestion.ipynb` for the executed analysis walkthrough. The source module is `casework/ingestion.py`.

The supported POST endpoint is `/webhooks/payments`. Supply `X-Timestamp` (Unix seconds), `X-Signature` (hex HMAC-SHA256 of timestamp + dot + raw request body), and JSON with `event_id`, `event_type="payment.received"`, and `data={"occurred_at":"2025-01-01T12:00:00Z","currency":"USD","amount":"12.34"}`. Run `process_pending(database_path)` from `casework.ingestion` as a separate scheduled worker. Stale messages must be re-signed by the sender; the event ID protects against recounting. Dead items require operator investigation, not automatic deletion.
