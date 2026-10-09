import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import pytest
from casework.ingestion import create_app, sign, process_pending

NOW = 1700000000
SECRET = "test-key"


def payment(id="p1", amount="12.34", currency="USD"):
    return {"event_id": id, "event_type": "payment.received", "data": {"occurred_at": "2025-01-01T23:30:00-02:00", "amount": amount, "currency": currency}}


def send(app, payload, stamp=NOW, key=SECRET):
    body = json.dumps(payload).encode()
    return app.test_client().post("/webhooks/payments", data=body, headers={"X-Timestamp": str(stamp), "X-Signature": sign(key, stamp, body), "Content-Type": "application/json"})


@pytest.fixture
def service(tmp_path):
    path = tmp_path / "events.sqlite"
    return create_app(path, SECRET, lambda: NOW), path


def test_duplicate_receipt_and_worker_replay_count_money_once(service):
    app, path = service
    assert send(app, payment()).status_code == 202
    assert send(app, payment()).status_code == 200
    assert process_pending(path) == [("2025-01-02", "USD", 1234, 1)]
    assert process_pending(path) == [("2025-01-02", "USD", 1234, 1)]


def test_conflicting_reuse_is_rejected(service):
    app, _ = service
    send(app, payment())
    assert send(app, payment(amount="50.00")).status_code == 409


def test_currency_totals_are_separate(service):
    app, path = service
    send(app, payment()); send(app, payment("p2", "10.00", "EUR"))
    assert len(process_pending(path)) == 2


@pytest.mark.parametrize("stamp,key", [(NOW - 301, SECRET), (NOW, "wrong")])
def test_expired_or_invalid_signature_is_rejected(service, stamp, key):
    app, path = service
    assert send(app, payment(), stamp, key).status_code == 401
    assert process_pending(path) == []


@pytest.mark.parametrize("amount", ["NaN", "-1.00", "1.001", "Infinity", 1.5, True])
def test_invalid_money_is_rejected_without_event(service, amount):
    app, path = service
    assert send(app, payment(amount=amount)).status_code == 422
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


def test_concurrent_duplicate_delivery(service):
    app, path = service
    with ThreadPoolExecutor(max_workers=4) as pool:
        codes = list(pool.map(lambda _: send(app, payment()).status_code, range(8)))
    assert codes.count(202) == 1 and codes.count(200) == 7
    assert process_pending(path)[0][-1] == 1


def test_corrupted_outbox_retries_then_dead_letters(service):
    app, path = service
    send(app, payment())
    with sqlite3.connect(path) as db:
        db.execute("UPDATE events SET payload='{}'")
    for _ in range(4):
        assert process_pending(path) == []
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT state,attempts FROM outbox").fetchone() == ("dead", 3)
