import json
import joblib
import numpy as np
import pandas as pd
import pytest
from casework.common import ROOT
from casework.churn import FEATURES
from casework.demand import FEATURES as DEMAND_FEATURES
from casework.routing import redact, infer
from casework.serve import create_app


def test_retention_split_disjoint_and_target_not_feature():
    manifest = pd.read_csv(ROOT / "artifacts/retention/split_manifest.csv")
    assert manifest.row.is_unique
    assert set(manifest.split) == {"train", "validation", "test"}
    assert not {"Churn", "customerID"} & set(FEATURES)


def test_future_period_predictions_exclude_target_components():
    report = json.loads((ROOT / "artifacts/demand/metrics.json").read_text())
    predictions = pd.read_csv(ROOT / "artifacts/demand/holdout_predictions.csv")
    assert not {"casual", "registered", "cnt", "instant"} & set(DEMAND_FEATURES)
    assert pd.to_datetime(predictions.timestamp).min() >= pd.Timestamp("2012-07-01")
    assert (predictions.lower >= 0).all() and (predictions.upper >= predictions.lower).all()
    assert 0 <= report["test"]["empirical_interval_coverage"] <= 1


def test_redaction_masks_example_email_and_long_number():
    text = redact("My email is a@example.test and phone is +201012345678")
    assert "a@example.test" not in text and "201012345678" not in text


@pytest.fixture(scope="module")
def router():
    return joblib.load(ROOT / "artifacts/routing/model.joblib")


def test_out_of_domain_and_short_query_go_to_review(router):
    assert infer(router, "Please recommend a pasta recipe")["action"] == "manual_review"
    assert infer(router, "help")["action"] == "manual_review"


def test_sensitive_requests_never_route_automatically(router):
    result = infer(router, "My debit card was stolen")
    assert result["urgency"] == "high" and result["action"] == "manual_review"


@pytest.mark.parametrize("text", ["", " ", None, "x" * 4001])
def test_invalid_text_rejected(router, text):
    with pytest.raises(ValueError):
        infer(router, text)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    return create_app(api_key="test-key", ticket_db=tmp_path_factory.mktemp("tickets") / "tickets.sqlite").test_client()


def test_api_auth_and_invalid_contracts(client):
    assert client.post("/route", json={"text": "card delivery"}).status_code == 401
    headers = {"X-API-Key": "test-key"}
    assert client.post("/route", json={"text": ""}, headers=headers).status_code == 422
    assert client.post("/predict/retention", json={"records": [{"Churn": "Yes"}]}, headers=headers).status_code == 422


def test_retention_api_valid_prediction_and_malformed_values(client):
    headers = {"X-API-Key": "test-key"}
    row = {"tenure": 12, "MonthlyCharges": 60, "TotalCharges": 720, "Contract": "Month-to-month", "InternetService": "DSL", "PaymentMethod": "Electronic check", "TechSupport": "No", "PaperlessBilling": "Yes"}
    response = client.post("/predict/retention", json={"records": [row]}, headers=headers)
    assert response.status_code == 200
    assert 0 <= response.json["predictions"][0]["risk_score"] <= 1
    for invalid in ["garbage", -1, None, True]:
        row["MonthlyCharges"] = invalid
        assert client.post("/predict/retention", json={"records": [row]}, headers=headers).status_code == 422


def test_routing_api_structured_response(client):
    response = client.post("/route", json={"text": "My card has not arrived yet"}, headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    assert {"intent", "action", "alternatives", "references", "review_reasons"} <= set(response.json)
    assert np.isfinite(response.json["model_score"])


def test_ticket_duplicate_reuses_result_and_conflicting_id_rejected(client):
    headers = {"X-API-Key": "test-key"}
    payload = {"ticket_id": "T001", "text": "My card has not arrived yet"}
    first = client.post("/tickets", json=payload, headers=headers)
    second = client.post("/tickets", json=payload, headers=headers)
    assert first.status_code == 201 and second.status_code == 200
    assert first.json["result"] == second.json["result"]
    payload["text"] = "I cannot withdraw cash"
    assert client.post("/tickets", json=payload, headers=headers).status_code == 409
