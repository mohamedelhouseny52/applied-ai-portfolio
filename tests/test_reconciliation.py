import pandas as pd
import pytest
from casework.reconcile import reconcile, prepare


def record(id, email="", name="Nadia", ip="192.0.2.1", date="2025-01-01", segment="S1"):
    return {"record_id": id, "email": email, "name": name, "ip": ip, "event_date": date, "segment": segment, "user_agent": "Browser A"}


def test_empty_email_is_not_a_duplicate_identity():
    left = pd.DataFrame([record("a"), record("b", name="Karim")])
    clean, duplicate = prepare(left)
    assert len(clean) == 2 and duplicate.empty


def test_email_case_dedup_keeps_file_order():
    a = pd.DataFrame([record("a", "Nadia@example.test"), record("b", "nadia@example.test"), record("c", "karim@example.test")])
    merged, _, _, duplicate = reconcile(a, pd.DataFrame([record("r", "karim@example.test")]))
    assert merged.record_id.tolist() == ["a", "c"]
    assert duplicate.record_id.tolist() == ["b"]


def test_shared_ip_name_and_date_are_not_enough_when_ambiguous():
    a = pd.DataFrame([record("a", "one@example.test"), record("b", "two@example.test")])
    _, decisions, _, _ = reconcile(a, pd.DataFrame([record("r")]))
    assert decisions.iloc[0].status == "review"
    assert decisions.iloc[0].margin == 0


def test_conflicting_email_requires_review_even_with_other_evidence():
    _, decisions, _, _ = reconcile(pd.DataFrame([record("a", "one@example.test")]), pd.DataFrame([record("b", "two@example.test")]))
    assert decisions.iloc[0].status == "review"


def test_unique_exact_email_takes_precedence_over_ambiguous_name():
    a = pd.DataFrame([record("a", "one@example.test"), record("b", "two@example.test")])
    _, decisions, _, _ = reconcile(a, pd.DataFrame([record("r", "one@example.test")]))
    assert decisions.iloc[0].status == "matched"
    assert decisions.iloc[0].left_id == "a"


def test_many_to_one_collision_does_not_silently_overwrite():
    _, decisions, _, _ = reconcile(pd.DataFrame([record("a")]), pd.DataFrame([record("b"), record("c")]))
    assert decisions.status.tolist() == ["review", "review"]


@pytest.mark.parametrize("date", ["not-a-date", "2025-12-01"])
def test_missing_or_distant_date_blocks_fuzzy_auto_match(date):
    _, decisions, _, _ = reconcile(pd.DataFrame([record("a")]), pd.DataFrame([record("b", date=date)]))
    assert decisions.iloc[0].status == "review"


def test_duplicate_record_id_is_rejected():
    with pytest.raises(ValueError):
        prepare(pd.DataFrame([record("a"), record("a")]))
