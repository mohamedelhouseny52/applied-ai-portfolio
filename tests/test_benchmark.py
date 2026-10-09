import json
from casework.common import ROOT


def test_controlled_fixture_has_no_false_automatic_links_and_preserves_rows():
    report = json.loads((ROOT / "artifacts/reconciliation/benchmark/audit.json").read_text())
    checks = report["controlled_fixture_checks"]
    assert checks["false_auto_links"] == 0
    assert checks["status_agreement"] == 1
    assert report["output_rows"] == report["source_a"]["rows"] - report["duplicates_removed"]
