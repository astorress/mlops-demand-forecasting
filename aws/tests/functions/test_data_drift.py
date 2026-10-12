import base64
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.features.schema import FEATURES, MONITORED_FEATURES
from src.functions.monitoring import data_drift
from src.monitoring.drift import build_profile


def _capture(rows: list, base64_encoded: bool = False) -> str:
    body = "\n".join(",".join(str(row[name])
                              for name in FEATURES)
                     for row in rows)
    data = base64.b64encode(body.encode()).decode() if base64_encoded else body
    return json.dumps({
        "captureData": {
            "endpointInput": {
                "observedContentType": "text/csv",
                "mode": "INPUT",
                "data": data,
                "encoding": "BASE64" if base64_encoded else "CSV",
            },
        },
    })


@pytest.fixture
def rows(curated_df):
    return curated_df.to_dict("records")


@pytest.fixture
def aws(monkeypatch, rows):
    s3 = MagicMock()
    cloudwatch = MagicMock()
    profile = build_profile(rows, MONITORED_FEATURES, [])
    s3.read_json.side_effect = lambda uri: (profile
                                            if "statistics" in uri else {
                                                "psi_threshold": 0.2,
                                                "min_samples": 30,
                                            })
    s3.list_keys.return_value = ["capture/a.jsonl", "capture/b.jsonl"]
    s3.read_text.side_effect = [
        _capture(rows[:2000:5]) + "\n",
        _capture(rows[2000:4000:5], base64_encoded=True) + "\n",
    ]
    monkeypatch.setattr(data_drift, "s3", s3)
    monkeypatch.setattr(data_drift, "cloudwatch", cloudwatch)
    monkeypatch.setattr(data_drift, "ENDPOINT_NAME", "demand-forecasting")
    monkeypatch.setattr(data_drift, "CAPTURE_PREFIX", "data-capture/df")
    monkeypatch.setattr(data_drift, "LOOKBACK_HOURS", 1)
    monkeypatch.setattr(data_drift, "BASELINE_STATISTICS_URI",
                        "s3://b/statistics.json")
    monkeypatch.setattr(data_drift, "BASELINE_CONSTRAINTS_URI",
                        "s3://b/constraints.json")
    return {
        "s3": s3,
        "cloudwatch": cloudwatch,
    }


def test_capture_prefixes_cover_previous_hours(monkeypatch):
    monkeypatch.setattr(data_drift, "CAPTURE_PREFIX", "data-capture/df")
    monkeypatch.setattr(data_drift, "ENDPOINT_NAME", "demand-forecasting")
    monkeypatch.setattr(data_drift, "LOOKBACK_HOURS", 2)
    now = datetime(2026, 10, 5, 3, 7, tzinfo=timezone.utc)
    assert data_drift.capture_prefixes(now) == [
        "data-capture/df/demand-forecasting/AllTraffic/2026/10/05/01/",
        "data-capture/df/demand-forecasting/AllTraffic/2026/10/05/02/",
    ]


def test_rows_from_csv_and_base64_captures(rows):
    plain = data_drift.rows_from_capture(_capture(rows[:3]))
    encoded = data_drift.rows_from_capture(_capture(rows[:3], True))
    assert len(plain) == 3
    assert plain == encoded
    assert set(plain[0]) == set(FEATURES)


def test_handler_reports_and_publishes_metrics(aws):
    result = data_drift.handler({}, None)

    assert result["status"] == "ok"
    report = json.loads(aws["s3"].put_object.call_args.args[2])
    assert report["samples"] == 800
    assert set(report["features"]) == set(MONITORED_FEATURES)
    metrics = aws["cloudwatch"].put_metrics.call_args.args[1]
    names = {metric["MetricName"]
             for metric in metrics}
    assert {"MaxFeatureDrift", "FeatureDrift", "SampleCount"} <= names


def test_handler_without_traffic_only_counts_samples(aws):
    aws["s3"].list_keys.return_value = []
    result = data_drift.handler({}, None)
    assert result["status"] == "insufficient_data"
    metrics = aws["cloudwatch"].put_metrics.call_args.args[1]
    assert [metric["MetricName"] for metric in metrics] == ["SampleCount"]
