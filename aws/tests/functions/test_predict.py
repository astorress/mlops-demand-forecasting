import json
from unittest.mock import MagicMock

import pytest

from src.features.schema import FEATURES
from src.functions.inference import predict


@pytest.fixture
def record():
    return {
        "timestamp": "2024-12-25T18:00",
        "temperature_c": 29.4,
        "demand_lag_24h": 1180.2,
        "demand_lag_168h": 1150.7,
        "demand_rolling_mean_7d": 1012.3,
    }


def test_derives_calendar_and_calls_endpoint(monkeypatch, record):
    runtime = MagicMock()
    runtime.invoke_endpoint.return_value = json.dumps({
        "predictions": [{
            "demand_kwh": 1100.5
        }],
    }).encode()
    monkeypatch.setattr(predict, "sagemaker_runtime", runtime)

    result = predict.handler({
        "body": json.dumps({
            "records": [record]
        })
    }, None)

    assert result["statusCode"] == 200
    body = json.loads(result["body"])
    assert body["predictions"][0] == {
        "demand_kwh": 1100.5,
        "timestamp": "2024-12-25 18:00:00",
    }
    sent = runtime.invoke_endpoint.call_args.args[1].decode().strip()
    values = dict(zip(FEATURES, sent.split(",")))
    assert len(values) == len(FEATURES)
    # Miércoles de Navidad a las 18 h.
    assert (values["hour"], values["day_of_week"],
            values["is_holiday"]) == ("18", "2", "1")


@pytest.mark.parametrize(
    "body, message",
    [
        ("not json", "Invalid JSON"),
        ("{}", "No records"),
        (json.dumps({
            "timestamp": "ayer"
        }), "missing"),
    ],
)
def test_rejects_invalid_bodies(body, message):
    result = predict.handler({
        "body": body
    }, None)
    assert result["statusCode"] == 400
    assert message in json.loads(result["body"])["message"]


def test_rejects_bad_timestamp(record):
    record["timestamp"] = "ayer"
    result = predict.handler({
        "body": json.dumps(record)
    }, None)
    assert result["statusCode"] == 400
    assert "Invalid timestamp" in json.loads(result["body"])["message"]


def test_rejects_too_many_records(monkeypatch, record):
    monkeypatch.setattr(predict, "MAX_RECORDS", 2)
    result = predict.handler({
        "body": json.dumps([record] * 3)
    }, None)
    assert result["statusCode"] == 400
