import csv
import io
from unittest.mock import MagicMock

import pytest

from src.functions.ingestion import ingest_dataset
from src.utilities.errors import RequestValidationError


@pytest.fixture
def s3(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(ingest_dataset, "s3", mock)
    monkeypatch.setattr(ingest_dataset, "DATA_LAKE_BUCKET_NAME", "lake")
    return mock


def test_writes_raw_partition(s3):
    result = ingest_dataset.handler(
        {
            "ingestion_date": "2026-01-02",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        }, None)

    bucket, key, body = s3.put_object.call_args.args[:3]
    assert bucket == "lake"
    assert key.endswith("/ingestion_date=2026-01-02/demand.csv")
    rows = list(csv.DictReader(io.StringIO(body.decode())))
    assert len(rows) == result["rows"] == 31 * 24
    assert list(rows[0]) == ["timestamp", "demand_kwh", "temperature_c"]
    assert result["last_timestamp"] == "2024-01-31 23:00:00"


def test_rejects_inverted_range(s3):
    with pytest.raises(RequestValidationError):
        ingest_dataset.handler(
            {
                "start_date": "2024-02-01",
                "end_date": "2024-01-01",
            }, None)
    s3.put_object.assert_not_called()
