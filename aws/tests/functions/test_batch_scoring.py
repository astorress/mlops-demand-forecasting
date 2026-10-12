from unittest.mock import MagicMock

import pytest

from src.functions.inference import batch_scoring
from src.utilities.errors import NoApprovedModelError

ARN = "arn:aws:sagemaker:us-east-1:1:model-package/demand-forecasting/3"


@pytest.fixture
def aws(monkeypatch):
    mocks = {
        name: MagicMock()
        for name in ("athena", "sagemaker", "ssm")
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(batch_scoring, name, mock)
    monkeypatch.setattr(batch_scoring, "GLUE_DATABASE", "db")
    monkeypatch.setattr(batch_scoring, "CURATED_TABLE", "demand")
    return mocks


def test_unload_query(aws):
    query = batch_scoring.build_unload_query("s3://b/in/", "2026-01-01", 3)
    assert query.startswith('UNLOAD (SELECT "timestamp", temperature_c, hour')
    assert "ingestion_date = '2026-01-01'" in query
    assert "date_add('day', -3," in query
    latest = batch_scoring.build_unload_query("s3://b/in/")
    assert 'SELECT max(ingestion_date) FROM "db"."demand"' in latest


def test_requires_approved_model(aws):
    aws["ssm"].get_parameter.return_value = "none"
    with pytest.raises(NoApprovedModelError):
        batch_scoring.handler({}, None)
    aws["athena"].run_query.assert_not_called()


def test_starts_transform_with_approved_model(aws):
    aws["ssm"].get_parameter.return_value = ARN

    result = batch_scoring.handler({
        "days": 2
    }, None)

    assert "-2," in aws["athena"].run_query.call_args.args[0]
    aws["sagemaker"].create_model_from_package.assert_called_once()
    request = aws["sagemaker"].create_transform_job.call_args.kwargs
    assert request["DataProcessing"]["InputFilter"] == "$[1:]"
    assert request["DataProcessing"]["OutputFilter"] == "$[0,-1]"
    assert request["TransformInput"]["CompressionType"] == "Gzip"
    assert result["model_package_arn"] == ARN
