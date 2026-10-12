from unittest.mock import MagicMock

import pytest

from src.functions.registry import on_model_approved
from src.utilities.errors import MissingBaselineError

ARN = "arn:aws:sagemaker:us-east-1:1:model-package/demand-forecasting/3"


@pytest.fixture
def aws(monkeypatch):
    mocks = {
        name: MagicMock()
        for name in ("sagemaker", "ssm", "sns")
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(on_model_approved, name, mock)
    return mocks


def test_updates_deployment_contract(aws):
    aws["sagemaker"].describe_model_package.return_value = {
        "ModelPackageVersion": 3,
        "DriftCheckBaselines": {
            "ModelDataQuality": {
                "Constraints": {
                    "S3Uri": "s3://b/constraints.json",
                },
                "Statistics": {
                    "S3Uri": "s3://b/statistics.json",
                },
            },
        },
    }

    result = on_model_approved.handler({
        "detail": {
            "ModelPackageArn": ARN
        }
    }, None)

    assert result["model_package_arn"] == ARN
    values = [call.args[1] for call in aws["ssm"].put_parameter.call_args_list]
    assert values == [ARN, "s3://b/constraints.json", "s3://b/statistics.json"]
    aws["sns"].publish.assert_called_once()


def test_requires_baselines(aws):
    aws["sagemaker"].describe_model_package.return_value = {}
    with pytest.raises(MissingBaselineError):
        on_model_approved.handler({
            "detail": {
                "ModelPackageArn": ARN
            }
        }, None)
    aws["ssm"].put_parameter.assert_not_called()
