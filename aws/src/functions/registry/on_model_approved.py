"""Puente Model Registry → despliegue.

Lo dispara EventBridge cuando un model package del grupo del proyecto pasa a
`Approved`. Publica en SSM el ARN del package aprobado y sus baselines de
drift — el contrato de despliegue que serverless/endpoint y serverless/batch
resuelven al desplegar/ejecutar — y avisa al equipo con el comando de deploy.
Es el equivalente del alias @champion del Model Registry de MLflow.
"""
import json
import os

from src.services import (
    sagemaker,
    sns,
    ssm,
)
from src.utilities.errors import MissingBaselineError
from src.utilities.logger import get_logger

logger = get_logger(__name__)

MODEL_PACKAGE_ARN_PARAMETER = os.environ.get(
    "APPROVED_MODEL_PACKAGE_ARN_PARAMETER", "")
BASELINE_CONSTRAINTS_PARAMETER = os.environ.get(
    "APPROVED_BASELINE_CONSTRAINTS_PARAMETER", "")
BASELINE_STATISTICS_PARAMETER = os.environ.get(
    "APPROVED_BASELINE_STATISTICS_PARAMETER", "")
NOTIFICATIONS_TOPIC_ARN = os.environ.get("NOTIFICATIONS_TOPIC_ARN", "")
STAGE = os.environ.get("STAGE", "dev")


def get_baselines(model_package: dict) -> dict:
    data_quality = model_package.get("DriftCheckBaselines",
                                     {}).get("ModelDataQuality", {})
    constraints = data_quality.get("Constraints", {}).get("S3Uri")
    statistics = data_quality.get("Statistics", {}).get("S3Uri")
    if not constraints or not statistics:
        raise MissingBaselineError(
            "Model package has no ModelDataQuality drift baselines")
    return {
        "constraints": constraints,
        "statistics": statistics,
    }


def handler(event, context):
    model_package_arn = event["detail"]["ModelPackageArn"]
    logger.info("Model package approved: %s", model_package_arn)

    model_package = sagemaker.describe_model_package(model_package_arn)
    baselines = get_baselines(model_package)

    ssm.put_parameter(MODEL_PACKAGE_ARN_PARAMETER, model_package_arn)
    ssm.put_parameter(BASELINE_CONSTRAINTS_PARAMETER, baselines["constraints"])
    ssm.put_parameter(BASELINE_STATISTICS_PARAMETER, baselines["statistics"])

    version = model_package.get("ModelPackageVersion")
    sns.publish(
        NOTIFICATIONS_TOPIC_ARN,
        f"[{STAGE}] demand-forecasting model v{version} approved",
        "\n".join([
            f"Model package aprobado: {model_package_arn}",
            "",
            "Contrato de despliegue (SSM) actualizado. Despliégala con:",
            f"  npm run deploy-endpoint:{STAGE}",
            "El batch scoring la usa en su próxima ejecución.",
        ]),
    )
    result = {
        "model_package_arn": model_package_arn,
        **baselines,
    }
    logger.info(json.dumps(result))
    return result
