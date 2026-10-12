"""Inferencia batch: pronostica las últimas DAYS días de la última partición
curada con el último modelo *aprobado* y deja las predicciones junto a la
demanda real, consultables en Athena (MAPE en producción = monitoreo de la
calidad del modelo).

    1. Athena UNLOAD → CSV sin encabezado (timestamp + FEATURES) en el bucket
       de artefactos.
    2. CreateModel desde el model package aprobado (Model Registry).
    3. Batch Transform: quita el timestamp antes del modelo (InputFilter),
       vuelve a unir el pronóstico (JoinSource/OutputFilter) y escribe en el
       data lake, tabla `predictions`.

Evento (opcional): {"ingestion_date": "YYYY-MM-DD", "days": 7}.
"""
import json
import os
from datetime import datetime, timezone

from src.features.schema import FEATURES, TIMESTAMP
from src.services import (
    athena,
    sagemaker,
    ssm,
)
from src.utilities.errors import NoApprovedModelError
from src.utilities.logger import get_logger

logger = get_logger(__name__)

ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "")
GLUE_DATABASE = os.environ.get("GLUE_DATABASE", "")
CURATED_TABLE = os.environ.get("CURATED_TABLE", "")
ARTIFACTS_BUCKET_NAME = os.environ.get("ARTIFACTS_BUCKET_NAME", "")
BATCH_INPUT_PREFIX = os.environ.get("BATCH_INPUT_PREFIX", "")
DATA_LAKE_BUCKET_NAME = os.environ.get("DATA_LAKE_BUCKET_NAME", "")
PREDICTIONS_PREFIX = os.environ.get("PREDICTIONS_PREFIX", "")
MODEL_PACKAGE_ARN_PARAMETER = os.environ.get(
    "APPROVED_MODEL_PACKAGE_ARN_PARAMETER", "")
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "")
RESOURCE_PREFIX = os.environ.get("RESOURCE_PREFIX", "demand-forecasting")
TRANSFORM_INSTANCE_TYPE = os.environ.get("TRANSFORM_INSTANCE_TYPE",
                                         "ml.m5.large")
DEFAULT_DAYS = int(os.environ.get("DEFAULT_DAYS", "7"))
ATHENA_TIMESTAMP = "'%Y-%m-%d %H:%i:%s'"


def build_unload_query(output_uri: str,
                       ingestion_date: str | None = None,
                       days: int = DEFAULT_DAYS) -> str:
    table = f'"{GLUE_DATABASE}"."{CURATED_TABLE}"'
    partition = (f"'{ingestion_date}'" if ingestion_date else
                 f"(SELECT max(ingestion_date) FROM {table})")
    moment = f'date_parse("{TIMESTAMP}", {ATHENA_TIMESTAMP})'
    columns = ", ".join([f'"{TIMESTAMP}"'] + FEATURES)
    return (f"UNLOAD (SELECT {columns} FROM {table} "
            f"WHERE ingestion_date = {partition} "
            f"AND {moment} > (SELECT date_add('day', -{int(days)}, "
            f"max({moment})) FROM {table} "
            f"WHERE ingestion_date = {partition})) "
            f"TO '{output_uri}' "
            "WITH (format = 'TEXTFILE', field_delimiter = ',')")


def get_approved_model_package() -> str:
    arn = ssm.get_parameter(MODEL_PACKAGE_ARN_PARAMETER)
    if not arn.startswith("arn:"):
        raise NoApprovedModelError(
            "No approved model yet: approve a model package in the registry")
    return arn


def transform_request(job_name: str, model_name: str, input_uri: str,
                      output_uri: str, model_package_arn: str) -> dict:
    return {
        "TransformJobName": job_name,
        "ModelName": model_name,
        "BatchStrategy": "MultiRecord",
        "MaxPayloadInMB": 6,
        "TransformInput": {
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": input_uri,
                },
            },
            "ContentType": "text/csv",
            # Athena UNLOAD escribe archivos de texto comprimidos con gzip.
            "CompressionType": "Gzip",
            "SplitType": "Line",
        },
        "TransformOutput": {
            "S3OutputPath": output_uri,
            "Accept": "text/csv",
            "AssembleWith": "Line",
        },
        "TransformResources": {
            "InstanceType": TRANSFORM_INSTANCE_TYPE,
            "InstanceCount": 1,
        },
        "DataProcessing": {
            # Quita el timestamp antes del modelo y deja timestamp + pronóstico.
            "InputFilter": "$[1:]",
            "JoinSource": "Input",
            "OutputFilter": "$[0,-1]",
        },
        "Tags": [{
            "Key": "ModelPackageArn",
            "Value": model_package_arn,
        }],
    }


def handler(event, context):
    event = event or {}
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%d-%H%M%S")
    input_uri = f"s3://{ARTIFACTS_BUCKET_NAME}/{BATCH_INPUT_PREFIX}/{run_id}/"
    output_uri = (f"s3://{DATA_LAKE_BUCKET_NAME}/{PREDICTIONS_PREFIX}/"
                  f"run_date={now.strftime('%Y-%m-%d')}/")
    model_package_arn = get_approved_model_package()

    athena.run_query(
        build_unload_query(
            input_uri,
            event.get("ingestion_date"),
            int(event.get("days", DEFAULT_DAYS)),
        ),
        ATHENA_WORKGROUP,
    )

    model_name = f"{RESOURCE_PREFIX}-batch-{run_id}"
    sagemaker.create_model_from_package(
        model_name,
        SAGEMAKER_ROLE_ARN,
        model_package_arn,
    )
    job_name = f"{RESOURCE_PREFIX}-{run_id}"
    sagemaker.create_transform_job(**transform_request(
        job_name,
        model_name,
        input_uri,
        output_uri,
        model_package_arn,
    ))
    result = {
        "transform_job_name": job_name,
        "model_package_arn": model_package_arn,
        "input": input_uri,
        "output": output_uri,
    }
    logger.info(json.dumps(result))
    return result
