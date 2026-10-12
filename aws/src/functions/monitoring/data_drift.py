"""Monitoreo de data drift del endpoint en tiempo real (reemplaza a SageMaker
Model Monitor, cerrado a clientes nuevos; equivalente al DAG de monitoreo con
Evidently del proyecto open source).

Cada día (schedule de EventBridge) lee las peticiones que el endpoint capturó
en las LOOKBACK_HOURS horas completas anteriores, las compara con el perfil de
referencia registrado con la versión desplegada (PSI por feature,
src/monitoring/drift.py) y:

    * escribe un reporte JSON en el bucket de artefactos,
    * publica métricas en CloudWatch (MaxFeatureDrift, FeatureDrift por
      feature, DriftedFeatures, SampleCount). Una alarma sobre MaxFeatureDrift
      avisa al equipo por SNS y dispara el reentrenamiento.
"""
import base64
import csv
import io
import json
import os
from datetime import (
    datetime,
    timedelta,
    timezone,
)

from src.features.schema import FEATURES
from src.monitoring.drift import evaluate
from src.services import cloudwatch, s3
from src.utilities.logger import get_logger

logger = get_logger(__name__)

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "")
VARIANT_NAME = os.environ.get("VARIANT_NAME", "AllTraffic")
ARTIFACTS_BUCKET_NAME = os.environ.get("ARTIFACTS_BUCKET_NAME", "")
CAPTURE_PREFIX = os.environ.get("CAPTURE_PREFIX", "")
REPORTS_PREFIX = os.environ.get("REPORTS_PREFIX", "")
BASELINE_STATISTICS_URI = os.environ.get("BASELINE_STATISTICS_URI", "")
BASELINE_CONSTRAINTS_URI = os.environ.get("BASELINE_CONSTRAINTS_URI", "")
METRIC_NAMESPACE = os.environ.get("METRIC_NAMESPACE", "")
LOOKBACK_HOURS = int(os.environ.get("LOOKBACK_HOURS", "24"))


def capture_prefixes(now: datetime) -> list:
    """Las capturas viven en <prefijo>/<endpoint>/<variante>/Y/m/d/H/."""
    hour = now.replace(minute=0, second=0, microsecond=0)
    return [
        f"{CAPTURE_PREFIX}/{ENDPOINT_NAME}/{VARIANT_NAME}/"
        f"{(hour - timedelta(hours=offset)):%Y/%m/%d/%H}/"
        for offset in range(LOOKBACK_HOURS, 0, -1)
    ]


def rows_from_capture(line: str) -> list:
    """Filas de features de una petición capturada (body CSV o JSON)."""
    record = json.loads(line)
    request = record["captureData"]["endpointInput"]
    data = request["data"]
    if request.get("encoding") == "BASE64":
        data = base64.b64decode(data).decode("utf-8")
    content_type = request.get("observedContentType", "")
    if content_type.startswith("text/csv"):
        return [
            dict(zip(FEATURES, values))
            for values in csv.reader(io.StringIO(data))
            if values
        ]
    if content_type.startswith("application/json"):
        payload = json.loads(data)
        instances = (payload.get("instances", payload) if isinstance(
            payload, dict) else payload)
        return instances if isinstance(instances, list) else [instances]
    return []


def load_rows(prefixes: list) -> list:
    rows = []
    for prefix in prefixes:
        for key in s3.list_keys(ARTIFACTS_BUCKET_NAME, prefix):
            for line in s3.read_text(ARTIFACTS_BUCKET_NAME, key).splitlines():
                if line.strip():
                    rows.extend(rows_from_capture(line))
    return rows


def metrics(report: dict) -> list:
    endpoint = [{
        "Name": "Endpoint",
        "Value": ENDPOINT_NAME,
    }]
    data = [{
        "MetricName": "SampleCount",
        "Dimensions": endpoint,
        "Value": report["samples"],
        "Unit": "Count",
    }]
    if report["status"] == "insufficient_data":
        return data
    data.append({
        "MetricName": "MaxFeatureDrift",
        "Dimensions": endpoint,
        "Value": report["max_psi"],
        "Unit": "None",
    })
    data.append({
        "MetricName": "DriftedFeatures",
        "Dimensions": endpoint,
        "Value": len(report["drifted_features"]),
        "Unit": "Count",
    })
    for feature, value in report["features"].items():
        data.append({
            "MetricName": "FeatureDrift",
            "Dimensions": endpoint + [{
                "Name": "Feature",
                "Value": feature,
            }],
            "Value": value,
            "Unit": "None",
        })
    return data


def handler(event, context):
    now = datetime.now(timezone.utc)
    prefixes = capture_prefixes(now)
    profile = s3.read_json(BASELINE_STATISTICS_URI)
    constraints = s3.read_json(BASELINE_CONSTRAINTS_URI)

    report = evaluate(profile, constraints, load_rows(prefixes))
    report.update({
        "endpoint": ENDPOINT_NAME,
        "analyzed_at": now.isoformat(),
        "capture_prefixes": prefixes,
        "baseline": BASELINE_STATISTICS_URI,
    })

    key = f"{REPORTS_PREFIX}/{now:%Y/%m/%d/%H}/report.json"
    s3.put_object(
        ARTIFACTS_BUCKET_NAME,
        key,
        json.dumps(report, indent=2).encode("utf-8"),
        "application/json",
    )
    cloudwatch.put_metrics(METRIC_NAMESPACE, metrics(report))
    logger.info(
        json.dumps({
            "status": report["status"],
            "samples": report["samples"],
            "max_psi": report["max_psi"],
            "drifted_features": report["drifted_features"],
            "report": key,
        }))
    return {
        "status": report["status"],
        "report": f"s3://{ARTIFACTS_BUCKET_NAME}/{key}",
    }
