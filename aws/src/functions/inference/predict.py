"""API de inferencia online: POST /predict (HTTP API, pública y con
throttling). Equivalente a /predict de la API Flask del proyecto open source.

Body: {"records": [{"timestamp": "2024-07-01 18:00", "temperature_c": 29.4,
"demand_lag_24h": 1180.2, "demand_lag_168h": 1150.7,
"demand_rolling_mean_7d": 1012.3}]} (también acepta un solo objeto). Las
features de calendario (hora, día, mes, fin de semana, festivo) se calculan
aquí a partir del timestamp.

Las filas viajan al endpoint como CSV sin encabezado en el orden de FEATURES:
el formato que el monitor DataDrift lee del tráfico capturado.
"""
import csv
import io
import json
import os

from src.features.calendar import calendar_features, parse_timestamp
from src.features.schema import (
    CALENDAR_FEATURES,
    FEATURES,
    TIMESTAMP,
    TIMESTAMP_FORMAT,
)
from src.services import sagemaker_runtime
from src.utilities.errors import RequestValidationError
from src.utilities.http import json_response
from src.utilities.logger import get_logger

logger = get_logger(__name__)

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "")
MAX_RECORDS = int(os.environ.get("MAX_RECORDS", "168"))
# El calendario se deriva del timestamp; el resto lo envía el cliente.
VALUE_FEATURES = [name for name in FEATURES if name not in CALENDAR_FEATURES]
REQUIRED = [TIMESTAMP] + VALUE_FEATURES


def build_record(index: int, raw: dict) -> dict:
    missing = [name for name in REQUIRED if raw.get(name) is None]
    if missing:
        raise RequestValidationError(f"Record {index} is missing {missing}")
    try:
        moment = parse_timestamp(raw[TIMESTAMP])
        values = {
            name: float(raw[name])
            for name in VALUE_FEATURES
        }
    except (TypeError, ValueError) as error:
        raise RequestValidationError(f"Record {index}: {error}") from error
    return {
        TIMESTAMP: moment.strftime(TIMESTAMP_FORMAT),
        **values,
        **calendar_features(moment),
    }


def parse_records(body: str | None) -> list:
    try:
        payload = json.loads(body or "{}")
    except json.JSONDecodeError as error:
        raise RequestValidationError(f"Invalid JSON: {error}") from error
    if isinstance(payload, dict):
        # {"records": [...]} o un solo registro ({} = sin registros).
        records = payload.get("records", [payload] if payload else [])
    else:
        records = payload
    if not records:
        raise RequestValidationError("No records in request body")
    if len(records) > MAX_RECORDS:
        raise RequestValidationError(
            f"At most {MAX_RECORDS} records per request")
    return [build_record(index, raw) for index, raw in enumerate(records)]


def to_csv(records: list) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    for record in records:
        writer.writerow([record[name] for name in FEATURES])
    return output.getvalue()


def handler(event, context):
    try:
        records = parse_records(event.get("body"))
    except RequestValidationError as error:
        return json_response(400, {
            "message": str(error)
        })

    body = sagemaker_runtime.invoke_endpoint(
        ENDPOINT_NAME,
        to_csv(records).encode("utf-8"),
        content_type="text/csv",
        accept="application/json",
    )
    predictions = json.loads(body)["predictions"]
    for record, prediction in zip(records, predictions):
        prediction[TIMESTAMP] = record[TIMESTAMP]
    return json_response(200, {
        "predictions": predictions
    })
