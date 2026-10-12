"""Ingesta: genera la serie horaria sintética de demanda (mediciones crudas)
y la escribe en la zona raw del data lake
(raw/<proyecto>/ingestion_date=YYYY-MM-DD/demand.csv).

Cada ingesta es un snapshot completo del histórico hasta end_date. El objeto
nuevo dispara (S3 → EventBridge) el workflow de Glue, que cura los datos y,
si todo sale bien, arranca el pipeline de entrenamiento de SageMaker.

Evento (todo opcional):
    ingestion_date   "YYYY-MM-DD" (por defecto: hoy, UTC)
    start_date       "YYYY-MM-DD" primera hora del histórico (2022-01-01)
    end_date         "YYYY-MM-DD" último día del histórico (2024-12-31)
    simulate_drift   true → ola de calor + más demanda en los últimos meses
    seed             semilla aleatoria (42)
"""
import csv
import io
import json
import os
from datetime import datetime, timezone

from src.datasets.synthetic_demand import COLUMNS, generate
from src.services import s3
from src.utilities.errors import RequestValidationError
from src.utilities.logger import get_logger

logger = get_logger(__name__)

DATA_LAKE_BUCKET_NAME = os.environ.get("DATA_LAKE_BUCKET_NAME", "")
RAW_PREFIX = os.environ.get("RAW_PREFIX", "raw/demand_forecasting")
DEFAULT_START_DATE = os.environ.get("DEFAULT_START_DATE", "2022-01-01")
DEFAULT_END_DATE = os.environ.get("DEFAULT_END_DATE", "2024-12-31")
FILE_NAME = "demand.csv"


def parse_range(event: dict) -> tuple[datetime, datetime]:
    start = datetime.strptime(
        event.get("start_date") or DEFAULT_START_DATE, "%Y-%m-%d")
    end = datetime.strptime(
        event.get("end_date") or DEFAULT_END_DATE, "%Y-%m-%d").replace(hour=23)
    if end <= start:
        raise RequestValidationError("end_date must be after start_date")
    return start, end


def to_csv(rows: list) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def handler(event, context):
    event = event or {}
    ingestion_date = event.get("ingestion_date") or datetime.now(
        timezone.utc).strftime("%Y-%m-%d")
    start, end = parse_range(event)
    simulate_drift = bool(event.get("simulate_drift", False))
    seed = int(event.get("seed", 42))

    rows = generate(start, end, seed, simulate_drift)
    key = f"{RAW_PREFIX}/ingestion_date={ingestion_date}/{FILE_NAME}"
    s3.put_object(DATA_LAKE_BUCKET_NAME, key,
                  to_csv(rows).encode("utf-8"), "text/csv")
    result = {
        "bucket": DATA_LAKE_BUCKET_NAME,
        "key": key,
        "rows": len(rows),
        "first_timestamp": rows[0]["timestamp"],
        "last_timestamp": rows[-1]["timestamp"],
        "simulate_drift": simulate_drift,
    }
    logger.info(json.dumps(result))
    return result
