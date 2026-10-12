"""Paso Processing de SageMaker: valida el dataset curado y lo parte en el
tiempo.

Entrada (/opt/ml/processing/input/data): la tabla curada, entregada por la
         definición de dataset de Athena como archivos JSON Lines (a veces
         comprimidos con gzip).
Salida (/opt/ml/processing/<nombre>):
    train/train.csv            timestamp + features + target, con encabezado
    validation/validation.csv  ídem (early stopping del entrenamiento)
    test/test.csv              ídem (evaluación offline)
    baseline/baseline.csv      solo features, con encabezado (baseline de
                               drift: misma forma que el tráfico del endpoint)

Split temporal (nunca aleatorio en series de tiempo): los últimos TEST_DAYS
días son test y los VALIDATION_DAYS anteriores, validación. Con el dataset por
defecto (2022-2024) da exactamente el split del proyecto open source:
train 2022-2023, validación 2024-S1, test 2024-S2.
"""
import argparse
import gzip
import json
import logging
import os

import pandas as pd

from src.features.schema import (
    FEATURES,
    TARGET,
    TIMESTAMP,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = "/opt/ml/processing"


class DataValidationError(Exception):
    pass


def _read_json_lines(path: str) -> list[dict]:
    with open(path, "rb") as file:
        content = file.read()
    if content[:2] == b"\x1f\x8b":
        content = gzip.decompress(content)
    return [
        json.loads(line)
        for line in content.decode("utf-8").splitlines()
        if line.strip()
    ]


def read_dataset(input_dir: str) -> pd.DataFrame:
    records = []
    for root, _, files in os.walk(input_dir):
        for name in sorted(files):
            path = os.path.join(root, name)
            if name.endswith(".csv"):
                records.extend(pd.read_csv(path).to_dict("records"))
            elif not name.startswith(".") and not name.endswith(".metadata"):
                records.extend(_read_json_lines(path))
    return pd.DataFrame.from_records(records)


def validate(df: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    """Gate de validación de datos: si falla, el pipeline se detiene antes
    de entrenar un modelo con datos malos (equivalente al paso de Great
    Expectations del proyecto open source)."""
    required = [TIMESTAMP, TARGET] + FEATURES
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")

    df = df[required].copy()
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
    df = df.drop_duplicates(subset=[TIMESTAMP], keep="last")
    df = df.sort_values(TIMESTAMP).reset_index(drop=True)
    for column in FEATURES + [TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if len(df) < min_rows:
        raise DataValidationError(
            f"Only {len(df)} rows; at least {min_rows} are required")
    nulls = df[FEATURES + [TARGET]].isna().sum()
    if nulls.any():
        raise DataValidationError(f"Null values: {nulls[nulls > 0].to_dict()}")
    if (df[TARGET] <= 0).any():
        raise DataValidationError("Non-positive demand_kwh found")
    if not df["hour"].between(0, 23).all():
        raise DataValidationError("hour outside 0-23")
    if not df["temperature_c"].between(-10, 50).all():
        raise DataValidationError("temperature_c outside -10..50 °C")

    logger.info("Validated %s rows (%s → %s)", len(df), df[TIMESTAMP].min(),
                df[TIMESTAMP].max())
    return df


def split(df: pd.DataFrame, test_days: int, validation_days: int):
    """Corta por fecha calendario (medianoche) para que cada split tenga días
    completos."""
    last_day = df[TIMESTAMP].max().normalize()
    test_start = last_day - pd.Timedelta(days=test_days - 1)
    validation_start = test_start - pd.Timedelta(days=validation_days)

    train = df[df[TIMESTAMP] < validation_start]
    validation = df[(df[TIMESTAMP] >= validation_start)
                    & (df[TIMESTAMP] < test_start)]
    test = df[df[TIMESTAMP] >= test_start]
    if train.empty or validation.empty or test.empty:
        raise DataValidationError(
            f"Not enough history for {validation_days} validation days and "
            f"{test_days} test days")
    logger.info("Temporal split: train=%s validation=%s test=%s", len(train),
                len(validation), len(test))
    return train, validation, test


def write_outputs(train, validation, test, base_dir: str) -> None:
    columns = [TIMESTAMP] + FEATURES + [TARGET]
    outputs = {
        "train": train[columns],
        "validation": validation[columns],
        "test": test[columns],
        "baseline": train[FEATURES],
    }
    for name, frame in outputs.items():
        directory = os.path.join(base_dir, name)
        os.makedirs(directory, exist_ok=True)
        frame.to_csv(os.path.join(directory, f"{name}.csv"), index=False)
        logger.info("Wrote %s rows to %s", len(frame), directory)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=f"{BASE_DIR}/input/data")
    parser.add_argument("--output-dir", default=BASE_DIR)
    parser.add_argument("--test-days", type=int, default=184)
    parser.add_argument("--validation-days", type=int, default=182)
    parser.add_argument("--min-rows", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    df = validate(read_dataset(args.input_dir), args.min_rows)
    train, validation, test = split(df, args.test_days, args.validation_days)
    write_outputs(train, validation, test, args.output_dir)


if __name__ == "__main__":
    main()
