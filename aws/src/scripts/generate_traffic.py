"""Envía horas de la serie local al modelo online para llenar el data capture
del endpoint, así el monitor DataDrift tiene tráfico que comparar con el
baseline.

    python3 -m src.scripts.generate_dataset
    python3 -m src.scripts.generate_traffic --api-url https://.../predict
    python3 -m src.scripts.generate_traffic --endpoint-name demand-forecasting \
        --requests 300 --drift

--drift simula una ola de calor y más consumo (temperatura +5 °C, lags y media
móvil +30 %): la siguiente ejecución de DataDrift debería reportar drift (y
la alarma, disparar el reentrenamiento).
"""
import argparse
import json
import random
import time
import urllib.request

import pandas as pd

from src.features.curation import curate
from src.features.schema import (
    CALENDAR_FEATURES,
    FEATURES,
    LAG_FEATURES,
    TIMESTAMP,
)
from src.scripts.paths import RAW_DATASET_PATH
from src.services import sagemaker_runtime


def load_records(drift: bool) -> list:
    df = curate(pd.read_csv(RAW_DATASET_PATH))
    if drift:
        df["temperature_c"] = df["temperature_c"] + 5
        for column in LAG_FEATURES:
            df[column] = (df[column] * 1.3).round(2)
    return df[[TIMESTAMP] + FEATURES].to_dict("records")


def send_api(url: str, records: list) -> dict:
    payload = [{
        key: value
        for key, value in record.items()
        if key not in CALENDAR_FEATURES
    }
               for record in records]
    request = urllib.request.Request(
        url,
        data=json.dumps({
            "records": payload
        }).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )
    with urllib.request.urlopen(request) as response:  # nosec B310
        return json.loads(response.read())


def send_endpoint(endpoint_name: str, records: list) -> dict:
    rows = "\n".join(",".join(str(record[name])
                              for name in FEATURES)
                     for record in records)
    body = sagemaker_runtime.invoke_endpoint(
        endpoint_name,
        rows.encode("utf-8"),
        content_type="text/csv",
        accept="application/json",
    )
    return json.loads(body)


def main():
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--api-url")
    target.add_argument("--endpoint-name")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--drift", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    records = load_records(args.drift)
    rng = random.Random(args.seed)

    for index in range(args.requests):
        batch = rng.sample(records, args.batch_size)
        if args.api_url:
            result = send_api(args.api_url, batch)
        else:
            result = send_endpoint(args.endpoint_name, batch)
        if index % 10 == 0:
            print(f"[{index}/{args.requests}] {result['predictions'][0]}")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
