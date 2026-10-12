"""Handlers de inferencia que usa serve.py (endpoint en tiempo real y Batch
Transform).

Content types de entrada:
    text/csv          filas sin encabezado en el orden de
                      src.features.schema.FEATURES (lo que el monitor de
                      drift lee del tráfico capturado)
    application/json  {"instances": [{"temperature_c": 28.1, ...}, ...]}
Accept de salida:
    application/json  {"predictions": [{"demand_kwh": 1043.2}, ...]}
    text/csv          un valor por línea (join de Batch Transform)
"""
import io
import json
import os

import joblib
import pandas as pd

from src.features.schema import FEATURES, TARGET

MODEL_FILE_NAME = "model.joblib"


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, MODEL_FILE_NAME))


def input_fn(request_body, request_content_type):
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")

    if request_content_type == "text/csv":
        df = pd.read_csv(
            io.StringIO(request_body),
            header=None,
            names=FEATURES,
            skipinitialspace=True,
        )
    elif request_content_type == "application/json":
        payload = json.loads(request_body)
        instances = payload.get("instances", payload) if isinstance(
            payload, dict) else payload
        if isinstance(instances, dict):
            instances = [instances]
        df = pd.DataFrame.from_records(instances)
        missing = [name for name in FEATURES if name not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        df = df[FEATURES]
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def predict_fn(input_data, model):
    return model.predict(input_data)


def output_fn(prediction, accept):
    if accept.split(";")[0].strip() == "text/csv":
        body = "\n".join(f"{value:.2f}" for value in prediction)
        return body, "text/csv"
    predictions = [{
        TARGET: round(float(value), 2)
    } for value in prediction]
    return json.dumps({
        "predictions": predictions
    }), "application/json"
