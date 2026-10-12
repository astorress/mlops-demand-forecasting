"""
API REST Flask para serving del modelo de predicción de demanda eléctrica.

Endpoints:
  GET  /health         → estado del servicio
  GET  /model-info     → información del modelo cargado
  POST /predict        → predicción para un registro
  POST /predict-batch  → predicciones para múltiples registros

Uso:
    python src/serving/flask_app.py
    FLASK_PORT=8001 python src/serving/flask_app.py
"""

import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

import mlflow.pyfunc
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Variables de entorno
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "demand_forecaster")
FLASK_HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.environ.get("FLASK_PORT", "8000"))

# ---------------------------------------------------------------------------
# Pydantic schemas para validación de inputs
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Schema para una predicción individual."""
    temperature_c: float = Field(..., ge=-10, le=50, description="Temperatura en °C")
    hour: int = Field(..., ge=0, le=23, description="Hora del día (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Día de la semana (0=lunes)")
    month: int = Field(..., ge=1, le=12, description="Mes (1-12)")
    is_weekend: int = Field(..., ge=0, le=1, description="1 si es fin de semana")
    is_holiday: int = Field(0, ge=0, le=1, description="1 si es feriado")
    demand_lag_24h: float = Field(..., description="Demanda de hace 24 horas (kWh)")
    demand_lag_168h: float = Field(..., description="Demanda de hace 168 horas (kWh)")
    demand_rolling_mean_7d: float = Field(..., description="Media móvil 7 días (kWh)")

    @field_validator("hour")
    @classmethod
    def validate_hour(cls, v: int) -> int:
        if not 0 <= v <= 23:
            raise ValueError(f"hour debe estar entre 0 y 23, recibido: {v}")
        return v


class PredictBatchRequest(BaseModel):
    """Schema para predicciones en batch."""
    records: list[PredictRequest]

    @field_validator("records")
    @classmethod
    def validate_non_empty(cls, v: list) -> list:
        if len(v) == 0:
            raise ValueError("records no puede estar vacío")
        if len(v) > 10000:
            raise ValueError("Máximo 10,000 registros por batch")
        return v


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------

class ModelManager:
    """Gestiona la carga y el acceso al modelo desde MLflow Registry."""

    def __init__(self) -> None:
        self.model = None
        self.model_version = None
        self.model_stage = "Production"
        self.model_name = MODEL_NAME
        self.load_time = None

    def load(self) -> None:
        """Carga el modelo desde MLflow Registry (stage=Production)."""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Intentar cargar por alias @champion, luego por stage Production
        for uri in [
            f"models:/{self.model_name}@champion",
            f"models:/{self.model_name}/Production",
        ]:
            try:
                logger.info(f"Intentando cargar modelo desde: {uri}")
                self.model = mlflow.pyfunc.load_model(uri)
                self.load_time = datetime.utcnow().isoformat()
                logger.info(f"Modelo cargado exitosamente desde: {uri}")
                return
            except Exception as e:
                logger.warning(f"No se pudo cargar desde {uri}: {e}")

        logger.warning(
            "No se encontró modelo en Production/champion. "
            "Endpoints de predicción retornarán error hasta que se cargue un modelo."
        )

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_loaded():
            raise RuntimeError("Modelo no cargado. Verifique que existe un modelo en Production.")
        return self.model.predict(X)


model_manager = ModelManager()

# ---------------------------------------------------------------------------
# Feature engineering (debe ser consistente con preprocessing.py)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "temperature_c", "hour", "day_of_week", "month",
    "is_weekend", "is_holiday",
    "demand_lag_24h", "demand_lag_168h", "demand_rolling_mean_7d",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "temp_x_hour", "is_peak_hour",
]


def build_features(record: dict) -> pd.DataFrame:
    """Agrega features derivadas a un registro de input."""
    import numpy as np
    hour = record["hour"]
    dow = record["day_of_week"]
    month = record["month"]

    record["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
    record["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
    record["dow_sin"] = float(np.sin(2 * np.pi * dow / 7))
    record["dow_cos"] = float(np.cos(2 * np.pi * dow / 7))
    record["month_sin"] = float(np.sin(2 * np.pi * month / 12))
    record["month_cos"] = float(np.cos(2 * np.pi * month / 12))
    record["temp_x_hour"] = record["temperature_c"] * hour
    record["is_peak_hour"] = int(18 <= hour <= 21)

    df = pd.DataFrame([record])
    cols_present = [c for c in FEATURE_COLS if c in df.columns]
    return df[cols_present]


# ---------------------------------------------------------------------------
# App Flask
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health() -> Any:
    """
    GET /health
    Retorna el estado del servicio.
    """
    return jsonify({
        "status": "ok" if model_manager.is_loaded() else "degraded",
        "model_loaded": model_manager.is_loaded(),
        "model_name": model_manager.model_name,
        "load_time": model_manager.load_time,
        "timestamp": datetime.utcnow().isoformat(),
    })


@app.route("/model-info", methods=["GET"])
def model_info() -> Any:
    """
    GET /model-info
    Retorna información del modelo actualmente cargado.
    """
    if not model_manager.is_loaded():
        return jsonify({"error": "Modelo no cargado"}), 503

    return jsonify({
        "model_name": model_manager.model_name,
        "stage": model_manager.model_stage,
        "load_time": model_manager.load_time,
        "tracking_uri": MLFLOW_TRACKING_URI,
    })


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    """
    POST /predict
    Body: JSON con los campos del schema PredictRequest.
    Retorna: {prediction, unit, model_name, timestamp}
    """
    if not model_manager.is_loaded():
        return jsonify({"error": "Modelo no disponible"}), 503

    try:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({"error": "Body JSON requerido"}), 400

        req = PredictRequest(**data)
        record = req.model_dump()

        X = build_features(record)

        t0 = time.time()
        prediction = float(model_manager.predict(X)[0])
        latency_ms = (time.time() - t0) * 1000

        logger.info(f"Predicción: {prediction:.2f} kWh (latencia: {latency_ms:.1f}ms)")

        return jsonify({
            "prediction": prediction,
            "unit": "kWh",
            "model_name": model_manager.model_name,
            "latency_ms": round(latency_ms, 2),
            "timestamp": datetime.utcnow().isoformat(),
        })

    except ValueError as e:
        return jsonify({"error": f"Input inválido: {str(e)}"}), 422
    except Exception as e:
        logger.error(f"Error en /predict: {e}", exc_info=True)
        return jsonify({"error": "Error interno del servidor"}), 500


@app.route("/predict-batch", methods=["POST"])
def predict_batch() -> Any:
    """
    POST /predict-batch
    Body: {"records": [<PredictRequest>, ...]}
    Retorna: {"predictions": [...], "count": N}
    """
    if not model_manager.is_loaded():
        return jsonify({"error": "Modelo no disponible"}), 503

    try:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({"error": "Body JSON requerido"}), 400

        batch_req = PredictBatchRequest(**data)
        records = [r.model_dump() for r in batch_req.records]

        dfs = [build_features(r) for r in records]
        X_batch = pd.concat(dfs, ignore_index=True)

        t0 = time.time()
        predictions = model_manager.predict(X_batch).tolist()
        latency_ms = (time.time() - t0) * 1000

        logger.info(f"Batch prediction: {len(predictions)} registros (latencia: {latency_ms:.1f}ms)")

        return jsonify({
            "predictions": [round(p, 4) for p in predictions],
            "count": len(predictions),
            "unit": "kWh",
            "model_name": model_manager.model_name,
            "latency_ms": round(latency_ms, 2),
            "timestamp": datetime.utcnow().isoformat(),
        })

    except ValueError as e:
        return jsonify({"error": f"Input inválido: {str(e)}"}), 422
    except Exception as e:
        logger.error(f"Error en /predict-batch: {e}", exc_info=True)
        return jsonify({"error": "Error interno del servidor"}), 500


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(f"Iniciando Flask API — {FLASK_HOST}:{FLASK_PORT}")
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"Modelo: {MODEL_NAME}")

    model_manager.load()

    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)
