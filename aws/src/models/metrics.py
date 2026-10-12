"""Métricas de regresión en el formato JSON que entiende SageMaker Model
Registry (`regression_metrics.<métrica>.value`), así aparecen en la pestaña
"Model quality" del model package. MAPE es la métrica del quality gate del
pipeline (mismas fórmulas que open-source/src/models/evaluate.py)."""
from typing import Any

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def _metric(value: float) -> dict[str, Any]:
    return {
        "value": float(value),
        "standard_deviation": "NaN",
    }


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Error porcentual absoluto medio (%), ignorando y_true = 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "regression_metrics": {
            "mse": _metric(mse),
            "rmse": _metric(np.sqrt(mse)),
            "mae": _metric(mean_absolute_error(y_true, y_pred)),
            "mape": _metric(mape(y_true, y_pred)),
            "r2": _metric(r2_score(y_true, y_pred)),
        },
        "samples": int(len(y_true)),
    }
