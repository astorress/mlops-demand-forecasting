"""
Evaluación de modelos y comparación con el modelo champion en MLflow Registry.

Funciones:
- evaluate_model: métricas completas en test set
- compare_with_champion: compara nuevo modelo vs Production
- plot_predictions: visualización de predicciones vs realidad
"""

import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "temperature_c", "hour", "day_of_week", "month",
    "is_weekend", "is_holiday",
    "demand_lag_24h", "demand_lag_168h", "demand_rolling_mean_7d",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "temp_x_hour", "is_peak_hour",
]
TARGET_COL = "demand_kwh"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula RMSE, MAE, MAPE y R2."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape), "r2": float(r2)}


def evaluate_model(
    run_id: str,
    test_path: str = "data/processed/test.parquet",
) -> dict[str, float]:
    """
    Evalúa un modelo MLflow en el test set.

    Args:
        run_id: ID del run de MLflow que contiene el modelo.
        test_path: Ruta al archivo Parquet del test set.

    Returns:
        Dict con métricas en test set.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Cargando modelo desde: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    test_df = pd.read_parquet(test_path)
    feature_cols = [c for c in FEATURE_COLS if c in test_df.columns]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test.values, y_pred)

    # Loguear métricas de test en el run original
    client = MlflowClient()
    for k, v in metrics.items():
        client.log_metric(run_id, f"test_{k}", v)

    logger.info(f"Test RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R2: {metrics['r2']:.4f}")
    return metrics


def compare_with_champion(
    run_id: str,
    model_name: str,
    test_path: str = "data/processed/test.parquet",
) -> bool:
    """
    Compara el nuevo modelo con el modelo en Production del Registry.

    Args:
        run_id: ID del run del nuevo modelo.
        model_name: Nombre del modelo en MLflow Registry.
        test_path: Ruta al test set.

    Returns:
        True si el nuevo modelo tiene menor RMSE en test.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    test_df = pd.read_parquet(test_path)
    feature_cols = [c for c in FEATURE_COLS if c in test_df.columns]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    # Evaluar nuevo modelo
    new_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    new_pred = new_model.predict(X_test)
    new_rmse = float(np.sqrt(mean_squared_error(y_test.values, new_pred)))
    logger.info(f"Nuevo modelo — RMSE en test: {new_rmse:.4f}")

    # Intentar cargar el champion
    try:
        champion_model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
        champ_pred = champion_model.predict(X_test)
        champ_rmse = float(np.sqrt(mean_squared_error(y_test.values, champ_pred)))
        logger.info(f"Champion actual — RMSE en test: {champ_rmse:.4f}")

        is_better = new_rmse < champ_rmse
        logger.info(
            f"Resultado: {'NUEVO MODELO ES MEJOR ✓' if is_better else 'CHAMPION SE MANTIENE'} "
            f"({new_rmse:.4f} vs {champ_rmse:.4f})"
        )
        return is_better

    except Exception as e:
        logger.warning(f"No hay modelo en Production ({e}). El nuevo modelo promoverá automáticamente.")
        return True


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicciones vs Real",
    output_path: str = "artifacts/predictions.png",
    n_samples: int = 168,
) -> str:
    """
    Genera un plot de predicciones vs valores reales.

    Args:
        y_true: Valores reales.
        y_pred: Predicciones del modelo.
        title: Título del plot.
        output_path: Ruta donde guardar el PNG.
        n_samples: Número de muestras a mostrar (default: 1 semana = 168h).

    Returns:
        Ruta del archivo PNG generado.
    """
    n = min(n_samples, len(y_true))
    x = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Serie temporal
    ax1.plot(x, y_true[:n], label="Real", alpha=0.8, linewidth=1.5, color="steelblue")
    ax1.plot(x, y_pred[:n], label="Predicción", alpha=0.8, linewidth=1.5,
             color="tomato", linestyle="--")
    ax1.set_xlabel("Hora")
    ax1.set_ylabel("Demanda (kWh)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Residuos
    residuals = y_true[:n] - y_pred[:n]
    ax2.plot(x, residuals, alpha=0.7, color="green", linewidth=1)
    ax2.axhline(0, color="red", linestyle="--", linewidth=1)
    ax2.set_xlabel("Hora")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residuos (Real - Predicción)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Evaluación de modelo MLflow")
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument("--test-path", default="data/processed/test.parquet")
    parser.add_argument("--model-name", default="demand_forecaster")
    args = parser.parse_args()

    metrics = evaluate_model(args.run_id, args.test_path)
    print("\nMétricas en test set:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    is_better = compare_with_champion(args.run_id, args.model_name, args.test_path)
    print(f"\n¿Nuevo modelo es mejor que el champion? {is_better}")
