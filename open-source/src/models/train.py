"""
Entrenamiento de modelo LightGBM con tracking en MLflow.

Uso:
    python -m src.models.train
    python -m src.models.train --data-path data/processed/ --n-estimators 500
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# Columnas de features (debe coincidir con preprocessing.py)
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
    # MAPE evitando división por cero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def plot_feature_importance(
    model: lgb.LGBMRegressor,
    feature_names: list[str],
    output_path: str,
) -> str:
    """Genera y guarda un plot de feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(20, len(feature_names))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(top_n),
        importances[indices[:top_n]][::-1],
        color="steelblue",
        alpha=0.8,
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[:top_n]][::-1])
    ax.set_xlabel("Importancia (split)")
    ax.set_title("Feature Importance — LightGBM")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def train_model(
    params: Optional[dict] = None,
    data_path: str = "data/processed/",
    experiment_name: str = "demand_forecasting",
) -> str:
    """
    Entrena un modelo LightGBM y loguea todo en MLflow.

    Args:
        params: Hiperparámetros del modelo. Si None, usa defaults.
        data_path: Directorio con train.parquet y val.parquet.
        experiment_name: Nombre del experimento en MLflow.

    Returns:
        run_id de MLflow del run creado.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Parámetros por defecto
    default_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        default_params.update(params)
    params = default_params

    data_path = Path(data_path)
    train_df = pd.read_parquet(data_path / "train.parquet")
    val_df = pd.read_parquet(data_path / "val.parquet")

    feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL]

    logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Features: {len(feature_cols)}")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        mlflow.log_params(params)
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=100)],
        )

        # Métricas en train
        y_pred_train = model.predict(X_train)
        train_metrics = compute_metrics(y_train.values, y_pred_train)
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)

        # Métricas en val
        y_pred_val = model.predict(X_val)
        val_metrics = compute_metrics(y_val.values, y_pred_val)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)

        logger.info(f"Train RMSE: {train_metrics['rmse']:.4f} | Val RMSE: {val_metrics['rmse']:.4f}")

        signature = infer_signature(X_train, y_pred_train)
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(5),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            fi_path = plot_feature_importance(model, feature_cols, f"{tmp_dir}/feature_importance.png")
            mlflow.log_artifact(fi_path, artifact_path="plots")

        mlflow.set_tag("model_type", "LightGBM")
        mlflow.set_tag("target", TARGET_COL)

        logger.info(f"Run completado: {run_id}")
        return run_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Entrenamiento LightGBM con MLflow tracking")
    parser.add_argument("--data-path", default="data/processed/")
    parser.add_argument("--experiment-name", default="demand_forecasting")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    args = parser.parse_args()

    params = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
    }

    run_id = train_model(
        params=params,
        data_path=args.data_path,
        experiment_name=args.experiment_name,
    )
    print(f"\nModelo entrenado. Run ID: {run_id}")
    print("Ver resultados en la UI de MLflow: http://localhost:5050")
