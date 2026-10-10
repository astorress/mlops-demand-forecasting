"""
Hyperparameter Optimization (HPO) con Optuna + MLflow.

Cada trial de Optuna se registra como un run MLflow separado dentro
del mismo experimento, permitiendo comparar todos los trials en la UI.

Uso:
    python -m src.models.hpo
    python -m src.models.hpo --n-trials 30 --data-path data/processed/
"""

import argparse
import logging
import os
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error

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


def run_hpo(
    data_path: str = "data/processed/",
    n_trials: int = 30,
    experiment_name: str = "demand_forecasting_hpo",
    study_name: str = "lgbm_hpo",
) -> dict:
    """
    Ejecuta HPO con Optuna, logueando cada trial como un run MLflow.

    El espacio de búsqueda cubre los hiperparámetros más influyentes de LightGBM.
    Usa MedianPruner para cortar tempranamente los trials con bajo potencial.

    Args:
        data_path: Directorio con train.parquet y val.parquet.
        n_trials: Número de trials de Optuna.
        experiment_name: Nombre del experimento MLflow para los trials.
        study_name: Nombre del estudio Optuna.

    Returns:
        Dict con los mejores hiperparámetros encontrados.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Cargar datos una sola vez (fuera del objetivo, para eficiencia)
    data_path = Path(data_path)
    train_df = pd.read_parquet(data_path / "train.parquet")
    val_df = pd.read_parquet(data_path / "val.parquet")

    feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_COL].values

    logger.info(f"HPO iniciado: {n_trials} trials | Features: {len(feature_cols)}")

    def objective(trial: optuna.Trial) -> float:
        """Función objetivo: entrena LightGBM con params sugeridos por Optuna."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        with mlflow.start_run(nested=False, run_name=f"trial_{trial.number}"):
            mlflow.log_params(params)
            mlflow.log_param("trial_number", trial.number)

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )

            y_pred = model.predict(X_val)
            val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))

            mlflow.log_metric("val_rmse", val_rmse)
            mlflow.set_tag("trial_number", trial.number)

            # Pruning: si el trial no es prometedor, se corta
            trial.report(val_rmse, step=params["n_estimators"])
            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_rmse

    # Configurar pruner
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50)
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=pruner,
        sampler=sampler,
    )

    # Suprimir logs de Optuna para no contaminar salida
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info("Iniciando optimización...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_value = study.best_value

    logger.info(f"HPO completado!")
    logger.info(f"Mejor RMSE val: {best_value:.4f}")
    logger.info(f"Mejores parámetros: {best_params}")

    # Loguear resumen del HPO en un run separado
    with mlflow.start_run(run_name="hpo_summary"):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_rmse", best_value)
        mlflow.log_metric("n_trials", len(study.trials))
        mlflow.log_metric("n_pruned", len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]))
        mlflow.set_tag("run_type", "hpo_summary")

    return best_params


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="HPO con Optuna + MLflow")
    parser.add_argument("--data-path", default="data/processed/")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--experiment-name", default="demand_forecasting_hpo")
    args = parser.parse_args()

    best_params = run_hpo(
        data_path=args.data_path,
        n_trials=args.n_trials,
        experiment_name=args.experiment_name,
    )
    print("\nMejores hiperparámetros encontrados:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
