"""Paso Training de SageMaker (imagen propia, `docker run <imagen> train`).

Contrato de entrenamiento de SageMaker (sin toolkit de por medio):
    /opt/ml/input/config/hyperparameters.json  hiperparámetros (todos string)
    /opt/ml/input/data/<canal>/                train, validation
    /opt/ml/model/                             modelo guardado → model.tar.gz
    /opt/ml/output/failure                     motivo de falla del job

El canal validation se usa para early stopping, como en el proyecto open
source. Las métricas se imprimen como `<split>:<métrica>=<valor>` para que
las MetricDefinitions del job las capturen (SageMaker Experiments /
CloudWatch). Los flags del CLI sobreescriben hiperparámetros en local.
"""
import argparse
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import pandas as pd

from src.features.engineering import add_features
from src.features.schema import FEATURES, TARGET
from src.models.estimator import DEFAULT_HYPERPARAMETERS, build_pipeline
from src.models.metrics import compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_FILE_NAME = "model.joblib"
BASE_DIR = Path(os.environ.get("SAGEMAKER_BASE_DIR", "/opt/ml"))


def load_split(channel_dir: str | Path) -> pd.DataFrame:
    files = sorted(Path(channel_dir).glob("*.csv"))
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def load_hyperparameters(path: Path, overrides: dict[str, Any]) -> dict:
    """SageMaker pasa cada hiperparámetro como string: se convierte cada uno
    al tipo de su valor por defecto."""
    provided = json.loads(path.read_text()) if path.exists() else {}
    unknown = set(provided) - set(DEFAULT_HYPERPARAMETERS)
    if unknown:
        logger.warning("Ignoring unknown hyperparameters: %s", unknown)
    hyperparameters = {}
    for name, default in DEFAULT_HYPERPARAMETERS.items():
        value = overrides.get(name)
        if value is None:
            value = provided.get(name, default)
        hyperparameters[name] = type(default)(value)
    return hyperparameters


def train(train_df: pd.DataFrame,
          hyperparameters: dict,
          validation_df: pd.DataFrame | None = None):
    pipeline = build_pipeline(hyperparameters)
    fit_params = {}
    if validation_df is not None and len(validation_df):
        rounds = int(
            hyperparameters.get(
                "early_stopping_rounds",
                DEFAULT_HYPERPARAMETERS["early_stopping_rounds"],
            ))
        fit_params = {
            "regressor__eval_X": add_features(validation_df[FEATURES]),
            "regressor__eval_y": validation_df[TARGET],
            "regressor__callbacks": [lgb.early_stopping(rounds, verbose=False)],
        }
    pipeline.fit(train_df[FEATURES], train_df[TARGET], **fit_params)
    return pipeline


def log_metrics(split_name: str, pipeline, df: pd.DataFrame) -> dict:
    predictions = pipeline.predict(df[FEATURES])
    metrics = compute_metrics(df[TARGET].values, predictions)
    for name in ("rmse", "mae", "mape", "r2"):
        value = metrics["regression_metrics"][name]["value"]
        print(f"{split_name}:{name}={value:.6f}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    for name, default in DEFAULT_HYPERPARAMETERS.items():
        parser.add_argument(f"--{name}", type=type(default), default=None)
    return parser.parse_args()


def run(overrides: dict[str, Any]) -> None:
    hyperparameters = load_hyperparameters(
        BASE_DIR / "input" / "config" / "hyperparameters.json",
        overrides,
    )
    logger.info("Hyperparameters: %s", hyperparameters)

    data_dir = BASE_DIR / "input" / "data"
    train_df = load_split(data_dir / "train")
    validation_df = None
    if (data_dir / "validation").exists():
        validation_df = load_split(data_dir / "validation")

    pipeline = train(train_df, hyperparameters, validation_df)
    logger.info("Best iteration: %s",
                pipeline.named_steps["regressor"].best_iteration_)
    log_metrics("train", pipeline, train_df)
    if validation_df is not None:
        log_metrics("validation", pipeline, validation_df)

    model_dir = BASE_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / MODEL_FILE_NAME)
    logger.info("Model saved to %s", model_dir)


def main():
    try:
        run(vars(parse_args()))
    except Exception:
        failure = BASE_DIR / "output" / "failure"
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
