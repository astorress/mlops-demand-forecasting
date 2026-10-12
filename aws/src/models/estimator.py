"""Definición del modelo: feature engineering + LightGBM en un solo Pipeline
de scikit-learn, así las mismas transformaciones corren al entrenar y al
servir (sin training/serving skew). Los hiperparámetros por defecto son los
del proyecto open source (open-source/src/models/train.py)."""
from typing import Any

from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.features.engineering import add_features

DEFAULT_HYPERPARAMETERS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "early_stopping_rounds": 50,
    "random_state": 42,
}


def build_pipeline(hyperparameters: dict[str, Any] | None = None) -> Pipeline:
    params = {
        **DEFAULT_HYPERPARAMETERS,
        **(hyperparameters or {})
    }
    params.pop("early_stopping_rounds")
    regressor = LGBMRegressor(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        min_child_samples=int(params["min_child_samples"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        random_state=int(params["random_state"]),
        n_jobs=-1,
        verbose=-1,
    )
    return Pipeline(steps=[
        ("features", FunctionTransformer(add_features)),
        ("regressor", regressor),
    ])
