"""Lógica de entrenamiento y contrato del training job de SageMaker para la
imagen propia (`<imagen> train`): rutas /opt/ml, hiperparámetros string,
archivo de falla."""
import json

import joblib
import pytest

from src.models import train as train_module
from src.models.train import load_hyperparameters


def test_training_learns_signal(trained_model, splits):
    from src.pipelines.training.evaluate import evaluate

    metrics = evaluate(trained_model, splits[2])["regression_metrics"]
    # Ruido gaussiano σ = 50 kWh sobre ~1 000 kWh: el piso es ~4 % de MAPE.
    assert metrics["mape"]["value"] < 8
    assert metrics["r2"]["value"] > 0.6


def test_early_stopping_uses_validation(trained_model):
    regressor = trained_model.named_steps["regressor"]
    assert 0 < regressor.best_iteration_ <= 200


def test_training_job_contract(tmp_path, monkeypatch, splits):
    train_df, validation_df, _ = splits
    for name, frame in (("train", train_df), ("validation", validation_df)):
        channel = tmp_path / "input" / "data" / name
        channel.mkdir(parents=True)
        frame.to_csv(channel / f"{name}.csv", index=False)
    config = tmp_path / "input" / "config"
    config.mkdir(parents=True)
    # SageMaker siempre envía strings.
    (config / "hyperparameters.json").write_text(
        json.dumps({
            "n_estimators": "40",
            "learning_rate": "0.1",
        }))
    monkeypatch.setattr(train_module, "BASE_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["train.py"])

    train_module.main()

    model = joblib.load(tmp_path / "model" / train_module.MODEL_FILE_NAME)
    assert model.named_steps["regressor"].n_estimators == 40


def test_hyperparameters_cast_and_override(tmp_path):
    path = tmp_path / "hyperparameters.json"
    path.write_text(json.dumps({
        "num_leaves": "31",
        "learning_rate": "0.2",
    }))
    params = load_hyperparameters(path, {
        "learning_rate": 0.3
    })
    assert params["num_leaves"] == 31
    assert params["learning_rate"] == 0.3
    assert params["n_estimators"] == 500


def test_training_failure_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "BASE_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["train.py"])
    with pytest.raises(Exception):
        train_module.main()
    assert (tmp_path / "output" / "failure").read_text()
