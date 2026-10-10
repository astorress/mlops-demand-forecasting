"""Fixtures compartidos: dataset generado una vez por sesión y splits procesados."""

import importlib.util
from pathlib import Path

import pytest

from src.data.preprocessing import create_features, save_processed_data, train_val_test_split

ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_dataset", ROOT / "data" / "generate_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def raw_df(tmp_path_factory):
    output = tmp_path_factory.mktemp("raw") / "demand.csv"
    return _load_generator().generate_dataset(output_path=str(output), seed=42)


@pytest.fixture(scope="session")
def processed_dir(raw_df, tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    train, val, test = train_val_test_split(create_features(raw_df))
    for name, split in [("train", train), ("val", val), ("test", test)]:
        save_processed_data(split, str(out), f"{name}.parquet")
    return out


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    """MLflow en una carpeta temporal: los tests no tocan el servidor real."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    return tmp_path
