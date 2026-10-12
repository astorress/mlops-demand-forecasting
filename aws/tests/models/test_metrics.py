import numpy as np
import pytest

from src.models.metrics import compute_metrics, mape


def test_mape_ignores_zero_targets():
    assert mape(np.array([100, 0, 200]), np.array([110, 5,
                                                   180])) == (pytest.approx(10))


def test_registry_layout():
    report = compute_metrics(np.array([10.0, 20.0]), np.array([12.0, 18.0]))
    metrics = report["regression_metrics"]
    assert set(metrics) == {"mse", "rmse", "mae", "mape", "r2"}
    assert metrics["rmse"]["value"] == pytest.approx(2)
    assert metrics["mape"]["value"] == pytest.approx(15)
    assert report["samples"] == 2
