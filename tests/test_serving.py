import numpy as np
import pytest

from src.serving import flask_app

PAYLOAD = {
    "temperature_c": 24.5,
    "hour": 19,
    "day_of_week": 2,
    "month": 7,
    "is_weekend": 0,
    "is_holiday": 0,
    "demand_lag_24h": 1450.0,
    "demand_lag_168h": 1420.0,
    "demand_rolling_mean_7d": 1300.0,
}


class FakeModel:
    def predict(self, X):
        return np.full(len(X), 1500.0)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(flask_app.model_manager, "model", FakeModel())
    return flask_app.app.test_client()


def test_build_features_matches_training_columns():
    from src.data.preprocessing import FEATURE_COLS

    X = flask_app.build_features(dict(PAYLOAD))
    assert list(X.columns) == FEATURE_COLS
    assert X.loc[0, "is_peak_hour"] == 1


def test_predict(client):
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 200
    assert response.get_json()["prediction"] == 1500.0


def test_predict_rejects_invalid_hour(client):
    response = client.post("/predict", json={**PAYLOAD, "hour": 30})
    assert response.status_code == 422


def test_health_reports_loaded_model(client):
    assert client.get("/health").get_json()["model_loaded"] is True
