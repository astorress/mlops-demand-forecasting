import joblib
import pytest

from src.features.schema import FEATURES
from src.serving import serve


@pytest.fixture
def client(tmp_path, monkeypatch, trained_model):
    joblib.dump(trained_model, tmp_path / "model.joblib")
    monkeypatch.setattr(serve, "MODEL_DIR", str(tmp_path))
    serve.get_model.cache_clear()
    yield serve.app.test_client()
    serve.get_model.cache_clear()


def test_ping(client):
    assert client.get("/ping").status_code == 200


def test_invocations_csv_to_json(client, curated_df):
    body = curated_df[FEATURES].head(4).to_csv(header=False, index=False)
    response = client.post("/invocations",
                           data=body,
                           content_type="text/csv",
                           headers={
                               "Accept": "application/json"
                           })
    assert response.status_code == 200
    assert len(response.get_json()["predictions"]) == 4


def test_invocations_bad_request(client):
    response = client.post("/invocations", data="{}", content_type="text/plain")
    assert response.status_code == 400


def test_ping_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "MODEL_DIR", str(tmp_path))
    serve.get_model.cache_clear()
    assert serve.app.test_client().get("/ping").status_code == 503
    serve.get_model.cache_clear()
