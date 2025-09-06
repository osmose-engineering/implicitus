from fastapi.testclient import TestClient
from design_api.main import app, models


def test_store_and_retrieve_model():
    client = TestClient(app)
    models.clear()
    model = {"id": "abc", "name": "test"}
    resp = client.post("/models", json=model)
    assert resp.status_code == 200
    assert resp.json() == {"id": "abc"}
    resp = client.get("/models/abc")
    assert resp.status_code == 200
    assert resp.json() == model


def test_get_missing_model_returns_404():
    client = TestClient(app)
    models.clear()
    resp = client.get("/models/missing")
    assert resp.status_code == 404
