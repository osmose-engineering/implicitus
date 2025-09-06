from design_api.main import models


def test_store_and_retrieve_model(client):
    model = {"id": "abc", "name": "test"}
    resp = client.post("/models", json=model)
    assert resp.status_code == 200
    assert resp.json() == {"id": "abc"}
    resp = client.get("/models/abc")
    assert resp.status_code == 200
    assert resp.json() == model


def test_store_model_missing_id_returns_400(client):
    resp = client.post("/models", json={"name": "test"})
    assert resp.status_code == 400


def test_get_missing_model_returns_404(client):
    resp = client.get("/models/missing")
    assert resp.status_code == 404
