import uuid

from design_api.main import models


def test_store_and_retrieve_model(client):
    model = {"id": "abc", "name": "test"}
    resp = client.post("/models", json=model)
    assert resp.status_code == 200
    assert resp.json() == {"id": "abc"}
    resp = client.get("/models/abc")
    assert resp.status_code == 200
    assert resp.json() == model


def test_store_model_missing_id_generates_uuid(client):
    resp = client.post("/models", json={"name": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    generated_id = data["id"]
    # validate UUID format
    uuid.UUID(generated_id)
    # verify model is stored under generated id
    resp = client.get(f"/models/{generated_id}")
    assert resp.status_code == 200
    assert resp.json() == {"id": generated_id, "name": "test"}


def test_get_missing_model_returns_404(client):
    resp = client.get("/models/missing")
    assert resp.status_code == 404
