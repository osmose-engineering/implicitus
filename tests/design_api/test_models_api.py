import importlib
import sys
import types
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    fake_inference = types.SimpleNamespace(generate=lambda *a, **k: "{}")
    monkeypatch.setitem(sys.modules, "ai_adapter.inference_pipeline", fake_inference)

    fake_csg = types.SimpleNamespace(
        review_request=lambda req: ([], "summary"),
        generate_summary=lambda spec: "summary",
        update_request=lambda sid, spec, raw: (spec, "summary"),
    )
    monkeypatch.setitem(sys.modules, "ai_adapter.csg_adapter", fake_csg)

    fake_mapping = types.SimpleNamespace(map_primitive=lambda spec: spec)
    monkeypatch.setitem(sys.modules, "design_api.services.mapping", fake_mapping)

    fake_validator = types.SimpleNamespace(validate_model_spec=lambda spec: spec)
    monkeypatch.setitem(sys.modules, "design_api.services.validator", fake_validator)

    monkeypatch.setenv("MODEL_STORE", str(tmp_path))
    monkeypatch.setenv("SLICER_URL", "http://slicer")

    app_module = importlib.reload(importlib.import_module("design_api.main"))
    app_module.design_states.clear()
    return TestClient(app_module.app)


def test_store_and_slice(client, tmp_path, monkeypatch):
    model = {"id": "test", "root": {"primitive": {"sphere": {"radius": 1.0}}}}

    resp = client.post("/models", json=model)
    assert resp.status_code == 200
    assert resp.json()["id"] == "test"

    # model persisted
    assert (tmp_path / "test.json").exists()

    import requests
    fake_resp = types.SimpleNamespace(json=lambda: {"contours": []}, raise_for_status=lambda: None)
    monkeypatch.setattr(requests, "post", lambda url, json: fake_resp)

    resp = client.get("/models/test/slices", params={"layer": 0.0})
    assert resp.status_code == 200
    assert "contours" in resp.json()
