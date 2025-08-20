import importlib
import sys
import types

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    """Return a TestClient with external deps stubbed out."""
    fake_inference = types.SimpleNamespace(generate=lambda *a, **k: "{}")
    monkeypatch.setitem(sys.modules, "ai_adapter.inference_pipeline", fake_inference)

    fake_csg = types.SimpleNamespace(
        review_request=lambda req: ([], "summary"),
        generate_summary=lambda spec: "summary",
        update_request=lambda sid, spec, raw: (spec, "summary"),
    )
    monkeypatch.setitem(sys.modules, "ai_adapter.csg_adapter", fake_csg)

    fake_mapping = types.SimpleNamespace(
        map_primitive=lambda spec: {
            "root": {"primitive": {"sphere": {"radius": spec.get("size_mm", 0) / 2}}}
        }
    )
    monkeypatch.setitem(sys.modules, "design_api.services.mapping", fake_mapping)

    fake_validator = types.SimpleNamespace(validate_model_spec=lambda spec: spec)
    monkeypatch.setitem(sys.modules, "design_api.services.validator", fake_validator)

    app_module = importlib.import_module("design_api.main")
    app_module.design_states.clear()
    return TestClient(app_module.app)


def test_design_happy(client, monkeypatch):
    import design_api.main as main

    monkeypatch.setattr(
        main, "generate_design_spec", lambda p: '{"shape":"sphere","size_mm":10}'
    )
    monkeypatch.setattr(main, "clean_llm_output", lambda raw: raw)

    resp = client.post("/design", json={"prompt": "foo"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["root"]["primitive"]["sphere"]["radius"] == 5


def test_design_bad_json(client, monkeypatch):
    import design_api.main as main

    monkeypatch.setattr(main, "generate_design_spec", lambda p: "not json")
    monkeypatch.setattr(main, "clean_llm_output", lambda raw: raw)

    resp = client.post("/design", json={"prompt": "foo"})
    assert resp.status_code == 502


def test_cors_headers(client):
    resp = client.options("/design")
    if resp.status_code == 200:
        assert "access-control-allow-origin" in resp.headers
        assert "access-control-allow-methods" in resp.headers
        assert "access-control-allow-headers" in resp.headers
    elif resp.status_code == 405:
        allowed = resp.headers.get("allow", "")
        assert "POST" in [m.strip() for m in allowed.split(",")]
    else:
        pytest.fail(f"Unexpected status code {resp.status_code}")


def test_review_update_submit_flow(client):
    # Start with review to create a session
    resp = client.post("/design/review", json={"raw": "spec"})
    assert resp.status_code == 200
    sid = resp.json()["sid"]

    # Update the session
    resp = client.post(
        "/design/update",
        json={"sid": sid, "raw": "more", "spec": []},
    )
    assert resp.status_code == 200

    # Submit the design
    resp = client.post(f"/design/submit?sid={sid}", json={})
    assert resp.status_code == 200
    assert resp.json()["sid"] == sid
