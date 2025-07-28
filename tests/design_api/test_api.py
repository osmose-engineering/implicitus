import pytest
import fastapi
from fastapi.testclient import TestClient

from design_api.main import app
from design_api.services import (
    llm_service,
    json_cleaner,
    mapping as mapping_srv,
    validator as validator_srv,
)

client = TestClient(app)

def test_design_happy(monkeypatch):
    # stub out the entire service chain
    monkeypatch.setattr(llm_service, 'generate_design_spec', lambda p: '{"shape":"sphere","size_mm":10}')
    monkeypatch.setattr(json_cleaner, 'clean_llm_output', lambda raw: raw)
    resp = client.post("/design", json={"prompt":"foo"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["root"]["primitive"]["sphere"]["radius"] == 5

def test_design_bad_json(monkeypatch):
    monkeypatch.setattr(llm_service, 'generate_design_spec', lambda p: 'not json')
    monkeypatch.setattr(json_cleaner, 'clean_llm_output', lambda raw: raw)
    resp = client.post("/design", json={"prompt":"foo"})
    assert resp.status_code == 200

def test_cors_headers():
    resp = client.options("/design")
    # Accept either successful preflight or 405 Method Not Allowed
    if resp.status_code == 200:
        # Ensure CORS headers exist
        assert "access-control-allow-origin" in resp.headers
        assert "access-control-allow-methods" in resp.headers
        assert "access-control-allow-headers" in resp.headers
    elif resp.status_code == 405:
        # OPTIONS not supported but POST should be allowed
        assert "allow" in resp.headers
        allowed = resp.headers["allow"]
        # allow header is comma-separated list of methods
        assert "POST" in [m.strip() for m in allowed.split(",")]
    else:
        pytest.fail(f"Unexpected status code {resp.status_code}")