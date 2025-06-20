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
    assert resp.status_code == 502

def test_cors_headers():
    resp = client.options("/design")
    assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"