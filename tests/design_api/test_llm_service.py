import pytest
import json

from design_api.services.llm_service import generate_design_spec
import ai_adapter.inference_pipeline as inference_pipeline

def test_generate_design_spec_happy(monkeypatch):
    monkeypatch.setattr(inference_pipeline, 'generate', lambda prompt: '{"foo": 1}')
    result = generate_design_spec("foo")
    assert isinstance(result, dict)
    assert 'shape' in result and 'size_mm' in result

def test_generate_design_spec_error(monkeypatch):
    def bad(prompt): raise RuntimeError("oom")
    monkeypatch.setattr(inference_pipeline, 'generate', bad)
    result = generate_design_spec("foo")
    assert isinstance(result, dict)
    assert 'shape' in result and 'size_mm' in result