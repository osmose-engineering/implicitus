import pytest

from design_api.services.llm_service import generate_design_spec
import ai_adapter.inference_pipeline as inference_pipeline

def test_generate_design_spec_happy(monkeypatch):
    monkeypatch.setattr(inference_pipeline, 'generate', lambda prompt: 'RAW_JSON')
    assert generate_design_spec("foo") == 'RAW_JSON'

def test_generate_design_spec_error(monkeypatch):
    def bad(prompt): raise RuntimeError("oom")
    monkeypatch.setattr(inference_pipeline, 'generate', bad)
    with pytest.raises(RuntimeError):
        generate_design_spec("foo")