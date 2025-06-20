# tests/ai_adapter/test_inference_pipeline.py
import pytest
from ai_adapter import inference_pipeline

class DummyPipe:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __call__(self, messages):
        # echo back the user content as JSON
        return [{"generated_text": messages[-1]["content"]}]

def test_prompt_and_return(monkeypatch):
    # stub out transformers.pipeline
    monkeypatch.setattr(inference_pipeline, "pipeline",
                        lambda task, **opts: DummyPipe(**opts))
    raw = inference_pipeline.generate("Make a 5mm cube")
    assert '"shape": "cube"' in raw
    # ensure system prompt included
    assert inference_pipeline.SYSTEM_PROMPT in inference_pipeline.last_prompt