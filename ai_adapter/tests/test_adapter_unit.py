# ai_adapter/tests/test_adapter_unit.py
import json
import pytest
from ai_adapter import adapter
from ai_adapter.schema import implicitus_pb2

class DummyTokenizer:
    eos_token_id = None
    def __call__(self, texts, return_tensors=None, padding=None):
        # Simulate tokenizer output structure
        return {"input_ids": None}
    def decode(self, output, skip_special_tokens=True):
        # If output is a string, return it directly; otherwise return the first element
        if isinstance(output, str):
            return output
        return output[0]

class DummyModel:
    def generate(self, **kwargs):
        # Return a list containing a JSON string
        return [json.dumps({
            "id": "test1",
            "root": { "primitive": { "sphere": { "radius": 1.0 } } }
        })]

class BadModel:
    def generate(self, **kwargs):
        return ["not a JSON!"]

def test_generate_model_parses_valid_json(monkeypatch):
    # Monkey-patch _load_llm to return our dummy tokenizer and model
    monkeypatch.setattr(adapter, "_load_llm", lambda: (DummyTokenizer(), DummyModel()))

    m = adapter.generate_model("make me a sphere", use_llm=True)
    assert isinstance(m, implicitus_pb2.Model)
    assert m.id == "test1"
    assert m.root.primitive.sphere.radius == 1.0

def test_generate_model_raises_on_bad_json(monkeypatch):
    # Monkey-patch _load_llm to return a tokenizer and bad model
    monkeypatch.setattr(adapter, "_load_llm", lambda: (DummyTokenizer(), BadModel()))

    with pytest.raises(ValueError) as exc:
        adapter.generate_model("give me nonsense", use_llm=True)
    assert "LLM output is not valid JSON" in str(exc.value)