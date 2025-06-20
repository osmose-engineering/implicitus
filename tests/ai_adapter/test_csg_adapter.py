# tests/ai_adapter/test_csg_adapter.py
import pytest
from ai_adapter import csg_adapter
from google.protobuf.json_format import MessageToDict

def test_parse_sphere_model():
    raw = '{"shape":"sphere","size_mm":10}'
    model = csg_adapter.parse_and_build_model(raw)
    d = MessageToDict(model)
    assert d["root"]["primitive"]["sphere"]["radius"] == 5.0

def test_invalid_shape():
    with pytest.raises(ValueError):
        csg_adapter.parse_and_build_model('{"shape":"torus","size_mm":10}')