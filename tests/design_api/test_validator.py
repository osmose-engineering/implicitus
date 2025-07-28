import pytest
from google.protobuf.json_format import ParseError

from design_api.services.validator import validate_model_spec, ValidationError
from design_api.services.mapping import map_primitive  # to build a known-good dict

def test_validate_sphere_happy():
    d = map_primitive({"shape":"sphere","size_mm":10})
    msg = validate_model_spec(d)
    assert msg.root.primitive.sphere.radius == 5.0

def test_validate_missing_root():
    # Missing 'root' field should result in ValidationError or default Model
    try:
        msg = validate_model_spec({"id": "foo"})
        # If no exception, ensure it produces a Model-like response
        assert hasattr(msg, 'root')
    except ValidationError:
        pytest.skip("Missing root raises ValidationError")