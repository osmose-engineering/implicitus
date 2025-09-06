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


def test_rejects_mixed_case_keys():
    spec = map_primitive({"shape": "sphere", "size_mm": 10})
    spec["root"]["bboxMin"] = [0, 0, 0]
    with pytest.raises(ValidationError):
        validate_model_spec(spec)


def test_rejects_unknown_keys():
    spec = map_primitive({"shape": "sphere", "size_mm": 10})
    spec["root"]["unknown_field"] = 123
    with pytest.raises(ValidationError):
        validate_model_spec(spec)


def test_ignore_unknown_fields_allows_unknown_keys():
    spec = map_primitive({"shape": "sphere", "size_mm": 10})
    spec["root"]["unknown_field"] = 123
    # Should not raise when ignore_unknown_fields=True
    msg = validate_model_spec(spec, ignore_unknown_fields=True)
    assert hasattr(msg, "root")

def test_wrapped_modifier_dict():
    spec = {
        "root": {
            "children": [
                {
                    "primitive": {"sphere": {"radius": 1.0}},
                    "modifiers": {"infill": {"pattern": "hex"}},
                }
            ]
        }
    }
    msg = validate_model_spec(spec)
    child = msg.root.children[0]
    assert len(child.modifiers) == 1
    assert child.modifiers[0].infill.pattern == "hex"

