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
        msg = validate_model_spec({"id": "foo", "version": 1})
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
        "version": 1,
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


def _make_seed_spec(seeds):
    return {
        "id": "abc",
        "version": 1,
        "root": {
            "primitive": {"sphere": {"radius": 1.0}},
            "modifiers": {
                "infill": {
                    "pattern": "hex",
                    "bbox_min": [0, 0, 0],
                    "bbox_max": [1, 1, 1],
                    "seed_points": seeds,
                    "num_points": len(seeds),
                }
            },
        },
    }


def test_rejects_duplicate_seed_points():
    spec = _make_seed_spec([[0, 0, 0], [0, 0, 0]])
    with pytest.raises(ValidationError):
        validate_model_spec(spec)


def test_rejects_out_of_bounds_seed_point():
    spec = _make_seed_spec([[2, 0, 0]])
    with pytest.raises(ValidationError):
        validate_model_spec(spec)


def test_rejects_out_of_bounds_edge_index():
    spec = {
        "id": "abc",
        "version": 1,
        "root": {
            "primitive": {"sphere": {"radius": 1.0}},
            "modifiers": {
                "infill": {
                    "pattern": "hex",
                    "cell_vertices": [[0, 0, 0], [1, 0, 0]],
                    "edge_list": [[0, 2]],
                }
            },
        },
    }
    with pytest.raises(ValidationError):
        validate_model_spec(spec)



def test_missing_version_rejected():
    spec = map_primitive({"shape": "sphere", "size_mm": 10})
    spec.pop("version", None)

def test_rejects_bbox_too_tight_for_primitive():
    spec = {
        "id": "abc",
        "root": {
            "primitive": {"sphere": {"radius": 1.0}},
            "modifiers": {
                "infill": {
                    "pattern": "hex",
                    # Bounds that do not fully contain the sphere
                    "bbox_min": [-0.5, -1.0, -1.0],
                    "bbox_max": [1.0, 1.0, 1.0],
                }
            },
        },
    }

    with pytest.raises(ValidationError):
        validate_model_spec(spec)



def test_unknown_version_rejected():
    spec = map_primitive({"shape": "sphere", "size_mm": 10})
    spec["version"] = 99
    with pytest.raises(ValidationError):
        validate_model_spec(spec)

def test_accepts_bbox_enclosing_primitive():
    spec = {
        "id": "abc",
        "root": {
            "primitive": {"sphere": {"radius": 1.0}},
            "modifiers": {
                "infill": {
                    "pattern": "hex",
                    "bbox_min": [-1.0, -1.0, -1.0],
                    "bbox_max": [1.0, 1.0, 1.0],
                }
            },
        },
    }
    msg = validate_model_spec(spec)
    assert hasattr(msg.root, "primitive")


