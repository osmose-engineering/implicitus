import pytest

from design_api.services.validator import validate_model_spec, ValidationError


def _base_spec():
    return {
        "id": "abc",
        "version": 1,
        "root": {
            "primitive": {"sphere": {"radius": 1.0}},
            "modifiers": {
                "infill": {
                    "pattern": "hex",
                    "cell_vertices": [
                        [0, 0, 0],
                        [1, 0, 0],
                        [1, 1, 0],
                    ],
                }
            },
        },
    }


def test_rejects_edges_not_forming_loop():
    spec = _base_spec()
    spec["root"]["modifiers"]["infill"]["edge_list"] = [[0, 1], [1, 2]]
    with pytest.raises(ValidationError):
        validate_model_spec(spec)


def test_rejects_duplicate_edges():
    spec = _base_spec()
    spec["root"]["modifiers"]["infill"]["edge_list"] = [
        [0, 1],
        [1, 2],
        [2, 0],
        [0, 1],
    ]
    with pytest.raises(ValidationError):
        validate_model_spec(spec)


def test_accepts_closed_loop():
    spec = _base_spec()
    spec["root"]["modifiers"]["infill"]["edge_list"] = [[0, 1], [1, 2], [2, 0]]
    validate_model_spec(spec, ignore_unknown_fields=True)
