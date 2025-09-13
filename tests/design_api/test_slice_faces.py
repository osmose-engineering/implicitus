import httpx
from .test_slice import DummyClient


def test_cells_include_faces_field(client, monkeypatch):
    capture = {}
    cells_raw = [
        {"vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "faces": [[0, 1, 2]]},
        {"vertices": [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]},
    ]

    from design_api.services.validator import validate_model_spec as real_validate

    def check_validate(model, ignore_unknown_fields=True):
        import json as _json
        assert "cells" not in _json.dumps(model)
        return real_validate(model, ignore_unknown_fields=ignore_unknown_fields)

    monkeypatch.setattr("design_api.main.validate_proto", check_validate)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: DummyClient(capture))

    client.post(
        "/models",
        json={
            "id": "abc",
            "version": 1,
            "root": {
                "children": [
                    {
                        "primitive": {"sphere": {"radius": 1.0}},
                        "modifiers": {
                            "infill": {
                                "pattern": "hex",
                                "cells": cells_raw,
                                "cell_vertices": [],
                                "edge_list": [],
                            }
                        },
                    }
                ]
            },
        },
    )

    resp = client.get("/models/abc/slices?layer=0")
    assert resp.status_code == 200
    expected = [
        {
            "vertices": [
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 1.0, "y": 0.0, "z": 0.0},
                {"x": 0.0, "y": 1.0, "z": 0.0},
            ],
            "faces": [{"vertex_indices": [0, 1, 2]}],
        },
        {
            "vertices": [
                {"x": 0.0, "y": 0.0, "z": 1.0},
                {"x": 1.0, "y": 0.0, "z": 1.0},
                {"x": 0.0, "y": 1.0, "z": 1.0},
            ],
            "faces": [],
        },
    ]
    assert capture["json"]["cells"] == expected
