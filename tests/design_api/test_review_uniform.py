import types
import sys
import numpy as np
import pytest


@pytest.mark.asyncio
async def test_review_generates_uniform_cells(monkeypatch):
    dummy_inf = types.ModuleType("ai_adapter.inference_pipeline")
    dummy_inf.generate = lambda prompt: "{}"
    sys.modules.setdefault("ai_adapter.inference_pipeline", dummy_inf)

    schema_pkg = types.ModuleType("ai_adapter.schema")
    dummy_proto = types.ModuleType("ai_adapter.schema.implicitus_pb2")
    class Dummy:
        pass
    for name in ["Primitive", "Modifier", "Infill", "Shell", "BooleanOp", "VoronoiLattice", "Model"]:
        setattr(dummy_proto, name, Dummy)
    sys.modules.setdefault("ai_adapter.schema", schema_pkg)
    sys.modules.setdefault("ai_adapter.schema.implicitus_pb2", dummy_proto)

    map_mod = types.ModuleType("design_api.services.mapping")
    map_mod.map_primitive = lambda spec: spec
    sys.modules.setdefault("design_api.services.mapping", map_mod)

    validator_mod = types.ModuleType("design_api.services.validator")
    validator_mod.validate_model_spec = lambda spec: spec
    sys.modules.setdefault("design_api.services.validator", validator_mod)

    csg_mod = types.ModuleType("ai_adapter.csg_adapter")
    csg_mod.review_request = lambda req: ({}, "")
    csg_mod.generate_summary = lambda *a, **k: ""
    csg_mod.update_request = lambda *a, **k: ({}, "")
    sys.modules.setdefault("ai_adapter.csg_adapter", csg_mod)

    from design_api import main

    spec = [{
        "primitive": {"sphere": {"radius": 1.0}},
        "modifiers": {
            "infill": {
                "pattern": "voronoi",
                "mode": "uniform",
                "seed_points": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                "bbox_min": [-1, -1, -1],
                "bbox_max": [1, 1, 1]
            }
        }
    }]

    def fake_review_request(req):
        return (spec, "summary")

    monkeypatch.setattr(main, "review_request", fake_review_request)
    monkeypatch.setattr(main, "log_turn", lambda *a, **k: None)

    resp = await main.review({})
    infill = resp["spec"][0]["modifiers"]["infill"]
    cells = infill.get("cells")
    assert cells and len(cells) == 3
    for poly in cells:
        assert len(poly) == 6
        assert np.isfinite(poly).all()
    # ensure uniform flag propagated from mode
    assert infill.get("uniform") is True
