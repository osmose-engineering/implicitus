import types
import sys
import pytest
from fastapi.testclient import TestClient


def test_review_returns_edges(monkeypatch):
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.pipeline = lambda *args, **kwargs: None
    transformers_stub.AutoTokenizer = object
    sys.modules.setdefault("transformers", transformers_stub)

    ai_adapter_mod = types.ModuleType("ai_adapter")
    ai_adapter_mod.__path__ = []  # mark as package
    schema_mod = types.ModuleType("ai_adapter.schema")
    schema_mod.__path__ = []
    pb2_mod = types.ModuleType("ai_adapter.schema.implicitus_pb2")
    for name in [
        "Primitive",
        "Modifier",
        "Infill",
        "Shell",
        "BooleanOp",
        "VoronoiLattice",
    ]:
        setattr(pb2_mod, name, object)
    schema_mod.implicitus_pb2 = pb2_mod
    ai_adapter_mod.schema = schema_mod
    csg_mod = types.ModuleType("ai_adapter.csg_adapter")
    csg_mod.review_request = lambda req: ({}, "")
    csg_mod.generate_summary = lambda *args, **kwargs: ""
    csg_mod.update_request = lambda *args, **kwargs: ({}, "")
    ai_adapter_mod.csg_adapter = csg_mod
    inference_mod = types.ModuleType("ai_adapter.inference_pipeline")
    inference_mod.generate = lambda *args, **kwargs: ""
    ai_adapter_mod.inference_pipeline = inference_mod
    sys.modules.setdefault("ai_adapter", ai_adapter_mod)
    sys.modules.setdefault("ai_adapter.schema", schema_mod)
    sys.modules.setdefault("ai_adapter.schema.implicitus_pb2", pb2_mod)
    sys.modules.setdefault("ai_adapter.csg_adapter", csg_mod)
    sys.modules.setdefault("ai_adapter.inference_pipeline", inference_mod)

    validator_stub = types.ModuleType("design_api.services.validator")
    validator_stub.validate_model_spec = lambda spec: spec
    sys.modules.setdefault("design_api.services.validator", validator_stub)

    from design_api.main import app
    import design_api.main as design_main

    client = TestClient(app)
    def fake_review_request(req):
        spec = [
            {
                "primitive": {"sphere": {"radius": 1.0}},
                "modifiers": {
                    "infill": {
                        "pattern": "voronoi",
                        "mode": "uniform",
                        "spacing": 1.0,
                        "seed_points": [[0.0, 0.0, 0.0]],
                        "bbox_min": [-1.0, -1.0, -1.0],
                        "bbox_max": [1.0, 1.0, 1.0],
                    }
                },
            }
        ]
        return spec, "summary"

    monkeypatch.setattr(design_main, "review_request", fake_review_request)
    monkeypatch.setattr(design_main, "log_turn", lambda *args, **kwargs: None)

    resp = client.post("/design/review", json={})
    assert resp.status_code == 200
    data = resp.json()
    edges = data["spec"][0]["modifiers"]["infill"]["edges"]
    points = data["spec"][0]["modifiers"]["infill"]["seed_points"]
    assert isinstance(edges, list)
    assert len(edges) > 0
    # ensure all edge indices reference valid points
    max_idx = max(max(e) for e in edges)
    assert max_idx < len(points)
