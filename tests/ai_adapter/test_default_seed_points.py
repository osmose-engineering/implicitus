import pytest

from ai_adapter import csg_adapter
from constants import DEFAULT_VORONOI_SEEDS


def test_interpret_llm_request_sets_default_num_points(monkeypatch):
    spec = {"shape": "cube", "size_mm": 20, "infill": {"pattern": "voronoi"}}

    result = csg_adapter.interpret_llm_request(spec)
    infill = result["primitives"][0]["modifiers"]["infill"]

    assert infill["num_points"] == DEFAULT_VORONOI_SEEDS
    assert "seed_points" not in infill


def test_update_request_sets_default_num_points(monkeypatch):
    monkeypatch.setattr(csg_adapter, "review_request", lambda data: (data["spec"], ""))
    spec = [
        {
            "primitive": {"cube": {"size": 20}},
            "modifiers": {"infill": {"pattern": "voronoi"}},
        }
    ]

    new_spec, _ = csg_adapter.update_request("sid", spec, "raw")
    infill = new_spec[0]["modifiers"]["infill"]

    assert infill["num_points"] == DEFAULT_VORONOI_SEEDS
    assert "seed_points" not in infill
