import pytest

from ai_adapter import csg_adapter


def _make_seeds(n):
    return [(float(i), 0.0, 0.0) for i in range(n)]


def test_interpret_llm_request_uses_default_seed_points(monkeypatch):
    # Arrange: stub seed sampler to return many points
    monkeypatch.setattr(
        csg_adapter.rust_primitives,
        "sample_inside",
        lambda shape, min_dist: _make_seeds(csg_adapter.DEFAULT_SEED_POINTS + 500),
    )
    spec = {"shape": "cube", "size_mm": 20, "infill": {"pattern": "voronoi"}}

    # Act
    result = csg_adapter.interpret_llm_request(spec)
    infill = result["primitives"][0]["modifiers"]["infill"]

    # Assert
    assert infill["num_points"] == csg_adapter.DEFAULT_SEED_POINTS
    assert len(infill["seed_points"]) == csg_adapter.DEFAULT_SEED_POINTS


def test_update_request_uses_default_seed_points(monkeypatch):
    # Arrange: stub sampler and review_request
    monkeypatch.setattr(
        csg_adapter.rust_primitives,
        "sample_inside",
        lambda shape, min_dist: _make_seeds(csg_adapter.DEFAULT_SEED_POINTS + 500),
    )
    monkeypatch.setattr(csg_adapter, "review_request", lambda data: (data["spec"], ""))
    spec = [
        {
            "primitive": {"cube": {"size": 20}},
            "modifiers": {"infill": {"pattern": "voronoi"}},
        }
    ]

    # Act
    new_spec, _ = csg_adapter.update_request("sid", spec, "raw")
    infill = new_spec[0]["modifiers"]["infill"]

    # Assert
    assert infill["num_points"] == csg_adapter.DEFAULT_SEED_POINTS
    assert len(infill["seed_points"]) == csg_adapter.DEFAULT_SEED_POINTS
