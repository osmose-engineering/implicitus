

import numpy as np
import pytest
import logging

from design_api.services.voronoi_gen.uniform.construct import compute_uniform_cells

class DummyMesh:
    def __init__(self, vertices):
        self.vertices = np.array(vertices)

def test_compute_uniform_cells_basic():
    # Two seeds in 3D space
    seeds = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ])
    # Simple tetrahedral mesh for medial-axis extraction
    mesh = DummyMesh([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    plane_normal = np.array([0.0, 0.0, 1.0])
    # Compute uniform cells with a fallback distance
    cells = compute_uniform_cells(seeds, mesh, plane_normal, max_distance=2.0)

    # Should return a dict mapping each seed index to a (6,3) array
    assert isinstance(cells, dict)
    assert set(cells.keys()) == {0, 1}
    for idx, pts in cells.items():
        assert isinstance(pts, np.ndarray)
        assert pts.shape == (6, 3)
        # Ensure each point is finite
        assert np.all(np.isfinite(pts))


def test_shared_vertex_alignment(monkeypatch, caplog):
    seeds = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mesh = DummyMesh([[0.0, 0.0, 0.0]])

    def fake_medial_axis(_mesh):  # pragma: no cover - simple stub
        return np.zeros((1, 3))

    base_hex = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.5, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [-0.5, 0.5, 0.0],
        ]
    )

    def fake_trace_hexagon(seed, medial, normal, max_distance):  # pragma: no cover
        if np.allclose(seed, seeds[0]):
            return base_hex.copy()
        perturbed = base_hex.copy()
        perturbed[0] += [8e-4, 0.0, 0.0]
        perturbed[1] += [-8e-4, 0.0, 0.0]
        return perturbed

    monkeypatch.setattr(
        "design_api.services.voronoi_gen.uniform.construct.compute_medial_axis", fake_medial_axis
    )
    monkeypatch.setattr(
        "design_api.services.voronoi_gen.uniform.construct.trace_hexagon", fake_trace_hexagon
    )

    plane_normal = np.array([0.0, 0.0, 1.0])
    with caplog.at_level(logging.INFO):
        cells = compute_uniform_cells(
            seeds, mesh, plane_normal, max_distance=2.0, vertex_tolerance=1e-4
        )

    # Vertices 0 and 1 should now be identical between the two cells
    assert np.allclose(cells[0][0], cells[1][0])
    assert np.allclose(cells[0][1], cells[1][1])

    # Expect a warning about mismatched edges
    warnings = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert warnings and "exceeds tolerance" in warnings[0].message
