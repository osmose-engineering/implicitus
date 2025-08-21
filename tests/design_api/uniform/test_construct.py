

import numpy as np
import pytest

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


def test_uniform_cells_hex_shared_edges(monkeypatch):
    import math
    from design_api.services.voronoi_gen.voronoi_gen import compute_voronoi_adjacency

    R = 1.0
    dx = math.sqrt(3) * R
    seeds = np.array([
        [0.0, 0.0, 0.0],
        [dx, 0.0, 0.0],
    ])

    monkeypatch.setattr(
        "design_api.services.voronoi_gen.uniform.sampler.compute_medial_axis",
        lambda mesh: np.zeros((0, 3)),
    )
    monkeypatch.setattr(
        "design_api.services.voronoi_gen.uniform.construct.compute_medial_axis",
        lambda mesh: np.zeros((0, 3)),
    )

    def fake_trace_hexagon(seed_pt, medial_points, plane_normal, max_distance=None):
        angles = np.radians(np.arange(6) * 60 + 30)
        pts = []
        for ang in angles:
            pts.append(seed_pt + np.array([math.cos(ang), math.sin(ang), 0.0]) * R)
        return np.array(pts)

    monkeypatch.setattr(
        "design_api.services.voronoi_gen.uniform.sampler.trace_hexagon",
        fake_trace_hexagon,
    )
    monkeypatch.setattr(
        "design_api.services.voronoi_gen.uniform.construct.trace_hexagon",
        fake_trace_hexagon,
    )

    mesh = DummyMesh([])
    plane_normal = np.array([0.0, 0.0, 1.0])
    cells = compute_uniform_cells(seeds, mesh, plane_normal)

    assert all(cell.shape == (6, 3) for cell in cells.values())

    shared = []
    for i, v in enumerate(cells[0]):
        for j, w in enumerate(cells[1]):
            if np.allclose(v, w, atol=1e-6):
                shared.append((i, j))
    assert len(shared) == 2
    i0, i1 = shared[0][0], shared[1][0]
    assert abs(i0 - i1) in (1, 5)

    bbox_min = (-1.0, -1.0, -1.0)
    bbox_max = (dx + 1.0, 1.0, 1.0)
    adj = compute_voronoi_adjacency(seeds.tolist(), bbox_min, bbox_max, resolution=(16, 16, 16))
    assert set(adj[0]) == {1}
    assert set(adj[1]) == {0}

