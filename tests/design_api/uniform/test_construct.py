

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
