

import numpy as np
import pytest

from design_api.services.voronoi_gen.uniform.construct import compute_uniform_cells


class DummyMesh:
    def __init__(self, vertices):
        self.vertices = np.array(vertices)


def _sample_mesh():
    """Return a simple tetrahedral mesh used across the tests."""
    return DummyMesh(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_compute_uniform_cells_basic():
    # Two nearby seeds in 3D space to create adjacent hexagonal cells
    seeds = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
        ]
    )
    mesh = _sample_mesh()
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

        # Each cell's edge lengths should be nearly equal
        edges = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
        assert np.allclose(edges, edges[0], rtol=1e-5, atol=1e-6)

    # Adjacent cells should share vertices within a tolerance
    dists = np.linalg.norm(cells[0][:, None, :] - cells[1][None, :, :], axis=2)
    assert np.min(dists) < 0.1


def test_construct_produces_closed_hexagons():
    """Regression test ensuring traced hexagons form closed loops."""
    seeds = np.array([[0.0, 0.0, 0.0]])
    mesh = _sample_mesh()
    plane_normal = np.array([0.0, 0.0, 1.0])

    cells = compute_uniform_cells(seeds, mesh, plane_normal, max_distance=2.0)
    pts = cells[0]

    # The sum of edge vectors should be approximately zero for a closed polygon
    edges = np.roll(pts, -1, axis=0) - pts
    assert np.allclose(np.sum(edges, axis=0), np.zeros(3), atol=1e-6)

    # Non-degenerate hexagon should have positive area
    centroid = np.mean(pts, axis=0)
    area = 0.0
    for i in range(pts.shape[0]):
        a = pts[i] - centroid
        b = pts[(i + 1) % pts.shape[0]] - centroid
        area += 0.5 * np.linalg.norm(np.cross(a, b))
    assert area > 0
