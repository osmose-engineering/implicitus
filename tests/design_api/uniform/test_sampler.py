import numpy as np
import pytest

from design_api.services.voronoi_gen.uniform.sampler import compute_medial_axis, trace_hexagon


class DummyMesh:
    def __init__(self, vertices):
        self.vertices = np.array(vertices)


def test_compute_medial_axis_simple():
    # Simple tetrahedron vertices
    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    mesh = DummyMesh(vertices)
    medial = compute_medial_axis(mesh)
    # Should return a non-empty (M,3) numpy array
    assert isinstance(medial, np.ndarray)
    assert medial.ndim == 2 and medial.shape[1] == 3
    assert medial.shape[0] > 0


def test_compute_medial_axis_bounds_clipping():
    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [100.0, 100.0, 100.0],
    ]
    mesh = DummyMesh(vertices)

    # With default tolerance the Voronoi vertex lies outside the bounding box
    medial_default = compute_medial_axis(mesh)
    assert medial_default.shape == (0, 3)

    # Expanding the bounds should retain the vertex
    medial_tolerant = compute_medial_axis(mesh, tol=60.0)
    assert medial_tolerant.shape[0] == 1
    assert np.allclose(medial_tolerant[0], np.array([0.5, 0.5, 149.0]))


def test_trace_hexagon_fallback():
    seed = np.array([0.0, 0.0, 0.0])
    # All medial points are behind the seed (negative x direction)
    medial = np.tile(np.array([-1.0, 0.0, 0.0]), (10, 1))
    plane_normal = np.array([0.0, 0.0, 1.0])
    hex_pts = trace_hexagon(seed, medial, plane_normal, max_distance=2.0)
    # Should be a (6,3) array of points not exceeding the expected radius
    assert isinstance(hex_pts, np.ndarray)
    assert hex_pts.shape == (6, 3)
    dists = np.linalg.norm(hex_pts - seed, axis=1)
    max_dist = 2.0
    assert np.all(dists >= 0)
    assert np.all(dists <= max_dist + 1e-6)
    # At least one ray should hit the bounding box before reaching max_dist
    assert np.any(dists < max_dist)


def test_trace_hexagon_basic():
    seed = np.array([0.0, 0.0, 0.0])
    # Medial points on a perfect circle of radius 5 in the XY plane
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    medial = np.stack([5 * np.cos(angles), 5 * np.sin(angles), np.zeros_like(angles)], axis=1)
    plane_normal = np.array([0.0, 0.0, 1.0])
    hex_pts = trace_hexagon(seed, medial, plane_normal)
    # All distances should be approximately the expected Voronoi radius (R / sqrt(3))
    dists = np.linalg.norm(hex_pts - seed, axis=1)
    assert np.allclose(dists, 5.0 / np.sqrt(3.0), atol=1e-6)
    # Should produce exactly 6 unique points
    unique_pts = {tuple(pt) for pt in hex_pts}
    assert len(unique_pts) == 6


def test_trace_hexagon_no_intersection_error():
    seed = np.array([0.0, 0.0, 0.0])
    medial = np.tile(np.array([-1.0, 0.0, 0.0]), (10, 1))
    plane_normal = np.array([0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        trace_hexagon(seed, medial, plane_normal)


def test_trace_hexagon_origin_medial_cluster(monkeypatch):
    seed = np.array([0.0, -1.0, 0.0])  # point on unit sphere in XY plane
    medial = np.zeros((12, 3))  # all medial points collapse at origin
    plane_normal = np.array([0.0, 0.0, 1.0])

    # Disable regularization to expose uneven edge lengths
    monkeypatch.setattr(
        "design_api.services.voronoi_gen.uniform.regularizer.regularize_hexagon",
        lambda pts, normal: pts,
    )

    hex_pts, used_fallback = trace_hexagon(
        seed, medial, plane_normal, max_distance=5.0, report_method=True
    )
    assert used_fallback is True

    edges = np.linalg.norm(np.roll(hex_pts, -1, axis=0) - hex_pts, axis=1)
    # Fallback rays yield a highly irregular hexagon
    assert np.ptp(edges) > 0.3

