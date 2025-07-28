

import numpy as np
import pytest

from design_api.services.voronoi_gen import (
    compute_voronoi_adjacency,
    _call_sdf,
    estimate_normal,
    estimate_hessian,
    compute_fillet_radius,
    smooth_union,
)

def test_compute_voronoi_adjacency_basic():
    seeds = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    adj = compute_voronoi_adjacency(seeds)
    assert isinstance(adj, dict)
    assert set(adj.keys()) == {0, 1, 2}
    for i, neighbors in adj.items():
        assert isinstance(neighbors, list)
        for n in neighbors:
            assert isinstance(n, int)
            assert n != i

def test_compute_voronoi_adjacency_single_site():
    seeds = np.array([[0.0, 0.0, 0.0]])
    adj = compute_voronoi_adjacency(seeds)
    assert isinstance(adj, dict)
    assert set(adj.keys()) == {0}
    assert adj[0] == []

def test_call_sdf_with_array_input():
    def sdf(pts):
        return np.linalg.norm(pts, axis=1)
    pts = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    vals = _call_sdf(sdf, pts)
    assert isinstance(vals, np.ndarray)
    assert np.allclose(vals, [1.0, 2.0])

def test_call_sdf_with_tuple_input():
    def sdf(x, y, z):
        return x + y + z
    pts = (1.0, 2.0, 3.0)
    val = _call_sdf(sdf, pts)
    assert isinstance(val, np.ndarray)
    # Scalar return is 0-d array
    assert val.shape == ()
    assert val == pytest.approx(6.0)

def test_estimate_normal_on_flat_plane():
    def sdf(pts):
        return pts[:, 2]
    point = np.array([[0.5, 0.5, 0.0]])
    normal = estimate_normal(sdf, point)
    assert isinstance(normal, np.ndarray)
    assert normal.shape == (1, 3)
    n = normal[0]
    assert np.allclose(np.abs(n), np.array([0.0, 0.0, 1.0]), atol=1e-3)

def test_estimate_hessian_of_linear_sdf_is_zero():
    def sdf(pts):
        return pts[:, 0] + 2 * pts[:, 1]
    point = np.array([[0.1, 0.2, 0.3]])
    hess = estimate_hessian(sdf, point)
    assert isinstance(hess, np.ndarray)
    assert hess.shape == (1, 3, 3)
    assert np.allclose(hess, 0.0, atol=1e-3)

def test_compute_fillet_radius_on_sphere():
    center = np.array([0.0, 0.0, 0.0])
    R = 2.0
    def sdf(pts):
        return np.linalg.norm(pts - center, axis=1) - R
    pts = np.array([[R, 0.0, 0.0]])
    rad = compute_fillet_radius(sdf, pts)
    assert isinstance(rad, np.ndarray)
    assert rad.shape == (1,)
    assert rad[0] == pytest.approx(R, rel=1e-3)

def test_smooth_union_behaves_like_max_with_small_r():
    d1 = np.array([1.0, 2.0, 3.0])
    d2 = np.array([2.0, 1.0, 4.0])
    r = 0.01
    res = smooth_union(d1, d2, r)
    expected = np.maximum(d1, d2)
    assert np.allclose(res, expected, atol=1e-2)