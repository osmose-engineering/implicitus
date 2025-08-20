

import numpy as np
import pytest

from design_api.services.voronoi_gen.uniform.regularizer import regularize_hexagon, hexagon_metrics

def create_irregular_hexagon(radius=1.0, noise_level=0.2):
    """
    Generate a hexagon around the Z=0 plane with radius perturbed by noise.
    """
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    # Base circle points
    pts = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)], axis=1)
    # Add radial noise
    noise = (np.random.rand(6) - 0.5) * noise_level
    vecs = pts - np.mean(pts, axis=0)
    lengths = np.linalg.norm(vecs, axis=1)
    unit_vecs = vecs / lengths[:, np.newaxis]
    noisy_pts = np.mean(pts, axis=0) + unit_vecs * (lengths + noise)[:, np.newaxis]
    return noisy_pts

def test_regularize_hexagon_preserves_centroid_and_edge_lengths():
    np.random.seed(42)
    hex_pts = create_irregular_hexagon(radius=5.0, noise_level=1.0)
    centroid_before = np.mean(hex_pts, axis=0)
    # Regularize
    new_pts = regularize_hexagon(hex_pts, np.array([0.0, 0.0, 1.0]))
    centroid_after = np.mean(new_pts, axis=0)
    # Centroid should remain effectively unchanged
    assert np.allclose(centroid_before, centroid_after, atol=1e-6)
    # Edge lengths should now be uniform
    edge_lengths = np.linalg.norm(new_pts - np.roll(new_pts, -1, axis=0), axis=1)
    assert pytest.approx(edge_lengths[0], rel=1e-6) == edge_lengths[1]
    assert np.allclose(edge_lengths, edge_lengths[0], atol=1e-6)

def test_hexagon_metrics_on_regular_hexagon():
    # Create a perfect regular hexagon of radius 3 in XY plane
    radius = 3.0
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    hex_pts = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)], axis=1)
    metrics = hexagon_metrics(hex_pts)
    # Edge length of a regular hexagon is radius
    expected_edge = 2 * radius * np.sin(np.pi / 6)
    assert pytest.approx(metrics['mean_edge_length'], rel=1e-6) == expected_edge
    # Standard deviation should be zero
    assert pytest.approx(metrics['std_edge_length'], abs=1e-8) == 0.0
    # Area of regular hexagon: (3*sqrt(3)/2)*radius^2
    expected_area = (3 * np.sqrt(3) / 2) * (radius ** 2)
    assert pytest.approx(metrics['area'], rel=1e-6) == expected_area
    # Edge_lengths array shape
    assert metrics['edge_lengths'].shape == (6,)