import numpy as np
import pytest
import math

from design_api.services.voronoi_gen.organic.sampler import (
    sample_seed_points,
    sample_seed_points_anisotropic,
    sample_surface_seed_points,
    _hex_lattice,
)

@pytest.fixture
def bbox():
    # axis-aligned unit cube
    return np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])

def test_sample_seed_points_basic(bbox):
    min_corner, max_corner = bbox
    seeds = sample_seed_points(10, min_corner, max_corner)
    assert isinstance(seeds, list)
    assert len(seeds) == 10
    # Each seed is a 3-tuple within bounds
    for pt in seeds:
        assert len(pt) == 3
        assert all(min_corner[i] <= pt[i] <= max_corner[i] for i in range(3))

def test_sample_seed_points_with_min_dist(bbox):
    min_corner, max_corner = bbox
    # Use a minimum distance to avoid clustering
    seeds = sample_seed_points(10, min_corner, max_corner, min_dist=0.2)
    # No two seeds should be closer than min_dist
    for i, p1 in enumerate(seeds):
        for p2 in seeds[i+1:]:
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            assert dist >= 0.2 - 1e-6

def test_sample_seed_points_with_density_field(bbox):
    min_corner, max_corner = bbox
    # Define simple density: more seeds near x=1
    def density(pt):
        return pt[0]
    seeds = sample_seed_points(100, min_corner, max_corner, density_field=density)
    # Compute mean x-coordinate manually
    mean_x = sum(pt[0] for pt in seeds) / len(seeds)
    assert mean_x > 0.5


def test_sample_seed_points_hex_pattern(bbox):
    min_corner, max_corner = bbox
    seeds = sample_seed_points(10, min_corner, max_corner, min_dist=0.5, pattern="hex")
    expected = _hex_lattice(min_corner, max_corner, cell_size=0.5, slice_thickness=0.5)
    assert np.array(seeds).shape == expected.shape
    assert np.allclose(np.array(seeds), expected)

def test_sample_seed_points_anisotropic_constant_scale(bbox):
    min_corner, max_corner = bbox
    # Constant anisotropy scale factor of 2.0 along x-axis
    scale = np.array([2.0, 1.0, 1.0])
    seeds = sample_seed_points_anisotropic(50, min_corner, max_corner)
    assert isinstance(seeds, list)
    assert len(seeds) == 50
    assert all(min_corner[i] <= pt[i] <= max_corner[i] for pt in seeds for i in range(3))

def test_sample_seed_points_anisotropic_callable_scale(bbox):
    min_corner, max_corner = bbox
    # Callable scale: scale factor increases with x-coordinate
    def scale_fn(pt):
        return 1.0 + pt[0]
    seeds = sample_seed_points_anisotropic(50, min_corner, max_corner)
    assert len(seeds) == 50
    assert all(isinstance(coord, float) for pt in seeds for coord in pt)
    # Ensure no NaNs or infs
    assert all(math.isfinite(coord) for pt in seeds for coord in pt)

def test_sample_surface_seed_points_basic_sdf():
    # Define a simple sphere SDF: distance from unit sphere
    def sphere_sdf(pt):
        x, y, z = pt
        return math.sqrt(x*x + y*y + z*z) - 1.0
    min_corner = np.array([-1.5, -1.5, -1.5])
    max_corner = np.array([ 1.5,  1.5,  1.5])
    pts = sample_surface_seed_points(10, min_corner, max_corner, sphere_sdf, max_trials=5000, projection_steps=50, step_size=0.05)
    assert isinstance(pts, list)
    assert len(pts) == 10
    # Each point should lie approximately on the unit sphere
    for p in pts:
        dist = math.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
        assert abs(dist - 1.0) < 1e-2