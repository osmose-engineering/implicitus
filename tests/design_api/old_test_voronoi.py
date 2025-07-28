import math
import numpy as np
import pytest

from design_api.services.voronoi_gen.organic.sampler import (
    sample_seed_points,
    sample_seed_points_anisotropic,
    sample_surface_seed_points,
)
from design_api.services.voronoi_gen.organic.construct import (
    construct_voronoi_cells,
    construct_surface_voronoi_cells,
)


def test_sample_seed_points_basic():
    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (1.0, 1.0, 1.0)
    num_points = 5
    seeds = sample_seed_points(num_points, bbox_min, bbox_max)
    assert isinstance(seeds, list)
    assert len(seeds) == num_points
    for pt in seeds:
        assert isinstance(pt, (tuple, list))
        assert len(pt) == 3
        for i in range(3):
            assert bbox_min[i] <= pt[i] <= bbox_max[i]


def test_sample_seed_points_with_min_dist():
    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (1.0, 1.0, 1.0)
    num_points = 10
    min_dist = 0.3
    seeds = sample_seed_points(num_points, bbox_min, bbox_max, min_dist=min_dist)
    # Should generate exactly num_points points given sufficient box size
    assert isinstance(seeds, list)
    assert len(seeds) == num_points
    # All pairwise distances must be at least min_dist
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            assert math.dist(seeds[i], seeds[j]) >= min_dist


def test_construct_voronoi_cells_basic():
    # Define a simple bounding box and two seed points
    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (1.0, 1.0, 1.0)
    points = [(0.2, 0.2, 0.2), (0.8, 0.8, 0.8)]
    # Construct Voronoi cells
    cells = construct_voronoi_cells(points, bbox_min, bbox_max)
    # Should return a list of cells matching input points
    assert isinstance(cells, list)
    assert len(cells) == len(points)
    # Each cell should be a dict with expected keys
    for cell, pt in zip(cells, points):
        assert isinstance(cell, dict)
        # 'site' should match the input point (as list or tuple)
        site = cell.get('site')
        assert site == list(pt) or site == pt
        # Must have vertices and volume
        assert 'vertices' in cell and isinstance(cell['vertices'], list)
        assert 'volume' in cell and isinstance(cell['volume'], float)


# New tests for surface sampling and surface Voronoi
def test_sample_surface_seed_points_basic():
    # Define a simple SDF for a sphere of radius 1 at origin
    def sphere_sdf(pt):
        x, y, z = pt
        return math.sqrt(x**2 + y**2 + z**2) - 1.0

    num_points = 8
    bbox_min = (-1.5, -1.5, -1.5)
    bbox_max = (1.5, 1.5, 1.5)
    surface_points = sample_surface_seed_points(
        num_points, bbox_min, bbox_max, sphere_sdf
    )
    assert isinstance(surface_points, list)
    assert len(surface_points) == num_points
    # Each point should be on the surface of the sphere (distance ≈ 1)
    for pt in surface_points:
        dist = math.sqrt(pt[0] ** 2 + pt[1] ** 2 + pt[2] ** 2)
        assert abs(dist - 1.0) < 1e-2


def test_construct_surface_voronoi_cells_basic():
    # Use two antipodal points on the unit sphere
    sites = [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
    ]
    # Sphere SDF as above
    def sphere_sdf(pt):
        x, y, z = pt
        return math.sqrt(x**2 + y**2 + z**2) - 1.0

    cells = construct_surface_voronoi_cells(sites, sphere_sdf)
    assert isinstance(cells, list)
    assert len(cells) == len(sites)
    for cell, site in zip(cells, sites):
        assert isinstance(cell, dict)
        assert 'site' in cell
        # site in output should match input site (list or tuple)
        assert cell['site'] == list(site) or cell['site'] == site
        assert 'vertices' in cell and isinstance(cell['vertices'], list)
        assert 'area' in cell and isinstance(cell['area'], float)


# Test for compute_voronoi_adjacency
def test_compute_voronoi_adjacency_basic():
    # Simple 3-site triangle in 2D plane (Z=0)
    sites = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    # derive bounding box from the sites
    xs, ys, zs = zip(*sites)
    bbox_min = (min(xs), min(ys), min(zs))
    bbox_max = (max(xs), max(ys), max(zs))
    adjacency = compute_voronoi_adjacency(sites, bbox_min, bbox_max)
    # Should return a dict mapping each site to its neighboring sites
    assert isinstance(adjacency, dict)
    assert len(adjacency) == len(sites)
    for site in sites:
        assert site in adjacency
        neighbors = adjacency[site]
        assert isinstance(neighbors, list)
        # Each neighbor should be one of the other input sites
        for nbr in neighbors:
            assert nbr in sites and nbr != site
    # In a triangle, each site should have two neighbors
    for site, neighbors in adjacency.items():
        assert len(neighbors) == 2


def test_call_sdf_with_tuple_and_separate_args():
    # function expecting a tuple
    def sum_tuple(pt):
        return pt[0] + pt[1] + pt[2]
    # function expecting separate x,y,z
    def prod_sep(x, y, z):
        return x * y * z

    assert _call_sdf(sum_tuple, (1, 2, 3)) == 6
    assert _call_sdf(prod_sep, (2, 3, 4)) == 24

def test_estimate_normal_on_flat_plane():
    # plane z=0 has SDF(z)=z, normal should be (0,0,1)
    def plane_sdf(pt):
        x, y, z = pt
        return z

    n = estimate_normal((0.1, -0.2, 0.0), plane_sdf, eps=1e-3)
    assert isinstance(n, np.ndarray)
    assert np.allclose(n, np.array([0.0, 0.0, 1.0]), atol=1e-2)

def test_estimate_hessian_of_linear_sdf_is_zero():
    # linear SDF x+y+z has zero second derivatives
    def linear_sdf(pt):
        x, y, z = pt
        return x + y + z

    H = estimate_hessian((0.3, 0.4, 0.5), linear_sdf, eps=1e-3)
    assert isinstance(H, np.ndarray) and H.shape == (3, 3)
    assert np.allclose(H, np.zeros((3, 3)), atol=1e-2)

def test_compute_fillet_radius_on_sphere():
    # sphere of radius R: mean curvature = 1/R => fillet radius = alpha / (1/R) = alpha*R
    R = 2.0
    alpha = 1.0
    def sphere_sdf(pt):
        x, y, z = pt
        return math.sqrt(x*x + y*y + z*z) - R

    # pick a point on the surface
    p = (R, 0.0, 0.0)
    r = compute_fillet_radius(p, sphere_sdf, alpha=alpha, min_r=0.1, max_r=10.0, eps=1e-3)
    # should be close to R (i.e. 2.0)
    assert pytest.approx(R, rel=1e-2) == r

def test_smooth_union_behaves_like_max_with_small_r():
    a, b, r = -1.0, 1.0, 0.5
    u = smooth_union(a, b, r)
    # with these values, h → 1, so union ≈ max(a,b) = 1.0
    assert pytest.approx(1.0, abs=1e-6) == u

    # symmetric case
    u2 = smooth_union(b, a, r)
    assert pytest.approx(1.0, abs=1e-6) == u2

def test_construct_voronoi_cells_sdf_and_neighbors():
    # Two seed points in opposite corners
    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (1.0, 1.0, 1.0)
    points = [(0.2, 0.2, 0.2), (0.8, 0.8, 0.8)]

    # Default resolution behavior
    cells_default = construct_voronoi_cells(points, bbox_min, bbox_max)
    for cell in cells_default:
        # SDF grid must exist and be 3D
        sdf = cell.get('sdf')
        assert isinstance(sdf, np.ndarray)
        assert sdf.ndim == 3
        # Neighbors must list exactly the other seed
        neighbors = cell.get('neighbors')
        assert isinstance(neighbors, list)
        assert len(neighbors) == 1
        # Determine expected neighbor
        current_site = cell['site']
        expected = points[1] if current_site == points[0] or current_site == list(points[0]) else points[0]
        assert neighbors[0] == expected

    # Custom resolution and wall_thickness
    resolution = (5, 6, 7)
    wall_thickness = 0.1
    cells_custom = construct_voronoi_cells(
        points, bbox_min, bbox_max,
        resolution=resolution, wall_thickness=wall_thickness
    )
    for cell in cells_custom:
        sdf = cell.get('sdf')
        assert isinstance(sdf, np.ndarray)
        assert sdf.shape == resolution

def test_compute_voronoi_adjacency_single_site():
    # Edge case: only one site => no neighbors
    sites = [(0.0, 0.0, 0.0)]
    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (0.0, 0.0, 0.0)
    adjacency = compute_voronoi_adjacency(sites, bbox_min, bbox_max)
    assert isinstance(adjacency, dict)
    assert len(adjacency) == 1
    assert (0.0, 0.0, 0.0) in adjacency
    assert adjacency[(0.0, 0.0, 0.0)] == []

def test_sample_seed_points_with_density_field():
    # Constant density => points spread within bbox
    def density(p):
        return 8.0

    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (1.0, 1.0, 1.0)
    num_points = 20

    pts = sample_seed_points(
        num_points,
        bbox_min,
        bbox_max,
        density_field=density
    )
    assert isinstance(pts, list)
    assert len(pts) == num_points
    for x, y, z in pts:
        assert bbox_min[0] <= x <= bbox_max[0]
        assert bbox_min[1] <= y <= bbox_max[1]
        assert bbox_min[2] <= z <= bbox_max[2]

def test_sample_seed_points_anisotropic_constant_scale():
    # Use a constant density and anisotropic scale to test bounding behavior
    def density(p):
        return 8.0

    scale = (2.0, 1.0, 0.5)
    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (2.0, 1.0, 1.0)
    num_points = 30

    pts = sample_seed_points_anisotropic(
        num_points,
        bbox_min,
        bbox_max,
        scale_field=scale,
        density_field=density
    )
    assert isinstance(pts, list)
    assert len(pts) == num_points
    for x, y, z in pts:
        assert bbox_min[0] <= x <= bbox_max[0]
        assert bbox_min[1] <= y <= bbox_max[1]
        assert bbox_min[2] <= z <= bbox_max[2]

def test_sample_seed_points_anisotropic_callable_scale():
    # Scale increases linearly with x coordinate
    def scale_field(p):
        sx = 1.0 + p[0]
        return (sx, 1.0, 1.0)

    def density(p):
        return 10.0

    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (1.0, 1.0, 1.0)
    num_points = 20

    pts = sample_seed_points_anisotropic(
        num_points,
        bbox_min,
        bbox_max,
        scale_field=scale_field,
        density_field=density
    )
    assert isinstance(pts, list)
    assert len(pts) == num_points
    for x, y, z in pts:
        assert bbox_min[0] <= x <= bbox_max[0]
        assert bbox_min[1] <= y <= bbox_max[1]
        assert bbox_min[2] <= z <= bbox_max[2]

def test_construct_voronoi_cells_adaptive_fallback(monkeypatch):
    # Fallback when Delaunay is disabled: neighbors should be empty
    import design_api.services.voronoi_gen as vg
    monkeypatch.setattr(vg, 'Delaunay', None)

    seeds = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (1.0, 1.0, 1.0)
    root = OctreeNode(bbox_min, bbox_max)
    cells = construct_voronoi_cells(
        seeds, bbox_min, bbox_max,
        adaptive_grid=root
    )
    # Each cell should have samples and no neighbors
    for cell in cells:
        assert 'samples' in cell
        assert isinstance(cell['samples'], list)
        assert 'neighbors' in cell
        assert cell['neighbors'] == []

def test_construct_voronoi_cells_adaptive_delaunay():
    # Test adjacency computed via Delaunay for tetrahedral seeds
    import pytest
    pytest.importorskip('scipy.spatial')
    seeds = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (1.0, 1.0, 1.0)
    root = OctreeNode(bbox_min, bbox_max)
    cells = construct_voronoi_cells(
        seeds, bbox_min, bbox_max,
        adaptive_grid=root
    )
    # Each cell should have 3 neighbors in tetrahedral adjacency
    for cell in cells:
        neigh = cell['neighbors']
        assert isinstance(neigh, list)
        assert len(neigh) == 3
        # neighbors should be among the other seeds
        for n in neigh:
            assert n in seeds and n != cell['site']

def test_construct_surface_voronoi_cells_adaptive_simple():
    # Simple plane SDF with two seeds; adaptive grid yields one sample per cell
    seeds = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    def plane_sdf(p):
        # plane at z=0
        return p[2]
    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (1.0, 1.0, 1.0)
    root = OctreeNode(bbox_min, bbox_max)
    cells = construct_surface_voronoi_cells(
        seeds, plane_sdf,
        wall_thickness_mm=1.0,
        thickness_field=None,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        adaptive_grid=root,
        resolution=(2, 2, 2)
    )
    assert len(cells) == len(seeds)
    for cell in cells:
        # Should have one sample at center
        samples = cell['samples']
        assert isinstance(samples, list) and len(samples) == 1
        p, val = samples[0]
        # center sample at (0.5,0.5,0.5)
        assert pytest.approx((0.5, 0.5, 0.5)) == p
        # body_sdf at center is 0.5; since abs(0.5)<1, val should equal body_sdf or >0
        assert isinstance(val, float)
        assert pytest.approx(0.5, rel=1e-6) == val
        # neighbors assigned (using voxel adjacency)
        assert 'neighbors' in cell and isinstance(cell['neighbors'], list)


# CSG operation tests
def test_construct_voronoi_cells_csg_uniform_intersection():
    # Single seed, resolution 1x1x1, intersection with constant -1.0 SDF
    def const_neg(p):
        return -1.0

    seeds = [(0.0, 0.0, 0.0)]
    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (1.0, 1.0, 1.0)

    cells = construct_voronoi_cells(
        seeds, bbox_min, bbox_max,
        resolution=(1, 1, 1),
        wall_thickness=0.0,
        csg_ops=[{'op': 'intersection', 'sdf': const_neg, 'r': 1e-3}]
    )
    assert len(cells) == 1
    sdf = cells[0]['sdf']
    assert isinstance(sdf, np.ndarray)
    assert sdf.shape == (1, 1, 1)
    # After intersection with -1, the SDF should be approximately -1
    assert pytest.approx(-1.0, rel=1e-2) == sdf[0, 0, 0]


def test_construct_voronoi_cells_csg_uniform_union_nochange():
    # Single seed, resolution 1x1x1, union with constant -1.0 SDF should leave positive region unchanged
    def const_neg(p):
        return -1.0

    seeds = [(0.0, 0.0, 0.0)]
    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (1.0, 1.0, 1.0)

    cells = construct_voronoi_cells(
        seeds, bbox_min, bbox_max,
        resolution=(1, 1, 1),
        wall_thickness=0.0,
        csg_ops=[{'op': 'union', 'sdf': const_neg, 'r': 1e-3}]
    )
    sdf = cells[0]['sdf']
    # Original raw_sdf is zero; union with -1 should keep zero
    assert pytest.approx(0.0, abs=1e-6) == sdf[0, 0, 0]


def test_construct_surface_voronoi_cells_csg_uniform_intersection():
    # Single seed on plane, resolution 1x1x1, intersection with constant -1.0 SDF
    def const_neg(p):
        return -1.0

    seeds = [(0.0, 0.0, 0.0)]
    def plane_sdf(p):
        return p[2]  # plane z=0

    cells = construct_surface_voronoi_cells(
        seeds, plane_sdf,
        wall_thickness_mm=1.0,
        thickness_field=None,
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(1.0, 1.0, 1.0),
        csg_ops=[{'op': 'intersection', 'sdf': const_neg, 'r': 1e-3}],
        resolution=(1, 1, 1)
    )
    assert len(cells) == 1
    grid = cells[0]['sdf']
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (1, 1, 1)
    # The initial val is positive (0.5), intersection with -1 yields -1
    assert pytest.approx(-1.0, rel=1e-2) == grid[0, 0, 0]
def test_construct_surface_voronoi_cells_hybrid_shell_offset():
    # Two seeds, plane SDF, shell_offset negative to blend toward inner volume
    seeds = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    def plane_sdf(p):
        return p[2]  # plane at z=0

    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (1.0, 1.0, 1.0)
    cells = construct_surface_voronoi_cells(
        seeds, plane_sdf,
        wall_thickness_mm=1.0,
        thickness_field=None,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        blend_curve=None,
        shell_offset=-0.5,
        resolution=(1, 1, 1)
    )
    assert len(cells) == 2
    # All grid values should be ~0.75 after blending
    for cell in cells:
        grid = cell['sdf']
        assert isinstance(grid, np.ndarray)
        assert grid.shape == (1, 1, 1)
        val = grid[0, 0, 0]
        assert pytest.approx(0.75, rel=1e-2) == val

def test_construct_surface_voronoi_cells_hybrid_blend_curve_full():
    # Two seeds, plane SDF; blend_curve=1 yields full inner volumetric SDF
    seeds = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    def plane_sdf(p):
        return p[2]

    def full_blend(t):
        return 1.0

    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (1.0, 1.0, 1.0)
    cells = construct_surface_voronoi_cells(
        seeds, plane_sdf,
        wall_thickness_mm=1.0,
        thickness_field=None,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        csg_ops=None,
        blend_curve=full_blend,
        shell_offset=0.0,
        resolution=(1, 1, 1)
    )
    assert len(cells) == 2
    # Expected vol_grid values: vol_cells sdf + 0.5 => [1.0, 1.5]
    expected = [1.0, 1.5]
    for cell, exp in zip(cells, expected):
        grid = cell['sdf']
        val = grid[0, 0, 0]
        assert pytest.approx(exp, rel=1e-2) == val


# Additional tests for auto_cap behavior
def test_construct_voronoi_cells_auto_cap_nochange():
    # Single seed at center: auto_cap should not alter interior lattice
    seeds = [(0.5, 0.5, 0.5)]
    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (1.0, 1.0, 1.0)
    resolution = (2, 2, 2)

    cells_no_cap = construct_voronoi_cells(
        seeds, bbox_min, bbox_max,
        resolution=resolution,
        auto_cap=False
    )
    cells_cap = construct_voronoi_cells(
        seeds, bbox_min, bbox_max,
        resolution=resolution,
        auto_cap=True,
        cap_blend=0.2
    )
    # Verify same number of cells
    assert len(cells_no_cap) == len(cells_cap) == 1
    sdf0 = cells_no_cap[0]['sdf']
    sdf1 = cells_cap[0]['sdf']
    # Values should be unchanged for interior points
    assert isinstance(sdf0, np.ndarray) and isinstance(sdf1, np.ndarray)
    assert sdf0.shape == resolution and sdf1.shape == resolution
    assert np.allclose(sdf0, sdf1)


def test_construct_voronoi_cells_adaptive_auto_cap_nochange():
    # Two seeds with adaptive grid: auto_cap should not alter interior samples
    seeds = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (1.0, 1.0, 1.0)
    root = OctreeNode(bbox_min, bbox_max)

    cells_no_cap = construct_voronoi_cells(
        seeds, bbox_min, bbox_max,
        adaptive_grid=root,
        auto_cap=False
    )
    cells_cap = construct_voronoi_cells(
        seeds, bbox_min, bbox_max,
        adaptive_grid=root,
        auto_cap=True,
        cap_blend=0.1
    )
    assert len(cells_no_cap) == len(cells_cap) == len(seeds)
    # Compare sample values
    for c0, c1 in zip(cells_no_cap, cells_cap):
        vals0 = [v for _, v in c0['samples']]
        vals1 = [v for _, v in c1['samples']]
        assert vals0 == vals1