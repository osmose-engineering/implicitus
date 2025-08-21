

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


def test_uniform_cells_adjacent_share_edge(monkeypatch):
    import math

    monkeypatch.setattr(
        "design_api.services.voronoi_gen.uniform.construct.compute_voronoi_cells",
        None,
    )

    # Generate hexagonal lattice of seeds (radius=2) so interior cells have 6 neighbors
    rad = 2
    coords = []
    for q in range(-rad, rad + 1):
        r1 = max(-rad, -q - rad)
        r2 = min(rad, -q + rad)
        for r in range(r1, r2 + 1):
            x = math.sqrt(3) * (q + r / 2)
            y = 1.5 * r
            coords.append([x, y, 0.0])
    seeds = np.array(coords)

    mesh = DummyMesh([])
    plane_normal = np.array([0.0, 0.0, 1.0])
    cells = compute_uniform_cells(seeds, mesh, plane_normal, max_distance=2.0)

    idx_center = int(np.where((seeds == np.array([0.0, 0.0, 0.0])).all(axis=1))[0][0])
    idx_neighbor = int(
        np.where((seeds == np.array([math.sqrt(3) * 1.0, 0.0, 0.0])).all(axis=1))[0][0]
    )

    cell_a = cells[idx_center]
    cell_b = cells[idx_neighbor]

    shared = []
    for i, v in enumerate(cell_a):
        for j, w in enumerate(cell_b):
            if np.allclose(v, w, atol=1e-6):
                shared.append((i, j))

    assert len(shared) == 2

    (i0, j0), (i1, j1) = shared
    assert abs(i0 - i1) in (1, 5)
    assert abs(j0 - j1) in (1, 5)


def test_solver_cells_adjacent_share_edge(monkeypatch):
    import math
    from design_api.services.voronoi_gen.voronoi_gen import compute_voronoi_adjacency

    # Hexagonal lattice of seeds (radius=2)
    rad = 2
    coords = []
    for q in range(-rad, rad + 1):
        r1 = max(-rad, -q - rad)
        r2 = min(rad, -q + rad)
        for r in range(r1, r2 + 1):
            x = math.sqrt(3) * (q + r / 2)
            y = 1.5 * r
            coords.append([x, y, 0.0])
    seeds = np.array(coords)

    adjacency = compute_voronoi_adjacency(seeds.tolist())

    def fake_solver(seeds_arg, bbox):
        return [{"polygons": [], "neighbors": adjacency[i]} for i in range(len(seeds_arg))]

    monkeypatch.setattr(
        "design_api.services.voronoi_gen.uniform.construct.compute_voronoi_cells",
        fake_solver,
    )

    mesh = DummyMesh([])
    plane_normal = np.array([0.0, 0.0, 1.0])
    cells = compute_uniform_cells(seeds, mesh, plane_normal, max_distance=2.0)

    idx_center = int(np.where((seeds == np.array([0.0, 0.0, 0.0])).all(axis=1))[0][0])
    idx_neighbor = int(
        np.where((seeds == np.array([math.sqrt(3) * 1.0, 0.0, 0.0])).all(axis=1))[0][0]
    )

    cell_a = cells[idx_center]
    cell_b = cells[idx_neighbor]

    shared = []
    for i, v in enumerate(cell_a):
        for j, w in enumerate(cell_b):
            if np.allclose(v, w, atol=1e-6):
                shared.append((i, j))

    assert len(shared) == 2

    (i0, j0), (i1, j1) = shared
    assert abs(i0 - i1) in (1, 5)
    assert abs(j0 - j1) in (1, 5)

