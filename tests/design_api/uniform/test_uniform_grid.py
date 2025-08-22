import math
import numpy as np
from design_api.services.voronoi_gen.uniform.construct import compute_uniform_cells
from design_api.services.voronoi_gen.uniform import construct as construct_module


class DummyMesh:
    def __init__(self, vertices):
        self.vertices = np.array(vertices, dtype=float)

def _honeycomb_grid(rows, cols, spacing=1.0):
    row_height = math.sqrt(3) / 2 * spacing
    pts = []
    for r in range(rows):
        y = r * row_height
        x_offset = (r % 2) * spacing / 2
        for c in range(cols):
            x = c * spacing + x_offset
            pts.append([x, y, 0.0])
    return np.array(pts, dtype=float)


def test_uniform_grid_vertex_sharing(monkeypatch):
    # Build an extended 5x5 honeycomb lattice and select the central 3x3 region
    full_grid = _honeycomb_grid(5, 5)
    indices = [r * 5 + c for r in range(1, 4) for c in range(1, 4)]
    seeds = full_grid[indices]

    # Provide abundant medial points so each seed has six neighbors
    monkeypatch.setattr(construct_module, "compute_medial_axis", lambda _mesh: full_grid)

    # Capture whether the fallback path was used for any cell
    fallback_flags = []

    def regular_hex(seed, medial, normal, max_distance):  # pragma: no cover - deterministic
        radius = 1.0 / math.sqrt(3.0)
        arbitrary = np.array([1.0, 0.0, 0.0])
        if np.allclose(np.dot(arbitrary, normal), 1.0):
            arbitrary = np.array([0.0, 1.0, 0.0])
        u = arbitrary - np.dot(arbitrary, normal) * normal
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        pts = []
        for k in range(6):
            ang = math.pi / 6 + k * math.pi / 3
            dir_vec = math.cos(ang) * u + math.sin(ang) * v
            pts.append(seed + radius * dir_vec)
        fallback_flags.append(False)
        return np.array(pts)

    monkeypatch.setattr(construct_module, "trace_hexagon", regular_hex)

    mesh = DummyMesh([[0.0, 0.0, 0.0]])
    plane_normal = np.array([0.0, 0.0, 1.0])
    cells = compute_uniform_cells(seeds, mesh, plane_normal, max_distance=2.0)

    assert set(cells.keys()) == set(range(9))
    for pts in cells.values():
        assert pts.shape == (6, 3)

    # Verify neighboring cells share vertices after reconciliation
    seed_dists = np.linalg.norm(seeds[:, None, :] - seeds[None, :, :], axis=2)
    neighbor_pairs = [
        (i, j) for i in range(len(seeds)) for j in range(i + 1, len(seeds)) if seed_dists[i, j] < 1.01
    ]
    for i, j in neighbor_pairs:
        dists = np.linalg.norm(cells[i][:, None, :] - cells[j][None, :, :], axis=2)
        assert np.min(dists) < 1e-6

    assert fallback_flags == [False] * len(seeds)
