import math
from collections import Counter

from design_api.services.voronoi_gen.voronoi_gen import build_hex_lattice

def test_hex_lattice_patch_structure():
    spacing = 1.0
    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (3.0, 3.0, 0.0)

    verts, edges = build_hex_lattice(
        bbox_min,
        bbox_max,
        spacing,
        primitive={},
        use_voronoi_edges=True,
    )

    assert verts and edges

    deg = Counter()
    for i, j in edges:
        deg[i] += 1

    # At least one vertex should have degree 3
    assert any(v == 3 for v in deg.values())

    lengths = []
    for i, j in edges:
        (x0, y0, z0) = verts[i]
        (x1, y1, z1) = verts[j]
        dist = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2 + (z0 - z1) ** 2)
        lengths.append(dist)
    targets = sorted(set(lengths))
    for dist in lengths:
        assert any(math.isclose(dist, t, rel_tol=0.1) for t in targets)

    edge_set = set(edges)
    for i, j in edges:
        assert (j, i) in edge_set
