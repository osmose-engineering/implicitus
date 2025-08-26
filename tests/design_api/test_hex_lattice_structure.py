import math
from collections import Counter

from design_api.services.infill_service import generate_hex_lattice


def test_hex_lattice_patch_structure():
    spacing = 1.0
    spec = {
        "pattern": "voronoi",
        "mode": "organic",
        "spacing": spacing,
        "bbox_min": (0.0, 0.0, 0.0),
        "bbox_max": (3.0, 3.0, 0.0),
        "primitive": {},
    }
    result = generate_hex_lattice(spec)
    verts = result["seed_points"]
    edges = result["edges"]
    assert verts and edges

    deg = Counter()
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    assert any(v >= 3 for v in deg.values())

    lengths = []
    for i, j in edges:
        (x0, y0, z0) = verts[i]
        (x1, y1, z1) = verts[j]
        dist = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2 + (z0 - z1) ** 2)
        lengths.append(dist)
    assert all(dist <= 2 * spacing + 1e-6 for dist in lengths)
