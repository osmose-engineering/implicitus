from design_api.services.infill_service import generate_hex_lattice


def test_generate_hex_lattice_edges():
    spec = {
        "pattern": "voronoi",
        "mode": "uniform",
        "spacing": 1.0,
        "seed_points": [[0.0, 0.0, 0.0]],
        "bbox_min": [-1.0, -1.0, -1.0],
        "bbox_max": [1.0, 1.0, 1.0],
        "primitive": {"sphere": {"radius": 1.0}},
        "_is_voronoi": True,
        "uniform": True,
    }
    result = generate_hex_lattice(spec)
    edges = result["edges"]
    points = result["seed_points"]
    assert isinstance(edges, list)
    assert len(edges) > 0
    max_idx = max(max(e) for e in edges)
    assert max_idx < len(points)
