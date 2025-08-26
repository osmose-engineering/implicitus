from design_api.services.infill_service import generate_hex_lattice


def test_generate_hex_lattice_returns_cells():
    spec = {
        "pattern": "voronoi",
        "mode": "organic",
        "spacing": 0.5,
        "bbox_min": (-1.0, -1.0, -1.0),
        "bbox_max": (1.0, 1.0, 1.0),
        "primitive": {"sphere": {"radius": 1.0}},
        "use_voronoi_edges": True,
        "resolution": (8, 8, 8),
    }
    result = generate_hex_lattice(spec)
    assert result["seed_points"] and result["edges"]
    assert result["cells"]


def test_generate_hex_lattice_points_inside():
    spec = {
        "pattern": "voronoi",
        "mode": "organic",
        "spacing": 0.5,
        "bbox_min": (-1.0, -1.0, -1.0),
        "bbox_max": (1.0, 1.0, 1.0),
        "primitive": {"sphere": {"radius": 1.0}},
        "use_voronoi_edges": False,
    }
    result = generate_hex_lattice(spec)
    pts = result["seed_points"]
    assert pts
    for x, y, z in pts:
        assert x * x + y * y + z * z <= 1.0 + 1e-6
