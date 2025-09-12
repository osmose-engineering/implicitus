import importlib.util


def test_generate_hex_lattice_populates_edges(monkeypatch):
    """Fallback to Voronoi adjacency when no edges are returned."""

    # The test environment stubs out the heavy Voronoi helpers. Provide
    # lightweight replacements that yield deterministic seed points and an empty
    # ``edge_list`` so the service must compute adjacency.
    import design_api.services.voronoi_gen.voronoi_gen as voro_stub

    def _fake_build_hex_lattice(bmin, bmax, spacing, primitive, **_):
        # Two arbitrary seed points with no edges
        return [[0, 0, 0], [1, 1, 1]], [], [], []

    monkeypatch.setattr(voro_stub, "build_hex_lattice", _fake_build_hex_lattice, raising=False)
    monkeypatch.setattr(
        voro_stub, "compute_voronoi_adjacency", lambda pts, spacing: [(0, 1)], raising=False
    )
    monkeypatch.setattr(voro_stub, "primitive_to_imds_mesh", lambda primitive: None, raising=False)

    # Load the real infill_service module under an alternate name so we can call
    # ``generate_hex_lattice`` without interfering with the test stubs used
    # elsewhere.
    spec = importlib.util.spec_from_file_location(
        "design_api.services.infill_real", "design_api/services/infill_service.py"
    )
    infill_real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infill_real)

    # Bypass the seed resolver to avoid unrelated dependencies.
    monkeypatch.setattr(
        infill_real,
        "resolve_seed_spec",
        lambda *args, **kwargs: {
            "bbox_min": [0, 0, 0],
            "bbox_max": [1, 1, 1],
            "spacing": 1.0,
            "mode": "uniform",
            "seed_points": [[0, 0, 0], [1, 1, 1]],
            "num_points": None,
        },
    )

    res = infill_real.generate_hex_lattice(
        {
            "bbox_min": [0, 0, 0],
            "bbox_max": [1, 1, 1],
            "spacing": 1.0,
            "primitive": {},
            "use_voronoi_edges": False,
        }
    )

    assert res["edge_list"] == [[0, 1]]

