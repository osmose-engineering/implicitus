from design_api.services.infill_service import generate_hex_lattice


def test_internal_flag_ignored(monkeypatch):
    captured = {}

    def fake_lattice(bbox_min, bbox_max, spacing, primitive, **kwargs):
        captured.update(kwargs)
        return [], [], [], []

    monkeypatch.setattr(
        'design_api.services.infill_service.build_hex_lattice', fake_lattice
    )

    spec = {"bbox_min": [0, 0, 0], "bbox_max": [1, 1, 1], "spacing": 0.5, "_is_voronoi": True}
    generate_hex_lattice(spec)
    assert '_is_voronoi' not in captured
