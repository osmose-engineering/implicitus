import importlib
import sys


def test_cells_normalized_list(monkeypatch):
    for mod in [
        "design_api.services.infill_service",
        "design_api.services.voronoi_gen.voronoi_gen",
        "design_api.services.voronoi_gen",
    ]:
        sys.modules.pop(mod, None)
    infill = importlib.import_module("design_api.services.infill_service")

    def fake_build_hex_lattice(*args, **kwargs):
        cells = {0: [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]}
        return [[0.0, 0.0, 0.0]], [], [], cells

    monkeypatch.setattr(infill, "build_hex_lattice", fake_build_hex_lattice)
    monkeypatch.setattr(infill, "compute_voronoi_adjacency", lambda *a, **k: [])

    spec = {
        "bbox_min": [0, 0, 0],
        "bbox_max": [1, 1, 1],
        "spacing": 1.0,
        "mode": "uniform",
    }
    res = infill.generate_hex_lattice(spec)
    assert isinstance(res["cells"], list)
    assert res["cells"][0]["vertices"][0] == [0.0, 0.0, 0.0]
    assert res["cells"][0]["faces"] == []
    assert res["debug"]["mode"] == "uniform"
