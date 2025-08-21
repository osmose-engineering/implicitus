import pytest
from design_api.services.voronoi_gen.organic.adaptive import OctreeNode
from design_api.services.voronoi_gen.organic import construct as construct_module
from design_api.services.voronoi_gen.organic.construct import construct_voronoi_cells


def test_construct_voronoi_cells_adaptive_fallback_neighbors(monkeypatch):
    # Simulate missing SciPy so Delaunay is None
    monkeypatch.setattr(construct_module, "Delaunay", None)
    seeds = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (1.0, 1.0, 1.0)
    root = OctreeNode(bbox_min, bbox_max)
    cells = construct_voronoi_cells(seeds, bbox_min, bbox_max, adaptive_grid=root)
    assert len(cells) == 2
    for cell in cells:
        expected = {s for s in seeds if s != cell["site"]}
        assert set(cell["neighbors"]) == expected


def test_construct_voronoi_cells_adaptive_delaunay_neighbors():
    pytest.importorskip("scipy.spatial")
    seeds = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    bbox_min = (0.0, 0.0, 0.0)
    bbox_max = (1.0, 1.0, 1.0)
    root = OctreeNode(bbox_min, bbox_max)
    cells = construct_voronoi_cells(seeds, bbox_min, bbox_max, adaptive_grid=root)
    for cell in cells:
        expected = {s for s in seeds if s != cell["site"]}
        assert set(cell["neighbors"]) == expected
