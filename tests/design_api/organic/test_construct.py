import numpy as np
import pytest

from design_api.services.voronoi_gen.organic.construct import (
    construct_voronoi_cells,
    construct_surface_voronoi_cells,
)
from design_api.services.voronoi_gen import _call_sdf

@pytest.fixture
def seeds():
    # Two sample seed points
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ])

@pytest.fixture
def bbox():
    # Axis-aligned bounding box
    return np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])

def test_construct_voronoi_cells_smoke(seeds, bbox):
    min_corner, max_corner = bbox
    cells = construct_voronoi_cells(seeds, min_corner, max_corner)
    assert isinstance(cells, list)
    assert all(isinstance(cell, dict) for cell in cells)
    for cell in cells:
        assert 'site' in cell and 'sdf' in cell and 'neighbors' in cell
        grid = cell['sdf']
        assert isinstance(grid, np.ndarray)
        assert grid.ndim == 3

def test_construct_surface_voronoi_cells_smoke(seeds, bbox):
    min_corner, max_corner = bbox
    cells = construct_surface_voronoi_cells(seeds, min_corner, max_corner)
    assert isinstance(cells, list)
    assert all(isinstance(cell, dict) for cell in cells)
    for cell in cells:
        assert 'site' in cell and 'sdf' in cell and 'neighbors' in cell
        grid = cell['sdf']
        assert isinstance(grid, np.ndarray)
        assert grid.ndim == 3
