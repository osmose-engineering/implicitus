import numpy as np
from design_api.services.voronoi_gen.voronoi_gen import (
    build_hex_lattice,
    primitive_to_imds_mesh,
)

def test_uniform_lattice_autogenerates_mesh():
    bbox_min = (-1.0, -1.0, -1.0)
    bbox_max = (1.0, 1.0, 1.0)
    spacing = 1.0
    primitive = {"sphere": {"radius": 1.0}}

    imds_mesh = primitive_to_imds_mesh(primitive)
    seed_pts, cell_vertices, edges, cells = build_hex_lattice(
        bbox_min,
        bbox_max,
        spacing,
        primitive,
        return_cells=True,
        mode="uniform",
        plane_normal=np.array([0.0, 0.0, 1.0]),
        max_distance=2.0,
        imds_mesh=imds_mesh,
    )

    assert seed_pts and cell_vertices and cells
    first = next(iter(cells.values()))
    assert isinstance(first, np.ndarray)
    assert first.shape == (6, 3)
