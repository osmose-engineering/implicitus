import numpy as np

from design_api.services.infill_service import generate_hex_lattice
from design_api.services.voronoi_gen.voronoi_gen import primitive_to_imds_mesh


def test_uniform_lattice_autogenerates_mesh():
    bbox_min = (-1.0, -1.0, -1.0)
    bbox_max = (1.0, 1.0, 1.0)
    spacing = 1.0
    primitive = {"sphere": {"radius": 1.0}}

    spec = {
        "pattern": "voronoi",
        "mode": "uniform",
        "spacing": spacing,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "primitive": primitive,
        "plane_normal": [0.0, 0.0, 1.0],
        "max_distance": 2.0,
        "imds_mesh": primitive_to_imds_mesh(primitive),
    }

    result = generate_hex_lattice(spec)
    seed_pts = result["seed_points"]
    cells = result["cells"]
    assert seed_pts and cells
    first = next(iter(cells.values()))
    assert isinstance(first, np.ndarray)
    assert first.shape == (6, 3)
