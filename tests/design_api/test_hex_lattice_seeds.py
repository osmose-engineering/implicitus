import numpy as np

from design_api.services.infill_service import generate_hex_lattice


def test_forwarded_seed_points_used_verbatim():
    seeds = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    spec = {
        "bbox_min": [-1.0, -1.0, -1.0],
        "bbox_max": [1.0, 1.0, 1.0],
        "spacing": 0.5,
        "seed_points": seeds,
        "mode": "organic",
    }

    res = generate_hex_lattice(spec)
    out = res.get("seed_points")

    assert len(out) == len(seeds)
    assert all(np.allclose(a, b) for a, b in zip(seeds, out))


def test_num_points_limits_generated_seeds():
    spec = {
        "bbox_min": [0.0, 0.0, 0.0],
        "bbox_max": [2.0, 2.0, 2.0],
        "spacing": 0.5,
        "num_points": 5,
        "primitive": {},
        "mode": "organic",
    }

    res = generate_hex_lattice(spec)
    seeds = res.get("seed_points", [])

    assert len(seeds) == 5
