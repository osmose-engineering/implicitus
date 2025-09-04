import numpy as np

from design_api.services.infill_service import generate_hex_lattice
from design_api.services.voronoi_gen.voronoi_gen import build_hex_lattice


BBOX_MIN = [0.0, 0.0, 0.0]
BBOX_MAX = [2.0, 2.0, 2.0]
SPACING = 0.5


def _spec(**kwargs):
    spec = {"bbox_min": BBOX_MIN, "bbox_max": BBOX_MAX, "spacing": SPACING}
    spec.update(kwargs)
    return spec


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
    spec = _spec(num_points=5, primitive={}, mode="organic")

    res = generate_hex_lattice(spec)
    seeds = res.get("seed_points", [])

    assert len(seeds) == 5


def test_uniform_mode_respects_bbox_and_spacing():
    spec = _spec(num_points=10, mode="uniform", primitive={"box": {"min": BBOX_MIN, "max": BBOX_MAX}})
    seeds = np.asarray(generate_hex_lattice(spec)["seed_points"])

    seeds_min = seeds.min(axis=0)
    seeds_max = seeds.max(axis=0)
    bbox_min = np.asarray(BBOX_MIN)
    bbox_max = np.asarray(BBOX_MAX)

    assert np.all(seeds_min >= bbox_min - 1e-6)
    assert np.all(seeds_max <= bbox_max + 1e-6)
    assert np.all(seeds_min - bbox_min <= SPACING)
    assert np.all(bbox_max - seeds_max <= SPACING)

    diff = seeds[:, None, :] - seeds[None, :, :]
    dists = np.sqrt((diff**2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    assert dists.min() > SPACING * 0.5


def test_organic_mode_random_seed_and_spread():
    seeds1, _ = build_hex_lattice(
        BBOX_MIN, BBOX_MAX, SPACING, {}, mode="organic", num_points=20, random_seed=42
    )
    seeds2, _ = build_hex_lattice(
        BBOX_MIN, BBOX_MAX, SPACING, {}, mode="organic", num_points=20, random_seed=42
    )
    assert seeds1 == seeds2

    seeds3, _ = build_hex_lattice(
        BBOX_MIN, BBOX_MAX, SPACING, {}, mode="organic", num_points=20, random_seed=43
    )
    assert seeds3 != seeds1

    seeds_arr = np.asarray(seeds1)
    span = seeds_arr.max(axis=0) - seeds_arr.min(axis=0)
    assert np.all(span > SPACING * 2)
