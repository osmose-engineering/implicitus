import numpy as np
import pytest

from design_api.services.infill_service import generate_hex_lattice


def test_voronoi_edges_do_not_include_seeds():
    spec = {
        "bbox_min": [-1.0, -1.0, -1.0],
        "bbox_max": [1.0, 1.0, 1.0],
        "spacing": 1.0,
        "primitive": {"sphere": {"radius": 1.0}},
        "use_voronoi_edges": True,
    }
    res = generate_hex_lattice(spec)

    vertices = res.get("vertices", [])
    edges = res.get("edges", [])
    seeds = np.asarray(res.get("seed_points", []))

    assert vertices, "Expected non-empty vertex list"
    assert edges, "Expected non-empty edge list"

    # Ensure every edge references valid vertex indices
    num_vertices = len(vertices)
    assert all(0 <= i < num_vertices and 0 <= j < num_vertices for i, j in edges)

    # Collect vertices referenced by edges
    referenced = {idx for edge in edges for idx in edge}
    for idx in referenced:
        v = np.asarray(vertices[idx])
        # No vertex referenced by an edge should coincide with any seed point
        assert not np.any(np.all(np.isclose(seeds, v), axis=1))


def test_voronoi_edges_have_nonzero_length():
    spec = {
        "bbox_min": [-1.0, -1.0, -1.0],
        "bbox_max": [1.0, 1.0, 1.0],
        "spacing": 1.0,
        "primitive": {"sphere": {"radius": 1.0}},
        "use_voronoi_edges": True,
    }
    res = generate_hex_lattice(spec)

    vertices = np.asarray(res.get("vertices", []))
    edges = res.get("edges", [])

    lengths = [np.linalg.norm(vertices[i] - vertices[j]) for i, j in edges]
    assert lengths, "Expected non-empty edge list"
    assert all(length > 1e-8 for length in lengths)


def test_voronoi_edges_have_z_variation():
    spec = {
        "bbox_min": [-1.0, -1.0, -1.0],
        "bbox_max": [1.0, 1.0, 1.0],
        "spacing": 1.0,
        "primitive": {"sphere": {"radius": 1.0}},
        "use_voronoi_edges": True,
    }
    res = generate_hex_lattice(spec)

    vertices = np.asarray(res.get("vertices", []))
    edges = res.get("edges", [])

    z_diffs = [abs(vertices[i][2] - vertices[j][2]) for i, j in edges]
    assert z_diffs, "Expected non-empty edge list"
    assert any(z > 1e-5 for z in z_diffs), "Voronoi edges are all flat in z"

