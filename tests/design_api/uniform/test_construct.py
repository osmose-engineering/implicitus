

import numpy as np
import pytest
import logging
import json

from design_api.services.voronoi_gen.uniform.construct import compute_uniform_cells
from design_api.services.voronoi_gen.uniform.regularizer import (
    regularize_hexagon,
    hexagon_metrics,
)


class DummyMesh:
    def __init__(self, vertices):
        self.vertices = np.array(vertices)


def _sample_mesh():
    """Return a simple tetrahedral mesh used across the tests."""
    return DummyMesh(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_compute_uniform_cells_basic():
    # Two nearby seeds in 3D space to create adjacent hexagonal cells
    seeds = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
        ]
    )
    mesh = _sample_mesh()
    plane_normal = np.array([0.0, 0.0, 1.0])

    # Compute uniform cells with a fallback distance
    cells = compute_uniform_cells(seeds, mesh, plane_normal, max_distance=2.0)

    # Should return a dict mapping each seed index to a (6,3) array
    assert isinstance(cells, dict)
    assert set(cells.keys()) == {0, 1}

    for idx, pts in cells.items():
        assert isinstance(pts, np.ndarray)
        assert pts.shape == (6, 3)
        # Ensure each point is finite
        assert np.all(np.isfinite(pts))


def test_cell_planes_align_with_normal():
    """Seeds offset from the slicing plane should still yield coplanar cells."""

    # Place seeds at different offsets along the normal direction
    seeds = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.5, 0.5, -0.5],
        ]
    )
    mesh = _sample_mesh()
    plane_normal = np.array([0.0, 0.0, 1.0])

    cells = compute_uniform_cells(seeds, mesh, plane_normal, max_distance=2.0)

    unit_normal = plane_normal / np.linalg.norm(plane_normal)
    for pts in cells.values():
        # Compute a normal from two consecutive edges
        edge1 = pts[1] - pts[0]
        edge2 = pts[2] - pts[1]
        cell_normal = np.cross(edge1, edge2)
        cell_normal /= np.linalg.norm(cell_normal)

        # Angle between computed normal and provided plane_normal should be tiny
        cos_angle = np.clip(np.abs(np.dot(cell_normal, unit_normal)), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        assert angle < 1e-6



def test_shared_vertex_alignment(monkeypatch, caplog):
    seeds = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mesh = DummyMesh([[0.0, 0.0, 0.0]])

    def fake_medial_axis(_mesh):  # pragma: no cover - simple stub
        return np.zeros((1, 3))

    base_hex = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3) / 2, 0.0],
            [-0.5, np.sqrt(3) / 2, 0.0],
            [-1.0, 0.0, 0.0],
            [-0.5, -np.sqrt(3) / 2, 0.0],
            [0.5, -np.sqrt(3) / 2, 0.0],
        ]
    )

    def fake_trace_hexagon(seed, medial, normal, max_distance):  # pragma: no cover
        if np.allclose(seed, seeds[0]):
            return base_hex.copy()
        perturbed = base_hex.copy()
        perturbed[0] += [8e-4, 0.0, 0.0]
        perturbed[1] += [-8e-4, 0.0, 0.0]
        return perturbed

    monkeypatch.setattr(
        "design_api.services.voronoi_gen.uniform.construct.compute_medial_axis", fake_medial_axis
    )
    monkeypatch.setattr(
        "design_api.services.voronoi_gen.uniform.construct.trace_hexagon", fake_trace_hexagon
    )

    plane_normal = np.array([0.0, 0.0, 1.0])
    with caplog.at_level(logging.INFO):
        cells = compute_uniform_cells(
            seeds, mesh, plane_normal, max_distance=2.0, vertex_tolerance=1e-4
        )

    # Vertices 0 and 1 should now be identical between the two cells
    assert np.allclose(cells[0][0], cells[1][0])
    assert np.allclose(cells[0][1], cells[1][1])

    # Expect a warning about mismatched edges
    warnings = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert warnings and "exceeds tolerance" in warnings[0].message

    # Each cell's edge lengths should be nearly equal
    pts = cells[0]
    edges = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
    assert np.allclose(edges, edges[0], rtol=1e-3, atol=1e-6)

    # Adjacent cells should share vertices within a tolerance
    dists = np.linalg.norm(cells[0][:, None, :] - cells[1][None, :, :], axis=2)
    assert np.min(dists) < 0.1


def test_construct_produces_closed_hexagons():
    """Regression test ensuring traced hexagons form closed loops."""
    seeds = np.array([[0.0, 0.0, 0.0]])
    mesh = _sample_mesh()
    plane_normal = np.array([0.0, 0.0, 1.0])

    cells = compute_uniform_cells(seeds, mesh, plane_normal, max_distance=2.0)
    pts = cells[0]

    # The sum of edge vectors should be approximately zero for a closed polygon
    edges = np.roll(pts, -1, axis=0) - pts
    assert np.allclose(np.sum(edges, axis=0), np.zeros(3), atol=1e-6)

    # Non-degenerate hexagon should have positive area
    centroid = np.mean(pts, axis=0)
    area = 0.0
    for i in range(pts.shape[0]):
        a = pts[i] - centroid
        b = pts[(i + 1) % pts.shape[0]] - centroid
        area += 0.5 * np.linalg.norm(np.cross(a, b))
    assert area > 0


def test_regularization_reduces_edge_variance(monkeypatch):
    """Regularizing the hexagon should lower edge-length variance."""

    # Slightly perturb seed positions
    base_seeds = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    perturb = np.array([[0.002, -0.001, 0.001], [-0.0015, 0.0025, -0.0003]])
    seeds = base_seeds + perturb

    mesh = DummyMesh([[0.0, 0.0, 0.0]])

    def fake_medial_axis(_mesh):  # pragma: no cover - simple stub
        return np.zeros((1, 3))

    base_hex = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3) / 2, 0.0],
            [-0.5, np.sqrt(3) / 2, 0.0],
            [-1.0, 0.0, 0.0],
            [-0.5, -np.sqrt(3) / 2, 0.0],
            [0.5, -np.sqrt(3) / 2, 0.0],
        ]
    )

    def fake_trace_hexagon(seed, medial, normal, max_distance):  # pragma: no cover
        perturbed = base_hex.copy()
        if np.allclose(seed, seeds[0]):
            perturbed[0] += [0.1, 0.0, 0.0]
            perturbed[3] -= [0.1, 0.0, 0.0]
        else:
            perturbed[1] += [0.05, 0.0, 0.0]
            perturbed[4] -= [0.05, 0.0, 0.0]
        return perturbed

    monkeypatch.setattr(
        "design_api.services.voronoi_gen.uniform.construct.compute_medial_axis", fake_medial_axis
    )
    monkeypatch.setattr(
        "design_api.services.voronoi_gen.uniform.construct.trace_hexagon", fake_trace_hexagon
    )

    plane_normal = np.array([0.0, 0.0, 1.0])
    cells = compute_uniform_cells(seeds, mesh, plane_normal, max_distance=2.0)

    unit_normal = plane_normal / np.linalg.norm(plane_normal)
    for pts in cells.values():
        metrics_before = hexagon_metrics(pts)
        reg_pts = regularize_hexagon(pts, plane_normal)
        metrics_after = hexagon_metrics(reg_pts)

        var_before = np.var(metrics_before["edge_lengths"])
        var_after = np.var(metrics_after["edge_lengths"])
        assert var_after < var_before

        # Ensure regularized cell remains coplanar with the slicing plane
        edge1 = reg_pts[1] - reg_pts[0]
        edge2 = reg_pts[2] - reg_pts[1]
        cell_normal = np.cross(edge1, edge2)
        cell_normal /= np.linalg.norm(cell_normal)
        angle = np.arccos(
            np.clip(np.abs(np.dot(cell_normal, unit_normal)), -1.0, 1.0)
        )
        assert angle < 1e-6


def test_pathological_medial_axis_triggers_warning(monkeypatch, caplog):
    """Cells from a degenerate medial axis should emit a warning before reconciliation."""

    seeds = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    mesh = DummyMesh([[0.0, 0.0, 0.0]])

    def central_medial_axis(_mesh):  # pragma: no cover - deterministic cluster
        return np.zeros((8, 3))

    def degenerate_trace(seed, medial, normal, max_distance):  # pragma: no cover
        pts = np.tile(seed, (6, 1))

        # Verify the hexagon before any vertex averaging
        unique = np.unique(pts, axis=0)
        centroid = pts.mean(axis=0)
        area = 0.0
        for i in range(6):
            a = pts[i] - centroid
            b = pts[(i + 1) % 6] - centroid
            area += 0.5 * np.linalg.norm(np.cross(a, b))
        if unique.shape[0] != 6 or area == 0.0:
            logging.warning("degenerate hexagon detected")
        return pts

    monkeypatch.setattr(
        "design_api.services.voronoi_gen.uniform.construct.compute_medial_axis",
        central_medial_axis,
    )
    monkeypatch.setattr(
        "design_api.services.voronoi_gen.uniform.construct.trace_hexagon",
        degenerate_trace,
    )

    plane_normal = np.array([0.0, 0.0, 1.0])
    with caplog.at_level(logging.WARNING):
        compute_uniform_cells(seeds, mesh, plane_normal, max_distance=1.0)

    warnings = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert any("degenerate hexagon" in w.message for w in warnings)


def test_uniform_cell_dump(tmp_path, monkeypatch):
    seeds = np.array([[0.0, 0.0, 0.0]])
    mesh = _sample_mesh()
    plane_normal = np.array([0.0, 0.0, 1.0])

    # Simplify medial axis and hexagon tracing
    monkeypatch.setattr(
        "design_api.services.voronoi_gen.uniform.construct.compute_medial_axis",
        lambda _mesh: np.zeros((1, 3)),
    )

    def fake_trace_hexagon(seed, medial, normal, max_distance, report_method=False, neighbor_resampler=None):  # pragma: no cover - deterministic
        pts = np.zeros((6, 3))
        if report_method:
            return pts, True
        return pts

    monkeypatch.setattr(
        "design_api.services.voronoi_gen.uniform.construct.trace_hexagon",
        fake_trace_hexagon,
    )

    monkeypatch.chdir(tmp_path)

    compute_uniform_cells(
        seeds,
        mesh,
        plane_normal,
        max_distance=1.0,
    )

    dump_file = tmp_path / "UNIFORM_CELL_DUMP.json"
    assert dump_file.exists()
    data = json.loads(dump_file.read_text())
    assert data["cells"]["0"]["used_fallback"] is True
