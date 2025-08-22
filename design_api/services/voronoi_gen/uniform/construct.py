import numpy as np
import logging

from typing import Any, Dict, Tuple, List, Optional
import json
from pathlib import Path

from .sampler import compute_medial_axis, trace_hexagon
from .regularizer import hexagon_metrics

def compute_uniform_cells(
    seeds: np.ndarray,
    imds_mesh: Any,
    plane_normal: np.ndarray,
    max_distance: Optional[float] = None,
    vertex_tolerance: float = 1e-5,
) -> Dict[int, np.ndarray]:
    """
    Compute near-uniform hexagonal Voronoi cells for each seed point.
    Args:
        seeds: (N,3) array of seed point locations.
        imds_mesh: mesh object with `.vertices` attribute for medial axis extraction.
        plane_normal: (3,) array defining slicing plane normal.
        max_distance: fallback distance for ray casting when no medial point is found.
        vertex_tolerance: tolerance used when reconciling shared vertices between
            adjacent cells. A warning is emitted if mismatches above this tolerance
            are detected.
    Returns:
        cells: dict mapping seed index to (6,3) array of hexagon vertices.
    """
    # Extract medial axis points
    medial_points = compute_medial_axis(imds_mesh)

    # Derive an axis-aligned bounding box from the interface mesh to provide
    # additional sampling locations if the medial axis alone is insufficient.
    verts = getattr(imds_mesh, "vertices", None)
    if verts is None:
        raise ValueError("imds_mesh must have a 'vertices' attribute")
    bbox_min = np.min(verts, axis=0)
    bbox_max = np.max(verts, axis=0)
    rng = np.random.default_rng(0)


    dump_data: Dict[str, Any] = {
        "seeds": seeds.tolist(),
        "plane_normal": plane_normal.tolist(),
        "max_distance": max_distance,
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "medial_points": medial_points.tolist(),
        "cells": {},
    }


    def _resample() -> np.ndarray:
        """Return extra candidate points within the mesh bounds."""
        return rng.uniform(bbox_min, bbox_max, size=(30, 3))

    cells: Dict[int, np.ndarray] = {}
    for idx, seed in enumerate(seeds):
        # Provide the resampler so that trace_hexagon has enough neighbor
        # directions and avoids the axis-aligned bounding-box fallback that
        # produces cubic cells. Older ``trace_hexagon`` implementations may not

        # accept the ``neighbor_resampler`` or ``report_method`` arguments, so we
        # fall back to calling it with fewer parameters when necessary.
        try:
            hex_pts, used_fallback = trace_hexagon(

                seed,
                medial_points,
                plane_normal,
                max_distance,

                report_method=True,
                neighbor_resampler=_resample,
            )
        except TypeError:  # pragma: no cover - legacy signature
            try:
                hex_pts, used_fallback = trace_hexagon(
                    seed,
                    medial_points,
                    plane_normal,
                    max_distance,
                    report_method=True,
                )
            except TypeError:  # pragma: no cover - legacy signature
                hex_pts = trace_hexagon(
                    seed,
                    medial_points,
                    plane_normal,
                    max_distance,
                )
                used_fallback = False

        # Optionally log metrics
        metrics = hexagon_metrics(hex_pts)
        logging.debug(
            f"Uniform cell {idx}: mean_edge="
            f"{metrics['mean_edge_length']:.3f}, std_edge="
            f"{metrics['std_edge_length']:.3f}, area="
            f"{metrics['area']:.3f}"
        )
        cells[idx] = hex_pts

        dump_data["cells"][str(idx)] = {
            "seed": seed.tolist(),
            "vertices": hex_pts.tolist(),
            "used_fallback": bool(used_fallback),
        }


    # --------------------
    # Reconcile shared vertices
    # --------------------
    all_vertices: List[np.ndarray] = []
    cell_slices: Dict[int, slice] = {}
    for idx in sorted(cells.keys()):
        start = len(all_vertices)
        all_vertices.extend(cells[idx])
        cell_slices[idx] = slice(start, start + len(cells[idx]))

    if all_vertices:
        verts = np.vstack(all_vertices)
        n = len(verts)
        parent = np.arange(n)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        merge_tol = vertex_tolerance * 10.0
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(verts[i] - verts[j]) <= merge_tol:
                    union(i, j)

        groups: Dict[int, List[int]] = {}
        for idx in range(n):
            root = find(idx)
            groups.setdefault(root, []).append(idx)

        deltas: List[float] = []
        for inds in groups.values():
            if len(inds) > 1:
                pts = verts[inds]
                avg = pts.mean(axis=0)
                dev = np.linalg.norm(pts - avg, axis=1)
                deltas.extend(dev.tolist())
                verts[inds] = avg

        for idx, sl in cell_slices.items():
            cells[idx] = verts[sl]

        if deltas:
            mean_delta = float(np.mean(deltas))
            max_delta = float(np.max(deltas))
            logging.info(
                "Shared vertex adjustment: mean_delta=%0.6e, max_delta=%0.6e, adjusted=%d",
                mean_delta,
                max_delta,
                len(deltas),
            )
            if max_delta > vertex_tolerance:
                logging.warning(
                    "Max vertex mismatch %0.6e exceeds tolerance %0.6e",
                    max_delta,
                    vertex_tolerance,
                )
        else:
            logging.info("Shared vertex adjustment: no coincident vertices found")

    dump_path = Path("UNIFORM_CELL_DUMP.json")
    try:
        with dump_path.open("w", encoding="utf-8") as f:
            json.dump(dump_data, f)
    except Exception as exc:  # pragma: no cover - best effort
        logging.warning("Failed to write uniform cell dump to %s: %s", dump_path, exc)


    return cells
