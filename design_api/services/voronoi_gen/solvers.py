"""Voronoi cell solver helpers.

This module attempts to use `pyvoro` for 3D Voronoi tessellation. If
`pyvoro` is unavailable, a fallback implementation using `scipy` or the
existing grid-based adjacency function is employed.  The public function
`compute_voronoi_cells` returns, for each seed, the face polygons and
adjacent cell indices.
"""
from __future__ import annotations

from typing import List, Dict, Any

import numpy as np

try:  # pragma: no cover - optional dependency
    import pyvoro  # type: ignore

    _PYVORO_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    pyvoro = None  # type: ignore
    _PYVORO_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    from scipy.spatial import Voronoi

    _SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    Voronoi = None  # type: ignore
    _SCIPY_AVAILABLE = False

from .voronoi_gen import compute_voronoi_adjacency


def _compute_with_pyvoro(seeds: np.ndarray, bbox: List[List[float]]) -> List[Dict[str, Any]]:
    """Compute Voronoi cells using :mod:`pyvoro`.

    Each returned element corresponds to a seed and contains:
        ``polygons`` – list of face polygons (each polygon is a list of 3D vertices)
        ``neighbors`` – list of neighboring seed indices.
    """

    cells = pyvoro.compute_voronoi(seeds.tolist(), bbox, 1.0)
    result: List[Dict[str, Any]] = []
    for cell in cells:
        faces = cell.get("faces", [])
        polys = [face.get("vertices", []) for face in faces]
        neighs = [face["adjacent_cell"] for face in faces if face.get("adjacent_cell") is not None]
        result.append({"polygons": polys, "neighbors": neighs})
    return result


def _compute_with_scipy(seeds: np.ndarray) -> List[Dict[str, Any]]:
    """Compute Voronoi cells using :mod:`scipy` as a fallback."""

    if len(seeds) < 5:  # scipy Voronoi in 3D needs at least 5 points
        adjacency = compute_voronoi_adjacency(seeds.tolist())
        return [{"polygons": [], "neighbors": adjacency.get(i, [])} for i in range(len(seeds))]

    vor = Voronoi(seeds, qhull_options="QJ")
    result: List[Dict[str, Any]] = [{"polygons": [], "neighbors": []} for _ in range(len(seeds))]
    for (p1, p2), verts in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in verts:
            continue
        poly = [vor.vertices[v].tolist() for v in verts]
        result[p1]["polygons"].append(poly)
        result[p2]["polygons"].append(poly)
        result[p1]["neighbors"].append(int(p2))
        result[p2]["neighbors"].append(int(p1))
    return result


def compute_voronoi_cells(seeds: List[List[float]] | np.ndarray,
                          bbox: List[List[float]]) -> List[Dict[str, Any]]:
    """Return Voronoi cell information for ``seeds`` inside ``bbox``.

    Parameters
    ----------
    seeds:
        Sequence of 3D seed coordinates.
    bbox:
        Bounding box ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]``.

    Returns
    -------
    List of dictionaries, one per seed, each with keys ``polygons`` and
    ``neighbors``.
    """

    seeds_arr = np.asarray(seeds, dtype=float)

    if _PYVORO_AVAILABLE:
        return _compute_with_pyvoro(seeds_arr, bbox)

    if _SCIPY_AVAILABLE:
        return _compute_with_scipy(seeds_arr)

    # Final fallback – use voxel-based adjacency; polygons remain empty
    bbox_min = (bbox[0][0], bbox[1][0], bbox[2][0])
    bbox_max = (bbox[0][1], bbox[1][1], bbox[2][1])
    adjacency = compute_voronoi_adjacency(seeds_arr.tolist(), bbox_min=bbox_min, bbox_max=bbox_max)
    result: List[Dict[str, Any]] = []
    for i in range(len(seeds_arr)):
        result.append({"polygons": [], "neighbors": adjacency.get(i, [])})
    return result
