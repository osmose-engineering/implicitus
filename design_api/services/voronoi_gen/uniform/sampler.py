import logging
from typing import Any, Optional

import numpy as np


def compute_medial_axis(imds_mesh: Any) -> np.ndarray:
    """Compute an approximation of the medial axis of ``imds_mesh``."""
    vertices = getattr(imds_mesh, "vertices", None)
    if vertices is None:
        raise ValueError("imds_mesh must have a 'vertices' attribute")
    from scipy.spatial import Voronoi

    vor = Voronoi(vertices)
    return vor.vertices


def trace_hexagon(
    seed: np.ndarray,
    medial: np.ndarray,
    plane_normal: np.ndarray = np.array([0.0, 0.0, 1.0]),
    max_distance: Optional[float] = None,
) -> np.ndarray:
    """Trace a hexagon around ``seed`` using nearby ``medial`` points."""
    n = plane_normal / np.linalg.norm(plane_normal)
    arbitrary = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.cross(arbitrary, n), 0):
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    if medial.size == 0:
        radius = max_distance if max_distance is not None else 1.0
    else:
        radius = np.mean(np.linalg.norm(medial - seed, axis=1))
        if max_distance is not None:
            radius = min(radius, max_distance)
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    hex_pts = seed + np.outer(np.cos(angles), u) * radius + np.outer(np.sin(angles), v) * radius
    return hex_pts

