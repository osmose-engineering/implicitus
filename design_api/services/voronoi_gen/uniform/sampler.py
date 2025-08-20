import numpy as np
import logging
import random
import math
from typing import Union, Any
from typing import Tuple, List, Optional
from typing import Dict, Callable
import itertools
from scipy.spatial import Voronoi

def compute_medial_axis(imds_mesh: Any) -> np.ndarray:
    """
    Compute an approximation of the medial axis of the interface mesh.
    imds_mesh should have a `.vertices` attribute as an (N,3) numpy array.
    Returns:
        medial_points: (M,3) numpy array of medial axis vertex positions.
    """
    vertices = getattr(imds_mesh, "vertices", None)
    if vertices is None:
        raise ValueError("imds_mesh must have a 'vertices' attribute")
    # Compute full Voronoi diagram of the mesh vertices
    vor = Voronoi(vertices)
    # TODO: Prune Voronoi vertices to medial axis subset within mesh boundaries
    medial_points = vor.vertices
    return medial_points


def trace_hexagon(seed_pt: np.ndarray, medial_points: np.ndarray, plane_normal: np.ndarray, max_distance: Optional[float] = None) -> np.ndarray:
    """
    Trace approximate hexagonal cell vertices around seed_pt by casting six rays
    in the plane defined by plane_normal and intersecting with medial_points.
    Args:
        seed_pt: (3,) array, seed point location.
        medial_points: (M,3) array of medial axis points.
        plane_normal: (3,) array normal to the slicing plane.
        max_distance: fallback ray length if no medial intersection found.
    Returns:
        hex_pts: (6,3) array of hexagon vertex positions.
    """
    # Create orthonormal basis (u, v) spanning the plane
    arbitrary = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.cross(arbitrary, plane_normal), 0):
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(plane_normal, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(plane_normal, u)
    v /= np.linalg.norm(v)

    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    hex_pts: List[np.ndarray] = []
    for theta in angles:
        dir_vec = np.cos(theta) * u + np.sin(theta) * v
        # Vector from seed to all medial points
        vecs = medial_points - seed_pt
        # Projection lengths along dir_vec
        t = vecs.dot(dir_vec)
        mask = t > 0
        if not np.any(mask):
            # Fallback: point at fixed max distance
            length = max_distance if max_distance is not None else 1.0
            hex_pts.append(seed_pt + dir_vec * length)
            continue
        pts_masked = medial_points[mask]
        t_masked = t[mask]
        # Compute perpendicular distances to the ray
        perp = pts_masked - seed_pt - np.outer(t_masked, dir_vec)
        perp_dists = np.linalg.norm(perp, axis=1)
        idx = np.argmin(perp_dists)
        hex_pts.append(pts_masked[idx])

    hex_pts = np.vstack(hex_pts)
    # Attempt to regularize edge lengths if available
    try:
        from design_api.services.voronoi_gen.uniform.regularizer import regularize_hexagon
        hex_pts = regularize_hexagon(hex_pts, plane_normal)
    except ImportError:
        pass

    return hex_pts
