import numpy as np
import logging
import random
import math
from typing import Union, Any
from typing import Tuple, List, Optional
from typing import Dict, Callable
import itertools
from scipy.spatial import Voronoi


def compute_medial_axis(imds_mesh: Any, tol: float = 1e-8) -> np.ndarray:
    """Compute an approximation of the medial axis of the interface mesh.

    Parameters
    ----------
    imds_mesh: Any
        Object exposing a ``vertices`` attribute as an ``(N, 3)`` array.
    tol: float, optional
        Extra tolerance to expand the mesh bounds when filtering Voronoi
        vertices.  Larger values keep more vertices; a value of ``0`` clips
        strictly to the axis-aligned bounding box of the mesh.

    Returns
    -------
    np.ndarray
        Array of Voronoi vertices that fall within the mesh bounds.
    """

    vertices = getattr(imds_mesh, "vertices", None)
    if vertices is None:
        raise ValueError("imds_mesh must have a 'vertices' attribute")

    # Compute full Voronoi diagram of the mesh vertices
    vor = Voronoi(vertices)

    # Clip Voronoi vertices to lie within the mesh's bounding box (+/- tol)
    bbox_min = vertices.min(axis=0) - tol
    bbox_max = vertices.max(axis=0) + tol
    inside = np.all((vor.vertices >= bbox_min) & (vor.vertices <= bbox_max), axis=1)
    medial_points = vor.vertices[inside]

    return medial_points


def trace_hexagon(
    seed_pt: np.ndarray,
    medial_points: np.ndarray,
    plane_normal: np.ndarray,
    max_distance: Optional[float] = None,
) -> np.ndarray:
    """
    Trace approximate hexagonal cell vertices around ``seed_pt`` using
    perpendicular bisectors to its nearest neighbors in the slicing plane.

    Args:
        seed_pt: (3,) array, seed point location.
        medial_points: (M,3) array of medial axis or seed points.
        plane_normal: (3,) array normal to the slicing plane.
        max_distance: fallback ray length if no neighbor data is available.

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

    hex_pts: List[np.ndarray] = []

    # Pre-compute bounding box of the medial points for ray fallback
    bbox_min = medial_points.min(axis=0)
    bbox_max = medial_points.max(axis=0)

    def _ray_box_intersection(origin: np.ndarray, direction: np.ndarray) -> Optional[float]:
        """Return distance to intersection with bbox or None if no hit."""
        tmin, tmax = -np.inf, np.inf
        for i in range(3):
            if abs(direction[i]) < 1e-12:
                if origin[i] < bbox_min[i] or origin[i] > bbox_max[i]:
                    return None
                continue
            t1 = (bbox_min[i] - origin[i]) / direction[i]
            t2 = (bbox_max[i] - origin[i]) / direction[i]
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
        if tmax < 0:
            return None
        return tmin if tmin > 0 else tmax

    # Vector from seed to all other points
    vecs = medial_points - seed_pt
    dists = np.linalg.norm(vecs, axis=1)
    mask = dists > 1e-8
    vecs = vecs[mask]
    dists = dists[mask]

    hex_success = False
    if vecs.shape[0] >= 6:
        # Select six nearest neighbors
        idx = np.argpartition(dists, 6)[:6]
        neighbor_vecs = vecs[idx]
        # 2D coordinates in plane basis
        neighbor_2d = np.column_stack((neighbor_vecs.dot(u), neighbor_vecs.dot(v)))
        # Sort neighbors by angle around the seed for consistency
        ang = np.arctan2(neighbor_2d[:, 1], neighbor_2d[:, 0])
        order = np.argsort(ang)
        neighbor_2d = neighbor_2d[order]

        normals = neighbor_2d
        bs = np.sum(neighbor_2d ** 2, axis=1) / 2.0
        verts_2d: List[np.ndarray] = []
        for i in range(6):
            j = (i + 1) % 6
            N = np.vstack([normals[i], normals[j]])
            B = np.array([bs[i], bs[j]])
            try:
                x = np.linalg.solve(N, B)
            except np.linalg.LinAlgError:
                verts_2d = []
                break
            verts_2d.append(x)

        if len(verts_2d) == 6:
            hex_pts = [seed_pt + x[0] * u + x[1] * v for x in verts_2d]
            hex_pts = np.vstack(hex_pts)
            hex_success = True

    if not hex_success:
        # Neighbor data unavailable or computation failed; fallback to bbox rays
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        for theta in angles:
            dir_vec = np.cos(theta) * u + np.sin(theta) * v
            length = _ray_box_intersection(seed_pt, dir_vec)
            if length is not None:
                hex_pts.append(seed_pt + dir_vec * length)
            elif max_distance is not None:
                hex_pts.append(seed_pt + dir_vec * max_distance)
            else:
                raise ValueError("No valid intersection for ray; resample seed point")
        hex_pts = np.vstack(hex_pts)

    # Attempt to regularize edge lengths if available
    try:
        from design_api.services.voronoi_gen.uniform.regularizer import regularize_hexagon
        hex_pts = regularize_hexagon(hex_pts, plane_normal)
    except ImportError:
        pass

    return hex_pts
