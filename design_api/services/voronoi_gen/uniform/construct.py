import numpy as np
import logging
from typing import Any, List, Optional, Dict
import itertools

from .regularizer import hexagon_metrics, regularize_hexagon


def construct_cell_from_bisectors(
    seed_idx: int,
    seeds: np.ndarray,
    neighbor_indices: List[int],
    plane_normal: np.ndarray,
    max_distance: Optional[float] = None,
) -> np.ndarray:
    """Construct a convex cell for a seed from bisector half-spaces.

    For each neighboring seed, a bisecting plane is created whose normal points
    toward the neighbor.  The intersection of these half-spaces with the slicing
    plane yields the polygonal cell for the seed.

    Args:
        seed_idx: Index of the seed in ``seeds``.
        seeds: Array of seed positions.
        neighbor_indices: Indices of neighboring seeds.
        plane_normal: Normal of slicing plane passing through the seed.
        max_distance: Optional distance bound to keep the intersection finite.

    Returns:
        (M,3) array of polygon vertices for the cell.
    """

    seed_pt = seeds[seed_idx]
    halfspaces: List[np.ndarray] = []

    # Fallback to regular hexagon if insufficient neighbors
    if len(neighbor_indices) < 3:
        radius = max_distance if max_distance is not None else 1.0
        arbitrary = np.array([1.0, 0.0, 0.0])
        if np.allclose(np.cross(arbitrary, plane_normal), 0):
            arbitrary = np.array([0.0, 1.0, 0.0])
        u = np.cross(plane_normal, arbitrary)
        u /= np.linalg.norm(u)
        v = np.cross(plane_normal, u)
        v /= np.linalg.norm(v)
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        hex_pts = seed_pt + np.outer(np.cos(angles), u) * radius + np.outer(
            np.sin(angles), v
        ) * radius
        return hex_pts

    for n_idx in neighbor_indices:
        neighbor = seeds[n_idx]
        normal = neighbor - seed_pt
        norm = np.linalg.norm(normal)
        if norm == 0:
            continue
        normal /= norm
        midpoint = 0.5 * (seed_pt + neighbor)
        halfspaces.append(np.append(normal, -np.dot(normal, midpoint)))

    # Constrain to slicing plane by adding a pair of opposite half-spaces
    n = plane_normal / np.linalg.norm(plane_normal)
    halfspaces.append(np.append( n, -np.dot(n, seed_pt)))
    halfspaces.append(np.append(-n,  np.dot(n, seed_pt)))

    # Optional bounding box in plane to avoid unbounded intersection
    if max_distance is not None:
        arbitrary = np.array([1.0, 0.0, 0.0])
        if np.allclose(np.cross(arbitrary, n), 0):
            arbitrary = np.array([0.0, 1.0, 0.0])
        u = np.cross(n, arbitrary)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        v /= np.linalg.norm(v)
        for vec in (u, v, -u, -v):
            pt = seed_pt + vec * max_distance
            halfspaces.append(np.append(vec, -np.dot(vec, pt)))

    arbitrary = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.cross(arbitrary, n), 0):
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, arbitrary); u /= np.linalg.norm(u)
    v = np.cross(n, u); v /= np.linalg.norm(v)
    interior = seed_pt + n*1e-6 + u*1e-6 + v*1e-6
    try:
        hs = HalfspaceIntersection(np.array(halfspaces), interior, qhull_options="QJ")
        return hs.intersections
    except Exception:
        radius = max_distance if max_distance is not None else 1.0
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        return seed_pt + np.outer(np.cos(angles), u)*radius + np.outer(np.sin(angles), v)*radius

def compute_uniform_cells(
    seeds: np.ndarray,
    imds_mesh: Any,
    plane_normal: np.ndarray,
    max_distance: Optional[float] = None,
) -> Dict[int, np.ndarray]:
    """Compute simple regular hexagonal cells around each seed."""
    n = plane_normal / np.linalg.norm(plane_normal)
    arbitrary = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.cross(arbitrary, n), 0):
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    cells: Dict[int, np.ndarray] = {}
    for idx, seed in enumerate(seeds):
        dists = np.linalg.norm(seeds - seed, axis=1)
        nearest = np.min(dists[dists > 0])
        radius = nearest / np.sqrt(3)
        if max_distance is not None:
            radius = min(radius, max_distance)
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        poly = seed + np.outer(np.cos(angles), u) * radius + np.outer(np.sin(angles), v) * radius
        cells[idx] = poly
    return cells
