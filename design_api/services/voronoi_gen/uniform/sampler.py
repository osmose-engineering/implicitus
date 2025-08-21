import numpy as np
import logging
import random
import math
from typing import Union, Any
from typing import Tuple, List, Optional
from typing import Dict, Callable
import itertools


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
    from scipy.spatial import Voronoi

    vor = Voronoi(vertices)
    # TODO: Prune Voronoi vertices to medial axis subset within mesh boundaries
    medial_points = vor.vertices
    return medial_points


def trace_hexagon(
    seed_idx: int,
    seeds: np.ndarray,
    neighbor_indices: List[int],
    plane_normal: np.ndarray,
    max_distance: Optional[float] = None,
) -> np.ndarray:
    """
    Construct a hexagonal cross-section for a seed by intersecting the slicing
    plane with bisector planes formed between the seed and its Voronoi
    neighbors.

    Args:
        seed_idx: index of the seed within ``seeds``.
        seeds: (N,3) array of all seed locations.
        neighbor_indices: list of indices of neighboring seeds.
        plane_normal: normal of the slicing plane passing through the seed.
        max_distance: fallback distance if intersections fail or insufficient
            neighbors are provided.

    Returns:
        (6,3) array of hexagon vertices.
    """

    seed_pt = seeds[seed_idx]

    # Build orthonormal basis for the slicing plane
    arbitrary = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.cross(arbitrary, plane_normal), 0):
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(plane_normal, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(plane_normal, u)
    v /= np.linalg.norm(v)

    # Fallback: if fewer than 3 neighbors, return a regular hexagon
    if len(neighbor_indices) < 3:
        radius = max_distance if max_distance is not None else 1.0
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        hex_pts = seed_pt + np.outer(np.cos(angles), u) * radius + np.outer(
            np.sin(angles), v
        ) * radius
        from design_api.services.voronoi_gen.uniform.regularizer import (
            regularize_hexagon,
        )

        return regularize_hexagon(hex_pts, plane_normal)

    neighbor_pts = seeds[neighbor_indices]
    vecs = neighbor_pts - seed_pt
    proj_u = vecs.dot(u)
    proj_v = vecs.dot(v)
    angles = np.arctan2(proj_v, proj_u)
    order = np.argsort(angles)
    ordered = [neighbor_indices[i] for i in order]

    vertices: List[np.ndarray] = []
    for i in range(len(ordered)):
        a = ordered[i]
        b = ordered[(i + 1) % len(ordered)]
        normals = [plane_normal, seeds[a] - seed_pt, seeds[b] - seed_pt]
        points = [seed_pt, 0.5 * (seed_pt + seeds[a]), 0.5 * (seed_pt + seeds[b])]
        N = np.vstack(normals)
        d = np.array([np.dot(n, p) for n, p in zip(normals, points)])
        try:
            x = np.linalg.solve(N, d)
        except np.linalg.LinAlgError:
            dir_vec = normals[1] + normals[2]
            dir_vec -= dir_vec.dot(plane_normal) * plane_normal
            dir_vec /= np.linalg.norm(dir_vec)
            length = max_distance if max_distance is not None else 1.0
            x = seed_pt + dir_vec * length
        vertices.append(x)

    hex_pts = np.vstack(vertices)
    if hex_pts.shape[0] < 6:
        hex_pts = np.vstack([hex_pts, hex_pts[: 6 - hex_pts.shape[0]]])
    elif hex_pts.shape[0] > 6:
        hex_pts = hex_pts[:6]

    from design_api.services.voronoi_gen.uniform.regularizer import regularize_hexagon

    return regularize_hexagon(hex_pts, plane_normal)
