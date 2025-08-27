import numpy as np
import logging
import random
import math
from typing import Union, Any
from typing import Tuple, List, Optional
from typing import Dict, Callable
import itertools

def regularize_hexagon(hex_pts: np.ndarray, plane_normal: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """
    Lightly regularize a hexagon by projecting its vertices onto the plane
    defined by ``plane_normal`` while preserving their in-plane positions.

    Args:
        hex_pts: (N,3) array of hexagon vertices in order.
        plane_normal: (3,) array normal to the slicing plane.

    Returns:
        (N,3) array of vertices constrained to the provided plane.
    """
    centroid = np.mean(hex_pts, axis=0)
    n = plane_normal / np.linalg.norm(plane_normal)
    # project points onto plane
    rel = hex_pts - centroid
    rel -= np.outer(rel.dot(n), n)
    # build orthonormal basis in plane
    arbitrary = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.cross(arbitrary, n), 0):
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    radius = np.mean(np.linalg.norm(rel, axis=1))
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    regular = np.outer(np.cos(angles), u) * radius + np.outer(np.sin(angles), v) * radius
    return centroid + regular

def hexagon_metrics(hex_pts: np.ndarray) -> Dict[str, Any]:
    """
    Compute edge-length and area metrics for a single hexagon.
    Args:
        hex_pts: (6,3) array of hexagon vertices in order.
    Returns:
        metrics: dict with keys:
            'edge_lengths': (6,) array of edge lengths,
            'mean_edge_length': float,
            'std_edge_length': float,
            'area': float
    """
    centroid = np.mean(hex_pts, axis=0)
    # Edge vectors
    edges = hex_pts - np.roll(hex_pts, -1, axis=0)
    edge_lengths = np.linalg.norm(edges, axis=1)
    mean_edge = np.mean(edge_lengths)
    std_edge = np.std(edge_lengths)
    # Compute area by summing triangular areas from centroid
    area = 0.0
    for i in range(hex_pts.shape[0]):
        a = hex_pts[i] - centroid
        b = hex_pts[(i+1)%hex_pts.shape[0]] - centroid
        area += 0.5 * np.linalg.norm(np.cross(a, b))
    return {
        'edge_lengths': edge_lengths,
        'mean_edge_length': mean_edge,
        'std_edge_length': std_edge,
        'area': area
    }
