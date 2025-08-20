import numpy as np
import logging
import random
import math
from typing import Union, Any
from typing import Tuple, List, Optional
from typing import Dict, Callable
import itertools

def regularize_hexagon(hex_pts: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """
    Normalize a hexagon to uniform edge lengths, preserving centroid.

    Args:
        hex_pts: (6,3) array of hexagon vertices in order.
        plane_normal: (3,) array normal to the plane in which the hexagon lies.

    Returns:
        new_pts: (6,3) array of regularized hexagon vertices.
    """
    # Compute centroid
    centroid = np.mean(hex_pts, axis=0)
    # Compute current edge lengths
    edges = hex_pts - np.roll(hex_pts, -1, axis=0)
    edge_lengths = np.linalg.norm(edges, axis=1)
    # Average edge length
    avg_edge = float(np.mean(edge_lengths))

    # Build an orthonormal basis (u, v) spanning the plane defined by plane_normal
    n = np.asarray(plane_normal, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        raise ValueError("plane_normal must be non-zero")
    n = n / n_norm
    # Choose a helper vector not parallel to the normal
    if abs(n[0]) < 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    else:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    # Generate perfect hexagon directions in the local (u, v) basis
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    dirs = np.stack([
        np.cos(angles),
        np.sin(angles)
    ], axis=1)  # (6,2)

    # Map directions back into 3-D using the basis
    new_pts = centroid + avg_edge * (dirs[:, 0:1] * u + dirs[:, 1:2] * v)
    return new_pts

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
