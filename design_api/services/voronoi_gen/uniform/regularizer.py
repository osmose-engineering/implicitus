import numpy as np
import logging
import random
import math
from typing import Union, Any
from typing import Tuple, List, Optional
from typing import Dict, Callable
import itertools

def regularize_hexagon(hex_pts: np.ndarray) -> np.ndarray:
    """
    Normalize a hexagon to uniform edge lengths, preserving centroid.
    hex_pts: (6,3) array of hexagon vertices in order.
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
    # Generate perfect hexagon directions in the XY plane
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    dirs = np.stack([
        np.cos(angles),
        np.sin(angles),
        np.zeros_like(angles)
    ], axis=1)
    # Build new points at uniform edge length radius
    new_pts = centroid + dirs * avg_edge
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
