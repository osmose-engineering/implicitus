import numpy as np
import logging
import random
import math
from typing import Union, Any
from typing import Tuple, List, Optional
from typing import Dict, Callable
import itertools

from typing import Any, Dict, Optional
from design_api.services.voronoi_gen.voronoi_gen import compute_voronoi_adjacency
from .sampler import trace_hexagon
from .regularizer import hexagon_metrics

def compute_uniform_cells(
    seeds: np.ndarray,
    imds_mesh: Any,
    plane_normal: np.ndarray,
    max_distance: Optional[float] = None
) -> Dict[int, np.ndarray]:
    """
    Compute near-uniform hexagonal Voronoi cells for each seed point.
    Args:
        seeds: (N,3) array of seed point locations.
        imds_mesh: mesh object with `.vertices` attribute for medial axis extraction.
        plane_normal: (3,) array defining slicing plane normal.
        max_distance: fallback distance for ray casting when no medial point is found.
    Returns:
        cells: dict mapping seed index to (6,3) array of hexagon vertices.
    """
    adjacency = compute_voronoi_adjacency(seeds.tolist())
    cells: Dict[int, np.ndarray] = {}
    for idx, seed in enumerate(seeds):
        neighbors = adjacency.get(idx, [])
        hex_pts = trace_hexagon(idx, seeds, neighbors, plane_normal, max_distance)
        # Optionally log metrics
        metrics = hexagon_metrics(hex_pts)
        logging.debug(
            f"Uniform cell {idx}: mean_edge="
            f"{metrics['mean_edge_length']:.3f}, std_edge="
            f"{metrics['std_edge_length']:.3f}, area="
            f"{metrics['area']:.3f}"
        )
        cells[idx] = hex_pts
    return cells
