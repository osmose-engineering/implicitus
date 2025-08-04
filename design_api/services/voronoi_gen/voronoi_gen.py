import numpy as np
import random
from typing import Union, Any, Dict
from typing import Tuple, List, Optional, Dict
from typing import Callable

import math
def build_spatial_index(
    seeds: List[Tuple[float, float, float]],
    spacing: float
) -> Dict[Tuple[int, int, int], List[int]]:
    """
    Build a 3D spatial hash grid of seed indices.

    Args:
        seeds: list of (x,y,z) seed coordinates
        spacing: target minimal seed spacing (r);
                 grid cell size will be 2 * spacing

    Returns:
        A dict mapping (i,j,k) cell keys to lists of seed indices in that cell.
    """
    grid: Dict[Tuple[int, int, int], List[int]] = {}
    cell_size = 2 * spacing
    for idx, (x, y, z) in enumerate(seeds):
        i = math.floor(x / cell_size)
        j = math.floor(y / cell_size)
        k = math.floor(z / cell_size)
        key = (i, j, k)
        grid.setdefault(key, []).append(idx)
    return grid


# --- Prune adjacency using spatial hash grid (efficient) ---
def prune_adjacency_via_grid(
    seeds: List[Tuple[float, float, float]],
    spacing: float
) -> List[Tuple[int, int]]:
    """
    Prune adjacency by only checking seeds in the same or neighboring spatial hash cells.
    Args:
        seeds: list of (x,y,z) seed coordinates
        spacing: target minimal seed spacing (r)
    Returns:
        List of undirected edges (i,j) where j>i and distance ≤ 2*spacing.
    """
    # build cell grid
    grid = build_spatial_index(seeds, spacing)
    edges = []
    max_dist2 = (2 * spacing) ** 2
    # neighbor cell offsets (3×3×3)
    neighbor_offsets = [(di, dj, dk)
                        for di in (-1, 0, 1)
                        for dj in (-1, 0, 1)
                        for dk in (-1, 0, 1)]
    # for each cell and its seeds
    for cell_key, idx_list in grid.items():
        i0, j0, k0 = cell_key
        # gather seeds in neighboring cells
        for di, dj, dk in neighbor_offsets:
            neighbor_key = (i0 + di, j0 + dj, k0 + dk)
            for idx in idx_list:
                for jdx in grid.get(neighbor_key, []):
                    if jdx > idx:
                        x0, y0, z0 = seeds[idx]
                        x1, y1, z1 = seeds[jdx]
                        dx = x0 - x1
                        dy = y0 - y1
                        dz = z0 - z1
                        if dx*dx + dy*dy + dz*dz <= max_dist2:
                            edges.append((idx, jdx))
    return edges


import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def _call_sdf(sdf_func, pt):
    """
    Wrapper to invoke an SDF function.
    Accepts either array-style or separate coords.
    Returns either a float (for single-point queries) or numpy array.
    """
    arr = np.array(pt, dtype=float)
    # Ensure 2D input for array-based SDFs when pt is a single 3D point
    if arr.ndim == 1 and arr.shape[0] == 3:
        arr_arg = arr.reshape(1, 3)
    else:
        arr_arg = arr
    try:
        res = sdf_func(arr_arg)
    except TypeError:
        try:
            res = sdf_func(*tuple(arr))
        except TypeError:
            # Last resort: call with original pt
            res = sdf_func(pt)
    # Convert to numpy array
    res_arr = np.array(res, dtype=float)
    # If single-value array, return scalar
    if res_arr.ndim == 1 and res_arr.shape[0] == 1:
        return float(res_arr[0])
    return res_arr

# --- Adaptive Octree Grid for Level-of-Detail Sampling ---
class OctreeNode:
    """
    Represents an axis-aligned box region that can be recursively subdivided
    based on an error metric.
    """
    def __init__(self, bbox_min: Tuple[float,float,float], bbox_max: Tuple[float,float,float], depth: int = 0):
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.depth = depth
        self.children: List[OctreeNode] = []

    def subdivide(self,
                  error_metric: Callable[['OctreeNode'], float],
                  max_depth: int,
                  threshold: float):
        """
        Recursively subdivide this node if error_metric(node) > threshold
        and depth < max_depth.
        """
        err = error_metric(self)
        if self.depth >= max_depth or err <= threshold:
            return
        # compute midpoints
        x0,y0,z0 = self.bbox_min
        x1,y1,z1 = self.bbox_max
        mx, my, mz = (0.5*(x0+x1), 0.5*(y0+y1), 0.5*(z0+z1))
        boxes = [
            ((x0,y0,z0), (mx,my,mz)),
            ((mx,y0,z0), (x1,my,mz)),
            ((x0,my,z0), (mx,y1,mz)),
            ((mx,my,z0), (x1,y1,mz)),
            ((x0,y0,mz), (mx,my,z1)),
            ((mx,y0,mz), (x1,my,z1)),
            ((x0,my,mz), (mx,y1,z1)),
            ((mx,my,mz), (x1,y1,z1)),
        ]
        for bmin, bmax in boxes:
            child = OctreeNode(bmin, bmax, depth=self.depth+1)
            child.subdivide(error_metric, max_depth, threshold)
            self.children.append(child)


# --- Helper functions for normals, curvature, fillet radius, and smooth union ---
def estimate_normal(sdf_func, pts, eps: float = 1e-4) -> np.ndarray:
    """
    Estimate unit surface normals at points via central differences.
    pts: array-like of shape (N,3) or (3,)
    Returns: normals array of shape (N,3)
    """
    pts_arr = np.atleast_2d(pts)
    normals = []
    for p in pts_arr:
        x, y, z = p
        dx = (_call_sdf(sdf_func, (x+eps, y, z)) - _call_sdf(sdf_func, (x-eps, y, z))) / (2*eps)
        dy = (_call_sdf(sdf_func, (x, y+eps, z)) - _call_sdf(sdf_func, (x, y-eps, z))) / (2*eps)
        dz = (_call_sdf(sdf_func, (x, y, z+eps)) - _call_sdf(sdf_func, (x, y, z-eps))) / (2*eps)
        n = np.array([dx, dy, dz], dtype=float)
        norm = np.linalg.norm(n)
        if norm > 0:
            n /= norm
        normals.append(n)
    return np.vstack(normals)

def estimate_hessian(sdf_func, pts, eps: float = 1e-3) -> np.ndarray:
    """
    Estimate Hessian matrices at points. Returns zero Hessian by default.
    """
    pts_arr = np.atleast_2d(pts)
    N = pts_arr.shape[0]
    return np.zeros((N, 3, 3), dtype=float)

def compute_fillet_radius(sdf_func, pts, eps: float = 1e-4) -> np.ndarray:
    """
    Compute fillet radius at points by projecting points onto normals.
    """
    pts_arr = np.atleast_2d(pts)
    normals = estimate_normal(sdf_func, pts_arr, eps)
    radii = [float(np.dot(p, n)) for p, n in zip(pts_arr, normals)]
    return np.array(radii, dtype=float)

def smooth_union(a, b, r: float):
    """
    Smooth union of two SDF fields (elementwise).
    a, b can be scalars or numpy arrays.
    """
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    h = np.clip(0.5 + 0.5*(b_arr - a_arr)/r, 0.0, 1.0)
    return h * b_arr + (1.0 - h) * a_arr

def smooth_intersection(a: float, b: float, r: float) -> float:
    """
    Smooth boolean intersection of two SDF values a, b using radius r.
    """
    h = max(0.0, min(1.0, 0.5 - 0.5 * (b - a) / r))
    return a * (1 - h) + b * h + r * h * (1 - h)

def smooth_difference(a: float, b: float, r: float) -> float:
    """
    Smooth boolean difference (a minus b) of two SDF values using radius r.
    """
    # difference = intersection of a and complement of b
    return smooth_intersection(a, -b, r)




# --- Helper function for primitive clipping ---
def point_in_primitive(
    pt: Tuple[float, float, float],
    primitive: Dict[str, Any]
) -> bool:
    """
    Return True if the point pt lies within the given primitive spec.
    Supports:
      - sphere    : {'sphere': {'radius': r}}
      - box       : {'box': {'min': [...], 'max': [...]} or 'bbox_min','bbox_max'}
    """
    # Sphere (centered at origin)
    if 'sphere' in primitive:
        sph = primitive['sphere']
        r = float(sph.get('radius', 0))
        x, y, z = pt
        return (x*x + y*y + z*z) <= (r*r)
    # Axis-aligned box
    if 'box' in primitive:
        box = primitive['box']
        bmin = box.get('min') or primitive.get('bbox_min')
        bmax = box.get('max') or primitive.get('bbox_max')
        if bmin and bmax:
            return all(bmin[i] <= pt[i] <= bmax[i] for i in range(3))
    return False


# --- 3D Hexagonal Lattice Sampling ---
def build_hex_lattice(
    bbox_min: Tuple[float, float, float],
    bbox_max: Tuple[float, float, float],
    spacing: float,
    primitive: Dict[str, Any]
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int]]]:
    """
    Generate a 3D hexagonally-packed lattice of points within the given AABB,
    and return both the list of points and the nearest-neighbor edges.
    """
    # Unpack bounds
    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max

    # Calculate the vertical and horizontal offsets for hex packing
    dx = spacing
    dy = spacing * np.sqrt(3) / 2
    dz = spacing * np.sqrt(6) / 3

    coords = []
    pts = []
    # layer-by-layer hex grid with integer coords
    k = 0
    z = z0
    while z <= z1:
        offset_y = (k % 2) * dy / 2
        offset_x = (k % 3) * dx / 2
        j = 0
        y = y0 + offset_y
        while y <= y1:
            i = 0
            x = x0 + offset_x
            while x <= x1:
                coords.append((i, j, k))
                pts.append((x, y, z))
                i += 1
                x = x0 + offset_x + i * dx
            j += 1
            y = y0 + offset_y + j * dy
        k += 1
        z = z0 + k * dz

    # Clip to the target shape
    if primitive:
        filtered = [(c, p) for c, p in zip(coords, pts) if point_in_primitive(p, primitive)]
        if filtered:
            coords, pts = zip(*filtered)
            coords, pts = list(coords), list(pts)
        else:
            coords, pts = [], []

    # Build adjacency efficiently via spatial pruning
    edges = prune_adjacency_via_grid(pts, spacing)

    return pts, edges



# Alias for Voronoi infill: direct seed adjacency via grid pruning
compute_voronoi_adjacency = prune_adjacency_via_grid
