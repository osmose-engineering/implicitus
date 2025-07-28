import numpy as np
import random
import math
from typing import Union, Any
from typing import Tuple, List, Optional
from typing import Dict, Callable

import itertools
try:
    from scipy.spatial import Delaunay
except ImportError:
    Delaunay = None

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

# Compute Voronoi adjacency on a grid
def compute_voronoi_adjacency(
    sites: List[Tuple[float, float, float]],
    bbox_min: Tuple[float, float, float] = None,
    bbox_max: Tuple[float, float, float] = None,
    resolution: Tuple[int, int, int] = (64,64,64)
) -> Dict[Tuple[float,float,float], List[Tuple[float,float,float]]]:
    """
    Compute adjacency between Voronoi sites by voxelizing the domain and
    checking which sites' Voronoi regions are adjacent (6-connected).
    Returns a dict mapping each site (tuple) to a list of adjacent sites (tuples).
    """
    if bbox_min is None or bbox_max is None:
        seeds_np = np.array(sites)
        bbox_min = tuple(seeds_np.min(axis=0))
        bbox_max = tuple(seeds_np.max(axis=0))
    nx, ny, nz = resolution
    xs = np.linspace(bbox_min[0], bbox_max[0], nx)
    ys = np.linspace(bbox_min[1], bbox_max[1], ny)
    zs = np.linspace(bbox_min[2], bbox_max[2], nz)
    # Prepare label grid
    label_grid = np.empty((nx, ny, nz), dtype=int)
    sites_np = np.array(sites)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                p = np.array([x, y, z])
                dists = np.linalg.norm(sites_np - p, axis=1)
                label_grid[i, j, k] = int(np.argmin(dists))
    # Build adjacency sets
    N = len(sites)
    adjacency = {i: set() for i in range(N)}
    neighbor_offsets = [
        (-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)
    ]
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                label = label_grid[i, j, k]
                for dx, dy, dz in neighbor_offsets:
                    ni, nj, nk = i+dx, j+dy, k+dz
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        nlabel = label_grid[ni, nj, nk]
                        if nlabel != label:
                            adjacency[label].add(nlabel)
    # Convert sets to lists of Python ints
    adjacency_lists = {int(i): [int(n) for n in neighs] for i, neighs in adjacency.items()}
    return adjacency_lists



