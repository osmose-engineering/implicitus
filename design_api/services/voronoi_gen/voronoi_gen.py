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


def compute_delaunay_adjacency(
    sites: List[Tuple[float, float, float]]
) -> List[Tuple[int, int]]:
    """
    Compute Delaunay adjacency: for a set of 3D points, compute the
    Delaunay tetrahedralization and return unique edges between vertices.
    """
    if Delaunay is None:
        raise ImportError("SciPy Delaunay not available for compute_delaunay_adjacency")
    # Convert to numpy array of shape (N,3)
    pts_np = np.asarray(sites, dtype=float)
    # Compute Delaunay triangulation
    tri = Delaunay(pts_np)
    edges = set()
    # tri.simplices is an array of indices for each tetrahedron
    for simplex in tri.simplices:
        # For each pair of vertices in the simplex, add the edge
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                a, b = sorted((int(simplex[i]), int(simplex[j])))
                edges.add((a, b))
    # Return as a list of tuple pairs
    return list(edges)


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

# Compute Voronoi adjacency on a grid (SDF)
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

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.spatial import Delaunay

def circumcenter_3d(tet: np.ndarray) -> np.ndarray:
    """
    Compute the circumcenter of a tetrahedron given its 4 vertices (4x3 array).
    """
    A = tet[0]
    B = tet[1]
    C = tet[2]
    D = tet[3]
    # Solve using linear system: (P - A) dot (B - A) etc.
    # Build matrix M and RHS vector b for solving M x = b
    # where circumcenter x satisfies (x-A).(B-A)=|B-A|^2/2, etc.
    BA = B - A
    CA = C - A
    DA = D - A
    RHS = np.array([
        np.dot(BA, BA) / 2,
        np.dot(CA, CA) / 2,
        np.dot(DA, DA) / 2
    ])
    M = np.vstack((BA, CA, DA)).T  # 3x3
    # Solve for x-A
    try:
        x_rel = np.linalg.solve(M, RHS)
    except np.linalg.LinAlgError:
        # Fallback: return centroid
        return np.mean(tet, axis=0)
    return A + x_rel


def compute_voronoi_mesh(
    seeds: List[Tuple[float, float, float]],
    primitive: Optional[Dict[str, Any]] = None,
    surface_only: bool = False
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int]]]:
    """
    Given a list of 3D seed points, compute the Voronoi vertices (circumcenters of Delaunay tetrahedra)
    and the adjacency edges between those vertices, corresponding to shared faces.
    Optionally, cap open boundary faces to a primitive.
    Returns:
      - vertices: list of circumcenters (3D tuples)
      - edges: list of pairs of vertex indices
    """
    logger.debug(f"compute_voronoi_mesh: received {len(seeds)} seeds, primitive={primitive}")
    pts = np.array(seeds)
    tri = Delaunay(pts)
    simplices = tri.simplices  # shape (M,4)
    logger.debug(f"  Delaunay produced {len(simplices)} tetrahedra")
    # Compute circumcenters for each tetrahedron
    centers = []
    # compute circumcenters and log first few
    for tid, s in enumerate(simplices):
        c_pt = tuple(circumcenter_3d(pts[s]))
        if tid < 5:
            logger.debug(f"  circumcenter[{tid}] = {c_pt}")
        centers.append(c_pt)

    # Map each triangular face to the list of tet indices sharing it
    from itertools import combinations
    face2tets: Dict[Tuple[int, int, int], List[int]] = {}
    for tid, simplex in enumerate(simplices):
        for face in combinations(simplex, 3):
            key = tuple(sorted(face))
            face2tets.setdefault(key, []).append(tid)

    total_faces = len(face2tets)
    boundary_faces = sum(1 for t in face2tets.values() if len(t) == 1)
    interior_faces = total_faces - boundary_faces
    logger.debug(f"  faces: total={total_faces}, interior={interior_faces}, boundary={boundary_faces}")

    # Build edges: two tetrahedra sharing a face produce an edge between their centers
    edges: List[Tuple[int, int]] = []
    for tets in face2tets.values():
        if len(tets) == 2:
            a, b = sorted(tets)
            edges.append((a, b))

    # Cap boundary faces to create finite cell walls
    # Each face with only one incident tetrahedron is on the hull
    # Create a cap vertex on the primitive boundary for each such face
    for face, tets in face2tets.items():
        if surface_only and len(tets) == 1 and primitive:
            tid = tets[0]
            # compute centroid of the three seed points
            pts_arr = np.array(seeds)
            verts = pts_arr[list(face)]
            centroid = tuple(np.mean(verts, axis=0))

            # compute cap point depending on primitive type
            cap_pt = None
            if 'sphere' in primitive:
                # project onto sphere
                r = float(primitive['sphere'].get('radius', 1.0))
                dir_vec = np.array(centroid)
                norm = np.linalg.norm(dir_vec)
                if norm > 0:
                    cap_pt = tuple((dir_vec / norm) * r)
            elif 'box' in primitive:
                # clamp to box extents
                bmin = primitive.get('bbox_min') or primitive['box'].get('min')
                bmax = primitive.get('bbox_max') or primitive['box'].get('max')
                if bmin and bmax:
                    cap_pt = tuple(
                        max(bmin[i], min(centroid[i], bmax[i]))
                        for i in range(3)
                    )

            if cap_pt is not None:
                logger.debug(f"  cap for face {face}: centroid={centroid} -> cap_pt={cap_pt}")
                # append cap vertex and connect it to the interior center
                cap_idx = len(centers)
                centers.append(cap_pt)
                edges.append((tid, cap_idx))

    cap_count = len(centers) - len(simplices)
    logger.debug(f"  after capping: centers={len(centers)}, edges={len(edges)}, cap_vertices={cap_count}")
    # log final bounding box of centers
    arr = np.array(centers)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    logger.debug(f"  final centers bbox: min={tuple(mins)}, max={tuple(maxs)}")

    # --- Filter out outlier circumcenters beyond the sphere radius ---
    if primitive and 'sphere' in primitive and not surface_only:
        # allow slight tolerance beyond the exact radius
        r = float(primitive['sphere']['radius'])
        tol = 1.01 * r
        valid_idxs = [i for i, c in enumerate(centers) if np.linalg.norm(c) <= tol]
        # build index remapping
        idx_map = {old: new for new, old in enumerate(valid_idxs)}
        # remap centers and edges
        filtered_centers = [centers[i] for i in valid_idxs]
        filtered_edges = [
            (idx_map[a], idx_map[b])
            for (a, b) in edges
            if a in idx_map and b in idx_map
        ]
        logger.debug(f"  filtered centers from {len(centers)} to {len(filtered_centers)}, "
                     f"edges from {len(edges)} to {len(filtered_edges)}")
        centers, edges = filtered_centers, filtered_edges

    if surface_only and primitive:
        # keep only edges touching at least one cap-vertex
        num_tets = len(simplices)
        surface_edges = [(a, b) for (a, b) in edges if a >= num_tets or b >= num_tets]
        used = sorted({i for e in surface_edges for i in e})
        idx_map = {old: new for new, old in enumerate(used)}
        centers = [centers[i] for i in used]
        edges = [(idx_map[a], idx_map[b]) for (a, b) in surface_edges]
    return centers, edges
