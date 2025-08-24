import numpy as np
import random
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from types import SimpleNamespace

import math
def derive_bbox_from_primitive(primitive: Dict[str, Any]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute axis-aligned bounding box (min and max) for the given primitive spec.
    Supports:
      - sphere: {'sphere': {'radius': r}}
      - box: {'box': {'min': [x,y,z], 'max': [x,y,z]}}
    """
    # Sphere centered at origin
    if 'sphere' in primitive:
        r = float(primitive['sphere'].get('radius', 0))
        return (-r, -r, -r), (r, r, r)
    # Axis-aligned box
    if 'box' in primitive:
        box = primitive['box']
        bmin = box.get('min') or primitive.get('bbox_min')
        bmax = box.get('max') or primitive.get('bbox_max')
        if bmin and bmax:
            return tuple(bmin), tuple(bmax)
    # Fallback to zero-sized box
    return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)


def primitive_to_imds_mesh(primitive: Dict[str, Any]) -> Optional[Any]:
    """Generate a simple mesh approximation for a primitive.

    Returns a ``SimpleNamespace`` with a ``vertices`` attribute or ``None`` if the
    primitive type is unsupported or insufficiently specified.
    """

    if not primitive:
        return None

    if "sphere" in primitive:
        r = float(primitive["sphere"].get("radius", 0))
        if r <= 0:
            return None
        # Sample a coarse latitude/longitude grid on the sphere
        phi = np.linspace(0.0, np.pi, 6)
        theta = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
        verts = []
        for p in phi:
            for t in theta:
                verts.append(
                    (
                        r * np.sin(p) * np.cos(t),
                        r * np.sin(p) * np.sin(t),
                        r * np.cos(p),
                    )
                )
        return SimpleNamespace(vertices=np.asarray(verts))

    if "box" in primitive:
        bmin, bmax = derive_bbox_from_primitive(primitive)
        x0, y0, z0 = bmin
        x1, y1, z1 = bmax
        verts = [
            (x0, y0, z0),
            (x1, y0, z0),
            (x0, y1, z0),
            (x1, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x0, y1, z1),
            (x1, y1, z1),
        ]
        return SimpleNamespace(vertices=np.asarray(verts))

    return None

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
    cell_size = max(2 * spacing, 1e-9)
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
    primitive: Dict[str, Any],
    *,
    return_cells: bool = False,
    use_voronoi_edges: bool = False,
    mode: Literal["organic", "uniform"] = "organic",
    **cell_kwargs: Any,
) -> Union[
    Tuple[List[Tuple[float, float, float]], List[Tuple[int, int]]],
    Tuple[List[Tuple[float, float, float]], List[Tuple[int, int]], Any],
]:
    """
    Generate a 3D hexagonally-packed lattice of points within the given AABB,
    returning a list of points and edges.  By default the edges are derived
    from a full Voronoi diagram (using SciPy when available).  If
    ``use_voronoi_edges`` is ``False``, edges are instead approximated via
    ``prune_adjacency_via_grid``/``compute_voronoi_adjacency`` and represented
    by the midpoints of adjacent seed pairs.

    If ``return_cells`` is True, the function also constructs Voronoi cell
    geometry for each seed. Set ``mode`` to ``"organic"`` (default) to call
    :func:`organic.construct_voronoi_cells`, or ``"uniform"`` to invoke
    :func:`uniform.compute_uniform_cells`. Additional keyword arguments are
    forwarded to the selected function. When ``mode`` is ``"uniform"``, the
    returned cells are a mapping from seed indices to vertex arrays.
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
                pts.append((float(x), float(y), float(z)))
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

    adjacency = prune_adjacency_via_grid(pts, spacing * 0.5)
    if use_voronoi_edges:
        # Build Voronoi diagram and extract ridge segments (finite edges)
        try:
            from scipy.spatial import Voronoi  # type: ignore
            arr = np.array(pts)
            if len(arr) == 0:
                raise ValueError("No seed points for Voronoi")
            if np.allclose(arr[:, 2], arr[0, 2]):
                # 2D planar case: compute Voronoi in XY plane
                vor = Voronoi(arr[:, :2])
                z_val = float(arr[0, 2])
                get_vertex = lambda v: (float(v[0]), float(v[1]), z_val)
            else:
                vor = Voronoi(arr)
                get_vertex = lambda v: (float(v[0]), float(v[1]), float(v[2]))
            vertex_map: Dict[int, int] = {}
            verts: List[Tuple[float, float, float]] = []
            edge_list: List[Tuple[int, int]] = []
            for rv in vor.ridge_vertices:
                if len(rv) != 2 or rv[0] == -1 or rv[1] == -1:
                    continue  # skip infinite ridges
                v0, v1 = rv
                p0 = vor.vertices[v0]
                p1 = vor.vertices[v1]
                if primitive:
                    if not point_in_primitive(tuple(p0), primitive) or not point_in_primitive(tuple(p1), primitive):
                        continue
                for vi, p in ((v0, p0), (v1, p1)):
                    if vi not in vertex_map:
                        vertex_map[vi] = len(verts)
                        verts.append(get_vertex(p))
                edge_list.append((vertex_map[v0], vertex_map[v1]))
        except Exception:
            # Fall back to seed adjacency when SciPy is unavailable
            verts = list(pts)
            edge_list = adjacency
    else:
        # Approximate edges using midpoints of adjacent seed pairs
        verts = []
        for i, j in adjacency:
            mid = (
                (pts[i][0] + pts[j][0]) * 0.5,
                (pts[i][1] + pts[j][1]) * 0.5,
                (pts[i][2] + pts[j][2]) * 0.5,
            )
            if primitive and not point_in_primitive(mid, primitive):
                continue
            verts.append(mid)
        edge_list = []

    if return_cells:
        if mode == "uniform":
            from .uniform.construct import compute_uniform_cells

            cells, edge_list = compute_uniform_cells(
                np.asarray(pts), return_edges=True, **cell_kwargs
            )
            # Reconstruct the reconciled vertex list in the same order used
            # when computing ``edge_list`` so edge indices remain valid.
            verts = [
                tuple(map(float, xyz))
                for idx in sorted(cells.keys())
                for xyz in cells[idx]
            ]
        else:
            from .organic.construct import construct_voronoi_cells

            cells = construct_voronoi_cells(
                pts, bbox_min, bbox_max, **cell_kwargs
            )
        return verts, edge_list, cells

    # Ensure edges are bidirectional
    edge_list = edge_list + [(j, i) for i, j in edge_list]
    return verts, edge_list



def compute_voronoi_adjacency(
    points: List[Tuple[float, float, float]],
    *args: Any,
    spacing: Optional[float] = None,
    **kwargs: Any,
) -> List[Tuple[int, int]]:
    """
    Compute a list of neighboring seed index pairs using a spatial hash grid.

    Parameters
    ----------
    points
        Seed coordinates as ``(x, y, z)`` tuples or arrays.
    spacing
        Optional minimum spacing between seeds. If omitted, it is inferred
        from the closest pair of points.
    *args, **kwargs
        Additional positional/keyword arguments are ignored for backward
        compatibility with older call signatures.

    Returns
    -------
    List[Tuple[int, int]]
        Each tuple ``(i, j)`` denotes an undirected edge between seeds ``i`` and
        ``j``.  The list contains only ``i < j`` pairs.
    """
    # Ignore legacy positional arguments (e.g., bbox, resolution)
    if spacing is None or isinstance(spacing, (tuple, list)):
        spacing = None

    if spacing is None and len(points) > 0:
        min_dist = float("inf")
        for i, (x0, y0, z0) in enumerate(points):
            for j in range(i + 1, len(points)):
                x1, y1, z1 = points[j]
                dx, dy, dz = x0 - x1, y0 - y1, z0 - z1
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist < min_dist:
                    min_dist = dist
        spacing = min_dist / 2.0 if min_dist < float("inf") else 0.0

    return prune_adjacency_via_grid(points, spacing or 0.0)
