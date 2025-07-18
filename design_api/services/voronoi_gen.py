import numpy as np
import numpy as np  # ensure numpy is available for cap SDF
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

def generate_adaptive_grid(
    bbox_min: Tuple[float,float,float],
    bbox_max: Tuple[float,float,float],
    max_depth: int = 4,
    threshold: float = 0.1,
    error_metric: Callable[[OctreeNode], float] = None
) -> OctreeNode:
    """
    Generate an adaptive octree over the domain [bbox_min, bbox_max].
    error_metric should take an OctreeNode and return a scalar error.
    By default, use the magnitude of mean curvature from estimate_hessian.
    """
    if error_metric is None:
        # default: sample mean curvature at cell center
        def error_metric(node: OctreeNode) -> float:
            (x0,y0,z0), (x1,y1,z1) = node.bbox_min, node.bbox_max
            center = ((x0+x1)/2, (y0+y1)/2, (z0+z1)/2)
            # approximate curvature by eigenvalues of Hessian
            H = estimate_hessian(center, lambda p: body_sdf(p))
            kappa = np.real(np.linalg.eigvals(H))
            return abs(kappa).sum() / 2.0
    root = OctreeNode(bbox_min, bbox_max, depth=0)
    root.subdivide(error_metric, max_depth, threshold)
    return root

# --- Helper functions for normals, curvature, fillet radius, and smooth union ---
def estimate_normal(p: Tuple[float, float, float], sdf_func: callable, eps: float = 1e-4) -> np.ndarray:
    """
    Estimate the unit surface normal at point p by central differences of the SDF.
    """
    x, y, z = p
    dx = (_call_sdf(sdf_func, (x + eps, y, z)) - _call_sdf(sdf_func, (x - eps, y, z))) / (2 * eps)
    dy = (_call_sdf(sdf_func, (x, y + eps, z)) - _call_sdf(sdf_func, (x, y - eps, z))) / (2 * eps)
    dz = (_call_sdf(sdf_func, (x, y, z + eps)) - _call_sdf(sdf_func, (x, y, z - eps))) / (2 * eps)
    n = np.array([dx, dy, dz])
    norm = np.linalg.norm(n)
    return n / norm if norm > 0 else n

def estimate_hessian(p: Tuple[float, float, float], sdf_func: callable, eps: float = 1e-3) -> np.ndarray:
    """
    Estimate the Hessian matrix of the SDF at point p via central differences.
    """
    x, y, z = p
    def d(i, j):
        coords = [x, y, z]
        coords[i] += eps
        coords[j] += eps
        fpp = _call_sdf(sdf_func, tuple(coords))
        coords[j] -= 2*eps
        fpm = _call_sdf(sdf_func, tuple(coords))
        coords[i] -= 2*eps
        fmm = _call_sdf(sdf_func, tuple(coords))
        coords[j] += 2*eps
        fmp = _call_sdf(sdf_func, tuple(coords))
        return (fpp - fpm - fmp + fmm) / (4 * eps**2)
    H = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            H[i,j] = d(i,j)
    return H

def compute_fillet_radius(p: Tuple[float, float, float], sdf_func: callable, alpha: float = 1.0, min_r: float = 0.1, max_r: float = 5.0, eps: float = 1e-4) -> float:
    """
    Compute an adaptive fillet radius at point p based on mean curvature.
    """
    # estimate normal and hessian
    n = estimate_normal(p, sdf_func, eps)
    H = estimate_hessian(p, sdf_func, eps)
    # project Hessian to tangent plane: S = (I - n n^T) H (I - n n^T)
    P = np.eye(3) - np.outer(n, n)
    S = P @ H @ P
    # eigenvalues are principal curvatures
    kappa = np.linalg.eigvals(S)
    mean_curv = np.real(kappa).sum() / 2.0
    r = alpha / (abs(mean_curv) + 1e-6)
    return float(max(min_r, min(max_r, r)))


def smooth_union(a: float, b: float, r: float) -> float:
    """
    Smooth boolean union of two SDF values a, b using radius r.
    """
    h = max(0.0, min(1.0, 0.5 + 0.5*(b - a)/r))
    return a * (1 - h) + b * h - r * h * (1 - h)

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

# Helper to call SDF with either tuple or separate coords
def _call_sdf(func, point):
    try:
        # Try tuple argument
        return func(point)
    except TypeError:
        # Try separate coords
        return func(*point)

def sample_surface_seed_points(
    num_points: int,
    bbox_min: Tuple[float, float, float],
    bbox_max: Tuple[float, float, float],
    sdf_func: callable,
    max_trials: int = 10000,
    projection_steps: int = 20,
    step_size: float = 0.1
) -> List[Tuple[float, float, float]]:
    """
    Sample points on the zero‐level isosurface of an SDF.
    
    1. Randomly pick candidates in the AABB.
    2. For each, do a short sphere‐trace / gradient‐descent toward sdf=0.
    3. Accept if you converge to |sdf| < tol within projection_steps.
    
    Returns:
        List of 3D points lying on the surface.
    """
    seeds: List[Tuple[float, float, float]] = []
    tol = 1e-3
    for _ in range(max_trials):
        if len(seeds) >= num_points:
            break
        # random sample in bounding box
        p = np.array([
            random.uniform(bbox_min[0], bbox_max[0]),
            random.uniform(bbox_min[1], bbox_max[1]),
            random.uniform(bbox_min[2], bbox_max[2])
        ])
        # project to surface
        for _ in range(projection_steps):
            d = _call_sdf(sdf_func, p)
            if abs(d) < tol:
                break
            # finite-difference gradient
            eps = 1e-4
            grad = np.array([
                _call_sdf(sdf_func, (p[0]+eps, p[1], p[2])) - _call_sdf(sdf_func, (p[0]-eps, p[1], p[2])),
                _call_sdf(sdf_func, (p[0], p[1]+eps, p[2])) - _call_sdf(sdf_func, (p[0], p[1]-eps, p[2])),
                _call_sdf(sdf_func, (p[0], p[1], p[2]+eps)) - _call_sdf(sdf_func, (p[0], p[1], p[2]-eps)),
            ]) / (2 * eps)
            norm = np.linalg.norm(grad)
            if norm == 0:
                break
            p = p - step_size * d * grad / norm
        if abs(_call_sdf(sdf_func, p)) < tol:
            seeds.append(tuple(p))
    return seeds


def construct_surface_voronoi_cells(
    seed_points: List[Tuple[float, float, float]],
    body_sdf: callable,
    wall_thickness_mm: float = 1.0,
    thickness_field: Callable[[Tuple[float,float,float]], float] = None,
    bbox_min: Optional[Tuple[float, float, float]] = None,
    bbox_max: Optional[Tuple[float, float, float]] = None,
    csg_ops: Optional[List[Dict[str, Any]]] = None,
    blend_curve: Optional[Callable[[float], float]] = None,
    shell_offset: float = 0.0,
    adaptive_grid: Optional[OctreeNode] = None,
    resolution: Tuple[int, int, int] = (64, 64, 64)
) -> list:
    """
    Build a surface‐only Voronoi lattice SDF on a 3D grid, clipped to a thin shell.

    This constructs the surface Voronoi cells:
      1. For each grid cell, compute distance to nearest seed point.
      2. Subtract half the wall_thickness to get the raw lattice SDF.
      3. Mask it so it only occupies a band around the body surface:
         mask(p) = abs(body_sdf(p)) < wall_thickness_mm
      4. Return a list of dicts, each with keys "site" (seed point) and "sdf" (3D array of final SDF values for marching‐cubes).
    """
    # Adaptive grid sampling: use octree leaves as sample points
    if adaptive_grid is not None:
        # collect leaf nodes
        leaves: List[OctreeNode] = []
        def collect(node: OctreeNode):
            if not node.children:
                leaves.append(node)
            else:
                for child in node.children:
                    collect(child)
        collect(adaptive_grid)
        # sample at leaf centers
        grid_points = [
            tuple((np.array(node.bbox_min) + np.array(node.bbox_max)) * 0.5)
            for node in leaves
        ]
        cells = []
        for seed in seed_points:
            samples = []
            for p in grid_points:
                p_arr = np.array(p)
                # compute lattice SDF
                d_seed = np.linalg.norm(np.array(seed) - p_arr)
                t = thickness_field(p) if thickness_field is not None else wall_thickness_mm
                lattice_sdf = d_seed - (t / 2.0)
                body_d = _call_sdf(body_sdf, p_arr)
                # initial val as before
                if abs(body_d) < wall_thickness_mm:
                    val0 = max(body_d, -lattice_sdf)
                else:
                    val0 = body_d
                val = val0
                # Hybrid lattice shell blending
                if blend_curve is not None or shell_offset != 0.0:
                    # recover volumetric d1-d0 from raw_val
                    # Compute raw_val as in construct_voronoi_cells
                    dists = [np.linalg.norm(np.array(s) - p_arr) for s in seed_points]
                    dists_sorted = sorted(dists)
                    d0 = dists_sorted[0]
                    d1 = dists_sorted[1] if len(dists_sorted) > 1 else d0
                    raw_val = (d1 - d0)
                    raw_vol = raw_val + (wall_thickness_mm / 2.0)
                    dist = abs(body_d) - shell_offset
                    t_norm = (min(max(dist / wall_thickness_mm, 0.0), 1.0)
                              if wall_thickness_mm > 0 else 1.0)
                    w = blend_curve(t_norm) if blend_curve else t_norm
                    val = (1.0 - w) * val0 + w * raw_vol
                if csg_ops:
                    for op in csg_ops:
                        typ, func, blend_r = op['op'], op['sdf'], op['r']
                        other_v = _call_sdf(func, p_arr)
                        if typ == 'union':
                            val = smooth_union(val, other_v, blend_r)
                        elif typ == 'intersection':
                            val = smooth_intersection(val, other_v, blend_r)
                        elif typ == 'difference':
                            val = smooth_difference(val, other_v, blend_r)
                samples.append((p, val))
            cells.append({
                "site": seed,
                "samples": samples,
                "vertices": [],
                "volume": 0.0,
                "area": 0.0,
                "neighbors": []
            })
        # Compute adjacency via compute_voronoi_adjacency
        adjacency = compute_voronoi_adjacency(
            seed_points, bbox_min, bbox_max, resolution
        )
        for cell in cells:
            cell["neighbors"] = adjacency.get(cell["site"], [])
        return cells
    if bbox_min is None or bbox_max is None:
        xs, ys, zs = zip(*seed_points)
        if bbox_min is None:
            bbox_min = (min(xs), min(ys), min(zs))
        if bbox_max is None:
            bbox_max = (max(xs), max(ys), max(zs))
    nx, ny, nz = resolution
    xs = np.linspace(bbox_min[0], bbox_max[0], nx)
    ys = np.linspace(bbox_min[1], bbox_max[1], ny)
    zs = np.linspace(bbox_min[2], bbox_max[2], nz)
    # Precompute full-volume Voronoi grids for hybrid blending
    vol_cells = None
    if blend_curve is not None or shell_offset != 0.0:
        vol_cells = construct_voronoi_cells(
            seed_points, bbox_min, bbox_max,
            resolution=resolution,
            wall_thickness=wall_thickness_mm,
            csg_ops=None,
            adaptive_grid=None
        )
    cells = []
    for idx, seed in enumerate(seed_points):
        grid = np.zeros((nx, ny, nz), dtype=float)
        vol_grid = (vol_cells[idx]['sdf']
                    + (wall_thickness_mm / 2.0)) if vol_cells is not None else None
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for k, z in enumerate(zs):
                    p = np.array([x, y, z])
                    d_seed = np.linalg.norm(np.array(seed) - p)
                    # allow variable wall thickness
                    t = thickness_field(tuple(p)) if thickness_field is not None else wall_thickness_mm
                    lattice_sdf = d_seed - (t / 2.0)
                    body_d = _call_sdf(body_sdf, p)
                    if abs(body_d) < wall_thickness_mm:
                        val0 = max(body_d, -lattice_sdf)
                    else:
                        val0 = body_d
                    val = val0
                    # Hybrid blending
                    if vol_grid is not None:
                        raw_vol = vol_grid[i, j, k]
                        dist = abs(body_d) - shell_offset
                        t_norm = (min(max(dist / wall_thickness_mm, 0.0), 1.0)
                                  if wall_thickness_mm > 0 else 1.0)
                        w = blend_curve(t_norm) if blend_curve else t_norm
                        val = (1.0 - w) * val0 + w * raw_vol
                    if csg_ops:
                        for op in csg_ops:
                            typ, func, blend_r = op['op'], op['sdf'], op['r']
                            other_v = _call_sdf(func, p)
                            if typ == 'union':
                                val = smooth_union(val, other_v, blend_r)
                            elif typ == 'intersection':
                                val = smooth_intersection(val, other_v, blend_r)
                            elif typ == 'difference':
                                val = smooth_difference(val, other_v, blend_r)
                    grid[i, j, k] = val
        cells.append({
            "site": seed,
            "sdf": grid,
            "vertices": [],
            "volume": 0.0,
            "area": 0.0
        })
    # Compute adjacency and add to each cell dict
    adjacency = compute_voronoi_adjacency(
        seed_points, bbox_min, bbox_max, resolution
    )
    for cell in cells:
        site = cell["site"]
        cell["neighbors"] = adjacency.get(site, [])
    return cells


# Compute Voronoi adjacency on a grid
def compute_voronoi_adjacency(
    sites: List[Tuple[float, float, float]],
    bbox_min: Tuple[float, float, float],
    bbox_max: Tuple[float, float, float],
    resolution: Tuple[int, int, int] = (64,64,64)
) -> Dict[Tuple[float,float,float], List[Tuple[float,float,float]]]:
    """
    Compute adjacency between Voronoi sites by voxelizing the domain and
    checking which sites' Voronoi regions are adjacent (6-connected).
    Returns a dict mapping each site (tuple) to a list of adjacent sites (tuples).
    """
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
    adjacency = {tuple(site): set() for site in sites}
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
                            s1 = tuple(sites[label])
                            s2 = tuple(sites[nlabel])
                            adjacency[s1].add(s2)
    # Convert sets to lists
    adjacency_lists = {site: list(neighs) for site, neighs in adjacency.items()}
    return adjacency_lists

def sample_seed_points(
    num_points: int,
    bbox_min: Tuple[float, float, float],
    bbox_max: Tuple[float, float, float],
    *,
    density_field: Optional[Callable[[Tuple[float, float, float]], float]] = None,
    min_dist: Optional[float] = None,
    max_trials: int = 10000
) -> List[Tuple[float, float, float]]:
    """
    Sample seed points using Poisson-disk sampling (Bridson's algorithm) within the axis-aligned bounding box.

    Args:
        num_points (int): Number of points to sample.
        bbox_min (tuple): Minimum (x, y, z) of bounding box.
        bbox_max (tuple): Maximum (x, y, z) of bounding box.
        min_dist (float, optional): Minimum distance between points. If not provided, spacing is chosen to target num_points by volume.
        max_trials (int): Maximum number of points to attempt to generate.
    """
    # Compute domain volume and minimal distance r
    xmin, ymin, zmin = bbox_min
    xmax, ymax, zmax = bbox_max
    volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
    if num_points <= 0 or volume <= 0:
        return []
    if min_dist is not None:
        r = min_dist
    else:
        r = (volume / num_points) ** (1/3)
    # Determine local Poisson radius
    if density_field is not None:
        def _get_r(p):
            d = density_field(p)
            if d is None or d <= 0:
                return float('inf')
            return (1.0 / d) ** (1/3)
    elif min_dist is not None:
        def _get_r(p):
            return min_dist
    else:
        def _get_r(p):
            return r
    cell_size = r / math.sqrt(3)
    nx = int(math.ceil((xmax - xmin) / cell_size))
    ny = int(math.ceil((ymax - ymin) / cell_size))
    nz = int(math.ceil((zmax - zmin) / cell_size))
    grid = [[[None for _ in range(nz)] for _ in range(ny)] for _ in range(nx)]
    def grid_coords(pt):
        return (
            int((pt[0] - xmin) / cell_size),
            int((pt[1] - ymin) / cell_size),
            int((pt[2] - zmin) / cell_size)
        )
    def in_bbox(pt):
        return (xmin <= pt[0] <= xmax and ymin <= pt[1] <= ymax and zmin <= pt[2] <= zmax)
    def neighbors(ix, iy, iz):
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                for dz in [-2, -1, 0, 1, 2]:
                    x = ix + dx
                    y = iy + dy
                    z = iz + dz
                    if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                        if grid[x][y][z] is not None:
                            yield grid[x][y][z]
    # Initialize with one random point
    first_pt = tuple(random.uniform(bbox_min[i], bbox_max[i]) for i in range(3))
    points = [first_pt]
    active_list = [first_pt]
    gx, gy, gz = grid_coords(first_pt)
    grid[gx][gy][gz] = first_pt
    k = 30
    while active_list and len(points) < num_points and len(points) < max_trials:
        idx = random.randrange(len(active_list))
        center = active_list[idx]
        found = False
        for _ in range(k):
            local_r = _get_r(center)
            # Random point in the spherical shell [local_r, 2*local_r]
            rr = random.uniform(local_r, 2*local_r)
            theta = random.uniform(0, 2*math.pi)
            phi = math.acos(2*random.uniform(0,1)-1)
            dx = rr * math.sin(phi) * math.cos(theta)
            dy = rr * math.sin(phi) * math.sin(theta)
            dz = rr * math.cos(phi)
            pt = (center[0] + dx, center[1] + dy, center[2] + dz)
            if not in_bbox(pt):
                continue
            gx, gy, gz = grid_coords(pt)
            # Check min distance to neighbors
            too_close = False
            for neighbor in neighbors(gx, gy, gz):
                dist = math.sqrt((pt[0]-neighbor[0])**2 + (pt[1]-neighbor[1])**2 + (pt[2]-neighbor[2])**2)
                if dist < local_r:
                    too_close = True
                    break
            if not too_close:
                points.append(pt)
                active_list.append(pt)
                grid[gx][gy][gz] = pt
                found = True
                break
        if not found:
            active_list.pop(idx)
    # Adjust if too many or too few points
    if len(points) > num_points:
        points = random.sample(points, num_points)
    elif len(points) < num_points:
        # Fill remainder with uniform random points
        n_extra = num_points - len(points)
        for _ in range(n_extra):
            pt = tuple(random.uniform(bbox_min[i], bbox_max[i]) for i in range(3))
            points.append(pt)
    return points

def sample_seed_points_anisotropic(
    num_points: int,
    bbox_min: Tuple[float, float, float],
    bbox_max: Tuple[float, float, float],
    *,
    scale_field: Optional[Union[Tuple[float, float, float],
                                 Callable[[Tuple[float, float, float]],
                                          Tuple[float, float, float]]]] = None,
    density_field: Optional[Callable[[Tuple[float, float, float]], float]] = None,
    min_dist: Optional[float] = None,
    max_trials: int = 10000
) -> List[Tuple[float, float, float]]:
    """
    Sample points under an anisotropic metric defined by scale_field.
    We warp input space by dividing coordinates by scale, run the
    isotropic Poisson-disk sampler there, and then multiply back
    by scale to return to original space.
    """
    # 1) Build warp/unwarp
    if scale_field is None:
        warp = lambda p: p
        unwarp = lambda q: q
    else:
        if callable(scale_field):
            def warp(p):
                sx, sy, sz = scale_field(p)
                return (p[0]/sx, p[1]/sy, p[2]/sz)
            def unwarp(q):
                sx, sy, sz = scale_field(q)
                return (q[0]*sx, q[1]*sy, q[2]*sz)
        else:
            sx, sy, sz = scale_field
            warp   = lambda p: (p[0]/sx, p[1]/sy, p[2]/sz)
            unwarp = lambda q: (q[0]*sx, q[1]*sy, q[2]*sz)

    # 2) Warp the bbox
    warped_min = warp(bbox_min)
    warped_max = warp(bbox_max)

    # 3) Wrap density for warped space
    warped_density = None
    if density_field is not None:
        def warped_density(q):
            return density_field(unwarp(q))

    # 4) Sample in warped space
    warped_pts = sample_seed_points(
        num_points,
        warped_min,
        warped_max,
        density_field=warped_density,
        min_dist=min_dist,
        max_trials=max_trials
    )

    # 5) Unwarp back and return
    return [unwarp(q) for q in warped_pts]

def construct_voronoi_cells(
    points: List[Tuple[float, float, float]],
    bbox_min: Tuple[float, float, float],
    bbox_max: Tuple[float, float, float],
    resolution: Tuple[int, int, int] = (64, 64, 64),
    wall_thickness: float = 0.0,
    csg_ops: Optional[List[Dict[str, Any]]] = None,
    auto_cap: bool = False,
    cap_blend: float = 0.0,
    adaptive_grid: Optional[OctreeNode] = None
) -> List[Dict]:
    """
    Build full-volume Voronoi cell SDFs on a 3D grid.

    For each voxel in a regular grid:
      1. Compute distances to all seed points.
      2. Identify the nearest (d0) and second-nearest (d1) distances.
      3. If the voxel belongs to a cell (nearest == this seed), set SDF = (d1 - d0) - (wall_thickness/2).
         Otherwise assign a large positive 'outside' value.
    Returns per-cell dicts with keys:
      - "site": the seed point
      - "sdf": 3D numpy array of SDF values
      - "vertices": [] (for later marching-cubes output)
      - "volume": 0.0  (to fill in later)
      - "neighbors": list of adjacent seed points
    """
    # Prepare box SDF for automatic capping
    bmin_np = np.array(bbox_min)
    bmax_np = np.array(bbox_max)
    def box_sdf(p: np.ndarray) -> float:
        # Signed distance to AABB: negative inside, positive outside
        q = np.maximum(bmin_np - p, p - bmax_np)
        outside = np.linalg.norm(np.maximum(q, 0.0))
        inside = min(max(q[0], max(q[1], q[2])), 0.0)
        return outside + inside

    # Adaptive grid sampling: use octree leaves as sample points
    if adaptive_grid is not None:
        # collect leaf nodes
        leaves: List[OctreeNode] = []
        def collect(node: OctreeNode):
            if not node.children:
                leaves.append(node)
            else:
                for child in node.children:
                    collect(child)
        collect(adaptive_grid)
        # sample at leaf centers
        grid_points = [
            tuple((np.array(node.bbox_min) + np.array(node.bbox_max)) * 0.5)
            for node in leaves
        ]
        # build cells with SDF samples per leaf
        cells = []
        for seed in points:
            samples = []
            for p in grid_points:
                p_arr = np.array(p)
                # compute nearest and second-nearest distances
                dists = [np.linalg.norm(np.array(s) - p_arr) for s in points]
                dists_sorted = sorted(dists)
                d0 = dists_sorted[0]
                d1 = dists_sorted[1] if len(dists_sorted) > 1 else d0
                raw_val = (d1 - d0) - (wall_thickness / 2.0)
                val = raw_val
                if csg_ops:
                    for op in csg_ops:
                        typ, func, blend_r = op['op'], op['sdf'], op['r']
                        other_v = _call_sdf(func, p_arr)
                        if typ == 'union':
                            val = smooth_union(val, other_v, blend_r)
                        elif typ == 'intersection':
                            val = smooth_intersection(val, other_v, blend_r)
                        elif typ == 'difference':
                            val = smooth_difference(val, other_v, blend_r)
                # Automatic capping for adaptive samples
                if auto_cap:
                    cap_val = box_sdf(p_arr)
                    if cap_val > 0:
                        val = smooth_intersection(val, cap_val, cap_blend)
                samples.append((p, val))
            cells.append({
                "site": seed,
                "samples": samples,
                "vertices": [],
                "volume": 0.0,
                "neighbors": []
            })
        # Compute adjacency via Delaunay triangulation of seed points
        if Delaunay is not None and len(points) >= 2:
            try:
                pts_np = np.array(points)
                tri = Delaunay(pts_np)
                adj_map = {tuple(seed): set() for seed in points}
                for simplex in tri.simplices:
                    for i in range(len(simplex)):
                        for j in range(i+1, len(simplex)):
                            si = tuple(points[simplex[i]])
                            sj = tuple(points[simplex[j]])
                            adj_map[si].add(sj)
                            adj_map[sj].add(si)
                # Assign neighbor lists
                for cell in cells:
                    cell['neighbors'] = list(adj_map[cell['site']])
            except Exception:
                # Fallback with no neighbors
                for cell in cells:
                    cell['neighbors'] = []
        else:
            # Fallback with no neighbors
            for cell in cells:
                cell['neighbors'] = []
        return cells

    nx, ny, nz = resolution
    xs = np.linspace(bbox_min[0], bbox_max[0], nx)
    ys = np.linspace(bbox_min[1], bbox_max[1], ny)
    zs = np.linspace(bbox_min[2], bbox_max[2], nz)
    sites_np = np.array(points)

    # Allocate grids for labels and the two smallest distances
    label_grid = np.empty((nx, ny, nz), dtype=int)
    d0_grid = np.empty((nx, ny, nz), dtype=float)
    d1_grid = np.empty((nx, ny, nz), dtype=float)

    # Compute distance fields
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                p = np.array([x, y, z])
                dists = np.linalg.norm(sites_np - p, axis=1)
                # Get indices of the two smallest distances (handles single-site case)
                if dists.size >= 2:
                    # argpartition at 1 ensures first two elements are the smallest distances
                    idxs = np.argpartition(dists, 1)[:2]
                else:
                    # single-site: duplicate index 0
                    idxs = np.array([0, 0], dtype=int)
                d_sorted = dists[idxs]
                # Order them so d0 <= d1
                if d_sorted[0] <= d_sorted[1]:
                    d0, d1 = d_sorted[0], d_sorted[1]
                    owner = idxs[0]
                else:
                    d0, d1 = d_sorted[1], d_sorted[0]
                    owner = idxs[1]
                label_grid[i, j, k] = owner
                d0_grid[i, j, k] = d0
                d1_grid[i, j, k] = d1

    # Define an 'outside' SDF value for non-cell voxels
    outside = float(np.max(d1_grid) - np.min(d0_grid))
    cells = []

    # Build each cell's SDF grid
    for s, pt in enumerate(points):
        raw_sdf = (d1_grid - d0_grid) - (wall_thickness / 2.0)
        cell_sdf = np.where(label_grid == s, raw_sdf, outside)
        # Apply CSG operations if provided
        if csg_ops:
            final_sdf = cell_sdf.copy()
            # Precompute other SDF grids for each op
            for op in csg_ops:
                op_type = op['op']
                sdf_func = op['sdf']
                blend_r = op['r']
                other = np.zeros_like(final_sdf)
                for ii in range(nx):
                    for jj in range(ny):
                        for kk in range(nz):
                            p_pt = np.array([xs[ii], ys[jj], zs[kk]])
                            other[ii, jj, kk] = _call_sdf(sdf_func, p_pt)
                if op_type == 'union':
                    final_sdf = smooth_union(final_sdf, other, blend_r)
                elif op_type == 'intersection':
                    final_sdf = smooth_intersection(final_sdf, other, blend_r)
                elif op_type == 'difference':
                    final_sdf = smooth_difference(final_sdf, other, blend_r)
            cell_sdf = final_sdf
        # Automatic capping at bounding box
        if auto_cap:
            for ii in range(nx):
                for jj in range(ny):
                    for kk in range(nz):
                        p = np.array([xs[ii], ys[jj], zs[kk]])
                        cap_val = box_sdf(p)
                        if cap_val > 0:
                            cell_sdf[ii, jj, kk] = smooth_intersection(cell_sdf[ii, jj, kk], cap_val, cap_blend)
        cells.append({
            "site": pt,
            "sdf": cell_sdf,
            "vertices": [],
            "volume": 0.0
        })

    # Attach neighbor lists
    adjacency = compute_voronoi_adjacency(points, bbox_min, bbox_max, resolution)
    for cell in cells:
        cell["neighbors"] = adjacency.get(cell["site"], [])

    return cells
