import numpy as np
import logging
import random
import math
from typing import Union, Any
from typing import Tuple, List, Optional
from typing import Dict, Callable
import itertools
from .adaptive import OctreeNode, generate_adaptive_grid

try:  # SciPy is optional
    from scipy.spatial import Delaunay  # type: ignore
except Exception:  # pragma: no cover - fallback when SciPy missing
    Delaunay = None  # type: ignore


def _voronoi_helpers():
    """Lazy import of core Voronoi routines to avoid circular imports."""
    from .. import voronoi_gen as vg

    return vg


def compute_voronoi_adjacency(*args, **kwargs):
    """Proxy to :mod:`voronoi_gen.compute_voronoi_adjacency` for monkeypatching."""
    vg = _voronoi_helpers()
    return vg.compute_voronoi_adjacency(*args, **kwargs)

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
    vg = _voronoi_helpers()
    _call_sdf = vg._call_sdf
    smooth_union = vg.smooth_union
    smooth_intersection = vg.smooth_intersection
    smooth_difference = vg.smooth_difference
    compute_voronoi_adjacency = vg.compute_voronoi_adjacency
    # Ensure integer-indexed neighbors: convert NumPy array of points to list of tuples
    if isinstance(points, np.ndarray):
        points = [tuple(pt) for pt in points]
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
        def _assign_neighbors_from_adjacency():
            adjacency_raw = compute_voronoi_adjacency(
                points, bbox_min, bbox_max, resolution
            )
            if isinstance(adjacency_raw, list):
                adjacency_idx = {i: [] for i in range(len(points))}
                for i, j in adjacency_raw:
                    adjacency_idx[i].append(j)
                    adjacency_idx[j].append(i)
            else:
                adjacency_idx = adjacency_raw
            for idx, cell in enumerate(cells):
                cell["neighbors"] = [points[j] for j in adjacency_idx.get(idx, [])]

        if Delaunay is not None and len(points) >= 2:
            try:
                pts_np = np.array(points)
                tri = Delaunay(pts_np)
                adj_map = {tuple(seed): set() for seed in points}
                for simplex in tri.simplices:
                    for i in range(len(simplex)):
                        for j in range(i + 1, len(simplex)):
                            si = tuple(points[simplex[i]])
                            sj = tuple(points[simplex[j]])
                            adj_map[si].add(sj)
                            adj_map[sj].add(si)
                for cell in cells:
                    cell["neighbors"] = list(adj_map[cell["site"]])
            except Exception:
                _assign_neighbors_from_adjacency()
        else:
            _assign_neighbors_from_adjacency()
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

    # Attach neighbor lists by index
    adjacency_raw = compute_voronoi_adjacency(points, bbox_min, bbox_max, resolution)
    if isinstance(adjacency_raw, list):
        adjacency = {i: [] for i in range(len(points))}
        for i, j in adjacency_raw:
            adjacency[i].append(j)
            adjacency[j].append(i)
    else:
        adjacency = adjacency_raw
    for idx, cell in enumerate(cells):
        cell["neighbors"] = adjacency.get(idx, [])

    return cells

def construct_surface_voronoi_cells(
    seed_points: List[Tuple[float, float, float]],
    body_sdf: Callable[[np.ndarray], float],
    wall_thickness_mm: float = 1.0,
    thickness_field: Callable[[Tuple[float, float, float]], float] = None,
    bbox_min: Optional[Tuple[float, float, float]] = None,
    bbox_max: Optional[Tuple[float, float, float]] = None,
    csg_ops: Optional[List[Dict[str, Any]]] = None,
    blend_curve: Optional[Callable[[float], float]] = None,
    shell_offset: float = 0.0,
    adaptive_grid: Optional[OctreeNode] = None,
    resolution: Tuple[int, int, int] = (64, 64, 64)
) -> list:
    vg = _voronoi_helpers()
    _call_sdf = vg._call_sdf
    smooth_union = vg.smooth_union
    smooth_intersection = vg.smooth_intersection
    smooth_difference = vg.smooth_difference
    compute_voronoi_adjacency = vg.compute_voronoi_adjacency
    # Support shorthand signature: shift args if body_sdf is array
    if not callable(body_sdf):
        # Called as (seed_points, bbox_min, bbox_max)
        seed_points, bbox_min, bbox_max = seed_points, body_sdf, wall_thickness_mm
        body_sdf = lambda p: 0.0
        wall_thickness_mm = 1.0
    # Ensure seed_points is a list of tuples
    if isinstance(seed_points, np.ndarray):
        seed_points = [tuple(pt) for pt in seed_points]
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
        adjacency_raw = compute_voronoi_adjacency(
            seed_points, bbox_min, bbox_max, resolution
        )
        if isinstance(adjacency_raw, list):
            adjacency = {i: [] for i in range(len(seed_points))}
            for i, j in adjacency_raw:
                adjacency[i].append(j)
                adjacency[j].append(i)
        else:
            adjacency = adjacency_raw
        for idx, cell in enumerate(cells):
            cell["neighbors"] = adjacency.get(idx, [])
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
    adjacency_raw = compute_voronoi_adjacency(
        seed_points, bbox_min, bbox_max, resolution
    )
    if isinstance(adjacency_raw, list):
        adjacency = {i: [] for i in range(len(seed_points))}
        for i, j in adjacency_raw:
            adjacency[i].append(j)
            adjacency[j].append(i)
    else:
        adjacency = adjacency_raw
    for idx, cell in enumerate(cells):
        cell["neighbors"] = adjacency.get(idx, [])
    return cells

