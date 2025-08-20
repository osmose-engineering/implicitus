import numpy as np
import logging
import random
import math
from typing import Union, Any
from typing import Tuple, List, Optional
from typing import Dict, Callable
import itertools


def _hex_lattice(bbox_min: Tuple[float, float, float],
                 bbox_max: Tuple[float, float, float],
                 cell_size: float,
                 slice_thickness: float) -> np.ndarray:
    """Generate a 3D hexagonal seed grid within a bounding box."""
    xmin, ymin, zmin = bbox_min
    xmax, ymax, zmax = bbox_max

    z_range = zmax - zmin
    n_layers = int(math.ceil(z_range / slice_thickness))

    vert_spacing = cell_size * math.sqrt(3) / 2.0

    points_xy = []
    n_rows = int(math.ceil((ymax - ymin) / vert_spacing))
    for row in range(n_rows + 1):
        y = ymin + row * vert_spacing
        if y > ymax:
            break
        x_start = xmin + (cell_size / 2.0 if row % 2 else 0.0)
        x = x_start
        while x <= xmax:
            points_xy.append((x, y))
            x += cell_size

    seeds = []
    for layer in range(n_layers + 1):
        z = zmin + layer * slice_thickness
        if z > zmax:
            break
        for x, y in points_xy:
            seeds.append([x, y, z])

    return np.array(seeds)


# --- SDF helper for surface sampling
def _call_sdf(sdf_func, pt):
    """
    Wrapper to invoke an SDF function, accepting numpy arrays or tuples.
    """
    try:
        return sdf_func(pt)
    except TypeError:
        return sdf_func(tuple(pt))

def sample_seed_points(
    num_points: int,
    bbox_min: Tuple[float, float, float],
    bbox_max: Tuple[float, float, float],
    *,
    density_field: Optional[Callable[[Tuple[float, float, float]], float]] = None,
    min_dist: Optional[float] = None,
    max_trials: int = 10000,
    pattern: str = "poisson"
) -> List[Tuple[float, float, float]]:
    """
    Sample seed points within the axis-aligned bounding box.

    When ``pattern`` is ``"poisson"`` (the default) Bridson's Poisson-disk
    sampling is used.  When ``pattern`` is ``"hex"`` a deterministic hexagonal
    lattice is generated via an internal seed generator.

    Args:
        num_points (int): Number of points to sample.
        bbox_min (tuple): Minimum (x, y, z) of bounding box.
        bbox_max (tuple): Maximum (x, y, z) of bounding box.
        min_dist (float, optional): Minimum distance between points. If not provided, spacing is chosen to target num_points by volume.
        max_trials (int): Maximum number of points to attempt to generate.
        pattern (str): ``"poisson"`` or ``"hex"``.
    """
    logging.debug(
        f"[sample_seed_points] called with num_points={num_points}, bbox_min={bbox_min}, bbox_max={bbox_max}, min_dist={min_dist}, density_field={'yes' if density_field is not None else 'no'}, pattern={pattern}"
    )
    # Compute domain volume and minimal distance r
    xmin, ymin, zmin = bbox_min
    xmax, ymax, zmax = bbox_max
    volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
    if num_points <= 0 or volume <= 0:
        return []

    if pattern == "hex":
        if min_dist is not None:
            r = min_dist
        else:
            r = (volume / num_points) ** (1 / 3)
        seeds = _hex_lattice(bbox_min, bbox_max, cell_size=r, slice_thickness=r)
        logging.debug(
            f"[sample_seed_points] returning {len(seeds)} hex lattice points with spacing {r:.3f}"
        )
        return [tuple(pt) for pt in seeds.tolist()]

    if pattern != "poisson":
        raise ValueError(f"Unsupported pattern '{pattern}'")

    if min_dist is not None:
        r = min_dist
    else:
        r = (volume / num_points) ** (1/3)
    logging.debug(f"[sample_seed_points] volume={volume:.3f}, computed spacing r={r:.3f}, max_trials={max_trials}")
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
                if density_field is not None:
                    prob = density_field(pt)
                    if random.random() > prob:
                        continue
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
        # Fill remainder with uniform random points, respecting density_field
        n_extra = num_points - len(points)
        for _ in range(n_extra):
            while True:
                pt = tuple(random.uniform(bbox_min[i], bbox_max[i]) for i in range(3))
                if density_field is None or random.random() <= density_field(pt):
                    points.append(pt)
                    break
    logging.debug(f"[sample_seed_points] returning {len(points)} seed points (requested {num_points})")
    return points

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

