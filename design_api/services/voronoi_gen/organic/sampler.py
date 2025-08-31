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

import importlib.util
import importlib.machinery
import pathlib
import subprocess
import sys
import os



def _load_core_engine():
    """Import the Rust extension, building it on the fly if necessary."""
    # First, attempt to import if it's already installed in site-packages
    spec = importlib.util.find_spec("core_engine.core_engine")
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # If not installed, build the cdylib with cargo and load directly
    crate_dir = pathlib.Path(__file__).resolve().parents[4] / "core_engine"
    try:

        env = os.environ.copy()
        env.setdefault("PYO3_USE_ABI3_FORWARD_COMPATIBILITY", "1")
        env.setdefault("PYO3_PYTHON", sys.executable)
        subprocess.run(
            ["cargo", "build", "--lib", "--features", "extension-module"],
            cwd=crate_dir,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    except Exception as exc:  # pragma: no cover - build failure
        raise ImportError("core_engine.core_engine build failed") from exc

    if sys.platform.startswith("win"):
        lib_name = "core_engine.dll"
    elif sys.platform == "darwin":
        lib_name = "libcore_engine.dylib"
    else:
        lib_name = "libcore_engine.so"

    lib_path = crate_dir / "target" / "debug" / lib_name
    if not lib_path.exists():  # pragma: no cover - unexpected build output
        raise ImportError(f"built library {lib_path} not found")

    loader = importlib.machinery.ExtensionFileLoader("core_engine.core_engine", str(lib_path))
    spec = importlib.util.spec_from_loader("core_engine.core_engine", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)

    return module

_core = _load_core_engine()
_sample_seed_points_rust = _core.sample_seed_points


def sample_seed_points(
    num_points: int,
    bbox_min: Tuple[float, float, float],
    bbox_max: Tuple[float, float, float],
    *,
    density_field: Optional[Callable[[Tuple[float, float, float]], float]] = None,
    min_dist: Optional[float] = None,
    max_trials: int = 10000,
    pattern: str = "poisson",
) -> List[Tuple[float, float, float]]:
    """Sample seed points using Rust bindings."""
    logging.debug(
        f"[sample_seed_points] called with num_points={num_points}, bbox_min={bbox_min}, bbox_max={bbox_max}, min_dist={min_dist}, density_field={'yes' if density_field is not None else 'no'}, pattern={pattern}"
    )
    return _sample_seed_points_rust(
        num_points,
        tuple(bbox_min),
        tuple(bbox_max),
        density_field=density_field,
        min_dist=min_dist,
        max_trials=max_trials,
        pattern=pattern,
    )

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

