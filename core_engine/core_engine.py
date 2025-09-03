"""Lightweight Python stub for the Rust-backed ``core_engine`` module.

This stub implements only the minimal surface area required by the tests.
It provides naive Python implementations of ``prune_adjacency_via_grid`` and
placeholders for other symbols expected by the real module.  The functions are
not optimized and should only be used in test environments where the compiled
extension is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Dict


def prune_adjacency_via_grid(
    points: Sequence[Tuple[float, float, float]], spacing: float
) -> List[Tuple[int, int]]:
    """Return index pairs for seeds within ``spacing`` distance.

    This naive implementation performs an ``O(n^2)`` scan which is sufficient
    for the small point sets used in tests.
    """

    adj: List[Tuple[int, int]] = []
    thresh = spacing * spacing
    for i, p in enumerate(points):
        for j in range(i + 1, len(points)):
            q = points[j]
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            dz = p[2] - q[2]
            if dx * dx + dy * dy + dz * dz <= thresh:
                adj.append((i, j))
    return adj


@dataclass
class OctreeNode:  # pragma: no cover - placeholder for API compatibility
    pass


def generate_adaptive_grid(*args, **kwargs):  # pragma: no cover - stub
    return []


def compute_uniform_cells(*args, **kwargs):  # pragma: no cover - stub
    return {}, []


def sample_inside(shape_spec: Dict[str, Dict[str, float]], spacing: float):  # pragma: no cover - stub
    """Return a coarse grid of points within the primitive described by ``shape_spec``.

    This simplified implementation only supports a handful of primitive types and
    does *not* guarantee points lie strictly inside the surface – it merely
    produces a regular grid within the primitive's axis-aligned bounding box. The
    real Rust implementation performs accurate sampling; this stub just provides
    enough structure for tests and local development when the compiled extension
    is unavailable.
    """

    # Determine an approximate bounding box for a few common primitives. All
    # primitives are assumed to be centred at the origin which is sufficient for
    # the way tests construct shape specs.
    if "cube" in shape_spec:
        size = shape_spec["cube"].get("size") or shape_spec["cube"].get("size_mm") or 1.0
        half = float(size) / 2.0
        bbox_min = (-half, -half, -half)
        bbox_max = (half, half, half)
    elif "ball" in shape_spec or "sphere" in shape_spec:
        params = shape_spec.get("ball") or shape_spec.get("sphere") or {}
        radius = float(params.get("radius") or params.get("radius_mm") or 1.0)
        bbox_min = (-radius, -radius, -radius)
        bbox_max = (radius, radius, radius)
    elif "cylinder" in shape_spec:
        params = shape_spec["cylinder"]
        radius = float(params.get("radius") or params.get("radius_mm") or 1.0)
        height = float(params.get("height") or params.get("height_mm") or 1.0)
        bbox_min = (-radius, -radius, -height / 2.0)
        bbox_max = (radius, radius, height / 2.0)
    else:
        # Fallback to a small unit cube around the origin
        bbox_min = (-0.5, -0.5, -0.5)
        bbox_max = (0.5, 0.5, 0.5)

    # Build a simple grid within the bounding box.  The grid step is the given
    # spacing, clamped to avoid generating an excessive number of points.
    if spacing <= 0:
        spacing = 1.0
    spacing = float(spacing)

    def frange(start: float, stop: float, step: float):
        cur = start
        # Include ``stop`` by using <= with a tiny tolerance.
        while cur <= stop + 1e-9:
            yield cur
            cur += step

    xs = list(frange(bbox_min[0], bbox_max[0], spacing))
    ys = list(frange(bbox_min[1], bbox_max[1], spacing))
    zs = list(frange(bbox_min[2], bbox_max[2], spacing))

    # Clamp to a reasonable maximum to prevent pathological test cases from
    # allocating huge lists.  This mirrors behaviour in ``ai_adapter`` which also
    # caps seed counts.
    MAX_SEEDS = 10_000
    points: List[Tuple[float, float, float]] = []
    for x in xs:
        for y in ys:
            for z in zs:
                points.append((x, y, z))
                if len(points) >= MAX_SEEDS:
                    return points
    return points

