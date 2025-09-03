"""Lightweight Python stub for the Rust-backed ``core_engine`` module.

This stub implements only the minimal surface area required by the tests.
It provides naive Python implementations of ``prune_adjacency_via_grid`` and
placeholders for other symbols expected by the real module.  The functions are
not optimized and should only be used in test environments where the compiled
extension is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


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

