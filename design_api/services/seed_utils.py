from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from constants import DEFAULT_VORONOI_SEEDS
from .voronoi_gen.voronoi_gen import derive_bbox_from_primitive

def resolve_seed_spec(
    primitive: Dict[str, Any],
    bbox_min: Optional[List[float]],
    bbox_max: Optional[List[float]],
    seed_points: Optional[List[List[float]]] = None,
    num_points: Optional[int] = None,
    spacing: Optional[float] = None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize seed generation parameters for lattice construction.

    Parameters
    ----------
    primitive:
        Primitive specification used to derive a bounding box when one is not
        explicitly supplied.
    bbox_min, bbox_max:
        Existing bounding-box extents, if any.
    seed_points:
        Explicit seed coordinates. When provided, ``num_points`` defaults to the
        number of points given.
    num_points:
        Desired number of seed points when ``seed_points`` is omitted. Defaults
        to :data:`DEFAULT_VORONOI_SEEDS` when both are ``None``.
    spacing:
        Desired spacing between seed points. When omitted but ``num_points`` and
        a valid bounding box are available, an approximate spacing is inferred
        from the bounding-box volume assuming a hexagonal close packing.
    mode:
        Sampling mode, either ``"organic"`` or ``"uniform"``. Defaults to
        ``"uniform"``.
    """

    # Derive bounding box from the primitive when needed
    if bbox_min is None or bbox_max is None:
        if primitive:
            derived_min, derived_max = derive_bbox_from_primitive(primitive)
            if bbox_min is None:
                bbox_min = list(derived_min)
            if bbox_max is None:
                bbox_max = list(derived_max)

    mode = mode or "uniform"

    if seed_points is not None:
        num_points = num_points or len(seed_points)
    elif num_points is None:
        num_points = DEFAULT_VORONOI_SEEDS

    if spacing is None and num_points is not None and bbox_min is not None and bbox_max is not None:
        bbox_min_arr = np.asarray(bbox_min, dtype=float)
        bbox_max_arr = np.asarray(bbox_max, dtype=float)
        vol = float(np.prod(bbox_max_arr - bbox_min_arr))
        if vol > 0 and num_points > 0:
            vol_per_seed = vol / float(num_points)
            spacing = float(
                2.0 * (vol_per_seed / (4.0 * np.sqrt(2.0))) ** (1.0 / 3.0)
            )

    spacing = spacing or 2.0

    return {
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "seed_points": seed_points,
        "num_points": num_points,
        "spacing": spacing,
        "mode": mode,
    }
