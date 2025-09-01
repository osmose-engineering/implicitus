from __future__ import annotations

from typing import Any, Dict, List
from types import SimpleNamespace
import numpy as np
import inspect


from .voronoi_gen.voronoi_gen import (
    compute_voronoi_adjacency,
    build_hex_lattice,
    primitive_to_imds_mesh,
)


def _edge_list_from_adjacency(adjacency: Any) -> List[List[int]]:
    """Normalize adjacency output into a list of [i, j] edges with i<j."""
    edges: List[List[int]] = []
    if isinstance(adjacency, dict):
        for i, nbrs in adjacency.items():
            for j in nbrs:
                if j > i:
                    edges.append([i, j])
    else:
        for i, j in adjacency:
            if j > i:
                edges.append([i, j])
            else:
                edges.append([j, i])
    return edges


def generate_voronoi(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Voronoi-based adjacency for an infill spec.

    Parameters
    ----------
    spec
        Dictionary containing at minimum ``seed_points`` and either ``spacing``
        or ``min_dist``. Optional ``bbox_min``/``bbox_max`` are forwarded in the
        return structure.
    """

    pts: List[List[float]] = spec.get("seed_points", [])
    spacing = spec.get("spacing") or spec.get("min_dist") or 2.0
    adjacency = compute_voronoi_adjacency(pts, spacing=spacing * 0.5)
    edge_list = _edge_list_from_adjacency(adjacency)

    return {
        "seed_points": pts,
        "vertices": pts,
        "edges": edge_list,
        "cells": spec.get("cells"),
        "bbox_min": spec.get("bbox_min"),
        "bbox_max": spec.get("bbox_max"),
    }


def generate_hex_lattice(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a hexagonal lattice for the given spec and return adjacency."""

    bbox_min = spec.get("bbox_min")
    bbox_max = spec.get("bbox_max")
    spacing = spec.get("spacing") or spec.get("min_dist") or 2.0
    primitive = spec.get("primitive", {})
    mode = spec.get("mode", "uniform")

    if "mode" not in spec:
        uniform_flag = spec.get("uniform")
        if isinstance(uniform_flag, str):
            uniform_flag = uniform_flag.lower() == "true"
        if uniform_flag is not None:
            mode = "uniform" if uniform_flag else "organic"

    use_voronoi_edges = spec.get("use_voronoi_edges", False)

    reserved_keys = {
        "pattern",
        "mode",
        "spacing",
        "min_dist",
        "primitive",
        "imds_mesh",
        "plane_normal",
        "max_distance",
        "bbox_min",
        "bbox_max",
        "seed_points",
        "use_voronoi_edges",
        "_is_voronoi",
        "uniform",
    }
    extra_kwargs = {k: v for k, v in spec.items() if k not in reserved_keys}

    # Forward only keyword arguments supported by the target cell builder to
    # avoid leaking unrelated fields (e.g. "wall_thickness") into
    # ``compute_uniform_cells`` or ``construct_voronoi_cells``.
    if extra_kwargs:
        if mode == "uniform":
            from .voronoi_gen.uniform import compute_uniform_cells  # type: ignore

            allowed = set(inspect.signature(compute_uniform_cells).parameters)
            allowed -= {"seeds", "imds_mesh", "plane_normal"}
        else:
            from .voronoi_gen.organic.construct import (  # type: ignore
                construct_voronoi_cells,
            )

            allowed = set(inspect.signature(construct_voronoi_cells).parameters)
            allowed -= {"points", "bbox_min", "bbox_max"}
        extra_kwargs = {k: v for k, v in extra_kwargs.items() if k in allowed}


    imds_mesh = spec.get("imds_mesh")
    if isinstance(imds_mesh, dict):
        verts = imds_mesh.get("vertices")
        if verts is not None:
            imds_mesh = SimpleNamespace(vertices=np.asarray(verts))
    if getattr(imds_mesh, "vertices", None) is None:
        imds_mesh = primitive_to_imds_mesh(primitive)

    plane_normal = spec.get("plane_normal") or [0.0, 0.0, 1.0]
    max_distance = spec.get("max_distance")

    lattice_kwargs = {
        "return_cells": True,
        "use_voronoi_edges": use_voronoi_edges,
        "mode": mode,
    }
    if mode == "uniform":
        lattice_kwargs.update(
            {
                "imds_mesh": imds_mesh,
                "plane_normal": np.asarray(plane_normal),
                "max_distance": max_distance,
            }
        )

    seed_pts, cell_vertices, edge_list, cells = build_hex_lattice(
        bbox_min,
        bbox_max,
        spacing,
        primitive,
        **lattice_kwargs,
        **extra_kwargs,
    )

    if use_voronoi_edges:
        return {
            "seed_points": seed_pts,
            "vertices": cell_vertices,
            "edges": edge_list,
            "cells": cells,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
        }

    adjacency = compute_voronoi_adjacency(seed_pts, spacing=spacing * 0.5)
    edge_list = _edge_list_from_adjacency(adjacency)

    return {
        "seed_points": seed_pts,
        "edges": edge_list,
        "cells": cells,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
    }
