import numpy as np
import logging

from typing import Any, Dict, Tuple, List, Optional, Union
import json
from pathlib import Path

from .sampler import compute_medial_axis, trace_hexagon
from .regularizer import hexagon_metrics

logger = logging.getLogger(__name__)

def compute_uniform_cells(
    seeds: np.ndarray,
    imds_mesh: Any,
    plane_normal: np.ndarray,
    max_distance: Optional[float] = None,
    vertex_tolerance: float = 1e-5,

    mean_edge_limit: Optional[float] = None,
    area_limit: Optional[float] = None,
    return_status: bool = False,
    resample_points: int = 60,
    resample_min_distance: float = 0.0,
    mean_edge_factor: Optional[float] = 2.0,
    std_edge_factor: Optional[float] = 2.0,
) -> Union[
    Dict[int, np.ndarray],
    Tuple[Dict[int, np.ndarray], int, List[Dict[str, Any]]],
]:

    

    """
    Compute near-uniform hexagonal Voronoi cells for each seed point.

    After each cell is traced its raw edge metrics are compared against
    thresholds derived from the running global averages. Cells with mean or
    standard deviation of edge lengths exceeding ``mean_edge_factor`` or
    ``std_edge_factor`` times their respective global means are resampled once
    via ``_resample``.  Persistent outliers are logged and omitted from the
    results.

    Args:
        seeds: (N,3) array of seed point locations.
        imds_mesh: mesh object with `.vertices` attribute for medial axis extraction.
        plane_normal: (3,) array defining slicing plane normal.
        max_distance: fallback distance for ray casting when no medial point is found.
        vertex_tolerance: tolerance used when reconciling shared vertices between
            adjacent cells. A warning is emitted if mismatches above this tolerance
            are detected.
        mean_edge_limit: optional threshold for the mean edge length of a cell.
            Cells exceeding this limit are resampled once. If the retry still
            fails the cell is omitted and the overall status set to ``1``.
        area_limit: optional threshold for the area of a cell. Cells exceeding
            this limit are resampled once. If the retry still fails the cell is
            omitted and the overall status set to ``1``.
        return_status: when ``True`` the function returns a tuple
            ``(cells, status, failed_indices)`` where ``status`` is ``0`` for
            success and ``1`` if any cell exceeded limits after resampling.
            ``failed_indices`` is a list of diagnostic dictionaries for each
            dropped seed containing ``index``, ``seed`` coordinates,
            ``neighbor_count`` and ``used_fallback``. When ``False`` only
            ``cells`` are returned.
        resample_points: number of random points to draw when ``trace_hexagon``
            requests additional neighbors.  These help avoid the
            axis-aligned bounding-box fallback that produces cubic cells.
        resample_min_distance: minimum allowed distance from the current seed when
            resampling.  Points closer than this are rejected, providing a more
            even angular distribution around the seed.
        mean_edge_factor: multiplier applied to the running global mean of
            ``mean_edge_length``. If the metric for a cell exceeds this factor
            times the global mean the cell is resampled once. Remaining
            outliers are dropped and logged as errors.
        std_edge_factor: multiplier applied to the running global mean of
            ``std_edge_length``. Cells exceeding this threshold are resampled
            once and omitted if still outliers.

    Returns:
        cells: dict mapping seed index to (6,3) array of hexagon vertices. If
            ``return_status`` is ``True`` then a tuple ``(cells, status,
            failed_indices)`` is returned instead.
    """
    # Extract medial axis points
    medial_points = compute_medial_axis(imds_mesh)

    # Derive an axis-aligned bounding box from the interface mesh to provide
    # additional sampling locations if the medial axis alone is insufficient.
    verts = getattr(imds_mesh, "vertices", None)
    if verts is None:
        raise ValueError("imds_mesh must have a 'vertices' attribute")
    bbox_min = np.min(verts, axis=0)
    bbox_max = np.max(verts, axis=0)
    rng = np.random.default_rng(0)


    dump_data: Dict[str, Any] = {
        "seeds": seeds.tolist(),
        "plane_normal": plane_normal.tolist(),
        "max_distance": max_distance,
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "medial_points": medial_points.tolist(),
        "cells": {},
    }


    cells: Dict[int, np.ndarray] = {}

    status = 0

    fallback_indices: List[int] = []
    failed_indices: List[Dict[str, Any]] = []

    global_mean_edge = 0.0
    global_std_edge = 0.0
    sample_count = 0

    def _check_outlier(metrics: Dict[str, float], idx: int, level: int = logging.WARNING) -> bool:
        exceeded_local = False
        if sample_count > 0:
            if (
                mean_edge_factor is not None
                and global_mean_edge > 0.0
                and metrics["mean_edge_length"] > mean_edge_factor * global_mean_edge
            ):
                logger.log(
                    level,
                    "Cell %d mean edge length %.3f exceeds %gx global mean %.3f",
                    idx,
                    metrics["mean_edge_length"],
                    mean_edge_factor,
                    global_mean_edge,
                )
                exceeded_local = True
            if (
                std_edge_factor is not None
                and global_std_edge > 1e-9
                and metrics["std_edge_length"] > std_edge_factor * global_std_edge
            ):
                logger.log(
                    level,
                    "Cell %d std edge length %.3f exceeds %gx global mean %.3f",
                    idx,
                    metrics["std_edge_length"],
                    std_edge_factor,
                    global_std_edge,
                )
                exceeded_local = True
        return exceeded_local

    for idx, seed in enumerate(seeds):
        def _resample() -> np.ndarray:
            """Return extra candidate points within the mesh bounds.

            Implements simple rejection sampling so that candidate points keep a
            minimum distance from the ``seed``.  Sampling continues until
            ``resample_points`` valid points are gathered or a maximum number of
            attempts is reached.
            """

            pts = np.empty((0, 3), dtype=float)
            attempts = 0
            # Avoid infinite loops if ``resample_min_distance`` is too large
            while pts.shape[0] < resample_points and attempts < 10:
                batch = rng.uniform(bbox_min, bbox_max, size=(resample_points, 3))
                if resample_min_distance > 0.0:
                    mask = np.linalg.norm(batch - seed, axis=1) >= resample_min_distance
                    batch = batch[mask]
                pts = np.vstack([pts, batch])
                attempts += 1
            return pts[:resample_points]

        # Provide the resampler so that trace_hexagon has enough neighbor
        # directions and avoids the axis-aligned bounding-box fallback that
        # produces cubic cells. Older ``trace_hexagon`` implementations may not

        # accept the ``neighbor_resampler`` or ``report_method`` arguments, so we
        # fall back to calling it with fewer parameters when necessary.

        def _check_limits(metrics: Dict[str, float], level: int = logging.WARNING) -> bool:
            exceeded_local = False
            if mean_edge_limit is not None and metrics["mean_edge_length"] > mean_edge_limit:
                logger.log(
                    level,
                    "Cell %d mean edge length %.3f exceeds limit %.3f",
                    idx,
                    metrics["mean_edge_length"],
                    mean_edge_limit,
                )
                exceeded_local = True
            if area_limit is not None and metrics["area"] > area_limit:
                logger.log(
                    level,
                    "Cell %d area %.3f exceeds limit %.3f",
                    idx,
                    metrics["area"],
                    area_limit,
                )
                exceeded_local = True
            return exceeded_local
        try:
            hex_pts, used_fallback, raw_hex = trace_hexagon(

                seed,
                medial_points,
                plane_normal,
                max_distance,

                report_method=True,
                neighbor_resampler=_resample,
                return_raw=True,
            )
        except TypeError:  # pragma: no cover - legacy signature
            try:
                hex_pts, used_fallback, raw_hex = trace_hexagon(
                    seed,
                    medial_points,
                    plane_normal,
                    max_distance,
                    report_method=True,
                    return_raw=True,
                )
            except TypeError:  # pragma: no cover - legacy signature
                try:
                    hex_pts, used_fallback = trace_hexagon(
                        seed,
                        medial_points,
                        plane_normal,
                        max_distance,
                        report_method=True,
                    )
                    raw_hex = hex_pts.copy()
                except TypeError:  # pragma: no cover - legacy signature
                    hex_pts = trace_hexagon(
                        seed,
                        medial_points,
                        plane_normal,
                        max_distance,
                    )
                    used_fallback = False
                    raw_hex = hex_pts.copy()


        metrics = hexagon_metrics(raw_hex)
        if _check_outlier(metrics, idx):
            extra_pts = _resample()
            neighbors = np.vstack([medial_points, extra_pts])
            neighbor_count = neighbors.shape[0]
            try:
                hex_pts, used_fallback, raw_hex = trace_hexagon(
                    seed,
                    neighbors,
                    plane_normal,
                    max_distance,
                    report_method=True,
                    neighbor_resampler=_resample,
                    return_raw=True,
                )
            except TypeError:  # pragma: no cover - legacy signature
                try:
                    hex_pts, used_fallback, raw_hex = trace_hexagon(
                        seed,
                        neighbors,
                        plane_normal,
                        max_distance,
                        report_method=True,
                        return_raw=True,
                    )
                except TypeError:  # pragma: no cover - legacy signature
                    hex_pts = trace_hexagon(
                        seed,
                        neighbors,
                        plane_normal,
                        max_distance,
                    )
                    used_fallback = False
                    raw_hex = hex_pts.copy()
            metrics = hexagon_metrics(raw_hex)
            if _check_outlier(metrics, idx, level=logging.ERROR):
                failed_indices.append(
                    {
                        "index": idx,
                        "seed": seed.tolist(),
                        "neighbor_count": int(neighbor_count),
                        "used_fallback": bool(used_fallback),
                    }
                )
                status = 1
                logger.error("Skipping cell %d due to edge metric outlier", idx)
                continue

        if _check_limits(metrics):
            extra_pts = _resample()
            neighbors = np.vstack([medial_points, extra_pts])
            neighbor_count = neighbors.shape[0]
            try:
                hex_pts, used_fallback, raw_hex = trace_hexagon(
                    seed,
                    neighbors,
                    plane_normal,
                    max_distance,
                    report_method=True,
                    neighbor_resampler=_resample,
                    return_raw=True,
                )
            except TypeError:  # pragma: no cover - legacy signature
                try:
                    hex_pts, used_fallback, raw_hex = trace_hexagon(
                        seed,
                        neighbors,
                        plane_normal,
                        max_distance,
                        report_method=True,
                        return_raw=True,
                    )
                except TypeError:  # pragma: no cover - legacy signature
                    hex_pts = trace_hexagon(
                        seed,
                        neighbors,
                        plane_normal,
                        max_distance,
                    )
                    used_fallback = False
                    raw_hex = hex_pts.copy()
            metrics = hexagon_metrics(raw_hex)
            if _check_outlier(metrics, idx, level=logging.ERROR):
                failed_indices.append(
                    {
                        "index": idx,
                        "seed": seed.tolist(),
                        "neighbor_count": int(neighbor_count),
                        "used_fallback": bool(used_fallback),
                    }
                )
                status = 1
                logger.error("Skipping cell %d due to edge metric outlier", idx)
                continue
            if _check_limits(metrics, level=logging.ERROR):
                failed_indices.append(
                    {
                        "index": idx,
                        "seed": seed.tolist(),
                        "neighbor_count": int(neighbor_count),
                        "used_fallback": bool(used_fallback),
                    }
                )
                status = 1
                logger.error("Skipping cell %d due to metric limits", idx)
                continue

        if used_fallback:
            fallback_indices.append(idx)
            logger.warning("Seed %d at %s used trace_hexagon fallback", idx, seed.tolist())
            extra_pts = _resample()
            neighbors = np.vstack([medial_points, extra_pts])
            neighbor_count = neighbors.shape[0]
            try:
                hex_pts, used_fallback, raw_hex = trace_hexagon(
                    seed,
                    neighbors,
                    plane_normal,
                    max_distance,
                    report_method=True,
                    neighbor_resampler=_resample,
                    return_raw=True,
                )
                metrics = hexagon_metrics(raw_hex)
                if used_fallback:
                    logger.error(
                        "Fallback used after resampling for seed %d at %s", idx, seed.tolist()
                    )
            except TypeError:  # pragma: no cover - legacy signature
                hex_pts = trace_hexagon(
                    seed,
                    neighbors,
                    plane_normal,
                    max_distance,
                )
                used_fallback = False
                raw_hex = hex_pts.copy()
                metrics = hexagon_metrics(raw_hex)
            if _check_outlier(metrics, idx, level=logging.ERROR):
                failed_indices.append(
                    {
                        "index": idx,
                        "seed": seed.tolist(),
                        "neighbor_count": int(neighbor_count),
                        "used_fallback": bool(used_fallback),
                    }
                )
                status = 1
                logger.error("Skipping cell %d due to edge metric outlier", idx)
                continue
            if _check_limits(metrics, level=logging.ERROR):
                failed_indices.append(
                    {
                        "index": idx,
                        "seed": seed.tolist(),
                        "neighbor_count": int(neighbor_count),
                        "used_fallback": bool(used_fallback),
                    }
                )
                status = 1
                logger.error("Skipping cell %d due to metric limits", idx)
                continue


        # Optionally log metrics (throttled to avoid flooding output)
        if logger.isEnabledFor(logging.DEBUG) and (idx < 10 or idx % 1000 == 0):
            logger.debug(
                "Uniform cell %d: mean_edge=%.3f, std_edge=%.3f, area=%.3f",
                idx,
                metrics['mean_edge_length'],
                metrics['std_edge_length'],
                metrics['area'],
            )

        cells[idx] = hex_pts
        dump_data["cells"][str(idx)] = {
            "seed": seed.tolist(),
            "vertices": hex_pts.tolist(),
            "raw_vertices": raw_hex.tolist(),
            "metrics": {
                "edge_lengths": metrics["edge_lengths"].tolist(),
                "mean_edge_length": float(metrics["mean_edge_length"]),
                "std_edge_length": float(metrics["std_edge_length"]),
                "area": float(metrics["area"]),
            },
            "used_fallback": bool(used_fallback),
        }
        global_mean_edge = (global_mean_edge * sample_count + metrics["mean_edge_length"]) / (sample_count + 1)
        global_std_edge = (global_std_edge * sample_count + metrics["std_edge_length"]) / (sample_count + 1)
        sample_count += 1
    if fallback_indices:
        logger.info(
            "trace_hexagon fallback used for %d seeds", len(fallback_indices)
        )
    dump_data["fallback_indices"] = fallback_indices
    dump_data["failed_indices"] = failed_indices

    # --------------------
    # Reconcile shared vertices
    # --------------------
    all_vertices: List[np.ndarray] = []
    cell_slices: Dict[int, slice] = {}
    for idx in sorted(cells.keys()):
        start = len(all_vertices)
        all_vertices.extend(cells[idx])
        cell_slices[idx] = slice(start, start + len(cells[idx]))

    if all_vertices:
        verts = np.vstack(all_vertices)
        n = len(verts)
        parent = np.arange(n)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        merge_tol = vertex_tolerance * 10.0
        try:
            from scipy.spatial import cKDTree  # type: ignore

            tree = cKDTree(verts)
            for i, j in tree.query_pairs(r=merge_tol):
                union(i, j)
        except Exception:  # pragma: no cover - fallback when scipy missing
            for i in range(n):
                for j in range(i + 1, n):
                    if np.linalg.norm(verts[i] - verts[j]) <= merge_tol:
                        union(i, j)

        groups: Dict[int, List[int]] = {}
        for idx in range(n):
            root = find(idx)
            groups.setdefault(root, []).append(idx)

        deltas: List[float] = []
        for inds in groups.values():
            if len(inds) > 1:
                pts = verts[inds]
                avg = pts.mean(axis=0)
                dev = np.linalg.norm(pts - avg, axis=1)
                deltas.extend(dev.tolist())
                verts[inds] = avg

        for idx, sl in cell_slices.items():
            cells[idx] = verts[sl]

        if deltas:
            mean_delta = float(np.mean(deltas))
            max_delta = float(np.max(deltas))
            logging.info(
                "Shared vertex adjustment: mean_delta=%0.6e, max_delta=%0.6e, adjusted=%d",
                mean_delta,
                max_delta,
                len(deltas),
            )
            if max_delta > vertex_tolerance:
                logging.warning(
                    "Max vertex mismatch %0.6e exceeds tolerance %0.6e",
                    max_delta,
                    vertex_tolerance,
                )
        else:
            logging.info("Shared vertex adjustment: no coincident vertices found")


    # Determine a suitable repository root for the log. When the package is
    # installed, ``__file__`` may reside in ``site-packages`` where writing is
    # disallowed. Walk up the path looking for a ``logs`` directory or a ``.git``
    # folder; if neither is found, fall back to the current working directory.
    root_candidate = Path(__file__).resolve()
    repo_root = None
    for parent in root_candidate.parents:
        if (parent / "logs").exists() or (parent / ".git").exists():
            repo_root = parent
            break
    if repo_root is None:
        repo_root = Path.cwd()

    # Explicitly log the resolved repository root for troubleshooting. The
    # previous call used a comma which resulted in a formatting error when the
    # logger was configured for DEBUG. Using "%s" ensures the path is rendered
    # correctly without raising a logging exception.
    logging.debug("REPO ROOT: %s", repo_root)
    
    dump_path = repo_root / "logs" / "UNIFORM_CELL_DUMP.json"
    try:
        dump_path.parent.mkdir(parents=True, exist_ok=True)

        with dump_path.open("w", encoding="utf-8") as f:
            json.dump(dump_data, f)
    except Exception as exc:  # pragma: no cover - best effort
        logging.warning("Failed to write uniform cell dump to %s: %s", dump_path, exc)


    if return_status:
        return cells, status, failed_indices
    return cells
