import numpy as np
import logging
import random
import math
from typing import Union, Any
from typing import Tuple, List, Optional
from typing import Dict, Callable
import itertools
try:  # SciPy is heavy; allow tests to run without it
    from scipy.spatial import Voronoi  # type: ignore
except Exception:  # pragma: no cover - fallback when scipy missing
    Voronoi = None


def compute_medial_axis(imds_mesh: Any, tol: float = 1e-8) -> np.ndarray:
    """Compute an approximation of the medial axis of the interface mesh.

    Parameters
    ----------
    imds_mesh: Any
        Object exposing a ``vertices`` attribute as an ``(N, 3)`` array.
    tol: float, optional
        Extra tolerance to expand the mesh bounds when filtering Voronoi
        vertices.  Larger values keep more vertices; a value of ``0`` clips
        strictly to the axis-aligned bounding box of the mesh.

    Returns
    -------
    np.ndarray
        Array of Voronoi vertices that fall within the mesh bounds.
    """

    vertices = getattr(imds_mesh, "vertices", None)
    if vertices is None:
        raise ValueError("imds_mesh must have a 'vertices' attribute")

    if Voronoi is not None:
        # Use SciPy's robust Voronoi implementation when available
        vor = Voronoi(vertices)
        bbox_min = vertices.min(axis=0) - tol
        bbox_max = vertices.max(axis=0) + tol
        inside = np.all((vor.vertices >= bbox_min) & (vor.vertices <= bbox_max), axis=1)
        return vor.vertices[inside]

    # Fallback: approximate Voronoi vertices via circumcenters of tetrahedra
    verts = np.asarray(vertices, dtype=float)
    n = len(verts)
    centers: List[np.ndarray] = []

    def _circumcenter(pts: np.ndarray) -> Optional[np.ndarray]:
        p1, p2, p3, p4 = pts
        A = 2 * np.array([p2 - p1, p3 - p1, p4 - p1])
        b = np.array([
            np.dot(p2, p2) - np.dot(p1, p1),
            np.dot(p3, p3) - np.dot(p1, p1),
            np.dot(p4, p4) - np.dot(p1, p1),
        ])
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

    if n <= 20:
        combos = itertools.combinations(range(n), 4)
    else:
        rng = np.random.default_rng(0)
        combos = (rng.choice(n, 4, replace=False) for _ in range(min(500, n * 5)))

    for idxs in combos:
        pts = verts[list(idxs)]
        c = _circumcenter(pts)
        if c is not None:
            centers.append(c)

    if not centers:
        return np.empty((0, 3))

    centers = np.array(centers)
    bbox_min = verts.min(axis=0) - tol
    bbox_max = verts.max(axis=0) + tol
    inside = np.all((centers >= bbox_min) & (centers <= bbox_max), axis=1)
    return centers[inside]


def trace_hexagon(
    seed_pt: np.ndarray,
    medial_points: np.ndarray,
    plane_normal: np.ndarray,
    max_distance: Optional[float] = None,
    report_method: bool = False,
    neighbor_resampler: Optional[Callable[[], np.ndarray]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, bool]]:
    """
    Trace approximate hexagonal cell vertices around ``seed_pt`` using
    perpendicular bisectors to its nearest neighbors in the slicing plane.

    Args:
        seed_pt: (3,) array, seed point location.
        medial_points: (M,3) array of medial axis or seed points.
        plane_normal: (3,) array normal to the slicing plane.
        max_distance: fallback ray length if no neighbor data is available.
        neighbor_resampler: optional callable returning additional medial points
            if the initial neighbor set is insufficient.  It is invoked before
            resorting to the bounding-box ray fallback.

    Returns:
        If ``report_method`` is ``False`` (default), returns an ``(6,3)`` array
        of hexagon vertex positions.
        If ``report_method`` is ``True``, returns a tuple ``(hex_pts, used_fallback)``
        where ``used_fallback`` is ``True`` when the ray casting fallback was
        used instead of the bisector method.
    """
    # Create orthonormal basis (u, v) spanning the plane
    arbitrary = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.cross(arbitrary, plane_normal), 0):
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(plane_normal, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(plane_normal, u)
    v /= np.linalg.norm(v)

    hex_pts: List[np.ndarray] = []

    # Pre-compute bounding box of the medial points for ray fallback
    bbox_min = medial_points.min(axis=0)
    bbox_max = medial_points.max(axis=0)

    def _ray_box_intersection(origin: np.ndarray, direction: np.ndarray) -> Optional[float]:
        """Return distance to intersection with bbox or None if no hit."""
        tmin, tmax = -np.inf, np.inf
        for i in range(3):
            if abs(direction[i]) < 1e-12:
                if origin[i] < bbox_min[i] or origin[i] > bbox_max[i]:
                    return None
                continue
            t1 = (bbox_min[i] - origin[i]) / direction[i]
            t2 = (bbox_max[i] - origin[i]) / direction[i]
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
        if tmax < 0:
            return None
        return tmin if tmin > 0 else tmax

    def _construct_from_vecs(vecs: np.ndarray) -> Optional[np.ndarray]:
        """Attempt to construct hexagon from neighbor vectors."""
        if vecs.shape[0] < 6:
            return None
        neighbor_2d_all = np.column_stack((vecs.dot(u), vecs.dot(v)))
        angles = np.mod(np.arctan2(neighbor_2d_all[:, 1], neighbor_2d_all[:, 0]), 2 * np.pi)

        sel_2d: List[np.ndarray] = []
        used = np.zeros(len(angles), dtype=bool)
        for k in range(6):
            target = 2 * np.pi * k / 6
            diffs = np.abs(np.angle(np.exp(1j * (angles - target))))
            diffs[used] = np.inf
            idx = int(np.argmin(diffs))
            used[idx] = True
            sel_2d.append(neighbor_2d_all[idx])

        neighbor_2d = np.vstack(sel_2d)
        ang = np.arctan2(neighbor_2d[:, 1], neighbor_2d[:, 0])
        order = np.argsort(ang)
        neighbor_2d = neighbor_2d[order]

        normals = neighbor_2d
        bs = np.sum(neighbor_2d ** 2, axis=1) / 2.0
        verts_2d: List[np.ndarray] = []
        for i in range(6):
            j = (i + 1) % 6
            N = np.vstack([normals[i], normals[j]])
            B = np.array([bs[i], bs[j]])
            try:
                x = np.linalg.solve(N, B)
            except np.linalg.LinAlgError:
                return None
            verts_2d.append(x)

        if len(verts_2d) != 6:
            return None
        pts = [seed_pt + x[0] * u + x[1] * v for x in verts_2d]
        return np.vstack(pts)

    # Initial attempt using provided medial points
    vecs = medial_points - seed_pt
    dists = np.linalg.norm(vecs, axis=1)
    vecs = vecs[dists > 1e-8]
    hex_pts_arr = _construct_from_vecs(vecs)

    # Retry using adjacency-derived neighbors if necessary
    if hex_pts_arr is None:
        try:  # pragma: no cover - import guarded for optional module
            from design_api.services.voronoi_gen.voronoi_gen import compute_voronoi_adjacency

            pts = np.vstack([seed_pt, medial_points])
            pairs = compute_voronoi_adjacency(pts.tolist())
            neigh_idx: List[int] = []
            for i, j in pairs:
                if i == 0:
                    neigh_idx.append(j)
                elif j == 0:
                    neigh_idx.append(i)
            if neigh_idx:
                vecs_adj = pts[neigh_idx] - seed_pt
                hex_pts_arr = _construct_from_vecs(vecs_adj)
        except Exception:  # pragma: no cover - any failure just skips
            pass

    # Allow caller to supply extra neighbors before falling back
    if hex_pts_arr is None and neighbor_resampler is not None:
        extra = neighbor_resampler()
        if extra is not None and len(extra) > 0:
            all_points = np.vstack([medial_points, extra])
            vecs = all_points - seed_pt
            dists = np.linalg.norm(vecs, axis=1)
            vecs = vecs[dists > 1e-8]
            hex_pts_arr = _construct_from_vecs(vecs)

    if hex_pts_arr is not None:
        hex_pts = hex_pts_arr
        hex_success = True
    else:
        logging.warning("trace_hexagon: using bounding-box fallback; consider resampling or expanding search radius")
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        for theta in angles:
            dir_vec = np.cos(theta) * u + np.sin(theta) * v
            length = _ray_box_intersection(seed_pt, dir_vec)
            if length is not None:
                hex_pts.append(seed_pt + dir_vec * length)
            elif max_distance is not None:
                hex_pts.append(seed_pt + dir_vec * max_distance)
            else:
                raise ValueError("No valid intersection for ray; resample seed point")
        hex_pts = np.vstack(hex_pts)
        hex_success = False

    used_fallback = not hex_success

    # Attempt to regularize edge lengths if available
    try:
        from design_api.services.voronoi_gen.uniform.regularizer import regularize_hexagon
        hex_pts = regularize_hexagon(hex_pts, plane_normal)
    except ImportError:
        pass

    if report_method:
        return hex_pts, used_fallback
    return hex_pts
