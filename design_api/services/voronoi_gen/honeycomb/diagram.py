# diagram.py
"""Voronoi diagram generation module."""
from scipy.spatial import Voronoi
import numpy as np


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram to finite
    regions clipped to a bounding radius.
    Returns regions (list of index lists) and vertices (ndarray).
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Function only supports 2D")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Map ridge vertices to ridges
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct each infinite region
    for p_idx, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # otherwise infinite region: reconstruct
        ridges = all_ridges.get(p_idx, [])
        new_region = [v for v in vertices if v >= 0]
        if not ridges:
            # no ridges for this point, treat region as finite
            new_regions.append(new_region)
            continue

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1 = v1  # finite
                v2 = v2  # infinite
            if v1 < 0 or v2 < 0:
                # Compute the missing endpoint at infinity
                v_finite = vor.vertices[v1 if v1 >= 0 else v2]
                tangent = vor.points[p2] - vor.points[p_idx]
                tangent /= np.linalg.norm(tangent)
                normal = np.array([-tangent[1], tangent[0]])
                far_point = v_finite + normal * radius
                new_vertices.append(far_point.tolist())
                new_region.append(len(new_vertices) - 1)
        new_regions.append(new_region)

    return new_regions, np.array(new_vertices)


def generate_voronoi_diagram(points, bbox, radius=None, **kwargs):
    """
    Build a clipped 2D Voronoi diagram (for honeycomb infill) from seed points.
    :param points: (N,2) array of seed coordinates
    :param bbox: ((xmin, ymin), (xmax, ymax))
    :param radius: optional float to bound infinite cells
    :return: (vor, segments)
      - vor: the raw SciPy Voronoi object
      - segments: list of np.ndarray shape (2,2) giving each clipped edge
    """
    pts = np.asarray(points)
    # If points are 3D (x,y,z), project to 2D for Voronoi
    if pts.ndim == 2 and pts.shape[1] == 3:
        # drop Z coordinate
        pts2d = pts[:, :2]
        # if bbox is also 3D, drop Z bounds
        if len(bbox) == 2 and len(bbox[0]) == 3:
            bbox2d = ((bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]))
        else:
            bbox2d = bbox
    else:
        pts2d = pts
        bbox2d = bbox

    # 1) Build raw voronoi
    vor = Voronoi(pts2d)

    # 2) Close infinite regions
    regions, verts = voronoi_finite_polygons_2d(vor, radius=radius)

    # 3) Clip vertices to bbox
    (xmin, ymin), (xmax, ymax) = bbox2d
    verts[:, 0] = np.clip(verts[:, 0], xmin, xmax)
    verts[:, 1] = np.clip(verts[:, 1], ymin, ymax)

    # 4) Build edge list
    segments = []
    for region in regions:
        poly = verts[region]
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            segments.append(np.vstack([poly[i], poly[j]]))

    return vor, segments
