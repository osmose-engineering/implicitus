# mesh.py
"""Mesh extrusion module."""
import numpy as np
from scipy.spatial import Voronoi, ConvexHull

def generate_honeycomb_cells(seeds, bbox_padding=1):
    """
    Given an (N,3) array of seed points, build a honeycomb of finite Voronoi cells.
    We add an “external” padding of points to avoid
    boundary‐infinite cells, then only return the internal ones.
    
    :param seeds:      (N,3) numpy array of seed coordinates
    :param bbox_padding: number of cells to pad beyond the bounding box of seeds
    :returns:          list of dicts, each with:
                       {
                         'verts': (M,3)-array of vertex positions,
                         'faces': (K,3)-array of triangle indices
                       }
    """
    # 1) compute axis‐aligned bounding box of seeds
    mins = seeds.min(axis=0)
    maxs = seeds.max(axis=0)
    size = maxs - mins

    # 2) build a small grid of “external” offsets
    #    e.g. offsets in [-padding..nx+padding) in each axis
    #    so that internal cells are closed
    nx, ny, nz = 2, 2, 2  # just one internal cube; adjust if you tile multiple
    internal = seeds.tolist()
    external = []
    for ix in range(-bbox_padding, nx + bbox_padding):
        for iy in range(-bbox_padding, ny + bbox_padding):
            for iz in range(-bbox_padding, nz + bbox_padding):
                offset = np.array([ix, iy, iz]) * size
                for p in seeds:
                    pt = (p + offset).tolist()
                    if ix<0 or ix>=nx or iy<0 or iy>=ny or iz<0 or iz>=nz:
                        external.append(pt)

    # 3) run full 3D Voronoi
    all_pts = np.vstack([internal, external])
    vor = Voronoi(all_pts)

    # 4) extract only the “internal” cells (first len(internal) regions)
    cells = []
    for i in range(len(internal)):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region:
            # unbounded cell → skip
            continue

        verts = vor.vertices[region]
        # build a watertight triangulation of the (possibly non-convex) cell
        # by convex-hulling its vertices
        hull = ConvexHull(verts)
        faces = hull.simplices  # (K,3) triangle indices into verts

        cells.append({
            'verts': verts,
            'faces': faces
        })

    return cells

