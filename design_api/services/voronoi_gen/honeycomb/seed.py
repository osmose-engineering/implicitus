
# seed.py
"""Seed point sampling module."""
import numpy as np
import itertools
import math

def sample_seed_points(num_points, bbox, spacing, uniform=True, **kwargs):
    """
    Sample seed points within a bounding box.
    :param num_points: target number of points
    :param bbox: ((xmin, ymin, zmin), (xmax, ymax, zmax))
    :param spacing: minimum spacing between points
    :param uniform: if True, use uniform grid sampling; otherwise Poisson disk
    :return: np.ndarray of shape (N,3)
    """
    xmin, ymin, zmin = bbox[0]
    xmax, ymax, zmax = bbox[1]

    if uniform:
        # jittered grid sampling
        nx = int(math.ceil((xmax - xmin) / spacing))
        ny = int(math.ceil((ymax - ymin) / spacing))
        nz = int(math.ceil((zmax - zmin) / spacing))
        cells = list(itertools.product(range(nx), range(ny), range(nz)))
        np.random.shuffle(cells)
        points = []
        for i, j, k in cells[: min(num_points, len(cells))]:
            x0, x1 = xmin + i * spacing, xmin + (i + 1) * spacing
            y0, y1 = ymin + j * spacing, ymin + (j + 1) * spacing
            z0, z1 = zmin + k * spacing, zmin + (k + 1) * spacing
            points.append([
                np.random.uniform(x0, x1),
                np.random.uniform(y0, y1),
                np.random.uniform(z0, z1)
            ])
        return np.array(points)
    else:
        # simple Poisson-disk via rejection
        points = []
        max_attempts = num_points * 30
        attempts = 0
        while len(points) < num_points and attempts < max_attempts:
            p = np.random.uniform([xmin, ymin, zmin], [xmax, ymax, zmax])
            if all(np.linalg.norm(p - q) >= spacing for q in points):
                points.append(p)
            attempts += 1
        # if we didn't get enough, fill the rest randomly
        if len(points) < num_points:
            rem = num_points - len(points)
            extra = np.random.uniform([xmin, ymin, zmin], [xmax, ymax, zmax], size=(rem, 3))
            points.extend(extra)
        return np.array(points)
