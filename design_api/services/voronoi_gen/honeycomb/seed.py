
# seed.py
"""Seed point sampling module."""
import numpy as np
import math

def sample_seed_points(bbox_min, bbox_max, cell_size, slice_thickness, uniform=True, **kwargs):
    """
    Generate a 3D hexagonal (honeycomb) seed grid within a bounding box.
    :param bbox_min: (xmin, ymin, zmin) tuple
    :param bbox_max: (xmax, ymax, zmax) tuple
    :param cell_size: horizontal spacing between cell centers
    :param slice_thickness: layer height (Z spacing) for extrusion
    :return: np.ndarray of shape (N, 3) of seed coordinates
    """
    xmin, ymin, zmin = bbox_min
    xmax, ymax, zmax = bbox_max

    # Compute number of Z-layers
    z_range = zmax - zmin
    n_layers = int(math.ceil(z_range / slice_thickness))

    # Compute vertical spacing for hex grid rows
    vert_spacing = cell_size * math.sqrt(3) / 2.0

    # Generate 2D hex grid in XY plane
    points_xy = []
    # number of rows needed (plus extra to cover top)
    n_rows = int(math.ceil((ymax - ymin) / vert_spacing))
    for row in range(n_rows + 1):
        y = ymin + row * vert_spacing
        if y > ymax:
            break
        # stagger every other row
        x_start = xmin + (cell_size / 2.0 if row % 2 else 0.0)
        x = x_start
        while x <= xmax:
            points_xy.append((x, y))
            x += cell_size

    # Extrude 2D grid through Z layers
    seeds = []
    for layer in range(n_layers + 1):
        z = zmin + layer * slice_thickness
        if z > zmax:
            break
        for x, y in points_xy:
            seeds.append([x, y, z])

    return np.array(seeds)
