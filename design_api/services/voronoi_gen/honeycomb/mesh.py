import numpy as np


def generate_honeycomb_cells(seeds: np.ndarray) -> list:
    """Generate placeholder honeycomb cell data.

    Parameters
    ----------
    seeds : np.ndarray
        Seed point coordinates of shape ``(N, 3)``.

    Returns
    -------
    list
        List containing a single bounding-box cell that encloses all seeds.
        This is a minimal stand-in to avoid crashes when honeycomb preview
        generation is requested.
    """
    seeds_np = np.asarray(seeds)
    if seeds_np.size == 0:
        return []
    mins = seeds_np.min(axis=0).tolist()
    maxs = seeds_np.max(axis=0).tolist()
    return [{"bbox_min": mins, "bbox_max": maxs}]
