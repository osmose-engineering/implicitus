
# offset.py
"""Wall offsetting module."""
import pyclipper

def offset_polygons(polygons, offset_distance):
    """
    Offset each polygon by the given distance.
    :param polygons: list of list of (x,y) tuples
    :param offset_distance: float
    :return: list of offset polygons
    """
    pco = pyclipper.PyclipperOffset()
    result = []
    for poly in polygons:
        pco.Clear()
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        result.append(pco.Execute(offset_distance))
    return result
