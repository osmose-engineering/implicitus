
# graph.py
"""Seed adjacency graph module."""
import numpy as np

def build_adjacency_graph(voronoi):
    """
    Derive adjacency edges between seed cells from a Voronoi diagram.
    :param voronoi: a scipy.spatial.Voronoi instance
    :return: list of tuple(int, int)
    """
    edges = set()
    for ridge in voronoi.ridge_points:
        a, b = ridge
        edges.add(tuple(sorted((a, b))))
    return list(edges)
