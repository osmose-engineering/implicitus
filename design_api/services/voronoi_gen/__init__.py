from . import organic
from . import uniform

# Expose core helpers lazily to avoid circular imports during package init.
__all__ = [
    "organic",
    "uniform",
    "compute_voronoi_adjacency",
    "estimate_normal",
    "estimate_hessian",
    "compute_fillet_radius",
    "smooth_union",
    "_call_sdf",
    "OctreeNode",
]

_LAZY_ATTRS = {
    "compute_voronoi_adjacency",
    "estimate_normal",
    "estimate_hessian",
    "compute_fillet_radius",
    "smooth_union",
    "_call_sdf",
    "OctreeNode",
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        from . import voronoi_gen as _vg

        return getattr(_vg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
