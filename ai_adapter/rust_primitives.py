"""Python bindings for Rust primitive sampling functions."""
from design_api.services.voronoi_gen.organic.sampler import _load_core_engine

_core = _load_core_engine()
_sample_inside_rust = _core.sample_inside


def sample_inside(shape_spec, spacing):
    """Return seed points inside the given primitive at the specified spacing."""
    return _sample_inside_rust(shape_spec, spacing)
