import importlib.util
import pathlib
import sys
import types

import pytest

# Load ``compute_voronoi_adjacency`` directly from the module to avoid
# importing the full ``design_api.services.voronoi_gen`` package, which may
# require optional compiled extensions during import.
# Ensure repository root is on ``sys.path`` so the ``core_engine`` stub can be
# located when importing the Voronoi generator module.
_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.append(str(_ROOT))

_VG_PATH = pathlib.Path(__file__).resolve().parents[1] / "voronoi_gen" / "voronoi_gen.py"


def _stub_prune(points, spacing):
    thresh = (2 * spacing) ** 2
    out = []
    for i, (x0, y0, z0) in enumerate(points):
        for j in range(i + 1, len(points)):
            x1, y1, z1 = points[j]
            dx, dy, dz = x0 - x1, y0 - y1, z0 - z1
            if dx * dx + dy * dy + dz * dz <= thresh:
                out.append((i, j))
    return out


# Execute the module with a stubbed core engine to avoid building the Rust
# extension. We substitute the line that would import the real extension with
# our lightweight implementation.
source = _VG_PATH.read_text().replace(
    "_core = _load_core_engine()",
    "_core = _stub_core_engine",
)
ns: dict[str, object] = {
    "_stub_core_engine": types.SimpleNamespace(
        prune_adjacency_via_grid=_stub_prune,
        OctreeNode=object,
        generate_adaptive_grid=lambda *args, **kwargs: [],
    )
}
exec(source, ns)
compute_voronoi_adjacency = ns["compute_voronoi_adjacency"]


def test_compute_voronoi_adjacency_respects_z_axis():
    """Seed pairs separated along z should still be connected."""
    pts = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
    ]
    edges = set(compute_voronoi_adjacency(pts, spacing=0.5))
    # Horizontal neighbors on each layer
    assert (0, 1) in edges
    assert (2, 3) in edges
    # Vertical neighbors between layers
    assert (0, 2) in edges
    assert (1, 3) in edges
    # Diagonal points are farther than the inferred spacing
    assert (0, 3) not in edges
    assert (1, 2) not in edges
