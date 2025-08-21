from design_api.services.voronoi_gen.voronoi_gen import build_hex_lattice

def test_build_hex_lattice_returns_cells():
    bbox_min = (-1.0, -1.0, -1.0)
    bbox_max = (1.0, 1.0, 1.0)
    spacing = 0.5
    primitive = {"sphere": {"radius": 1.0}}

    pts, edges, cells = build_hex_lattice(
        bbox_min,
        bbox_max,
        spacing,
        primitive,
        return_cells=True,
        resolution=(8, 8, 8),
    )

    # Expect some Voronoi vertices and connecting edges
    assert pts and edges
    # Returned cells should contain SDF grids describing each seed cell
    assert cells and all("sdf" in cell for cell in cells)
