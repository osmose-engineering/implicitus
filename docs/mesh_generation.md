# Mesh Generation Roadmap

This document outlines how the `core_engine` will evolve to produce meshes for
both primitive solids and infill structures.

## Primitives

* Parse the `implicitus` model tree and evaluate SDFs for each primitive.
* Convert SDF surfaces into triangle meshes using marching cubes or similar
  algorithms.
* Return vertex and edge lists that other subsystems can consume directly.

## Infills

* Accept seed points and bounds describing the infill region.
* Sampling is controlled by two fields:
  * `mode` – either `"uniform"` or `"organic"`. `uniform` places seeds on a
    hexagonal lattice while `organic` uses blue‑noise sampling. The default is
    `"uniform"`.
  * `num_points` – the desired number of seeds. When omitted this defaults to
    `DEFAULT_VORONOI_SEEDS`.
* When only a seed count is provided, the lattice spacing is estimated from the
  bounding‑box volume so that fewer seeds produce a coarser pattern. Explicit
  `spacing` or `min_dist` values override this behavior.
* Construct Voronoi or lattice cells, yielding explicit vertex coordinates.
* Produce edge connectivity so downstream tools can build struts or surfaces.
* The meshing endpoint returns the `seed_points` used for sampling. Reusing
  these coordinates in later preview or slicing calls ensures both stages see
  the exact same lattice.

### Example: Sampling a Sphere

```python
from design_api.services.infill_service import generate_hex_lattice

spec = {
    "pattern": "voronoi",
    "primitive": {"type": "sphere", "radius": 20},
    "num_points": 200,
    "mode": "uniform",
}

mesh = generate_hex_lattice(spec)
seeds = mesh["seed_points"]

# Pass `seeds` back when slicing to guarantee preview/slicer parity
```

### Edge diagnostics

The front‑end’s `VoronoiCanvas` performs a lightweight sanity check on the
edges returned from the meshing pipeline. After filtering long edges it warns
when an edge’s endpoints share nearly the same **z** value—an indication that a
3D edge may have collapsed into a plane. In development this check can be made
strict by setting `VORONOI_ASSERT_Z=true`, causing tests to fail fast when such
degenerate edges appear.

## Unified Output

The long‑term goal is to have `core_engine` provide a single authoritative
geometry source.  Both the slicing API and the front‑end UI will consume the
meshes generated here rather than duplicating meshing logic elsewhere.

The same utilities described above will power future primitives and SDF-driven
meshes, allowing them to share seed sampling and lattice construction logic.
