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
* Construct Voronoi or lattice cells, yielding explicit vertex coordinates.
* Produce edge connectivity so downstream tools can build struts or surfaces.

## Unified Output

The long‑term goal is to have `core_engine` provide a single authoritative
geometry source.  Both the slicing API and the front‑end UI will consume the
meshes generated here rather than duplicating meshing logic elsewhere.
