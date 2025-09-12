
## Mesh Generation Fields

The `design/mesh` endpoint exposes several controls for seed sampling. Only the
following keys are interpreted; any additional fields are ignored:

- `pattern` – currently only `"voronoi"` is supported (with `"honeycomb"`
  mapped to `"voronoi"`).
- `mode` – either `"uniform"` or `"organic"`. Defaults to `"uniform"`.
- `num_points` – desired number of seed points. Defaults to
  `DEFAULT_VORONOI_SEEDS` when omitted.
- `seed_points` – optional explicit coordinates. When provided, these points
  are used verbatim and returned back to the caller so that previews can reuse
  them for slicer parity.
- `spacing` or `min_dist` – override the spacing inference. When both are
  omitted and only `num_points` is supplied, spacing is estimated from the
  bounding‑box volume so that fewer points produce a coarser lattice.

Internal flags used in earlier revisions (for example, `_is_voronoi`) are no
longer recognized.

## Voronoi Generation Parameters

`compute_uniform_cells` accepts several optional limits to guard against
pathological cells:

- `mean_edge_limit` – maximum allowed mean edge length for a traced cell.
- `area_limit` – maximum allowed polygon area.
- `raw_std_edge_limit` – caps the standard deviation of edge lengths. If the
  initial polygon exceeds this value the function resamples once before
  dropping the seed.

- `neighbor_variance_limit` – when the variance of distances from a seed to its
  medial neighbors exceeds this threshold the algorithm generates an additional
  set of medial points before resampling.

- `mean_edge_factor` – multiplier applied to the running global average of
  `mean_edge_length`. Cells exceeding this factor times the global mean are
  resampled once before being discarded.
- `std_edge_factor` – multiplier on the global average of `std_edge_length`.
  Like `mean_edge_factor`, cells whose raw edge standard deviation exceeds the
  scaled global mean are retried once and skipped if still out of bounds.


Any cell exceeding these thresholds after the retry is omitted and reported as
failed.
