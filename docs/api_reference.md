
## Voronoi Generation Parameters

`compute_uniform_cells` accepts several optional limits to guard against
pathological cells:

- `mean_edge_limit` – maximum allowed mean edge length for a traced cell.
- `area_limit` – maximum allowed polygon area.
- `raw_std_edge_limit` – caps the standard deviation of edge lengths. If the
  initial polygon exceeds this value the function resamples once before
  dropping the seed.
- `mean_edge_factor` – multiplier applied to the running global average of
  `mean_edge_length`. Cells exceeding this factor times the global mean are
  resampled once before being discarded.
- `std_edge_factor` – multiplier on the global average of `std_edge_length`.
  Like `mean_edge_factor`, cells whose raw edge standard deviation exceeds the
  scaled global mean are retried once and skipped if still out of bounds.

Any cell exceeding these thresholds after the retry is omitted and reported as
failed.
