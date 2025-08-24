
## Voronoi Generation Parameters

`compute_uniform_cells` accepts several optional limits to guard against
pathological cells:

- `mean_edge_limit` – maximum allowed mean edge length for a traced cell.
- `area_limit` – maximum allowed polygon area.
- `raw_std_edge_limit` – caps the standard deviation of edge lengths. If the
  initial polygon exceeds this value the function resamples once before
  dropping the seed.

Any cell exceeding these thresholds after the retry is omitted and reported as
failed.
