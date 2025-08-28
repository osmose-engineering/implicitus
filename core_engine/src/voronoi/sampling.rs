use pyo3::prelude::*;
use rand::Rng;
use std::collections::HashMap;

// -----------------------------------------------------------------------------
// Hex lattice generator
// -----------------------------------------------------------------------------
fn hex_lattice(
    bbox_min: (f64, f64, f64),
    bbox_max: (f64, f64, f64),
    cell_size: f64,
    slice_thickness: f64,
) -> Vec<(f64, f64, f64)> {
    let (xmin, ymin, zmin) = bbox_min;
    let (xmax, ymax, zmax) = bbox_max;
    let z_range = zmax - zmin;
    let n_layers = (z_range / slice_thickness).ceil() as i32;
    let vert_spacing = cell_size * (3.0f64).sqrt() / 2.0;
    let n_rows = ((ymax - ymin) / vert_spacing).ceil() as i32;

    let mut points_xy = Vec::new();
    for row in 0..=n_rows {
        let y = ymin + row as f64 * vert_spacing;
        if y > ymax { break; }
        let x_start = xmin + if row % 2 == 0 { 0.0 } else { cell_size / 2.0 };
        let mut x = x_start;
        while x <= xmax {
            points_xy.push((x, y));
            x += cell_size;
        }
    }

    let mut seeds = Vec::new();
    for layer in 0..=n_layers {
        let z = zmin + layer as f64 * slice_thickness;
        if z > zmax { break; }
        for &(x, y) in &points_xy {
            seeds.push((x, y, z));
        }
    }
    seeds
}

// -----------------------------------------------------------------------------
// Poisson-disk sampling (simplified Bridson)
// -----------------------------------------------------------------------------
#[pyfunction(signature = (
    num_points,
    bbox_min,
    bbox_max,
    density_field=None,
    min_dist=None,
    max_trials=None,
    pattern=None
))]
pub fn sample_seed_points(
    py: Python<'_>,
    num_points: usize,
    bbox_min: (f64, f64, f64),
    bbox_max: (f64, f64, f64),
    density_field: Option<PyObject>,
    min_dist: Option<f64>,
    max_trials: Option<usize>,
    pattern: Option<&str>,
) -> PyResult<Vec<(f64, f64, f64)>> {
    let pattern = pattern.unwrap_or("poisson");
    let (xmin, ymin, zmin) = bbox_min;
    let (xmax, ymax, zmax) = bbox_max;
    if num_points == 0 || xmax <= xmin || ymax <= ymin || zmax <= zmin {
        return Ok(Vec::new());
    }
    let volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin);
    let mut rng = rand::thread_rng();

    if pattern == "hex" {
        let r = min_dist.unwrap_or_else(|| (volume / num_points as f64).cbrt());
        let seeds = hex_lattice(bbox_min, bbox_max, r, r);
        return Ok(seeds);
    }

    let r = min_dist.unwrap_or(0.0);
    let max_trials = max_trials.unwrap_or(10_000);
    let mut points: Vec<(f64, f64, f64)> = Vec::new();
    let mut trials = 0usize;
    while points.len() < num_points && trials < max_trials {
        let x = rng.gen_range(xmin..xmax);
        let y = rng.gen_range(ymin..ymax);
        let z = rng.gen_range(zmin..zmax);
        let p = (x, y, z);
        if let Some(ref df) = density_field {
            let d: f64 = df.call1(py, (p,))?.extract(py)?;
            if d <= 0.0 || rng.gen::<f64>() > d {
                trials += 1;
                continue;
            }
        }
        if r > 0.0 {
            let mut ok = true;
            for &(px, py_, pz) in &points {
                let dx = px - x;
                let dy = py_ - y;
                let dz = pz - z;
                if dx * dx + dy * dy + dz * dz < r * r {
                    ok = false;
                    break;
                }
            }
            if !ok {
                trials += 1;
                continue;
            }
        }
        points.push(p);
        trials += 1;
    }
    while points.len() < num_points {
        let x = rng.gen_range(xmin..xmax);
        let y = rng.gen_range(ymin..ymax);
        let z = rng.gen_range(zmin..zmax);
        let p = (x, y, z);
        if let Some(ref df) = density_field {
            let d: f64 = df.call1(py, (p,))?.extract(py)?;
            if d <= 0.0 || rng.gen::<f64>() > d {
                continue;
            }
        }
        points.push(p);
    }
    Ok(points)
}

// -----------------------------------------------------------------------------
// Spatial hash grid for adjacency pruning
// -----------------------------------------------------------------------------
fn build_spatial_index(
    seeds: &[(f64, f64, f64)],
    spacing: f64,
) -> HashMap<(i64, i64, i64), Vec<usize>> {
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    let cell_size = (2.0 * spacing).max(1e-9);
    for (idx, &(x, y, z)) in seeds.iter().enumerate() {
        let i = (x / cell_size).floor() as i64;
        let j = (y / cell_size).floor() as i64;
        let k = (z / cell_size).floor() as i64;
        grid.entry((i, j, k)).or_insert_with(Vec::new).push(idx);
    }
    grid
}

#[pyfunction]
pub fn prune_adjacency_via_grid(
    seeds: Vec<(f64, f64, f64)>,
    spacing: f64,
) -> PyResult<Vec<(usize, usize)>> {
    let grid = build_spatial_index(&seeds, spacing);
    let cell_size = (2.0 * spacing).max(1e-9);
    let neighbor_offsets: Vec<(i64, i64, i64)> = (-1..=1)
        .flat_map(|di| {
            (-1..=1).flat_map(move |dj| (-1..=1).map(move |dk| (di, dj, dk)))
        })
        .collect();
    let max_dist2 = (2.0 * spacing) * (2.0 * spacing);
    let mut edges = Vec::new();
    for (i, &(x, y, z)) in seeds.iter().enumerate() {
        let ci = (x / cell_size).floor() as i64;
        let cj = (y / cell_size).floor() as i64;
        let ck = (z / cell_size).floor() as i64;
        for (di, dj, dk) in &neighbor_offsets {
            if let Some(indices) = grid.get(&(ci + di, cj + dj, ck + dk)) {
                for &j_idx in indices {
                    if j_idx <= i { continue; }
                    let (x2, y2, z2) = seeds[j_idx];
                    let dx = x - x2;
                    let dy = y - y2;
                    let dz = z - z2;
                    if dx * dx + dy * dy + dz * dz <= max_dist2 {
                        edges.push((i, j_idx));
                    }
                }
            }
        }
    }
    Ok(edges)
}
