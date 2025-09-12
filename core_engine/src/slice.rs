// slice.rs

//! Slice tracing using the Marching Squares algorithm.
//! Produces 2D contours (loops) at a given z-plane.

use crate::evaluate_sdf;
use crate::hex_lattice;
use crate::implicitus::Model;
use crate::uniform::hex::compute_uniform_cells;
use crate::voronoi_mesh;

use log::info;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A single contour loop: a series of (x, y) points.
pub type Contour = Vec<(f64, f64)>;

/// A single line segment represented by its start and end points.
pub type Segment = ((f64, f64), (f64, f64));

/// Maximum number of infill segments to log for debugging / hairball correlation.
const SEGMENT_LOG_LIMIT: usize = 20;

/// Output from a slice operation containing outer contours and infill segments.
pub struct SliceResult {
    pub contours: Vec<Contour>,
    pub segments: Vec<Segment>,
}

/// Slice parameters: bounding box and resolution.
pub struct SliceConfig {
    pub z: f64,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub nx: usize,
    pub ny: usize,
    pub seed_points: Vec<(f64, f64, f64)>,
    pub infill_pattern: Option<String>,
    pub wall_thickness: f64,
    pub mode: Option<String>,
    pub bbox_min: Option<(f64, f64, f64)>,
    pub bbox_max: Option<(f64, f64, f64)>,
}

/// Marching squares edge table mapping case index to edge pairs.
/// Edges are ordered: 0 - bottom, 1 - right, 2 - top, 3 - left.
const MARCHING_SQUARES_EDGES: [&[(usize, usize)]; 16] = [
    &[],
    &[(3, 0)],
    &[(0, 1)],
    &[(3, 1)],
    &[(1, 2)],
    &[(3, 0), (1, 2)],
    &[(0, 2)],
    &[(2, 3)],
    &[(2, 3)],
    &[(0, 2)],
    &[(0, 1), (2, 3)],
    &[(1, 2)],
    &[(3, 1)],
    &[(0, 1)],
    &[(3, 0)],
    &[],
];

/// Mapping from edge index to its two corner indices.
const EDGE_VERTEX_MAP: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];

/// Linear interpolation between two points given their SDF values.
fn interp(p0: (f64, f64), p1: (f64, f64), v0: f64, v1: f64) -> (f64, f64) {
    let t = v0 / (v0 - v1);
    (p0.0 + t * (p1.0 - p0.0), p0.1 + t * (p1.1 - p0.1))
}

/// Compute the corner point coordinates given cell indices, edge index, and grid spacing.
fn corner_point(
    i: usize,
    j: usize,
    edge: usize,
    dx: f64,
    dy: f64,
    config: &SliceConfig,
) -> (f64, f64) {
    let x0 = config.x_min + i as f64 * dx;
    let y0 = config.y_min + j as f64 * dy;
    match edge {
        0 => (x0, y0),
        1 => (x0 + dx, y0),
        2 => (x0 + dx, y0 + dy),
        3 => (x0, y0 + dy),
        _ => (x0, y0),
    }
}

/// Retrieve the SDF value for a given corner index.
fn corner_value(v00: f64, v10: f64, v11: f64, v01: f64, corner: usize) -> f64 {
    match corner {
        0 => v00,
        1 => v10,
        2 => v11,
        3 => v01,
        _ => v00,
    }
}

/// Perform a slice at height `config.z` and return contours and segments.
pub fn slice_model(model: &Model, config: &SliceConfig) -> SliceResult {
    let mut contours: Vec<Contour> = Vec::new();
    let mut segments: Vec<Segment> = Vec::new();

    // Optional infill generation for supported patterns
    if !config.seed_points.is_empty() {
        match config.infill_pattern.as_deref() {
            Some("voronoi") => {
                if config.mode.as_deref() == Some("uniform") {
                    let uniform_cells =
                        Python::with_gil(|py| -> PyResult<Vec<Vec<(f64, f64, f64)>>> {
                            let seed_rows: Vec<Vec<f64>> = config
                                .seed_points
                                .iter()
                                .map(|&(x, y, z)| vec![x, y, z])
                                .collect();
                            let seeds = PyArray2::from_vec2_bound(py, &seed_rows)?;
                            let plane = PyArray1::from_vec_bound(py, vec![0.0f64, 0.0, 1.0]);
                            let obj = compute_uniform_cells(
                                py,
                                seeds.readonly(),
                                py.None(),
                                plane.readonly(),
                                None,
                                1e-5,
                                false,
                                false,
                            )?;
                            let dict = obj.downcast_bound::<PyDict>(py)?;
                            let mut cells = Vec::new();
                            for idx in 0..seed_rows.len() {
                                if let Some(arr_obj) = dict.get_item(idx)? {
                                    let arr = arr_obj.downcast::<PyArray2<f64>>()?;
                                    let arr_ro = arr.readonly();
                                    let mut verts = Vec::new();
                                    for row in arr_ro.as_array().outer_iter() {
                                        verts.push((row[0], row[1], row[2]));
                                    }
                                    cells.push(verts);
                                }
                            }
                            Ok(cells)
                        });

                    if let Ok(cells) = uniform_cells {
                        for verts in cells {
                            let m = verts.len();
                            let mut points: Vec<(f64, f64)> = Vec::new();
                            for i in 0..m {
                                let p0 = verts[i];
                                let p1 = verts[(i + 1) % m];
                                let z0 = p0.2;
                                let z1 = p1.2;
                                if (z0 - config.z) * (z1 - config.z) <= 0.0
                                    && (z1 - z0).abs() > 1e-9
                                {
                                    let t = (config.z - z0) / (z1 - z0);
                                    if (0.0..=1.0).contains(&t) {
                                        let x = p0.0 + t * (p1.0 - p0.0);
                                        let y = p0.1 + t * (p1.1 - p0.1);
                                        points.push((x, y));
                                    }
                                }
                            }
                            if points.len() > 1 {
                                for i in 0..points.len() {
                                    let start = points[i];
                                    let end = points[(i + 1) % points.len()];
                                    segments.push((start, end));
                                }
                            }
                        }
                    } else {
                        let mesh = voronoi_mesh(&config.seed_points);
                        for (a, b) in mesh.edges {
                            let p0 = mesh.vertices[a];
                            let p1 = mesh.vertices[b];
                            let z0 = p0.2;
                            let z1 = p1.2;
                            if (z0 - config.z) * (z1 - config.z) <= 0.0 && (z1 - z0).abs() > 1e-9 {
                                let t = (config.z - z0) / (z1 - z0);
                                if (0.0..=1.0).contains(&t) {
                                    let x = p0.0 + t * (p1.0 - p0.0);
                                    let y = p0.1 + t * (p1.1 - p0.1);
                                    segments.push(((x, y), (x, y)));
                                }
                            }
                        }
                    }
                } else {
                    let mesh = voronoi_mesh(&config.seed_points);
                    for (a, b) in mesh.edges {
                        let p0 = mesh.vertices[a];
                        let p1 = mesh.vertices[b];
                        let z0 = p0.2;
                        let z1 = p1.2;
                        if (z0 - config.z) * (z1 - config.z) <= 0.0 && (z1 - z0).abs() > 1e-9 {
                            let t = (config.z - z0) / (z1 - z0);
                            if (0.0..=1.0).contains(&t) {
                                let x = p0.0 + t * (p1.0 - p0.0);
                                let y = p0.1 + t * (p1.1 - p0.1);
                                segments.push(((x, y), (x, y)));
                            }
                        }
                    }
                }
            }
            Some("hex") => {
                let mesh = hex_lattice(&config.seed_points);
                for (a, b) in mesh.edges {
                    let p0 = mesh.vertices[a];
                    let p1 = mesh.vertices[b];
                    let z0 = p0.2;
                    let z1 = p1.2;
                    if (z0 - config.z) * (z1 - config.z) <= 0.0 && (z1 - z0).abs() > 1e-9 {
                        let t = (config.z - z0) / (z1 - z0);
                        if (0.0..=1.0).contains(&t) {
                            let x = p0.0 + t * (p1.0 - p0.0);
                            let y = p0.1 + t * (p1.1 - p0.1);
                            segments.push(((x, y), (x, y)));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Log a limited set of infill segments for debugging and hairball visualization.
    if !segments.is_empty() {
        let log_count = segments.len().min(SEGMENT_LOG_LIMIT);
        for (i, &((x0, y0), (x1, y1))) in segments.iter().take(log_count).enumerate() {
            if (x0 - x1).abs() < 1e-9 && (y0 - y1).abs() < 1e-9 {
                info!("hairball segment {i}: point ({:.3}, {:.3})", x0, y0);
            } else {
                info!(
                    "hairball segment {i}: line ({:.3}, {:.3}) -> ({:.3}, {:.3})",
                    x0, y0, x1, y1
                );
            }
        }
        if segments.len() > SEGMENT_LOG_LIMIT {
            info!(
                "hairball segment log truncated to first {} of {} segments",
                SEGMENT_LOG_LIMIT,
                segments.len()
            );
        }
    }

    // Compute grid spacing
    let dx = (config.x_max - config.x_min) / (config.nx - 1) as f64;
    let dy = (config.y_max - config.y_min) / (config.ny - 1) as f64;

    // Sample SDF values on a 2D grid
    let mut grid = vec![vec![0.0; config.ny]; config.nx];
    for i in 0..config.nx {
        for j in 0..config.ny {
            let x = config.x_min + i as f64 * dx;
            let y = config.y_min + j as f64 * dy;
            grid[i][j] = evaluate_sdf(
                model,
                x,
                y,
                config.z,
                config.infill_pattern.as_deref(),
                &config.seed_points,
                config.wall_thickness,
                config.mode.as_deref(),
            );
        }
    }

    // Marching Squares contour extraction
    let mut contour_segments: Vec<Segment> = Vec::new();
    for i in 0..config.nx - 1 {
        for j in 0..config.ny - 1 {
            let v00 = grid[i][j];
            let v10 = grid[i + 1][j];
            let v11 = grid[i + 1][j + 1];
            let v01 = grid[i][j + 1];
            let mut case_index = 0;
            if v00 < 0.0 {
                case_index |= 1;
            }
            if v10 < 0.0 {
                case_index |= 2;
            }
            if v11 < 0.0 {
                case_index |= 4;
            }
            if v01 < 0.0 {
                case_index |= 8;
            }

            if MARCHING_SQUARES_EDGES[case_index].is_empty() {
                continue;
            }

            let corners = [
                corner_point(i, j, 0, dx, dy, config),
                corner_point(i, j, 1, dx, dy, config),
                corner_point(i, j, 2, dx, dy, config),
                corner_point(i, j, 3, dx, dy, config),
            ];
            let values = [v00, v10, v11, v01];

            // Compute segments for this cell
            for &(e0, e1) in MARCHING_SQUARES_EDGES[case_index].iter() {
                let (c0a, c0b) = EDGE_VERTEX_MAP[e0];
                let (c1a, c1b) = EDGE_VERTEX_MAP[e1];
                let p_start = interp(corners[c0a], corners[c0b], values[c0a], values[c0b]);
                let p_end = interp(corners[c1a], corners[c1b], values[c1a], values[c1b]);
                contour_segments.push((p_start, p_end));
            }
        }
    }

    // Assemble connected contour loops from the generated segments
    fn nearly_equal(a: (f64, f64), b: (f64, f64)) -> bool {
        (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9
    }

    let mut segments_left = contour_segments.clone();
    while let Some((start, end)) = segments_left.pop() {
        let mut loop_points = vec![start, end];
        let mut extended = true;
        while extended {
            extended = false;
            for idx in (0..segments_left.len()).rev() {
                let (s, e) = segments_left[idx];
                if nearly_equal(loop_points[0], e) {
                    loop_points.insert(0, s);
                    segments_left.swap_remove(idx);
                    extended = true;
                } else if nearly_equal(loop_points[0], s) {
                    loop_points.insert(0, e);
                    segments_left.swap_remove(idx);
                    extended = true;
                } else if nearly_equal(*loop_points.last().unwrap(), s) {
                    loop_points.push(e);
                    segments_left.swap_remove(idx);
                    extended = true;
                } else if nearly_equal(*loop_points.last().unwrap(), e) {
                    loop_points.push(s);
                    segments_left.swap_remove(idx);
                    extended = true;
                }
            }
        }
        if !loop_points.is_empty() && !nearly_equal(loop_points[0], *loop_points.last().unwrap()) {
            loop_points.push(loop_points[0]);
        }
        contours.push(loop_points);
    }

    SliceResult { contours, segments }
}
