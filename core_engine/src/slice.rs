// slice.rs

//! Slice tracing using the Marching Squares algorithm.
//! Produces 2D contours (loops) at a given z-plane.

use crate::evaluate_sdf;
use crate::implicitus::Model;
use crate::voronoi_mesh;

/// A single contour loop: a series of (x, y) points.
pub type Contour = Vec<(f64, f64)>;

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
}

/// Placeholder marching squares edge table: for each case index, list of edge pairs.
const MARCHING_SQUARES_EDGES: [&[(usize, usize)]; 16] = [
    &[], /* cases 0 and 15 produce no edges */
    /* 1 */ &[(0, 3)],
    /* 2 */ &[(0, 1)],
    /* 3 */ &[(1, 3)],
    /* 4 */ &[(1, 2)],
    /* 5 */ &[(0, 1), (2, 3)],
    /* 6 */ &[(0, 2)],
    /* 7 */ &[(2, 3)],
    /* 8 */ &[(2, 3)],
    /* 9 */ &[(0, 2)],
    /* 10*/ &[(0, 3), (1, 2)],
    /* 11*/ &[(1, 2)],
    /* 12*/ &[(1, 3)],
    /* 13*/ &[(0, 1)],
    /* 14*/ &[(0, 3)],
    &[],
];

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

/// Perform a slice at height `config.z` and return contours.
pub fn slice_model(model: &Model, config: &SliceConfig) -> Vec<Contour> {
    let mut contours: Vec<Contour> = Vec::new();

    // Optional infill generation via Voronoi edges
    if config.infill_pattern.as_deref() == Some("voronoi") && !config.seed_points.is_empty() {
        let mesh = voronoi_mesh(&config.seed_points);
        for (a, b) in mesh.edges {
            let p0 = mesh.vertices[a];
            let p1 = mesh.vertices[b];
            let z0 = p0.2;
            let z1 = p1.2;
            if (z0 - config.z) * (z1 - config.z) <= 0.0 && (z1 - z0).abs() > 1e-9 {
                let t = (config.z - z0) / (z1 - z0);
                if t >= 0.0 && t <= 1.0 {
                    let x = p0.0 + t * (p1.0 - p0.0);
                    let y = p0.1 + t * (p1.1 - p0.1);
                    contours.push(vec![(x, y), (x, y)]);
                }
            }
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
            );
        }
    }

    // Marching Squares contour extraction sketch
    // Precompute line segments
    let mut segments: Vec<(f64, f64)> = Vec::new();
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
            // Lookup edges for this case (placeholder)
            for &(e0, e1) in MARCHING_SQUARES_EDGES[case_index].iter() {
                let p0 = corner_point(i, j, e0, dx, dy, config);
                let p1 = corner_point(i, j, e1, dx, dy, config);
                let v0 = corner_value(v00, v10, v11, v01, e0);
                let v1 = corner_value(v00, v10, v11, v01, e1);
                let p_start = interp(p0, p1, v0, v1);
                let p_end = interp(p1, p0, v1, v0);
                segments.push(p_start);
                segments.push(p_end);
            }
        }
    }
    // Each segment was pushed twice (start and end), so take only start points
    let contour: Vec<(f64, f64)> = segments.into_iter().step_by(2).collect();

    // Compute angle for sorting
    fn point_angle(p: &(f64, f64)) -> f64 {
        p.1.atan2(p.0)
    }

    // Sort points by angle around origin
    let mut sorted = contour.clone();
    sorted.sort_by(|a, b| point_angle(a).partial_cmp(&point_angle(b)).unwrap());

    // Close the contour loop
    if let Some(first) = sorted.first().cloned() {
        sorted.push(first);
    }

    contours.push(sorted);
    contours
}
