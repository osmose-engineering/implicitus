#![allow(deprecated)]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

const MAX_SEED_POINTS: usize = 7500;

fn get_f64(dict: &Bound<PyDict>, key: &str) -> Option<f64> {
    dict
        .get_item(key)
        .ok()
        .flatten()
        .and_then(|v| v.extract::<f64>().ok())
}

fn get_dict<'py>(dict: &Bound<'py, PyDict>, key: &str) -> Option<Bound<'py, PyDict>> {
    dict
        .get_item(key)
        .ok()
        .flatten()
        .and_then(|v| v.downcast_into::<PyDict>().ok())
}

fn point_inside(shape: &str, params: &Bound<PyDict>, x: f64, y: f64, z: f64) -> bool {
    match shape {
        "sphere" => {
            let r = get_f64(params, "radius").unwrap_or(0.0);
            x * x + y * y + z * z <= r * r
        }
        "cube" | "box" => {
            if let Some(size_dict) = get_dict(params, "size") {
                let sx = get_f64(&size_dict, "x").unwrap_or(0.0) / 2.0;
                let sy = get_f64(&size_dict, "y").unwrap_or(0.0) / 2.0;
                let sz = get_f64(&size_dict, "z").unwrap_or(0.0) / 2.0;
                x.abs() <= sx && y.abs() <= sy && z.abs() <= sz
            } else if let Some(size_val) = get_f64(params, "size") {
                let half = size_val / 2.0;
                x.abs() <= half && y.abs() <= half && z.abs() <= half
            } else {
                true
            }
        }
        "cylinder" => {
            let r = get_f64(params, "radius").unwrap_or(0.0);
            let h = get_f64(params, "height").unwrap_or(0.0);
            x * x + y * y <= r * r && (0.0..=h).contains(&z)
        }
        _ => true,
    }
}

fn bounding_box(shape: &str, params: &Bound<PyDict>) -> ((f64, f64, f64), (f64, f64, f64)) {
    match shape {
        "box" => {
            if let Some(size_dict) = get_dict(params, "size") {
                let x = get_f64(&size_dict, "x").unwrap_or(0.0) / 2.0;
                let y = get_f64(&size_dict, "y").unwrap_or(0.0) / 2.0;
                let z = get_f64(&size_dict, "z").unwrap_or(0.0) / 2.0;
                return ((-x, -y, -z), (x, y, z));
            }
        }
        "cube" => {
            if let Some(size_val) = get_f64(params, "size") {
                let half = size_val / 2.0;
                return ((-half, -half, -half), (half, half, half));
            }
        }
        "sphere" => {
            let r = get_f64(params, "radius").unwrap_or(0.0);
            return ((-r, -r, -r), (r, r, r));
        }
        "cylinder" => {
            let r = get_f64(params, "radius").unwrap_or(0.0);
            let h = get_f64(params, "height").unwrap_or(0.0);
            return ((-r, -r, 0.0), (r, r, h));
        }
        _ => {}
    }
    ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
}

#[pyfunction]
pub fn sample_inside(shape_spec: &Bound<PyDict>, spacing: f64) -> PyResult<Vec<(f64, f64, f64)>> {
    if shape_spec.len() != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "shape_spec must contain exactly one primitive",
        ));
    }
    let (shape_name, params_any) = shape_spec.iter().next().unwrap();
    let shape: String = shape_name.extract()?;
    let params = params_any.downcast::<PyDict>()?;
    let (bbox_min, bbox_max) = bounding_box(&shape, &params);

    let nx = ((bbox_max.0 - bbox_min.0) / spacing).floor() as usize + 1;
    let ny = ((bbox_max.1 - bbox_min.1) / spacing).floor() as usize + 1;
    let nz = ((bbox_max.2 - bbox_min.2) / spacing).floor() as usize + 1;

    let mut seeds: Vec<(f64, f64, f64)> = Vec::new();
    for ix in 0..nx {
        let x = bbox_min.0 + ix as f64 * spacing;
        for iy in 0..ny {
            let y = bbox_min.1 + iy as f64 * spacing;
            for iz in 0..nz {
                let z = bbox_min.2 + iz as f64 * spacing;
                if point_inside(&shape, &params, x, y, z) {
                    seeds.push((x, y, z));
                    if seeds.len() >= MAX_SEED_POINTS {
                        return Ok(seeds);
                    }
                }
            }
        }
    }
    Ok(seeds)
}
