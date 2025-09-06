// core_engine/src/bin/slicer_server.rs

use bytes::Bytes;
use core_engine::implicitus::Model;
use core_engine::slice::{slice_model, SliceConfig};
use log::{info, warn};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use warp::http::{Method, StatusCode};
use warp::reject::Reject;
use warp::Filter;

#[derive(Deserialize, Serialize)]
pub struct SliceRequest {
    #[serde(rename = "model")]
    pub _model: Model,
    pub layer: f64,
    pub x_min: Option<f64>,
    pub x_max: Option<f64>,
    pub y_min: Option<f64>,
    pub y_max: Option<f64>,
    pub nx: Option<usize>,
    pub ny: Option<usize>,

    pub bbox_min: Option<(f64, f64, f64)>,
    pub bbox_max: Option<(f64, f64, f64)>,

    pub cell_vertices: Option<Vec<(f64, f64, f64)>>,
    pub edge_list: Option<Vec<(usize, usize)>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DebugInfo {
    pub seed_count: usize,
    pub infill_pattern: Option<String>,
    pub seed_points: Option<Vec<(f64, f64, f64)>>,
}

#[derive(Serialize, Deserialize)]
pub struct SliceResponse {
    pub contours: Vec<Vec<(f64, f64)>>,
    pub segments: Vec<((f64, f64), (f64, f64))>,
    pub debug: DebugInfo,
}

#[derive(Deserialize)]
struct VoronoiRequest {
    _seeds: Vec<(f64, f64, f64)>,
}

#[derive(Serialize, Clone)]
struct VoronoiResponse {
    status: &'static str,
    vertices: Vec<(f64, f64, f64)>,
    edges: Vec<(usize, usize)>,
}

type JobMap = Arc<Mutex<HashMap<String, Option<VoronoiResponse>>>>;

#[tokio::main]
async fn main() {
    pyo3::prepare_freethreaded_python();
    env_logger::init();
    // shared job state
    let jobs: JobMap = Arc::new(Mutex::new(HashMap::new()));
    let with_jobs = warp::any().map(move || jobs.clone());

    // POST /slice  with JSON body to perform slicing
    let slice_route = warp::post()
        .and(warp::path("slice"))
        .and(warp::body::bytes())
        .and_then(handle_slice);

    // POST /voronoi to generate a mesh from seed points
    let voronoi_route = warp::post()
        .and(warp::path("voronoi"))
        .and(warp::body::json())
        .and(with_jobs.clone())
        .and_then(handle_voronoi);

    // GET /voronoi/status/{job_id}
    let status_route = warp::get()
        .and(warp::path("voronoi"))
        .and(warp::path("status"))
        .and(warp::path::param())
        .and(with_jobs.clone())
        .and_then(handle_voronoi_status);

    let cors = warp::cors()
        .allow_any_origin()
        .allow_methods(vec![Method::POST, Method::GET])
        .allow_header("content-type");

    info!("Slicer server listening on 127.0.0.1:4000");
    let routes = slice_route.or(voronoi_route).or(status_route).with(cors);
    warp::serve(routes).run(([127, 0, 0, 1], 4000)).await;
}

#[derive(Debug)]
struct InvalidBody;
impl Reject for InvalidBody {}

pub async fn handle_slice(body: Bytes) -> Result<impl warp::Reply, warp::Rejection> {
    // Log a truncated preview of the body to avoid dumping large or sensitive data.
    let raw_len = body.len();
    let preview_len = raw_len.min(1024);
    let preview = String::from_utf8_lossy(&body[..preview_len]);
    if raw_len > preview_len {
        info!(
            "Slice request body ({} bytes, first {} shown): {}...",
            raw_len, preview_len, preview
        );
    } else {
        info!("Slice request body ({} bytes): {}", raw_len, preview);
    }

    let req: SliceRequest = serde_json::from_slice(&body).map_err(|e| {
        warn!("Failed to deserialize slice request: {}", e);
        warp::reject::custom(InvalidBody)
    })?;

    info!(
        "cell_vertices len: {:?}, edge_list len: {:?}",
        req.cell_vertices.as_ref().map(|v| v.len()),
        req.edge_list.as_ref().map(|e| e.len())
    );

    // Pull out infill or lattice data to forward to the slice configuration

    let (seed_points, infill_pattern, wall_thickness, mode, bbox_min, bbox_max) = parse_infill(
        &req._model,
        req.cell_vertices.as_deref(),
        req.edge_list.as_deref(),
    );

    let model_id = if req._model.id.is_empty() {
        "unknown".to_string()
    } else {
        req._model.id.clone()
    };

    info!("Slice request for model ID: {}", model_id);
    info!(
        "parse_infill -> seed_count: {}, first_seeds: {:?}, pattern: {:?}, wall_thickness: {}, mode: {:?}",
        seed_points.len(),
        seed_points.iter().take(3).collect::<Vec<_>>(),
        infill_pattern,
        wall_thickness,
        mode
    );

    if seed_points.is_empty() {
        warn!("No seed points found for model ID {}", model_id);
    }

    // Include seed points in the debug response by default. This can be disabled
    // by setting the `IMPLICITUS_DEBUG_SEEDS` environment variable to `0` or
    // `false`.
    let include_seed_points = std::env::var("IMPLICITUS_DEBUG_SEEDS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);

    let debug = DebugInfo {
        seed_count: seed_points.len(),
        infill_pattern: infill_pattern.clone(),
        seed_points: if include_seed_points {
            Some(seed_points.clone())
        } else {
            None
        },
    };

    let nx = req.nx.unwrap_or(50);
    let ny = req.ny.unwrap_or(50);
    if nx < 2 || ny < 2 {
        warn!("Invalid grid resolution: nx={} ny={}, expected >=2", nx, ny);
        return Err(warp::reject::custom(InvalidBody));
    }

    let config = SliceConfig {
        z: req.layer,
        x_min: req.x_min.unwrap_or(-1.0),
        x_max: req.x_max.unwrap_or(1.0),
        y_min: req.y_min.unwrap_or(-1.0),
        y_max: req.y_max.unwrap_or(1.0),
        nx,
        ny,
        seed_points,
        infill_pattern,
        // Forward wall thickness so slice_model can pass it through to evaluate_sdf
        wall_thickness,
        mode,
        bbox_min: req.bbox_min.or(bbox_min),
        bbox_max: req.bbox_max.or(bbox_max),
    };

    info!(
        "SliceConfig: bbox_min={:?} bbox_max={:?} mode={:?}",
        config.bbox_min, config.bbox_max, config.mode
    );
    if bbox_min.is_some() && config.bbox_min != bbox_min {
        warn!(
            "SliceConfig bbox_min {:?} differs from parsed {:?}",
            config.bbox_min, bbox_min
        );
    }
    if bbox_max.is_some() && config.bbox_max != bbox_max {
        warn!(
            "SliceConfig bbox_max {:?} differs from parsed {:?}",
            config.bbox_max, bbox_max
        );
    }

    let result = slice_model(&req._model, &config);

    Ok(warp::reply::with_status(
        warp::reply::json(&SliceResponse {
            contours: result.contours,
            segments: result.segments,
            debug,
        }),
        StatusCode::OK,
    ))
}

async fn handle_voronoi(
    _req: VoronoiRequest,
    _jobs: JobMap,
) -> Result<impl warp::Reply, warp::Rejection> {
    warn!("/voronoi route is deprecated; use /design/mesh instead");
    let resp = StatusResponse {
        status: "deprecated",
    };
    Ok(warp::reply::with_status(
        warp::reply::json(&resp),
        StatusCode::GONE,
    ))
}

async fn handle_voronoi_status(
    job_id: String,
    jobs: JobMap,
) -> Result<impl warp::Reply, warp::Rejection> {
    let map = jobs.lock().unwrap();
    match map.get(&job_id) {
        Some(Some(resp)) => Ok(warp::reply::with_status(
            warp::reply::json(resp),
            StatusCode::OK,
        )),
        Some(None) => Ok(warp::reply::with_status(
            warp::reply::json(&StatusResponse {
                status: "rendering",
            }),
            StatusCode::ACCEPTED,
        )),
        None => Ok(warp::reply::with_status(
            warp::reply::json(&StatusResponse {
                status: "not_found",
            }),
            StatusCode::NOT_FOUND,
        )),
    }
}

#[derive(Serialize)]
struct StatusResponse {
    status: &'static str,
}

pub fn parse_infill(
    model: &Model,
    cell_vertices: Option<&[(f64, f64, f64)]>,
    edge_list: Option<&[(usize, usize)]>,
) -> (
    Vec<(f64, f64, f64)>,
    Option<String>,
    f64,
    Option<String>,
    Option<(f64, f64, f64)>,
    Option<(f64, f64, f64)>,
) {
    // Recursively walk the JSON tree collecting infill/lattice blocks. Seed points from
    // all blocks are aggregated, while the first encountered pattern and wall thickness
    // take precedence. When precomputed ``cell_vertices`` and ``edge_list`` are
    // supplied, seeds are derived directly from those vertices to avoid lattice
    // recomputation.
    fn collect(
        v: &Value,
        seeds: &mut Vec<(f64, f64, f64)>,
        pattern: &mut Option<String>,
        wall: &mut Option<f64>,
        mode: &mut Option<String>,
        bbox_min: &mut Option<(f64, f64, f64)>,
        bbox_max: &mut Option<(f64, f64, f64)>,
    ) {
        if let Some(obj) = v.as_object() {
            if let Some(infill) = obj.get("infill").or_else(|| obj.get("lattice")) {
                if pattern.is_none() {
                    *pattern = infill
                        .get("pattern")
                        .and_then(|p| p.as_str())
                        .map(|s| s.to_string());
                }
                if wall.is_none() {
                    *wall = infill.get("wall_thickness").and_then(|w| w.as_f64());
                }
                if mode.is_none() {
                    *mode = infill
                        .get("mode")
                        .and_then(|m| m.as_str())
                        .map(|s| s.to_string());
                }
                if bbox_min.is_none() {
                    *bbox_min = infill
                        .get("bbox_min")
                        .and_then(|b| b.as_array())
                        .and_then(|arr| {
                            if arr.len() == 3 {
                                Some((
                                    arr[0].as_f64().unwrap_or(0.0),
                                    arr[1].as_f64().unwrap_or(0.0),
                                    arr[2].as_f64().unwrap_or(0.0),
                                ))
                            } else {
                                None
                            }
                        });
                }
                if bbox_max.is_none() {
                    *bbox_max = infill
                        .get("bbox_max")
                        .and_then(|b| b.as_array())
                        .and_then(|arr| {
                            if arr.len() == 3 {
                                Some((
                                    arr[0].as_f64().unwrap_or(0.0),
                                    arr[1].as_f64().unwrap_or(0.0),
                                    arr[2].as_f64().unwrap_or(0.0),
                                ))
                            } else {
                                None
                            }
                        });
                }
                if let Some(arr) = infill.get("seed_points").and_then(|sp| sp.as_array()) {
                    for pt in arr {
                        if let Some(coords) = pt.as_array() {
                            if coords.len() == 3 {
                                seeds.push((
                                    coords[0].as_f64().unwrap_or(0.0),
                                    coords[1].as_f64().unwrap_or(0.0),
                                    coords[2].as_f64().unwrap_or(0.0),
                                ));
                            }
                        }
                    }
                }
            }
            for val in obj.values() {
                collect(val, seeds, pattern, wall, mode, bbox_min, bbox_max);
            }
        } else if let Some(arr) = v.as_array() {
            for val in arr {
                collect(val, seeds, pattern, wall, mode, bbox_min, bbox_max);
            }
        }
    }

    let value = serde_json::to_value(model).unwrap_or_else(|_| Value::Null);

    let mut seeds = Vec::new();
    let mut pattern: Option<String> = None;
    let mut wall: Option<f64> = None;
    let mut mode: Option<String> = None;

    let mut bbox_min: Option<(f64, f64, f64)> = None;
    let mut bbox_max: Option<(f64, f64, f64)> = None;
    collect(
        &value,
        &mut seeds,
        &mut pattern,
        &mut wall,
        &mut mode,
        &mut bbox_min,
        &mut bbox_max,
    );

    // If a precomputed lattice was provided, derive seeds from the referenced
    // vertex list to avoid recomputing the lattice. Only vertices referenced by
    // ``edge_list`` are included and duplicates are removed.
    if let (Some(verts), Some(edges)) = (cell_vertices, edge_list) {
        let mut seen = HashSet::new();
        for (i, j) in edges {
            if seen.insert(*i) {
                if let Some(v) = verts.get(*i) {
                    seeds.push(*v);
                }
            }
            if seen.insert(*j) {
                if let Some(v) = verts.get(*j) {
                    seeds.push(*v);
                }
            }
        }
    }

    if let (Some(bmin), Some(bmax)) = (bbox_min, bbox_max) {
        let out_of_bounds = seeds.iter().any(|&(x, y, z)| {
            x < bmin.0 || x > bmax.0 || y < bmin.1 || y > bmax.1 || z < bmin.2 || z > bmax.2
        });
        if out_of_bounds {
            warn!(
                "parse_infill: seed points outside bbox_min={:?} bbox_max={:?}",
                bmin, bmax
            );
        }
    }

    info!(
        "parse_infill: bbox_min={:?} bbox_max={:?} mode={:?} first_seeds={:?}",
        bbox_min,
        bbox_max,
        mode,
        seeds.iter().take(3).collect::<Vec<_>>()
    );
    (
        seeds,
        pattern,
        wall.unwrap_or(0.0),
        mode,
        bbox_min,
        bbox_max,
    )
}
