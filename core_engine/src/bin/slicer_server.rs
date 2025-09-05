// core_engine/src/bin/slicer_server.rs

use core_engine::implicitus::Model;
use core_engine::slice::{slice_model, SliceConfig, SliceResult};
use log::{info, warn};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use warp::http::{Method, StatusCode};
use warp::Filter;

#[derive(Deserialize)]
pub struct SliceRequest {
    // TODO: replace Value with actual Model once JSON <-> Protobuf integration is set up
    #[serde(rename = "model")]
    pub _model: Value,
    pub layer: f64,
    pub x_min: Option<f64>,
    pub x_max: Option<f64>,
    pub y_min: Option<f64>,
    pub y_max: Option<f64>,
    pub nx: Option<usize>,
    pub ny: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DebugInfo {
    pub seed_count: usize,
    pub infill_pattern: Option<String>,
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
        .and(warp::body::json())
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

pub async fn handle_slice(req: SliceRequest) -> Result<impl warp::Reply, warp::Rejection> {
    // Pull out infill or lattice data to forward to the slice configuration
    let (seed_points, infill_pattern, wall_thickness) = parse_infill(&req._model);

    let model_id = req
        ._model
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    info!("Slice request for model ID: {}", model_id);
    info!(
        "parse_infill -> seed_count: {}, first_seeds: {:?}, pattern: {:?}, wall_thickness: {}",
        seed_points.len(),
        seed_points.iter().take(3).collect::<Vec<_>>(),
        infill_pattern,
        wall_thickness
    );

    if seed_points.is_empty() {
        warn!("No seed points found for model ID {}", model_id);
    }

    let debug = DebugInfo {
        seed_count: seed_points.len(),
        infill_pattern: infill_pattern.clone(),
    };

    let config = SliceConfig {
        z: req.layer,
        x_min: req.x_min.unwrap_or(-1.0),
        x_max: req.x_max.unwrap_or(1.0),
        y_min: req.y_min.unwrap_or(-1.0),
        y_max: req.y_max.unwrap_or(1.0),
        nx: req.nx.unwrap_or(50),
        ny: req.ny.unwrap_or(50),
        seed_points,
        infill_pattern,
        // Forward wall thickness so slice_model can pass it through to evaluate_sdf
        wall_thickness,
    };

    let result = match serde_json::from_value::<Model>(req._model) {
        Ok(model) => slice_model(&model, &config),
        Err(e) => {
            warn!(
                "Failed to deserialize model {}: {}. Returning empty slice.",
                model_id, e
            );
            SliceResult {
                contours: Vec::new(),
                segments: Vec::new(),
            }
        }
    };

    Ok(warp::reply::json(&SliceResponse {
        contours: result.contours,
        segments: result.segments,
        debug,
    }))
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

pub fn parse_infill(value: &Value) -> (Vec<(f64, f64, f64)>, Option<String>, f64) {
    // Recursively walk the JSON tree collecting infill/lattice blocks. Seed points from
    // all blocks are aggregated, while the first encountered pattern and wall thickness
    // take precedence.
    fn collect(
        v: &Value,
        seeds: &mut Vec<(f64, f64, f64)>,
        pattern: &mut Option<String>,
        wall: &mut Option<f64>,
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
                collect(val, seeds, pattern, wall);
            }
        } else if let Some(arr) = v.as_array() {
            for val in arr {
                collect(val, seeds, pattern, wall);
            }
        }
    }

    let mut seeds = Vec::new();
    let mut pattern: Option<String> = None;
    let mut wall: Option<f64> = None;
    collect(value, &mut seeds, &mut pattern, &mut wall);
    (seeds, pattern, wall.unwrap_or(0.0))
}
