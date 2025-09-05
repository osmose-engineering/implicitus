// core_engine/src/bin/slicer_server.rs

use core_engine::implicitus::Model;
use core_engine::slice::{slice_model, SliceConfig};
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

    println!("Slicer server listening on 127.0.0.1:4000");
    let routes = slice_route.or(voronoi_route).or(status_route).with(cors);
    warp::serve(routes).run(([127, 0, 0, 1], 4000)).await;
}

pub async fn handle_slice(req: SliceRequest) -> Result<impl warp::Reply, warp::Rejection> {
    // Extract debug info before consuming the model value
    let debug = extract_debug_info(&req._model);

    // Pull out infill data to forward to the slice configuration
    let (seed_points, infill_pattern) = parse_infill(&req._model);

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
    };

    let contours = match serde_json::from_value::<Model>(req._model) {
        Ok(model) => slice_model(&model, &config),
        Err(_) => Vec::new(),
    };

    Ok(warp::reply::json(&SliceResponse { contours, debug }))
}

async fn handle_voronoi(
    _req: VoronoiRequest,
    _jobs: JobMap,
) -> Result<impl warp::Reply, warp::Rejection> {
    println!("[slicer_server] /voronoi route is deprecated; use /design/mesh instead");
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

fn extract_debug_info(value: &Value) -> DebugInfo {
    let (seeds, infill_pattern) = parse_infill(value);
    DebugInfo {
        seed_count: seeds.len(),
        infill_pattern,
    }
}

fn parse_infill(value: &Value) -> (Vec<(f64, f64, f64)>, Option<String>) {
    fn find_infill(v: &Value) -> Option<&Value> {
        if let Some(obj) = v.as_object() {
            if let Some(infill) = obj.get("infill") {
                return Some(infill);
            }
            for val in obj.values() {
                if let Some(found) = find_infill(val) {
                    return Some(found);
                }
            }
        } else if let Some(arr) = v.as_array() {
            for val in arr {
                if let Some(found) = find_infill(val) {
                    return Some(found);
                }
            }
        }
        None
    }

    if let Some(infill) = find_infill(value) {
        let pattern = infill
            .get("pattern")
            .and_then(|p| p.as_str())
            .map(|s| s.to_string());
        let seed_points = infill
            .get("seed_points")
            .and_then(|sp| sp.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|pt| {
                        let coords = pt.as_array()?;
                        if coords.len() == 3 {
                            Some((
                                coords[0].as_f64().unwrap_or(0.0),
                                coords[1].as_f64().unwrap_or(0.0),
                                coords[2].as_f64().unwrap_or(0.0),
                            ))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(Vec::new);
        (seed_points, pattern)
    } else {
        (Vec::new(), None)
    }
}
