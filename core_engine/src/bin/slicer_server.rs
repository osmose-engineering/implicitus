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

    let config = SliceConfig {
        z: req.layer,
        x_min: req.x_min.unwrap_or(-1.0),
        x_max: req.x_max.unwrap_or(1.0),
        y_min: req.y_min.unwrap_or(-1.0),
        y_max: req.y_max.unwrap_or(1.0),
        nx: req.nx.unwrap_or(50),
        ny: req.ny.unwrap_or(50),
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
        let seed_count = infill
            .get("seed_points")
            .and_then(|sp| sp.as_array().map(|a| a.len()))
            .unwrap_or(0);
        let infill_pattern = infill
            .get("pattern")
            .and_then(|p| p.as_str())
            .map(|s| s.to_string());
        DebugInfo {
            seed_count,
            infill_pattern,
        }
    } else {
        DebugInfo {
            seed_count: 0,
            infill_pattern: None,
        }
    }
}
