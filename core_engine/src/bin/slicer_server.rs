// core_engine/src/bin/slicer_server.rs

use core_engine::implicitus::Model;
use core_engine::slice::{slice_model, SliceConfig};
use core_engine::voronoi_mesh;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid;
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

#[derive(Serialize, Deserialize)]
pub struct SliceResponse {
    pub contours: Vec<Vec<(f64, f64)>>,
}

#[derive(Deserialize)]
struct VoronoiRequest {
    seeds: Vec<(f64, f64, f64)>,
}

#[derive(Serialize, Clone)]
struct VoronoiResponse {
    status: &'static str,
    vertices: Vec<(f64, f64, f64)>,
    edges: Vec<(usize, usize)>,
}

#[derive(Serialize)]
struct JobResponse {
    job_id: String,
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

#[derive(Debug)]
struct InvalidModel;
impl warp::reject::Reject for InvalidModel {}

pub async fn handle_slice(req: SliceRequest) -> Result<impl warp::Reply, warp::Rejection> {
    // Deserialize the incoming model description.
    let model: Model = serde_json::from_value(req._model)
        .map_err(|_| warp::reject::custom(InvalidModel))?;

    let config = SliceConfig {
        z: req.layer,
        x_min: req.x_min.unwrap_or(-1.0),
        x_max: req.x_max.unwrap_or(1.0),
        y_min: req.y_min.unwrap_or(-1.0),
        y_max: req.y_max.unwrap_or(1.0),
        nx: req.nx.unwrap_or(50),
        ny: req.ny.unwrap_or(50),
    };

    let contours = slice_model(&model, &config);
    Ok(warp::reply::json(&SliceResponse { contours }))
}

async fn handle_voronoi(
    req: VoronoiRequest,
    jobs: JobMap,
) -> Result<impl warp::Reply, warp::Rejection> {
    println!(
        "[slicer_server] /voronoi request: {} seeds",
        req.seeds.len()
    );
    let job_id = Uuid::new_v4().to_string();
    {
        let mut map = jobs.lock().unwrap();
        map.insert(job_id.clone(), None);
    }
    let seeds = req.seeds.clone();
    let jobs_clone = jobs.clone();
    let job_id_clone = job_id.clone();
    tokio::spawn(async move {
        let mesh = voronoi_mesh(&seeds);
        println!(
            "[slicer_server] /voronoi job {job_id_clone} response: {} vertices, {} edges",
            mesh.vertices.len(),
            mesh.edges.len()
        );
        let resp = VoronoiResponse {
            status: "complete",
            vertices: mesh.vertices,
            edges: mesh.edges,
        };
        let mut map = jobs_clone.lock().unwrap();
        map.insert(job_id_clone, Some(resp));
    });
    let resp = JobResponse { job_id };
    Ok(warp::reply::with_status(
        warp::reply::json(&resp),
        StatusCode::ACCEPTED,
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
