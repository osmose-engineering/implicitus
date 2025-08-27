// core_engine/src/bin/slicer_server.rs

use core_engine::implicitus::Model;
use core_engine::slice::{slice_model, SliceConfig};
use core_engine::voronoi_mesh;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use warp::http::Method;
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

#[derive(Serialize)]
struct VoronoiResponse {
    vertices: Vec<(f64, f64, f64)>,
    edges: Vec<(usize, usize)>,
}

#[tokio::main]
async fn main() {
    // POST /slice  with JSON body to perform slicing
    let slice_route = warp::post()
        .and(warp::path("slice"))
        .and(warp::body::json())
        .and_then(handle_slice);

    // POST /voronoi to generate a mesh from seed points
    let voronoi_route = warp::post()
        .and(warp::path("voronoi"))
        .and(warp::body::json())
        .and_then(handle_voronoi);

    let cors = warp::cors()
        .allow_any_origin()
        .allow_methods(vec![Method::POST])
        .allow_header("content-type");

    println!("Slicer server listening on 127.0.0.1:4000");
    let routes = slice_route.or(voronoi_route).with(cors);
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

async fn handle_voronoi(req: VoronoiRequest) -> Result<impl warp::Reply, warp::Rejection> {
    let mesh = voronoi_mesh(&req.seeds);
    let resp = VoronoiResponse {
        vertices: mesh.vertices,
        edges: mesh.edges,
    };
    Ok(warp::reply::json(&resp))
}
