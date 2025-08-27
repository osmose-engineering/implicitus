

// core_engine/src/bin/slicer_server.rs

use warp::Filter;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use core_engine::slice::{slice_model, SliceConfig};
use core_engine::implicitus::{Model, Node, Primitive, Sphere};
use core_engine::implicitus::primitive::Shape;
use core_engine::implicitus::node::Body;
use core_engine::voronoi_mesh;

#[derive(Deserialize)]
struct SliceRequest {
    // TODO: replace Value with actual Model once JSON <-> Protobuf integration is set up
    #[serde(rename = "model")]
    _model: Value,
    layer: f64,
    x_min: Option<f64>,
    x_max: Option<f64>,
    y_min: Option<f64>,
    y_max: Option<f64>,
    nx: Option<usize>,
    ny: Option<usize>,
}

#[derive(Serialize)]
struct SliceResponse {
    contours: Vec<Vec<(f64, f64)>>,
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

    println!("Slicer server listening on 127.0.0.1:4000");
    warp::serve(slice_route.or(voronoi_route))
        .run(([127, 0, 0, 1], 4000))
        .await;
}

async fn handle_slice(req: SliceRequest) -> Result<impl warp::Reply, warp::Rejection> {
    // TODO: Deserialize Value into Model with proper JSON->Protobuf conversion.
    // For now, construct a dummy sphere model based on the prompt in req.model.
    let mut model = Model::default();
    model.id = "dummy".into();

    let mut prim = Primitive::default();
    let sphere = Sphere { radius: 1.0 };
    prim.shape = Some(Shape::Sphere(sphere));

    let mut root = Node::default();
    root.body = Some(Body::Primitive(prim));
    model.root = Some(root);

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