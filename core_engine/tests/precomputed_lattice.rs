#[path = "../src/bin/slicer_server.rs"]
mod slicer_server;

use bytes::Bytes;
use core_engine::implicitus::Model;
use serde_json::json;
use slicer_server::{handle_slice, SliceRequest, SliceResponse};
use warp::hyper::body::to_bytes;
use warp::Reply;

#[tokio::test]
async fn handle_slice_derives_seeds_from_lattice() {
    let model_json = json!({
        "id": "sphere",
        "constraints": [],
        "root": {
            "children": [],
            "primitive": {"sphere": {"radius": 1.0}},
            "modifiers": [{"infill": {"pattern": "voronoi"}}]
        }
    });
    let model: Model = serde_json::from_value(model_json).unwrap();
    let req = SliceRequest {
        _model: model,
        layer: 0.0,
        x_min: None,
        x_max: None,
        y_min: None,
        y_max: None,
        nx: None,
        ny: None,
        bbox_min: None,
        bbox_max: None,
        cell_vertices: Some(vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]),
        edge_list: Some(vec![(0usize, 1usize), (1, 2)]),
    };

    let body = serde_json::to_vec(&req).unwrap();
    let reply = handle_slice(Bytes::from(body)).await.unwrap();
    let bytes = to_bytes(reply.into_response().into_body()).await.unwrap();
    let resp: SliceResponse = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(resp.debug.seed_count, 3);
    let seeds = resp.debug.seed_points.unwrap();
    for v in &req.cell_vertices.clone().unwrap() {
        assert!(seeds.contains(v));
    }
}
