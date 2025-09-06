#[path = "../src/bin/slicer_server.rs"]
mod slicer_server;

use bytes::Bytes;
use serde_json::{json, Value};
use slicer_server::{handle_slice, parse_infill, SliceRequest, SliceResponse};
use warp::hyper::body::to_bytes;
use warp::Reply;
use core_engine::implicitus::{node::Body, primitive::Shape, Model, Node, Primitive, Sphere};

#[test]
fn parse_infill_uses_lattice_vertices_as_seeds() {
    let model_json = json!({
        "infill": {"pattern": "voronoi"}
    });
    let verts = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)];
    let edges = vec![(0usize, 1usize), (1, 2)];
    let (seeds, pattern, _, _, _, _) =
        parse_infill(&model_json, Some(&verts), Some(&edges));
    assert_eq!(seeds.len(), 3);
    assert_eq!(pattern.as_deref(), Some("voronoi"));
    for v in &verts {
        assert!(seeds.contains(v));
    }
}

#[tokio::test]
async fn handle_slice_derives_seeds_from_lattice() {
    // Build a minimal valid model containing a single sphere primitive.
    let mut model = Model::default();
    model.id = "precomputed_lattice".into();
    let sphere = Sphere { radius: 1.0 };
    let mut prim = Primitive::default();
    prim.shape = Some(Shape::Sphere(sphere));
    let mut node = Node::default();
    node.body = Some(Body::Primitive(prim));
    model.root = Some(node);

    // Convert the model to JSON and attach an infill pattern so `parse_infill`
    // picks it up when building the slice configuration.
    let mut model_json = serde_json::to_value(model).unwrap();
    if let Value::Object(ref mut obj) = model_json {
        obj.insert(
            "modifiers".into(),
            json!({"infill": {"pattern": "voronoi"}}),
        );
    }

    let req = SliceRequest {
        _model: model_json,
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
