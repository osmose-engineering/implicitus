#[path = "../src/bin/slicer_server.rs"]
mod slicer_server;

use bytes::Bytes;
use core_engine::implicitus::{node::Body, primitive::Shape, Box, Model, Node, Primitive, Vector3};
use serde_json::json;
use slicer_server::{handle_slice, SliceRequest, SliceResponse};
use warp::hyper::body::to_bytes;
use warp::Reply;

fn make_box_model() -> Model {
    let mut model = Model::default();
    model.id = "box".into();
    let box_shape = Box {
        size: Some(Vector3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        }),
    };
    let mut prim = Primitive::default();
    prim.shape = Some(Shape::Box(box_shape));
    let mut node = Node::default();
    node.body = Some(Body::Primitive(prim));
    model.root = Some(node);
    model
}

#[tokio::test]
async fn slice_box_model_returns_square_contour() {
    // Build a simple box model centered at the origin with side length 2.0
    let mut model = Model::default();
    model.id = "box".into();

    let box_shape = Box {
        size: Some(Vector3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        }),
    };
    let mut prim = Primitive::default();
    prim.shape = Some(Shape::Box(box_shape));

    let mut node = Node::default();
    node.body = Some(Body::Primitive(prim));
    model.root = Some(node);

    let req = SliceRequest {
        _model: model,
        layer: 0.0,
        x_min: Some(-1.5),
        x_max: Some(1.5),
        y_min: Some(-1.5),
        y_max: Some(1.5),
        nx: Some(5),
        ny: Some(5),

        bbox_min: None,
        bbox_max: None,
        cell_vertices: None,
        edge_list: None,
    };

    let body = serde_json::to_vec(&req).unwrap();
    let reply = handle_slice(Bytes::from(body)).await.unwrap();
    let body = reply.into_response().into_body();
    let bytes = to_bytes(body).await.unwrap();
    let resp: SliceResponse = serde_json::from_slice(&bytes).unwrap();
    let contour = &resp.contours[0];
    assert!(resp.segments.is_empty());

    // Verify contour bounds match the box dimensions
    let min_x = contour.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let max_x = contour
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = contour.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let max_y = contour
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max);

    assert!((min_x + 1.0).abs() < 1e-6, "min_x was {}", min_x);
    assert!((max_x - 1.0).abs() < 1e-6, "max_x was {}", max_x);
    assert!((min_y + 1.0).abs() < 1e-6, "min_y was {}", min_y);
    assert!((max_y - 1.0).abs() < 1e-6, "max_y was {}", max_y);

    // Debug info should be present with zero seeds and no pattern
    assert_eq!(resp.debug.seed_count, 0);
    assert!(resp.debug.infill_pattern.is_none());
    assert_eq!(resp.debug.seed_points.unwrap().len(), 0);
}

#[tokio::test]
async fn slice_returns_error_for_invalid_model() {
    let body = serde_json::to_vec(&json!({
        "model": {
            "primitive": {"sphere": {"radius": 1.0}},
            "modifiers": {"infill": {"pattern": "voronoi", "seed_points": [[0.0,0.0,0.0]]}}
        },
        "layer": 0.0
    }))
    .unwrap();
    let res = handle_slice(Bytes::from(body)).await;
    assert!(res.is_err());
}

#[tokio::test]
async fn slice_returns_error_for_lattice_primitive() {
    let body = serde_json::to_vec(&json!({
        "model": {
            "primitive": {"lattice": {
                "pattern": "voronoi",
                "wall_thickness": 0.2,
                "seed_points": [[0.0,0.0,0.0],[1.0,1.0,1.0]]
            }}
        },
        "layer": 0.0
    }))
    .unwrap();
    let res = handle_slice(Bytes::from(body)).await;
    assert!(res.is_err());
}

#[tokio::test]
async fn slice_rejects_nx_less_than_two() {
    for &invalid in &[0usize, 1usize] {
        let req = SliceRequest {
            _model: make_box_model(),
            layer: 0.0,
            x_min: Some(-1.5),
            x_max: Some(1.5),
            y_min: Some(-1.5),
            y_max: Some(1.5),
            nx: Some(invalid),
            ny: Some(5),
            bbox_min: None,
            bbox_max: None,
            cell_vertices: None,
            edge_list: None,
        };
        let body = serde_json::to_vec(&req).unwrap();
        let res = handle_slice(Bytes::from(body)).await;
        assert!(res.is_err(), "expected error for nx={}", invalid);
    }
}

#[tokio::test]
async fn slice_rejects_ny_less_than_two() {
    for &invalid in &[0usize, 1usize] {
        let req = SliceRequest {
            _model: make_box_model(),
            layer: 0.0,
            x_min: Some(-1.5),
            x_max: Some(1.5),
            y_min: Some(-1.5),
            y_max: Some(1.5),
            nx: Some(5),
            ny: Some(invalid),
            bbox_min: None,
            bbox_max: None,
            cell_vertices: None,
            edge_list: None,
        };
        let body = serde_json::to_vec(&req).unwrap();
        let res = handle_slice(Bytes::from(body)).await;
        assert!(res.is_err(), "expected error for ny={}", invalid);
    }
}
