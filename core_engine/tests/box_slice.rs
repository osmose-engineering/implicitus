#[path = "../src/bin/slicer_server.rs"]
mod slicer_server;

use core_engine::implicitus::{node::Body, primitive::Shape, Box, Model, Node, Primitive, Vector3};
use serde_json::{json, to_value};
use slicer_server::{handle_slice, SliceRequest, SliceResponse};
use warp::hyper::body::to_bytes;
use warp::Reply;

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
        _model: to_value(&model).unwrap(),
        layer: 0.0,
        x_min: Some(-1.5),
        x_max: Some(1.5),
        y_min: Some(-1.5),
        y_max: Some(1.5),
        nx: Some(5),
        ny: Some(5),
    };

    let reply = handle_slice(req).await.unwrap();
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
async fn slice_returns_debug_for_invalid_model() {
    let req = SliceRequest {
        _model: json!({
            "primitive": {"sphere": {"radius": 1.0}},
            "modifiers": {"infill": {"pattern": "voronoi", "seed_points": [[0.0,0.0,0.0]]}}
        }),
        layer: 0.0,
        x_min: None,
        x_max: None,
        y_min: None,
        y_max: None,
        nx: None,
        ny: None,
    };

    let reply = handle_slice(req).await.unwrap();
    let body = reply.into_response().into_body();
    let bytes = to_bytes(body).await.unwrap();
    let resp: SliceResponse = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(resp.debug.seed_count, 1);
    assert_eq!(resp.debug.seed_points.unwrap().len(), 1);
    assert!(resp.contours.is_empty());
    assert!(resp.segments.is_empty());
}

#[tokio::test]
async fn slice_returns_debug_for_lattice_primitive() {
    let req = SliceRequest {
        _model: json!({
            "primitive": {"lattice": {
                "pattern": "voronoi",
                "wall_thickness": 0.2,
                "seed_points": [[0.0,0.0,0.0],[1.0,1.0,1.0]]
            }}
        }),
        layer: 0.0,
        x_min: None,
        x_max: None,
        y_min: None,
        y_max: None,
        nx: None,
        ny: None,
    };

    let reply = handle_slice(req).await.unwrap();
    let body = reply.into_response().into_body();
    let bytes = to_bytes(body).await.unwrap();
    let resp: SliceResponse = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(resp.debug.seed_count, 2);
    assert_eq!(resp.debug.seed_points.unwrap().len(), 2);
    assert!(resp.contours.is_empty());
    assert!(resp.segments.is_empty());
}
