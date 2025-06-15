// core_engine/tests/slice_model.rs

use core_engine::slice::{slice_model, SliceConfig};
use core_engine::implicitus::{Model, Node, Primitive, Sphere, primitive::Shape, node::Body};

#[test]
fn slice_model_produces_segments() {
    // Build a simple sphere model
    let mut model = Model::default();
    model.id = "test_sphere".into();

    let sphere = Sphere { radius: 1.0 };
    let mut prim = Primitive::default();
    prim.shape = Some(Shape::Sphere(sphere));

    let mut node = Node::default();
    node.body = Some(Body::Primitive(prim));
    model.root = Some(node);

    // Define a basic slice configuration
    let config = SliceConfig {
        z: 0.0,
        x_min: -1.0,
        x_max: 1.0,
        y_min: -1.0,
        y_max: 1.0,
        nx: 3,
        ny: 3,
    };

    // Call the slice and verify it returns non-empty contours
    let contours = slice_model(&model, &config);
    assert!(!contours.is_empty(), "Expected non-empty contours from slice_model, got {:?}", contours);
    assert!(contours[0].len() >= 2, "Expected at least two points in a contour, got {:?}", contours[0]);
}