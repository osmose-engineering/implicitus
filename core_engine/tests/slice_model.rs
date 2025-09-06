// core_engine/tests/slice_model.rs

use core_engine::implicitus::{
    node::Body,
    primitive::Shape,
    transform,
    Model,
    Node,
    Primitive,
    Sphere,
    Transform,
    Translate,
    Vector3,
};
use core_engine::slice::{slice_model, SliceConfig};

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
        seed_points: Vec::new(),
        infill_pattern: None,
        wall_thickness: 0.0,
        mode: None,
        bbox_min: None,
        bbox_max: None,
    };

    // Call the slice and verify it returns non-empty contours and no segments
    let result = slice_model(&model, &config);
    assert!(
        !result.contours.is_empty(),
        "Expected non-empty contours from slice_model, got {:?}",
        result.contours
    );
    assert!(
        result.contours[0].len() >= 2,
        "Expected at least two points in a contour, got {:?}",
        result.contours[0]
    );
    assert!(
        result.segments.is_empty(),
        "Expected no segments for sphere slice, got {:?}",
        result.segments
    );
}

#[test]
fn sphere_slice_matches_expected_circle() {
    // Sphere centered at origin with radius 1
    let mut model = Model::default();
    model.id = "sphere".into();
    let sphere = Sphere { radius: 1.0 };
    let mut prim = Primitive::default();
    prim.shape = Some(Shape::Sphere(sphere));
    let mut node = Node::default();
    node.body = Some(Body::Primitive(prim));
    model.root = Some(node);

    // Higher resolution grid for accuracy
    let config = SliceConfig {
        z: 0.0,
        x_min: -1.2,
        x_max: 1.2,
        y_min: -1.2,
        y_max: 1.2,
        nx: 50,
        ny: 50,
        seed_points: Vec::new(),
        infill_pattern: None,
        wall_thickness: 0.0,
        mode: None,
        bbox_min: None,
        bbox_max: None,
    };

    let result = slice_model(&model, &config);
    assert_eq!(result.contours.len(), 1);
    let contour = &result.contours[0];
    // First point repeated at end
    assert!(contour.len() > 4);
    let expected_r = 1.0;
    for &(x, y) in contour.iter().take(contour.len() - 1) {
        let r = (x * x + y * y).sqrt();
        assert!(
            (r - expected_r).abs() < 0.1,
            "point ({}, {}) had radius {}",
            x,
            y,
            r
        );
    }
    let first = contour.first().unwrap();
    let last = contour.last().unwrap();
    assert!((first.0 - last.0).abs() < 1e-6 && (first.1 - last.1).abs() < 1e-6);
}

#[test]
fn two_spheres_produce_two_loops() {
    // Build model with two translated spheres
    let mut model = Model::default();
    model.id = "two_spheres".into();

    let sphere = Sphere { radius: 1.0 };
    let mut prim = Primitive::default();
    prim.shape = Some(Shape::Sphere(sphere));

    // helper to build a translated sphere node
    fn translated_sphere(x: f64, prim: &Primitive) -> Node {
        let mut trans = Transform::default();
        trans.op = Some(transform::Op::Translate(Translate {
            offset: Some(Vector3 { x, y: 0.0, z: 0.0 }),
        }));
        let mut tnode = Node::default();
        tnode.body = Some(Body::Transform(trans));
        let mut child = Node::default();
        child.body = Some(Body::Primitive(prim.clone()));
        tnode.children.push(child);
        tnode
    }

    let node_a = translated_sphere(-2.0, &prim);
    let node_b = translated_sphere(2.0, &prim);
    let mut root = Node::default();
    root.children.push(node_a);
    root.children.push(node_b);
    model.root = Some(root);

    let config = SliceConfig {
        z: 0.0,
        x_min: -4.0,
        x_max: 4.0,
        y_min: -2.0,
        y_max: 2.0,
        nx: 80,
        ny: 40,
        seed_points: Vec::new(),
        infill_pattern: None,
        wall_thickness: 0.0,
        mode: None,
        bbox_min: None,
        bbox_max: None,
    };

    let result = slice_model(&model, &config);
    assert_eq!(result.contours.len(), 2);

    // Validate each loop is a unit circle centered near +/-2
    let mut centers_found = Vec::new();
    for contour in &result.contours {
        let n = contour.len() - 1; // last == first
        let (sum_x, sum_y) = contour.iter().take(n).fold((0.0, 0.0), |acc, p| {
            (acc.0 + p.0, acc.1 + p.1)
        });
        let center = (sum_x / n as f64, sum_y / n as f64);
        centers_found.push(center);
        for &(x, y) in contour.iter().take(n) {
            let r = ((x - center.0).powi(2) + (y - center.1).powi(2)).sqrt();
            assert!((r - 1.0).abs() < 0.1);
        }
    }
    centers_found.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    assert!((centers_found[0].0 + 2.0).abs() < 0.2);
    assert!((centers_found[1].0 - 2.0).abs() < 0.2);
}
