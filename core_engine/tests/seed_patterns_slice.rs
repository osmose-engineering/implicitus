use core_engine::implicitus::{node::Body, primitive::Shape, Model, Node, Primitive, Sphere};
use core_engine::slice::{slice_model, SliceConfig};

fn build_model() -> Model {
    let mut model = Model::default();
    model.id = "seed_pattern".into();
    let sphere = Sphere { radius: 2.0 };
    let mut prim = Primitive::default();
    prim.shape = Some(Shape::Sphere(sphere));
    let mut node = Node::default();
    node.body = Some(Body::Primitive(prim));
    model.root = Some(node);
    model
}

fn approx_ne(a: f64, b: f64) -> bool {
    (a - b).abs() > 1e-6
}

#[test]
fn voronoi_custom_seeds_modify_segments() {
    let model = build_model();
    let seeds_a = vec![
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.0, 2.0),
        (2.0, 2.0, 3.0),
    ];
    let mut seeds_b = seeds_a.clone();
    seeds_b[1] = (3.0, 0.0, 0.0); // shift one seed

    let cfg = |seeds: Vec<(f64, f64, f64)>| SliceConfig {
        z: 1.1,
        x_min: -2.0,
        x_max: 3.0,
        y_min: -2.0,
        y_max: 2.0,
        nx: 3,
        ny: 3,
        seed_points: seeds,
        infill_pattern: Some("voronoi".into()),
        wall_thickness: 0.0,
        mode: None,
        bbox_min: None,
        bbox_max: None,
        cells: None,
    };

    let r1 = slice_model(&model, &cfg(seeds_a));
    let r2 = slice_model(&model, &cfg(seeds_b));
    assert!(!r1.segments.is_empty() && !r2.segments.is_empty());
    let ((x1, y1), _) = r1.segments[0];
    let ((x2, y2), _) = r2.segments[0];
    assert!(approx_ne(x1, x2) || approx_ne(y1, y2));
}

#[test]
fn hex_custom_seeds_modify_segments() {
    let model = build_model();
    let seeds_a = vec![(0.0, 0.0, 0.0)];
    let seeds_b = vec![(1.0, 0.0, 0.0)];

    let cfg = |seeds: Vec<(f64, f64, f64)>| SliceConfig {
        z: 0.0,
        x_min: -2.0,
        x_max: 2.0,
        y_min: -2.0,
        y_max: 2.0,
        nx: 3,
        ny: 3,
        seed_points: seeds,
        infill_pattern: Some("hex".into()),
        wall_thickness: 0.0,
        mode: None,
        bbox_min: None,
        bbox_max: None,
        cells: None,
    };

    let r1 = slice_model(&model, &cfg(seeds_a));
    let r2 = slice_model(&model, &cfg(seeds_b));
    assert!(!r1.segments.is_empty() && !r2.segments.is_empty());
    let ((x1, _), _) = r1.segments[0];
    let ((x2, _), _) = r2.segments[0];
    assert!(approx_ne(x1, x2));
}
