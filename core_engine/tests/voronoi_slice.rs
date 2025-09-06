use core_engine::implicitus::{node::Body, primitive::Shape, Model, Node, Primitive, Sphere};
use core_engine::slice::{slice_model, SliceConfig};

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn voronoi_infill_slice_matches_expected() {
    // Seed configuration yielding a single Voronoi edge
    let seeds = vec![
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.0, 2.0),
        (2.0, 2.0, 3.0),
    ];
    let target_z = 1.1;

    // Build a simple spherical model
    let mut model = Model::default();
    model.id = "voronoi_slice".into();
    let sphere = Sphere { radius: 2.0 };
    let mut prim = Primitive::default();
    prim.shape = Some(Shape::Sphere(sphere));
    let mut node = Node::default();
    node.body = Some(Body::Primitive(prim));
    model.root = Some(node);

    // Slice configuration with Voronoi infill enabled
    let config = SliceConfig {
        z: target_z,
        x_min: -2.0,
        x_max: 2.0,
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
    };

    let result = slice_model(&model, &config);
    assert!(!result.segments.is_empty());
    // First segment should correspond to the Voronoi edge intersection at (1.1, 1.1)
    let ((sx, sy), (ex, ey)) = result.segments[0];
    assert!(
        approx(sx, 1.1) && approx(sy, 1.1),
        "Expected intersection near (1.1,1.1), got ({:?}, {:?})",
        sx,
        sy
    );
    assert!(approx(sx, ex) && approx(sy, ey));
}
