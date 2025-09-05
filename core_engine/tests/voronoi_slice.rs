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
    };

    let contours = slice_model(&model, &config);
    assert!(!contours.is_empty());
    // First contour should correspond to the Voronoi edge intersection at (1.1, 1.1)
    let (cx, cy) = contours[0][0];
    assert!(
        approx(cx, 1.1) && approx(cy, 1.1),
        "Expected intersection near (1.1,1.1), got {:?}",
        contours[0][0]
    );
    assert_eq!(contours[0].len(), 2);
    assert!(
        approx(contours[0][0].0, contours[0][1].0) && approx(contours[0][0].1, contours[0][1].1)
    );
}
