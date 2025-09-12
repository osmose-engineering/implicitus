use core_engine::implicitus::{node::Body, primitive::Shape, Model, Node, Primitive, Sphere};
use core_engine::slice::{slice_model, SliceConfig};

fn min_radius(contour: &[(f64, f64)]) -> f64 {
    contour
        .iter()
        .map(|&(x, y)| (x * x + y * y).sqrt())
        .fold(f64::INFINITY, |a, b| a.min(b))
}

#[test]
fn wall_thickness_shifts_contours() {
    let seeds = vec![(0.0, 0.0, 0.0), (4.0, 0.0, 0.0)];

    let mut model = Model::default();
    model.id = "wall_thickness_slice".into();
    let sphere = Sphere { radius: 5.0 };
    let mut prim = Primitive::default();
    prim.shape = Some(Shape::Sphere(sphere));
    let mut node = Node::default();
    node.body = Some(Body::Primitive(prim));
    model.root = Some(node);

    let mut config = SliceConfig {
        z: 0.0,
        x_min: -5.0,
        x_max: 5.0,
        y_min: -5.0,
        y_max: 5.0,
        nx: 41,
        ny: 41,
        seed_points: seeds,
        infill_pattern: Some("voronoi".into()),
        wall_thickness: 0.5,
        mode: None,
        bbox_min: None,
        bbox_max: None,
        cells: None,
    };

    let thin = slice_model(&model, &config);
    assert!(
        !thin.contours[0].is_empty(),
        "Expected non-empty contour for thin wall",
    );
    let thin_min = min_radius(&thin.contours[0]);

    config.wall_thickness = 1.0;
    let thick = slice_model(&model, &config);
    assert!(
        !thick.contours[0].is_empty(),
        "Expected non-empty contour for thick wall",
    );
    let thick_min = min_radius(&thick.contours[0]);

    assert!(
        thick_min < thin_min,
        "Expected thicker wall to yield smaller interior radius ({} >= {})",
        thick_min,
        thin_min
    );
}
