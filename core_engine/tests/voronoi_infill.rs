use core_engine::evaluate_sdf;
use core_engine::implicitus::{node::Body, primitive::Shape, Model, Node, Primitive, Sphere};

#[test]
fn voronoi_infill_creates_voids() {
    // Sphere of radius 2.0 with Voronoi infill
    let mut model = Model::default();
    model.id = "voronoi".into();
    let sphere = Sphere { radius: 2.0 };
    let mut prim = Primitive::default();
    prim.shape = Some(Shape::Sphere(sphere));
    let mut node = Node::default();
    node.body = Some(Body::Primitive(prim));
    model.root = Some(node);

    let seeds = vec![(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)];
    let val = evaluate_sdf(&model, 0.25, 0.0, 0.0, Some("voronoi"), &seeds, 0.0, None);
    assert!(
        val > 0.0,
        "Expected positive SDF inside Voronoi void, got {}",
        val
    );
}
