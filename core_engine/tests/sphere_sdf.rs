// core_engine/tests/sphere_sdf.rs

use core_engine::evaluate_sdf;
use core_engine::implicitus::{node::Body, primitive::Shape, Model, Node, Primitive, Sphere};

#[test]
fn sphere_sdf() {
    // Build a sphere model of radius 2.0
    let mut model = Model::default();
    model.id = "test".into();

    // Define the sphere primitive
    let sphere = Sphere { radius: 2.0 };
    let mut prim = Primitive::default();
    prim.shape = Some(Shape::Sphere(sphere));

    // Attach primitive to a root node
    let mut node = Node::default();
    node.body = Some(Body::Primitive(prim));
    model.root = Some(node);

    // At the center, SDF = -radius
    let val = evaluate_sdf(&model, 0.0, 0.0, 0.0, None, &[], 0.0, None);
    assert!(
        (val + 2.0).abs() < 1e-6,
        "Expected SDF at center to be -2.0, got {}",
        val
    );

    // On the surface, SDF ~ 0
    let val2 = evaluate_sdf(&model, 2.0, 0.0, 0.0, None, &[], 0.0, None);
    assert!(
        val2.abs() < 1e-6,
        "Expected SDF at surface to be ~0, got {}",
        val2
    );
}
