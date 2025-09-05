use core_engine::evaluate_sdf;
use core_engine::implicitus::{node::Body, primitive::Shape, Model, Node, Primitive, Sphere};

#[test]
fn sphere_boundary_intact_with_voronoi_walls() {
    // Build a sphere model of radius 2.0
    let mut model = Model::default();
    model.id = "voronoi_wall".into();
    let sphere = Sphere { radius: 2.0 };
    let mut prim = Primitive::default();
    prim.shape = Some(Shape::Sphere(sphere));
    let mut node = Node::default();
    node.body = Some(Body::Primitive(prim));
    model.root = Some(node);

    let seeds = vec![(0.0, 0.0, 0.0), (0.0, 0.0, 0.8)];
    let wall_thickness = 0.1;

    // On the sphere surface, Voronoi walls should not erode the boundary
    let surface = evaluate_sdf(
        &model,
        2.0,
        0.0,
        0.0,
        Some("voronoi"),
        &seeds,
        wall_thickness,
        None,
    );
    assert!(
        surface.abs() < 1e-6,
        "Expected sphere boundary to remain at zero SDF, got {}",
        surface
    );

    // Inside the sphere along the seed axis we expect a positive value from the Voronoi wall
    let wall = evaluate_sdf(
        &model,
        0.0,
        0.0,
        1.5,
        Some("voronoi"),
        &seeds,
        wall_thickness,
        None,
    );
    assert!(
        wall > 0.0,
        "Expected positive SDF inside Voronoi wall, got {}",
        wall
    );
}
