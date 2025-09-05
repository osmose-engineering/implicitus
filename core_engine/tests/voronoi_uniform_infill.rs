use core_engine::evaluate_sdf;
use core_engine::implicitus::{node::Body, primitive::Shape, Model, Node, Primitive, Sphere};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::path::Path;
use std::sync::Once;

fn init_python() {
    static START: Once = Once::new();
    START.call_once(|| {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let sys = py.import_bound("sys").unwrap();
            let path = sys
                .getattr("path")
                .unwrap()
                .downcast::<PyList>()
                .unwrap()
                .clone();
            let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
            path.insert(0, repo_root.to_str().unwrap()).unwrap();
        });
    });
}

#[test]
fn voronoi_uniform_infill_returns_positive() {
    init_python();
    let mut model = Model::default();
    model.id = "voronoi_uniform".into();
    let sphere = Sphere { radius: 2.0 };
    let mut prim = Primitive::default();
    prim.shape = Some(Shape::Sphere(sphere));
    let mut node = Node::default();
    node.body = Some(Body::Primitive(prim));
    model.root = Some(node);

    let seeds = vec![(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)];
    let val = evaluate_sdf(
        &model,
        0.0,
        0.0,
        0.0,
        Some("voronoi"),
        &seeds,
        0.0,
        Some("uniform"),
    );
    assert!(val.is_finite());
}
