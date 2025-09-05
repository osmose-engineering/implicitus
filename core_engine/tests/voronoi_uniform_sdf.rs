use core_engine::core_engine as pymodule;
use core_engine::evaluate_sdf;
use core_engine::implicitus::{node::Body, primitive::Shape, Model, Node, Primitive, Sphere};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::path::Path;
use std::sync::Once;

fn init_python() {
    static START: Once = Once::new();
    START.call_once(|| {
        pyo3::append_to_inittab!(pymodule);
    });
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let sys = py.import_bound("sys").unwrap();
        let sys_path = sys.getattr("path").unwrap();
        let path = sys_path.downcast::<PyList>().unwrap();
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        path.insert(0, repo_root.to_str().unwrap()).unwrap();
    });
}

fn patch_hex_stubs(py: Python<'_>) {
    let code = r#"
import numpy as np

def fake_medial_axis(mesh):
    return np.zeros((0,3))

def fake_trace_hexagon(seed, medial_points, plane, *args, **kwargs):
    s = np.array(seed)
    verts = np.array([
        [s[0]+1.0, s[1]+0.0, s[2]-1.0],
        [s[0]+0.5, s[1]+0.866, s[2]+1.0],
        [s[0]-0.5, s[1]+0.866, s[2]-1.0],
        [s[0]-1.0, s[1]+0.0, s[2]+1.0],
        [s[0]-0.5, s[1]-0.866, s[2]-1.0],
        [s[0]+0.5, s[1]-0.866, s[2]+1.0],
    ])
    return verts, False, verts
"#;
    let module = PyModule::from_code_bound(py, code, "stub.py", "stub").unwrap();
    let construct = py
        .import_bound("design_api.services.voronoi_gen.uniform.construct")
        .unwrap();
    construct
        .setattr(
            "compute_medial_axis",
            module.getattr("fake_medial_axis").unwrap(),
        )
        .unwrap();
    construct
        .setattr(
            "trace_hexagon",
            module.getattr("fake_trace_hexagon").unwrap(),
        )
        .unwrap();
}

#[test]
fn evaluate_sdf_uniform_has_thicker_walls() {
    init_python();
    Python::with_gil(|py| patch_hex_stubs(py));

    let mut model = Model::default();
    model.id = "voronoi_uniform_sdf".into();
    let sphere = Sphere { radius: 2.0 };
    let mut prim = Primitive::default();
    prim.shape = Some(Shape::Sphere(sphere));
    let mut node = Node::default();
    node.body = Some(Body::Primitive(prim));
    model.root = Some(node);

    let seeds = vec![(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)];
    let point = (1.0, 0.0, 0.0);

    let standard = evaluate_sdf(
        &model,
        point.0,
        point.1,
        point.2,
        Some("voronoi"),
        &seeds,
        0.0,
        None,
    );
    let uniform = evaluate_sdf(
        &model,
        point.0,
        point.1,
        point.2,
        Some("voronoi"),
        &seeds,
        0.0,
        Some("uniform"),
    );

    assert!(uniform > standard, "expected uniform mode to yield thicker walls");
}
