use core_engine::core_engine as pymodule;
use core_engine::implicitus::{node::Body, primitive::Shape, Model, Node, Primitive, Sphere};
use core_engine::slice::{slice_model, SliceConfig};
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
fn slice_uniform_voronoi_produces_hex_segments() {
    init_python();
    Python::with_gil(|py| patch_hex_stubs(py));

    let mut model = Model::default();
    model.id = "uniform_hex".into();
    let sphere = Sphere { radius: 2.0 };
    let mut prim = Primitive::default();
    prim.shape = Some(Shape::Sphere(sphere));
    let mut node = Node::default();
    node.body = Some(Body::Primitive(prim));
    model.root = Some(node);

    let seeds = vec![(0.0, 0.0, 0.0)];

    let config = SliceConfig {
        z: 0.0,
        x_min: -2.0,
        x_max: 2.0,
        y_min: -2.0,
        y_max: 2.0,
        nx: 3,
        ny: 3,
        seed_points: seeds,
        infill_pattern: Some("voronoi".into()),
        wall_thickness: 0.0,
        mode: Some("uniform".into()),
        bbox_min: None,
        bbox_max: None,
        cells: None,
    };

    let result = slice_model(&model, &config);
    assert_eq!(
        result.segments.len(),
        6,
        "Expected 6 hexagonal segments, got {}",
        result.segments.len()
    );
}
