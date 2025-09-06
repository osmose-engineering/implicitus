use std::path::Path;
use std::sync::Once;

use core_engine::hex_lattice;
use core_engine::core_engine as pymodule;
use pyo3::prelude::*;
use pyo3::types::PyList;

// Initialize embedded Python with repository path for importing modules
fn init_python() {
    static START: Once = Once::new();
    START.call_once(|| {
        pyo3::append_to_inittab!(pymodule);
    });
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
}

#[test]
fn rust_edges_match_python_reference() {
    init_python();
    let seeds = vec![(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)];
    let mesh = hex_lattice(&seeds);

    Python::with_gil(|py| {
        let code = r#"
import math

def build_hex_lattice(seeds):
    verts = []
    edges = []
    for cx, cy, cz in seeds:
        for i in range(6):
            angle = math.pi/2 + i*math.pi/3
            x = cx + math.cos(angle)
            y = cy + math.sin(angle)
            low = (x, y, cz - 1.0)
            high = (x, y, cz + 1.0)
            idx = len(verts)
            verts.extend([low, high])
            edges.append((idx, idx + 1))
    return verts, edges
"#;
        let module = PyModule::from_code_bound(py, code, "hex_ref.py", "hex_ref").unwrap();
        let build_hex = module.getattr("build_hex_lattice").unwrap();
        let result: (Vec<(f64, f64, f64)>, Vec<(usize, usize)>) = build_hex
            .call1((seeds.clone(),))
            .unwrap()
            .extract()
            .unwrap();
        let py_edges = result.1;
        println!("Rust edges: {:?}\nPython edges: {:?}", mesh.edges, py_edges);
        assert_eq!(mesh.edges, py_edges);
    });
}
