use std::path::Path;
use std::sync::Once;

use core_engine::hex_lattice;
use core_engine::core_engine as pymodule;
use pyo3::prelude::*;
use pyo3::types::PyList;

// Initialize embedded Python with repository path for importing design_api modules
fn init_python() {
    static START: Once = Once::new();
    START.call_once(|| {
        pyo3::append_to_inittab!(pymodule);
    });
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let sys = py.import_bound("sys").unwrap();
        let path = sys.getattr("path").unwrap().downcast::<PyList>().unwrap().clone();
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        path.insert(0, repo_root.to_str().unwrap()).unwrap();
    });
}

#[test]
fn hex_lattice_geometry_matches_reference() {
    init_python();
    let seeds = vec![(0.0, 0.0, 0.0)];
    let mesh = hex_lattice(&seeds);
    assert_eq!(mesh.vertices.len(), 12);
    assert_eq!(mesh.edges.len(), 6);
    let expected_xy = vec![
        (0.0, 1.0),
        (-0.8660254037844386, 0.5),
        (-0.8660254037844387, -0.5),
        (-1.2246467991473532e-16, -1.0),
        (0.8660254037844384, -0.5),
        (0.866025403784439, 0.49999999999999917),
    ];
    for (i, (x, y)) in expected_xy.iter().enumerate() {
        let low = mesh.vertices[2 * i];
        let high = mesh.vertices[2 * i + 1];
        assert!((low.0 - x).abs() < 1e-6);
        assert!((low.1 - y).abs() < 1e-6);
        assert!((low.2 + 1.0).abs() < 1e-6);
        assert!((high.0 - x).abs() < 1e-6);
        assert!((high.1 - y).abs() < 1e-6);
        assert!((high.2 - 1.0).abs() < 1e-6);
    }
    let expected_edges = vec![(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)];
    assert_eq!(mesh.edges, expected_edges);
}
