use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::path::Path;
use std::sync::Once;
use core_engine::core_engine as pymodule;

// Initialize the embedded Python interpreter with the `core_engine` module
// registered so tests can `import core_engine` without a compiled extension.
fn init_python() {
    static START: Once = Once::new();
    START.call_once(|| {
        pyo3::append_to_inittab!(pymodule);
    });
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let sys = py.import("sys").unwrap();
        let path: &pyo3::types::PyList = sys.getattr("path").unwrap().downcast().unwrap();
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        path.insert(0, repo_root.to_str().unwrap()).unwrap();
    });
}

fn make_mesh<'py>(py: Python<'py>) -> PyResult<PyObject> {
    let code = "class DummyMesh:\n    def __init__(self, v):\n        import numpy as np\n        self.vertices = np.array(v)\n";
    let module = PyModule::from_code(py, code, "dummy.py", "dummy")?;
    let np = py.import("numpy")?;
    let verts = np.call_method1("array", (
        vec![
            vec![0.0,0.0,0.0],
            vec![1.0,0.0,0.0],
            vec![0.0,1.0,0.0],
            vec![0.0,0.0,1.0]
        ],
    ))?;
    let mesh = module.getattr("DummyMesh")?.call1((verts,))?;
    Ok(mesh.into())
}

#[test]
fn test_compute_uniform_cells_basic() {
    init_python();
    Python::with_gil(|py| {
        let core = py.import("core_engine").unwrap();
        let func = core.getattr("compute_uniform_cells").unwrap();
        let np = py.import("numpy").unwrap();
        let seeds = np.call_method1("array", (
            vec![vec![0.0,0.0,0.0], vec![0.1,0.0,0.0]],
        )).unwrap();
        let mesh = make_mesh(py).unwrap();
        let plane = np.call_method1("array", (vec![0.0,0.0,1.0],)).unwrap();
        let kwargs = PyDict::new(py);
        kwargs.set_item("max_distance", 2.0).unwrap();
        let result = func.call((seeds, mesh, plane), Some(kwargs)).unwrap();
        let cells: &pyo3::types::PyDict = result.downcast().unwrap();
        assert_eq!(cells.len(), 2);
        for (_k,v) in cells.iter() {
            let arr = v.downcast::<numpy::PyArray2<f64>>().unwrap();
            assert_eq!(arr.shape(), &[6,3]);
        }
    });
}

#[test]
fn test_edges_generated_for_simple_seed() {
    init_python();
    Python::with_gil(|py| {
        let core = py.import("core_engine").unwrap();
        let func = core.getattr("compute_uniform_cells").unwrap();
        let np = py.import("numpy").unwrap();
        let seeds = np.call_method1("array", (vec![vec![0.0,0.0,0.0]],)).unwrap();
        let mesh = make_mesh(py).unwrap();
        let plane = np.call_method1("array", (vec![0.0,0.0,1.0],)).unwrap();
        let kwargs = PyDict::new(py);
        kwargs.set_item("max_distance", 2.0).unwrap();
        kwargs.set_item("return_edges", true).unwrap();
        let result = func.call((seeds, mesh, plane), Some(kwargs)).unwrap();
        let tup: &PyTuple = result.downcast().unwrap();
        let cells: &pyo3::types::PyDict = tup.get_item(0).unwrap().downcast().unwrap();
        let edges: Vec<(usize,usize)> = tup.get_item(1).unwrap().extract().unwrap();
        assert_eq!(cells.len(),1);
        assert!(!edges.is_empty());
    });
}

#[test]
fn test_uniform_dump_file_created() {
    init_python();
    Python::with_gil(|py| {
        let core = py.import("core_engine").unwrap();
        let func = core.getattr("compute_uniform_cells").unwrap();
        let np = py.import("numpy").unwrap();
        let seeds = np.call_method1("array", (vec![vec![0.0,0.0,0.0]],)).unwrap();
        let mesh = make_mesh(py).unwrap();
        let plane = np.call_method1("array", (vec![0.0,0.0,1.0],)).unwrap();
        let kwargs = PyDict::new(py);
        kwargs.set_item("max_distance", 1.0).unwrap();
        let dump_path = Path::new("../logs/UNIFORM_CELL_DUMP.json");
        if dump_path.exists() { std::fs::remove_file(dump_path).unwrap(); }
        let _ = func.call((seeds, mesh, plane), Some(kwargs)).unwrap();
        assert!(dump_path.exists());
        if let Ok(meta) = std::fs::metadata(dump_path) {
            assert!(meta.len() >= 0);
        }
        let _ = std::fs::remove_file(dump_path);
    });
}
