use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use numpy::{PyReadonlyArray2, PyReadonlyArray1, PyArray1, PyArray2};
use std::collections::{HashMap, HashSet};

fn hexagon_metrics(pts: &Vec<[f64;3]>) -> (Vec<f64>, f64, f64, f64) {
    let mut edges = Vec::new();
    for i in 0..pts.len() {
        let a = pts[i];
        let b = pts[(i + 1) % pts.len()];
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        edges.push((dx*dx + dy*dy + dz*dz).sqrt());
    }
    let mean = edges.iter().sum::<f64>() / edges.len() as f64;
    let std = (edges.iter().map(|e| (e-mean).powi(2)).sum::<f64>() / edges.len() as f64).sqrt();
    let mut centroid = [0.0f64;3];
    for p in pts { centroid[0]+=p[0]; centroid[1]+=p[1]; centroid[2]+=p[2]; }
    centroid[0]/=pts.len() as f64; centroid[1]/=pts.len() as f64; centroid[2]/=pts.len() as f64;
    let mut area = 0.0;
    for i in 0..pts.len() {
        let a = [pts[i][0]-centroid[0], pts[i][1]-centroid[1], pts[i][2]-centroid[2]];
        let b = [pts[(i+1)%pts.len()][0]-centroid[0], pts[(i+1)%pts.len()][1]-centroid[1], pts[(i+1)%pts.len()][2]-centroid[2]];
        let cross = [
            a[1]*b[2]-a[2]*b[1],
            a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0],
        ];
        area += 0.5*((cross[0]*cross[0]+cross[1]*cross[1]+cross[2]*cross[2]).sqrt());
    }
    (edges, mean, std, area)
}

fn build_edge_list(cell_slices: &HashMap<usize,(usize,usize)>) -> Vec<(usize,usize)> {
    let mut edges = Vec::new();
    for (_idx, &(start, end)) in cell_slices.iter() {
        for i in start..end {
            let j = if i + 1 == end { start } else { i + 1 };
            edges.push((i, j));
        }
    }
    let mut seen: HashSet<(usize,usize)> = HashSet::new();
    let mut unique = Vec::new();
    for (a,b) in edges {
        let key = (a.min(b), a.max(b));
        if seen.insert(key) {
            unique.push((a,b));
        }
    }
    unique
}

#[pyfunction]
#[pyo3(signature=(
    seeds,
    imds_mesh,
    plane_normal,
    max_distance=None,
    vertex_tolerance=1e-5,
    return_status=false,
    return_edges=false
))]
pub fn compute_uniform_cells(
    py: Python<'_>,
    seeds: PyReadonlyArray2<f64>,
    imds_mesh: PyObject,
    plane_normal: PyReadonlyArray1<f64>,
    max_distance: Option<f64>,
    vertex_tolerance: f64,
    return_status: bool,
    return_edges: bool,
) -> PyResult<PyObject> {
    let module = py.import("design_api.services.voronoi_gen.uniform.construct")?;
    let compute_medial_axis = module.getattr("compute_medial_axis")?;
    let trace_hexagon = module.getattr("trace_hexagon")?;
    let dump_fn = module.getattr("dump_uniform_cell_map")?;

    let medial_points = compute_medial_axis.call1((imds_mesh.clone_ref(py),))?.into_py(py);

    let seeds_arr = seeds.as_array();
    let mut seed_list: Vec<Vec<f64>> = Vec::new();
    for row in seeds_arr.outer_iter() { seed_list.push(vec![row[0], row[1], row[2]]); }
    let plane_vec = plane_normal.as_array();
    let plane_list = vec![plane_vec[0], plane_vec[1], plane_vec[2]];

    let dump_cells = PyDict::new(py);
    let dump_data = PyDict::new(py);
    dump_data.set_item("seeds", &seed_list)?;
    dump_data.set_item("plane_normal", &plane_list)?;
    dump_data.set_item("max_distance", max_distance)?;
    dump_data.set_item("medial_points", medial_points.clone_ref(py))?;
    dump_data.set_item("cells", dump_cells)?;
    let fallback_list = PyList::empty(py);
    dump_data.set_item("fallback_indices", fallback_list)?;
    let failed_list = PyList::empty(py);
    dump_data.set_item("failed_indices", failed_list)?;
    let _ = vertex_tolerance;

    let mut all_vertices: Vec<[f64;3]> = Vec::new();
    let mut cell_slices: HashMap<usize,(usize,usize)> = HashMap::new();
    let cells_out = PyDict::new(py);

    for (idx, seed) in seed_list.iter().enumerate() {
        let seed_arr = PyArray1::from_slice(py, seed);
        let plane_arr = PyArray1::from_slice(py, &plane_list);
        let mut args: Vec<PyObject> = vec![
            seed_arr.into_py(py),
            medial_points.clone_ref(py),
            plane_arr.into_py(py),
        ];
        if let Some(md) = max_distance { args.push(md.into_py(py)); }
        let kwargs = PyDict::new(py);
        kwargs.set_item("report_method", true)?;
        kwargs.set_item("return_raw", true)?;
        let result = trace_hexagon.call(PyTuple::new(py, args), Some(kwargs))?;
        let tpl: &PyTuple = result.downcast()?;
        let hex_pts = tpl.get_item(0)?;
        let used_fallback: bool = tpl.get_item(1)?.extract()?;
        let raw_hex = tpl.get_item(2)?;
        let hex_array: &PyArray2<f64> = hex_pts.downcast()?;
        let hex_vec = unsafe { hex_array.as_array() };
        let mut pts_vec: Vec<[f64;3]> = Vec::new();
        for row in hex_vec.outer_iter() {
            pts_vec.push([row[0], row[1], row[2]]);
            all_vertices.push([row[0], row[1], row[2]]);
        }
        let start = all_vertices.len() - pts_vec.len();
        let end = all_vertices.len();
        cell_slices.insert(idx, (start, end));
        let (edge_lengths, mean_edge, std_edge, area) = hexagon_metrics(&pts_vec);
        let cell_info = PyDict::new(py);
        cell_info.set_item("seed", seed)?;
        cell_info.set_item("vertices", hex_pts)?;
        cell_info.set_item("raw_vertices", raw_hex)?;
        let metrics = PyDict::new(py);
        metrics.set_item("edge_lengths", edge_lengths)?;
        metrics.set_item("mean_edge_length", mean_edge)?;
        metrics.set_item("std_edge_length", std_edge)?;
        metrics.set_item("area", area)?;
        cell_info.set_item("metrics", metrics)?;
        cell_info.set_item("used_fallback", used_fallback)?;
        dump_cells.set_item(idx, cell_info)?;
        cells_out.set_item(idx, hex_pts)?;
    }

    let edges = build_edge_list(&cell_slices);
    dump_data.set_item("edges", &edges)?;
    let _ = dump_fn.call1((dump_data,));

    if return_status && return_edges {
        let tuple = PyTuple::new(py, &[cells_out.into_py(py), edges.clone().into_py(py), 0i32.into_py(py), PyList::empty(py).into_py(py)]);
        return Ok(tuple.into());
    }
    if return_status {
        let tuple = PyTuple::new(py, &[cells_out.into_py(py), 0i32.into_py(py), PyList::empty(py).into_py(py)]);
        return Ok(tuple.into());
    }
    if return_edges {
        let tuple = PyTuple::new(py, &[cells_out.into_py(py), edges.into_py(py)]);
        return Ok(tuple.into());
    }
    Ok(cells_out.into_py(py))
}
