use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct OctreeNode {
    #[pyo3(get)]
    pub bbox_min: (f64, f64, f64),
    #[pyo3(get)]
    pub bbox_max: (f64, f64, f64),
    #[pyo3(get)]
    pub depth: usize,
    #[pyo3(get)]
    pub children: Vec<OctreeNode>,
}

#[pymethods]
impl OctreeNode {
    #[new]
    #[pyo3(signature = (bbox_min, bbox_max, depth=0))]
    pub fn new(bbox_min: (f64, f64, f64), bbox_max: (f64, f64, f64), depth: usize) -> Self {
        Self {
            bbox_min,
            bbox_max,
            depth,
            children: Vec::new(),
        }
    }

    pub fn subdivide(
        &mut self,
        py: Python<'_>,
        sdf_fn: Bound<PyAny>,
        max_depth: usize,
        threshold: f64,
    ) -> PyResult<()> {
        self.subdivide_internal(py, &sdf_fn, max_depth, threshold)
    }
}

impl OctreeNode {
    fn subdivide_internal(
        &mut self,
        py: Python<'_>,
        sdf_fn: &Bound<PyAny>,
        max_depth: usize,
        threshold: f64,
    ) -> PyResult<()> {
        if self.depth >= max_depth {
            return Ok(());
        }
        let err = self.estimate_error(sdf_fn)?;
        if err <= threshold {
            return Ok(());
        }
        let (x0, y0, z0) = self.bbox_min;
        let (x1, y1, z1) = self.bbox_max;
        let mx = 0.5 * (x0 + x1);
        let my = 0.5 * (y0 + y1);
        let mz = 0.5 * (z0 + z1);
        let boxes = [
            ((x0, y0, z0), (mx, my, mz)),
            ((mx, y0, z0), (x1, my, mz)),
            ((x0, my, z0), (mx, y1, mz)),
            ((mx, my, z0), (x1, y1, mz)),
            ((x0, y0, mz), (mx, my, z1)),
            ((mx, y0, mz), (x1, my, z1)),
            ((x0, my, mz), (mx, y1, z1)),
            ((mx, my, mz), (x1, y1, z1)),
        ];
        for &(bmin, bmax) in &boxes {
            let mut child = OctreeNode::new(bmin, bmax, self.depth + 1);
            child.subdivide_internal(py, sdf_fn, max_depth, threshold)?;
            self.children.push(child);
        }
        Ok(())
    }

    fn estimate_error(&self, sdf_fn: &Bound<PyAny>) -> PyResult<f64> {
        let (x0, y0, z0) = self.bbox_min;
        let (x1, y1, z1) = self.bbox_max;
        let corners = [
            (x0, y0, z0),
            (x1, y0, z0),
            (x0, y1, z0),
            (x1, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x0, y1, z1),
            (x1, y1, z1),
        ];
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for &(x, y, z) in &corners {
            let v: f64 = sdf_fn.call1((x, y, z))?.extract()?;
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        Ok(max_v - min_v)
    }
}

#[pyfunction]
#[pyo3(signature = (bbox, sdf_fn, max_depth, threshold))]
pub fn generate_adaptive_grid(
    py: Python<'_>,
    bbox: ((f64, f64, f64), (f64, f64, f64)),
    sdf_fn: Bound<PyAny>,
    max_depth: usize,
    threshold: f64,
) -> PyResult<OctreeNode> {
    let (bbox_min, bbox_max) = bbox;
    let mut root = OctreeNode::new(bbox_min, bbox_max, 0);
    root.subdivide_internal(py, &sdf_fn, max_depth, threshold)?;
    Ok(root)
}
