use pyo3::prelude::*;

// Import the generated Protobuf definitions
pub mod implicitus {
    include!(concat!(env!("OUT_DIR"), "/implicitus.rs"));
}

use implicitus::Model;
use implicitus::node::Body;
use implicitus::primitive::Shape;
pub mod voronoi;

// A very basic SDF evaluator that handles a few primitive shapes.
pub fn evaluate_sdf(model: &Model, x: f64, y: f64, z: f64) -> f64 {
    // Check for a root node
    if let Some(root_node) = &model.root {
        // Look for a primitive in that node
        if let Some(Body::Primitive(p)) = &root_node.body {
            // Match on the inner shape
            if let Some(shape) = &p.shape {
                match shape {
                    Shape::Sphere(s) => {
                        let dist = (x*x + y*y + z*z).sqrt();
                        return dist - s.radius;
                    }
                    Shape::Box(b) => {
                        // Axis-aligned box centered at the origin
                        if let Some(size) = &b.size {
                            let hx = size.x / 2.0;
                            let hy = size.y / 2.0;
                            let hz = size.z / 2.0;
                            let qx = x.abs() - hx;
                            let qy = y.abs() - hy;
                            let qz = z.abs() - hz;
                            let outside = (qx.max(0.0).powi(2)
                                + qy.max(0.0).powi(2)
                                + qz.max(0.0).powi(2))
                                .sqrt();
                            let inside = qx.max(qy.max(qz)).min(0.0);
                            return outside + inside;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    // If nothing matches, return a large positive value (empty space)
    f64::MAX
}

/// Simple mesh structure containing explicit vertices and edge indices.
pub struct VoronoiMesh {
    pub vertices: Vec<(f64, f64, f64)>,
    pub edges: Vec<(usize, usize)>,
}

/// Prototype Voronoi mesher.
///
/// For now this merely echoes the seed points as vertices and links them in a
/// ring.  It provides a stand‑in API for downstream consumers.
pub fn voronoi_mesh(seeds: &[(f64, f64, f64)]) -> VoronoiMesh {
    let vertices = seeds.to_vec();
    let mut edges = Vec::new();
    if seeds.len() > 1 {
        for i in 0..seeds.len() {
            let j = (i + 1) % seeds.len();
            edges.push((i, j));
        }
    }
    VoronoiMesh { vertices, edges }
}

pub mod slice;

#[pymodule]

fn core_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(voronoi::sampling::sample_seed_points, m)?)?;
    m.add_function(wrap_pyfunction!(voronoi::sampling::prune_adjacency_via_grid, m)?)?;
    Ok(())
}
