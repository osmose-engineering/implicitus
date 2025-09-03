use pyo3::prelude::*;

// Import the generated Protobuf definitions
pub mod implicitus {
    include!(concat!(env!("OUT_DIR"), "/implicitus.rs"));
}

use implicitus::node::Body;
use implicitus::primitive::Shape;
use implicitus::Model;
pub mod primitives;
pub mod spatial;
pub mod uniform;
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
                        let dist = (x * x + y * y + z * z).sqrt();
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
                            let outside =
                                (qx.max(0.0).powi(2) + qy.max(0.0).powi(2) + qz.max(0.0).powi(2))
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

fn sub(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}

fn add(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
}

fn scale(a: (f64, f64, f64), s: f64) -> (f64, f64, f64) {
    (a.0 * s, a.1 * s, a.2 * s)
}

fn dot(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

fn cross(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}

fn norm_sq(a: (f64, f64, f64)) -> f64 {
    dot(a, a)
}

fn circumcenter(
    p1: (f64, f64, f64),
    p2: (f64, f64, f64),
    p3: (f64, f64, f64),
    p4: (f64, f64, f64),
) -> Option<(f64, f64, f64)> {
    let a = sub(p1, p4);
    let b = sub(p2, p4);
    let c = sub(p3, p4);
    let cross_bc = cross(b, c);
    let denom = 2.0 * dot(a, cross_bc);
    if denom.abs() < 1e-12 {
        return None;
    }
    let cross_ca = cross(c, a);
    let cross_ab = cross(a, b);
    let u = add(
        add(scale(cross_bc, norm_sq(a)), scale(cross_ca, norm_sq(b))),
        scale(cross_ab, norm_sq(c)),
    );
    let u = scale(u, 1.0 / denom);
    Some(add(p4, u))
}

/// Compute a 3D Voronoi diagram for the provided seed points.
///
/// The returned mesh lists Voronoi vertices (circumcenters of valid
/// tetrahedra) and edges connecting them.  Each edge corresponds to a triple of
/// neighboring seeds whose cells share a face.
pub fn voronoi_mesh(seeds: &[(f64, f64, f64)]) -> VoronoiMesh {
    use std::collections::{HashMap, HashSet};

    let n = seeds.len();
    let mut vertices: Vec<(f64, f64, f64)> = Vec::new();
    let mut quad_to_vertex: HashMap<[usize; 4], usize> = HashMap::new();

    // Enumerate all tetrahedra and collect valid circumcenters as Voronoi vertices
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                for l in (k + 1)..n {
                    if let Some(center) = circumcenter(seeds[i], seeds[j], seeds[k], seeds[l]) {
                        let dist2 = norm_sq(sub(center, seeds[i]));
                        let mut valid = true;
                        for m in 0..n {
                            if m == i || m == j || m == k || m == l {
                                continue;
                            }
                            let d2 = norm_sq(sub(center, seeds[m]));
                            if d2 < dist2 - 1e-9 {
                                valid = false;
                                break;
                            }
                        }
                        if valid {
                            let idx = vertices.len();
                            vertices.push(center);
                            let mut key = [i, j, k, l];
                            key.sort();
                            quad_to_vertex.insert(key, idx);
                        }
                    }
                }
            }
        }
    }

    // Build adjacency edges between Voronoi vertices sharing a triangle of seeds
    let mut triple_map: HashMap<[usize; 3], Vec<usize>> = HashMap::new();
    for (quad, &v_idx) in &quad_to_vertex {
        let [a, b, c, d] = *quad;
        let triples = [[a, b, c], [a, b, d], [a, c, d], [b, c, d]];
        for mut t in triples {
            t.sort();
            triple_map.entry(t).or_default().push(v_idx);
        }
    }

    let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
    for verts in triple_map.values() {
        if verts.len() >= 2 {
            for i in 0..verts.len() {
                for j in (i + 1)..verts.len() {
                    let a = verts[i];
                    let b = verts[j];
                    let e = if a < b { (a, b) } else { (b, a) };
                    edge_set.insert(e);
                }
            }
        }
    }

    let mut edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
    edges.sort();

    VoronoiMesh { vertices, edges }
}

pub mod slice;

#[pymodule]
pub fn core_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(voronoi::sampling::sample_seed_points, m)?)?;
    m.add_function(wrap_pyfunction!(
        voronoi::sampling::prune_adjacency_via_grid,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        voronoi::cells::construct_voronoi_cells,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        voronoi::cells::construct_surface_voronoi_cells,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(uniform::hex::compute_uniform_cells, m)?)?;
    m.add_function(wrap_pyfunction!(primitives::sample_inside, m)?)?;
    m.add_class::<spatial::octree::OctreeNode>()?;
    m.add_function(wrap_pyfunction!(
        spatial::octree::generate_adaptive_grid,
        m
    )?)?;
    Ok(())
}
