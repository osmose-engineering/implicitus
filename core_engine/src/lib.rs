use crate::voronoi::sdf::voronoi_sdf;
use kiddo::float::kdtree::KdTree;
use kiddo::SquaredEuclidean;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// Import the generated Protobuf definitions
pub mod implicitus {
    include!(concat!(env!("OUT_DIR"), "/implicitus.rs"));
}

use implicitus::node::Body;
use implicitus::primitive::Shape;
use implicitus::{transform, Model};
pub mod primitives;
pub mod spatial;
pub mod uniform;
pub mod voronoi;

// This constant is generated at build time from `constants.json` to ensure
// that all language layers share the same value.
include!(concat!(env!("OUT_DIR"), "/constants.rs"));

// Evaluate the signed distance for a node, handling primitives, simple
// translations and union of children.  The evaluator is intentionally
// minimal and only supports the few primitives required by tests.
fn eval_node(node: &implicitus::Node, x: f64, y: f64, z: f64) -> f64 {
    match &node.body {
        // Primitive shape at the current coordinate
        Some(Body::Primitive(p)) => {
            let mut sdf = if let Some(shape) = &p.shape {
                match shape {
                    Shape::Sphere(s) => {
                        let dist = (x * x + y * y + z * z).sqrt();
                        dist - s.radius
                    }
                    Shape::Box(b) => {
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
                            outside + inside
                        } else {
                            f64::MAX
                        }
                    }
                    _ => f64::MAX,
                }
            } else {
                f64::MAX
            };

            // Union any child nodes with this primitive
            for child in &node.children {
                let child_sdf = eval_node(child, x, y, z);
                sdf = sdf.min(child_sdf);
            }
            sdf
        }
        // Simple translate transform: evaluate children in translated coordinates
        Some(Body::Transform(t)) => {
            if let Some(transform::Op::Translate(tr)) = &t.op {
                if let Some(offset) = &tr.offset {
                    let nx = x - offset.x;
                    let ny = y - offset.y;
                    let nz = z - offset.z;
                    return node
                        .children
                        .iter()
                        .fold(f64::MAX, |acc, c| acc.min(eval_node(c, nx, ny, nz)));
                }
            }
            // Unknown transform, fall back to union of unmodified children
            node.children
                .iter()
                .fold(f64::MAX, |acc, c| acc.min(eval_node(c, x, y, z)))
        }
        // No body: union of children
        None => node
            .children
            .iter()
            .fold(f64::MAX, |acc, c| acc.min(eval_node(c, x, y, z))),
    }
}

// A very basic SDF evaluator that handles a few primitive shapes and optional infill.
pub fn evaluate_sdf(
    model: &Model,
    x: f64,
    y: f64,
    z: f64,
    infill_pattern: Option<&str>,
    seeds: &[(f64, f64, f64)],
    wall_thickness: f64,
    mode: Option<&str>,
) -> f64 {
    // Evaluate the base model SDF using the simple recursive evaluator
    let mut base_sdf = if let Some(root_node) = &model.root {
        eval_node(root_node, x, y, z)
    } else {
        f64::MAX
    };

    // Optional infill adjustments
    if infill_pattern == Some("voronoi") && mode == Some("uniform") && !seeds.is_empty() {
        let cells = get_uniform_hex_cells(seeds);
        if !cells.is_empty() {
            let infill_sdf = uniform_hex_sdf((x, y, z), &cells);
            base_sdf = base_sdf.max(infill_sdf - wall_thickness);
        }
    } else if infill_pattern == Some("voronoi") && !seeds.is_empty() {
        let infill_sdf = voronoi_sdf((x, y, z), seeds);
        base_sdf = base_sdf.max(infill_sdf - wall_thickness);
    } else if infill_pattern == Some("hex") && !seeds.is_empty() {
        let infill_sdf = hex_sdf((x, y, z), seeds);
        base_sdf = base_sdf.max(infill_sdf - wall_thickness);
    }

    base_sdf
}

/// Simple mesh structure containing explicit vertices and edge indices.
pub struct VoronoiMesh {
    pub vertices: Vec<(f64, f64, f64)>,
    pub edges: Vec<(usize, usize)>,
}

/// Simple SDF for a hexagonal infill pattern approximated by vertical cylinders
/// centered at the provided seed points.  The cylinder radius is fixed at 0.5.
fn hex_sdf(point: (f64, f64, f64), seeds: &[(f64, f64, f64)]) -> f64 {
    if seeds.is_empty() {
        return f64::INFINITY;
    }
    let mut best = f64::INFINITY;
    for &(sx, sy, _sz) in seeds {
        let dx = point.0 - sx;
        let dy = point.1 - sy;
        let dist = (dx * dx + dy * dy).sqrt();
        best = best.min(dist - 0.5);
    }
    best
}

/// Generate a hexagonal lattice using explicit cell vertices with vertical
/// edges suitable for slicing.
///
/// Each seed point defines the center of a regular hexagon in the X-Y plane.
/// For every vertex of that hexagon a vertical edge is emitted, represented by
/// two vertices at ``z-1`` and ``z+1``.  The returned mesh therefore contains
/// six vertical struts per seed point.
pub fn hex_lattice(seeds: &[(f64, f64, f64)]) -> VoronoiMesh {
    let mut vertices = Vec::new();
    let mut edges = Vec::new();
    for &(cx, cy, cz) in seeds {
        for i in 0..6 {
            let angle = std::f64::consts::FRAC_PI_2 + i as f64 * std::f64::consts::PI / 3.0;
            let x = cx + angle.cos();
            let y = cy + angle.sin();
            let base = vertices.len();
            vertices.push((x, y, cz - 1.0));
            vertices.push((x, y, cz + 1.0));
            edges.push((base, base + 1));
        }
    }
    VoronoiMesh { vertices, edges }
}

static UNIFORM_CELL_CACHE: Lazy<Mutex<HashMap<String, Vec<Vec<(f64, f64, f64)>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn get_uniform_hex_cells(seeds: &[(f64, f64, f64)]) -> Vec<Vec<(f64, f64, f64)>> {
    let key = format!("{:?}", seeds);
    let mut cache = UNIFORM_CELL_CACHE.lock().unwrap();
    if let Some(cells) = cache.get(&key) {
        return cells.clone();
    }
    let cells = Python::with_gil(|py| -> PyResult<Vec<Vec<(f64, f64, f64)>>> {
        let seed_rows: Vec<Vec<f64>> = seeds.iter().map(|&(x, y, z)| vec![x, y, z]).collect();
        let seeds_arr = PyArray2::from_vec2_bound(py, &seed_rows)?;
        let plane = PyArray1::from_vec_bound(py, vec![0.0f64, 0.0, 1.0]);
        let obj = uniform::hex::compute_uniform_cells(
            py,
            seeds_arr.readonly(),
            py.None(),
            plane.readonly(),
            None,
            1e-5,
            false,
            false,
        )?;
        let dict = obj.downcast_bound::<PyDict>(py)?;
        let mut cells = Vec::new();
        for idx in 0..seed_rows.len() {
            if let Some(arr_obj) = dict.get_item(idx)? {
                let arr = arr_obj.downcast::<PyArray2<f64>>()?;
                let arr_ro = arr.readonly();
                let mut verts = Vec::new();
                for row in arr_ro.as_array().outer_iter() {
                    verts.push((row[0], row[1], row[2]));
                }
                cells.push(verts);
            }
        }
        Ok(cells)
    })
    .unwrap_or_default();
    cache.insert(key, cells.clone());
    cells
}

fn uniform_hex_sdf(point: (f64, f64, f64), cells: &[Vec<(f64, f64, f64)>]) -> f64 {
    let mut best = f64::INFINITY;
    for verts in cells {
        let m = verts.len();
        for i in 0..m {
            let a = verts[i];
            let b = verts[(i + 1) % m];
            let v = sub(b, a);
            let w = sub(point, a);
            let c1 = dot(w, v);
            let c2 = dot(v, v);
            let t = if c1 <= 0.0 {
                0.0
            } else if c1 >= c2 {
                1.0
            } else {
                c1 / c2
            };
            let proj = add(a, scale(v, t));
            let diff = sub(point, proj);
            let dist = (diff.0 * diff.0 + diff.1 * diff.1 + diff.2 * diff.2).sqrt();
            if dist < best {
                best = dist;
            }
        }
    }
    best
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

    // Limit seed count via thinning when the input exceeds a reasonable cap.
    let seeds: Vec<(f64, f64, f64)> = if seeds.len() > MAX_VORONOI_SEEDS {
        voronoi::sampling::thin_points(seeds, MAX_VORONOI_SEEDS)
    } else {
        seeds.to_vec()
    };

    let n = seeds.len();
    let vertices: Arc<Mutex<Vec<(f64, f64, f64)>>> = Arc::new(Mutex::new(Vec::new()));
    let quad_to_vertex: Arc<Mutex<HashMap<[usize; 4], usize>>> =
        Arc::new(Mutex::new(HashMap::new()));

    // Estimate an average spacing to drive the grid adjacency pruning.
    let (mut xmin, mut ymin, mut zmin) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let (mut xmax, mut ymax, mut zmax) = (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &(x, y, z) in &seeds {
        xmin = xmin.min(x);
        ymin = ymin.min(y);
        zmin = zmin.min(z);
        xmax = xmax.max(x);
        ymax = ymax.max(y);
        zmax = zmax.max(z);
    }
    let volume = ((xmax - xmin) * (ymax - ymin) * (zmax - zmin)).max(1e-9);
    let spacing = 1.5 * (volume / n as f64).cbrt();

    // Derive candidate neighbor pairs via spatial hashing.
    let pairs =
        voronoi::sampling::prune_adjacency_via_grid(seeds.clone(), spacing).unwrap_or_default();
    let mut adjacency: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for (a, b) in pairs {
        adjacency[a].insert(b);
        adjacency[b].insert(a);
    }

    // Build a k-d tree of seeds for efficient radius queries
    let mut tree: KdTree<f64, u64, 3, 128, u32> = KdTree::with_capacity(n);
    for (idx, &(x, y, z)) in seeds.iter().enumerate() {
        tree.add(&[x, y, z], idx as u64);
    }

    // For each seed, form tetrahedra from localized neighbor sets.
    (0..n).into_par_iter().for_each(|i| {
        let mut neighbors: Vec<usize> = adjacency[i].iter().cloned().filter(|&j| j > i).collect();
        neighbors.sort();
        for a in 0..neighbors.len() {
            for b in (a + 1)..neighbors.len() {
                for c in (b + 1)..neighbors.len() {
                    let j = neighbors[a];
                    let k = neighbors[b];
                    let l = neighbors[c];
                    // All pairs must be adjacent to approximate Delaunay tetrahedra
                    if !(adjacency[j].contains(&k)
                        && adjacency[j].contains(&l)
                        && adjacency[k].contains(&l))
                    {
                        continue;
                    }
                    if let Some(center) = circumcenter(seeds[i], seeds[j], seeds[k], seeds[l]) {
                        let dist2 = norm_sq(sub(center, seeds[i]));
                        let search_radius = (dist2 - 1e-9).max(0.0);
                        let mut valid = true;
                        for nn in tree.within_unsorted::<SquaredEuclidean>(
                            &[center.0, center.1, center.2],
                            search_radius,
                        ) {
                            let m = nn.item as usize;
                            if m == i || m == j || m == k || m == l {
                                continue;
                            }
                            valid = false;
                            break;
                        }
                        if valid {
                            let mut key = [i, j, k, l];
                            key.sort();
                            let mut qtv = quad_to_vertex.lock().unwrap();
                            if !qtv.contains_key(&key) {
                                let mut verts = vertices.lock().unwrap();
                                let idx = verts.len();
                                verts.push(center);
                                qtv.insert(key, idx);
                            }
                        }
                    }
                }
            }
        }
    });

    let vertices = Arc::try_unwrap(vertices).unwrap().into_inner().unwrap();
    let quad_to_vertex = Arc::try_unwrap(quad_to_vertex)
        .unwrap()
        .into_inner()
        .unwrap();

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

    let edge_set: Arc<Mutex<HashSet<(usize, usize)>>> = Arc::new(Mutex::new(HashSet::new()));
    triple_map.par_iter().for_each(|(_, verts)| {
        if verts.len() >= 2 {
            for i in 0..verts.len() {
                for j in (i + 1)..verts.len() {
                    let a = verts[i];
                    let b = verts[j];
                    let e = if a < b { (a, b) } else { (b, a) };
                    edge_set.lock().unwrap().insert(e);
                }
            }
        }
    });

    let mut edges: Vec<(usize, usize)> = Arc::try_unwrap(edge_set)
        .unwrap()
        .into_inner()
        .unwrap()
        .into_iter()
        .collect();
    edges.sort();

    VoronoiMesh { vertices, edges }
}

#[pyfunction]
fn voronoi_mesh_py(
    seeds: Vec<(f64, f64, f64)>,
) -> PyResult<(Vec<(f64, f64, f64)>, Vec<(usize, usize)>)> {
    let mesh = voronoi_mesh(&seeds);
    Ok((mesh.vertices, mesh.edges))
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
    m.add_function(wrap_pyfunction!(voronoi_mesh_py, m)?)?;
    m.add_class::<spatial::octree::OctreeNode>()?;
    m.add_function(wrap_pyfunction!(
        spatial::octree::generate_adaptive_grid,
        m
    )?)?;
    Ok(())
}
