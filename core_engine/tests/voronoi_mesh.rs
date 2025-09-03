use core_engine::voronoi_mesh;

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn nondegenerate_voronoi_mesh() {
    // A simple 3D configuration yielding two Voronoi vertices and one edge
    let seeds = vec![
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.0, 2.0),
        (2.0, 2.0, 3.0),
    ];
    let mesh = voronoi_mesh(&seeds);
    assert_eq!(mesh.vertices.len(), 2);
    assert_eq!(mesh.edges, vec![(0, 1)]);
    let v0 = mesh.vertices[0];
    let v1 = mesh.vertices[1];
    assert!(approx(v0.0, 1.0) && approx(v0.1, 1.0) && approx(v0.2, 1.0));
    assert!(approx(v1.0, 1.3) && approx(v1.1, 1.3) && approx(v1.2, 1.3));
}

#[test]
fn grid_voronoi_counts() {
    // Deterministic grid of points that should yield a regular lattice of Voronoi cells.
    let n = 10u32;
    let mut seeds = Vec::new();
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                seeds.push((x as f64, y as f64, z as f64));
            }
        }
    }
    let mesh = voronoi_mesh(&seeds);
    // Counts observed for a 10x10x10 grid with the localized tetrahedra approach.
    assert_eq!(mesh.vertices.len(), 42282);
    assert_eq!(mesh.edges.len(), 392688);
}
