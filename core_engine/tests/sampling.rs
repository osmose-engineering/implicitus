use core_engine::{voronoi_mesh, MAX_VORONOI_SEEDS};
use core_engine::voronoi::sampling::thin_points;
use rand::Rng;

fn centroid(points: &[(f64, f64, f64)]) -> (f64, f64, f64) {
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    for &(x, y, z) in points {
        cx += x;
        cy += y;
        cz += z;
    }
    let n = points.len() as f64;
    (cx / n, cy / n, cz / n)
}

fn dist(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2)).sqrt()
}

#[test]
fn thinning_reduces_point_count_and_preserves_centroid() {
    let mut rng = rand::thread_rng();
    let seeds: Vec<_> = (0..1000)
        .map(|_| (rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>()))
        .collect();
    let thinned = thin_points(&seeds, 100);
    assert!(thinned.len() <= 100);
    let c_orig = centroid(&seeds);
    let c_thin = centroid(&thinned);
    assert!(dist(c_orig, c_thin) < 0.1);
}

#[test]
fn voronoi_mesh_respects_seed_limit() {
    let mut rng = rand::thread_rng();
    let seeds: Vec<_> = (0..100)
        .map(|_| (rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>()))
        .collect();
    let mesh = voronoi_mesh(&seeds);
    let max_verts = (MAX_VORONOI_SEEDS
        * (MAX_VORONOI_SEEDS - 1)
        * (MAX_VORONOI_SEEDS - 2)
        * (MAX_VORONOI_SEEDS - 3))
        / 24;
    assert!(mesh.vertices.len() <= max_verts);
}
