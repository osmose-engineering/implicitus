pub fn voronoi_sdf(point: (f64, f64, f64), seeds: &[(f64, f64, f64)]) -> f64 {
    if seeds.len() < 2 {
        return f64::INFINITY;
    }
    let mut d1 = f64::INFINITY;
    let mut d2 = f64::INFINITY;
    for &(sx, sy, sz) in seeds {
        let dx = point.0 - sx;
        let dy = point.1 - sy;
        let dz = point.2 - sz;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist < d1 {
            d2 = d1;
            d1 = dist;
        } else if dist < d2 {
            d2 = dist;
        }
    }
    (d2 - d1) * 0.5
}
