use core_engine::voronoi_mesh;

#[test]
fn triangle_mesh_edges() {
    let seeds = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)];
    let mesh = voronoi_mesh(&seeds);
    assert_eq!(mesh.vertices.len(), 3);
    assert_eq!(mesh.edges, vec![(0,1),(1,2),(2,0)]);
}
