use core_engine::implicitus::{node::Body, primitive::Shape, Model, Node, Primitive, Sphere};
use core_engine::slice::{slice_model, SliceConfig};
use serde_json::json;

#[path = "../src/bin/slicer_server.rs"]
mod slicer_server;

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn seeds_from_multiple_blocks_affect_slice() {
    // Seeds split across separate infill/lattice blocks
    let model_json = json!({
        "infill": {
            "pattern": "voronoi",
            "mode": "uniform",
            "seed_points": [[0.0,0.0,0.0],[2.0,0.0,0.0]]
        },
        "extra": {
            "lattice": {
                "mode": "uniform",
                "seed_points": [
                    [0.0,2.0,0.0],
                    [0.0,0.0,2.0],
                    [2.0,2.0,3.0]
                ]
            }
        }
    });

    let (seeds, pattern, _, mode) = slicer_server::parse_infill(&model_json);
    assert_eq!(seeds.len(), 5);
    assert_eq!(pattern.as_deref(), Some("voronoi"));
    assert_eq!(mode.as_deref(), Some("uniform"));

    // Basic spherical model
    let mut model = Model::default();
    model.id = "multi_infill".into();
    let sphere = Sphere { radius: 2.0 };
    let mut prim = Primitive::default();
    prim.shape = Some(Shape::Sphere(sphere));
    let mut node = Node::default();
    node.body = Some(Body::Primitive(prim));
    model.root = Some(node);

    let config = SliceConfig {
        z: 1.1,
        x_min: -2.0,
        x_max: 2.0,
        y_min: -2.0,
        y_max: 2.0,
        nx: 3,
        ny: 3,
        seed_points: seeds,
        infill_pattern: pattern,
        wall_thickness: 0.0,
        mode: None,
    };

    let result = slice_model(&model, &config);
    assert!(!result.segments.is_empty());
    let ((sx, sy), (ex, ey)) = result.segments[0];
    assert!(approx(sx, 1.1) && approx(sy, 1.1));
    assert!(approx(sx, ex) && approx(sy, ey));
}
