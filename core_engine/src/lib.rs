// src/lib.rs

// Import the generated Protobuf definitions
pub mod implicitus {
    include!(concat!(env!("OUT_DIR"), "/implicitus.rs"));
}

use implicitus::Model;
use implicitus::node::Body;
use implicitus::primitive::Shape;

// A very basic SDF evaluator that only handles a single sphere at the root.
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
                    _ => {},
                }
            }
        }
    }
    // If nothing matches, return a large positive value (empty space)
    f64::MAX
}

pub mod slice;
