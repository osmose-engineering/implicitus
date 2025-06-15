use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let proto_dir = manifest_dir.parent().unwrap().join("schema");
    let proto_file = proto_dir.join("implicitus.proto");

    prost_build::compile_protos(&[proto_file], &[proto_dir])
        .expect("Failed to compile protobufs");
}