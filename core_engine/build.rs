use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let proto_dir = manifest_dir.parent().unwrap().join("schema");
    let proto_file = proto_dir.join("implicitus.proto");

    let mut config = prost_build::Config::new();
    // Derive Serialize/Deserialize for all generated types so we can
    // convert JSON models directly into protobuf structs.
    config.type_attribute(
        ".",
        "#[derive(serde::Serialize, serde::Deserialize)]",
    );

    config
        .compile_protos(&[proto_file], &[proto_dir])
        .expect("Failed to compile protobufs");
}