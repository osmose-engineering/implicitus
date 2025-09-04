use std::path::PathBuf;
use std::fs;

fn main() {
    pyo3_build_config::use_pyo3_cfgs();

    #[cfg(feature = "extension-module")]
    {
        pyo3_build_config::add_extension_module_link_args();
    }
    #[cfg(not(feature = "extension-module"))]
    {
        if let Some(lib_dir) = pyo3_build_config::get().lib_dir.clone() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir);
        }
    }

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

    // Generate Rust constants from shared configuration
    let constants_path = manifest_dir.parent().unwrap().join("constants.json");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let dest = out_dir.join("constants.rs");
    let data = fs::read_to_string(constants_path).expect("read constants.json");
    let value: serde_json::Value = serde_json::from_str(&data).expect("parse constants.json");
    let max = value["MAX_VORONOI_SEEDS"].as_u64().expect("MAX_VORONOI_SEEDS");
    fs::write(dest, format!("pub const MAX_VORONOI_SEEDS: usize = {};", max))
        .expect("write constants.rs");
}