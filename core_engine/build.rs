use std::path::PathBuf;

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
}