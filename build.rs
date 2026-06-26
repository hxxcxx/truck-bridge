//! Build script: regenerate `include/truck_bridge.h` from the Rust sources via
//! cbindgen on every build. The header is a build product but is committed so
//! consumers can consume it without running cargo.

use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("include");
    let out_file = out_dir.join("truck_bridge.h");

    std::fs::create_dir_all(&out_dir).expect("create include/ dir");

    // Re-run if anything under src/ changes (cbindgen reads the sources).
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=Cargo.toml");

    let cfg = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml"))
        .expect("read cbindgen.toml");

    cbindgen::generate_with_config(&crate_dir, cfg)
        .expect("cbindgen generate")
        .write_to_file(&out_file);
}
