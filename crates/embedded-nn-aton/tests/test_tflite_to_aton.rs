use embedded_nn_aton::compiler::AtonCompiler;
use std::fs;
use std::path::PathBuf;

#[test]
fn test_tflite_dense_mlp_to_aton() {
    let tflite_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../embedded-nn-tflite/fixtures/dense_mlp.tflite");
    let bytes = fs::read(&tflite_path).expect("failed to read tflite fixture");

    let graph = embedded_nn_tflite::import_tflite(&bytes).expect("tflite import failed");

    let compiler = AtonCompiler::new().with_symbols("_in", "_out");
    let compiled = compiler.compile(&graph).expect("aton compilation failed");

    assert_eq!(compiled.input_size_bytes, 16);
    assert_eq!(compiled.output_size_bytes, 4);

    let words_u64 = compiled
        .build_blob_u64()
        .expect("failed to build u64 container");
    assert!(!words_u64.is_empty());

    println!("compiled name: {}", compiled.name);
    let c_header = compiled.emit_c_header().expect("emit C failed");
    println!("c_header:\n{}", c_header);
    assert!(c_header.contains("_ec_blob_"));

    let rust_code = compiled.emit_rust_code().expect("emit Rust failed");
    println!("rust_code:\n{}", rust_code);
    assert!(rust_code.contains("pub static EC_CONTAINER_"));
}
