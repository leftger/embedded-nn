//! Integration tests importing official Google TFLite-Micro models.
//!
//! Validates that `embedded-nn-tflite` can import, schedule arena memory, and generate
//! zero-allocation `#![no_std]` Rust code for official TFLite-Micro benchmark models.

use std::fs;
use std::path::Path;

#[test]
fn test_import_tflite_micro_keyword_scrambled() {
    let path = Path::new(
        "/home/usuario/Projects/open-source-repos/tflite-micro/tensorflow/lite/micro/models/keyword_scrambled_8bit.tflite",
    );
    if !path.exists() {
        return;
    }
    let bytes = fs::read(path).expect("keyword_scrambled_8bit.tflite must be readable");
    let graph = embedded_nn_tflite::import_tflite(&bytes)
        .expect("importing keyword_scrambled_8bit.tflite must succeed");

    assert!(
        !graph.layers.is_empty(),
        "ModelGraph layers should not be empty"
    );
    let code = embedded_nn_codegen::RustCodeGenerator::new("KeywordScrambledNet").generate(&graph);
    assert!(code.contains("pub struct KeywordScrambledNet"));
    assert!(code.contains("pub const ARENA_SIZE: usize"));
}

#[test]
fn test_import_tflite_micro_person_detect() {
    let path = Path::new(
        "/home/usuario/Projects/open-source-repos/tflite-micro/tensorflow/lite/micro/models/person_detect.tflite",
    );
    if !path.exists() {
        return;
    }
    let bytes = fs::read(path).expect("person_detect.tflite must be readable");
    let graph = embedded_nn_tflite::import_tflite(&bytes)
        .expect("importing person_detect.tflite must succeed");

    assert!(
        !graph.layers.is_empty(),
        "ModelGraph layers should not be empty"
    );
    let code = embedded_nn_codegen::RustCodeGenerator::new("PersonDetectNet").generate(&graph);
    assert!(code.contains("pub struct PersonDetectNet"));
    assert!(code.contains("pub const ARENA_SIZE: usize"));
}
