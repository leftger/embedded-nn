//! Integration tests importing and validating standard MLPerf Tiny models from `microflow-rs`.

use std::fs;
use std::path::Path;

#[test]
fn test_import_microflow_sine_model() {
    let path =
        Path::new("/home/usuario/Projects/open-source-repos/microflow-rs/models/sine.tflite");
    if !path.exists() {
        return;
    }
    let bytes = fs::read(path).expect("sine.tflite must be readable");
    let graph =
        embedded_nn_tflite::import_tflite(&bytes).expect("importing sine.tflite must succeed");

    assert!(
        !graph.layers.is_empty(),
        "ModelGraph layers should not be empty"
    );
    let code = embedded_nn_codegen::RustCodeGenerator::new("SineNet").generate(&graph);
    assert!(code.contains("pub struct SineNet"));
    assert!(code.contains("pub const ARENA_SIZE: usize"));
}

#[test]
fn test_import_microflow_speech_model() {
    let path =
        Path::new("/home/usuario/Projects/open-source-repos/microflow-rs/models/speech.tflite");
    if !path.exists() {
        return;
    }
    let bytes = fs::read(path).expect("speech.tflite must be readable");
    let graph =
        embedded_nn_tflite::import_tflite(&bytes).expect("importing speech.tflite must succeed");

    assert!(
        !graph.layers.is_empty(),
        "ModelGraph layers should not be empty"
    );
    let code = embedded_nn_codegen::RustCodeGenerator::new("MicroSpeechNet").generate(&graph);
    assert!(code.contains("pub struct MicroSpeechNet"));
    assert!(code.contains("pub const ARENA_SIZE: usize"));
}

#[test]
fn test_import_microflow_person_detect_model() {
    let path = Path::new(
        "/home/usuario/Projects/open-source-repos/microflow-rs/models/person_detect.tflite",
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
