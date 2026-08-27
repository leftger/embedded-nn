//! Integration tests importing and validating MicroFlow's public `.tflite` models.

use std::fs;
use std::path::{Path, PathBuf};

fn microflow_tflite(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/microflow")
        .join(name)
}

fn import_and_codegen(name: &str, struct_name: &str) {
    let path = microflow_tflite(name);
    let bytes = fs::read(&path).unwrap_or_else(|err| {
        panic!("{} must be vendored at {:?}: {err}", name, path);
    });
    let graph = embedded_nn_tflite::import_tflite(&bytes).unwrap_or_else(|err| {
        panic!("importing {name} must succeed: {err}");
    });

    assert!(
        !graph.layers.is_empty(),
        "{name}: ModelGraph layers should not be empty"
    );
    let code = embedded_nn_codegen::RustCodeGenerator::new(struct_name).generate(&graph);
    assert!(code.contains(&format!("pub struct {struct_name}")));
    assert!(code.contains("pub const ARENA_SIZE: usize"));
    assert!(code.contains("pub const INPUT_SHAPE: [usize; 4]"));
    assert!(code.contains("pub fn predict_from_f32("));
}

#[test]
fn test_import_microflow_sine_model() {
    import_and_codegen("sine.tflite", "SineNet");
}

#[test]
fn test_import_microflow_speech_model() {
    import_and_codegen("speech.tflite", "MicroSpeechNet");
}

#[test]
fn test_import_microflow_person_detect_model() {
    import_and_codegen("person_detect.tflite", "PersonDetectNet");
}
