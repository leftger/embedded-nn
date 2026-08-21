//! End-to-end ground-truth check: import a *real* TensorFlow-generated `.tflite` model
//! (`fixtures/dense_mlp.tflite`, see `fixtures/README.md`), generate `#![no_std]` Rust code for
//! it, compile+run that code in a scratch crate, and diff its `predict()` output against real
//! `tf.lite.Interpreter` reference output recorded in `fixtures/test_vectors.txt`.
//!
//! This is what caught a real bug: `emit_rust.rs`'s ReLU activation clamp used to hardcode
//! `Activation::new(0, i8::MAX)`, ignoring the output tensor's zero-point, which is wrong for
//! any asymmetrically-quantized (nonzero zero-point) output — the common case for real models.

use std::fs;
use std::path::Path;
use std::process::Command;

const RUNNER_MAIN: &str = r#"
mod model;
use model::DenseMlpNet;

fn parse_i8_list(s: &str) -> Vec<i8> {
    s.split(',').map(|v| v.trim().parse::<i32>().unwrap() as i8).collect()
}

fn main() {
    let test_vectors = include_str!("../test_vectors.txt");
    let mut mismatches = 0usize;
    let mut total = 0usize;
    for line in test_vectors.lines() {
        let parts: Vec<&str> = line.split('|').collect();
        let input = parse_i8_list(parts[0]);
        let expected = parse_i8_list(parts[1]);

        let mut arena = [0u8; DenseMlpNet::ARENA_SIZE];
        let output = DenseMlpNet::predict(&input, &mut arena).unwrap();

        total += 1;
        if output != expected.as_slice() {
            mismatches += 1;
            println!("MISMATCH: got {:?} expected {:?}", output, expected);
        }

        let input_f32: Vec<f32> = input
            .iter()
            .map(|value| {
                (*value as i32 - DenseMlpNet::INPUT_ZERO_POINT) as f32
                    * DenseMlpNet::INPUT_SCALE
            })
            .collect();
        let expected_f32: Vec<f32> = expected
            .iter()
            .map(|value| {
                (*value as i32 - DenseMlpNet::OUTPUT_ZERO_POINT) as f32
                    * DenseMlpNet::OUTPUT_SCALE
            })
            .collect();
        let mut quantized_input = vec![0i8; DenseMlpNet::INPUT_DIM];
        let mut output_f32 = vec![0.0f32; DenseMlpNet::OUTPUT_DIM];
        DenseMlpNet::predict_f32(
            &input_f32,
            &mut quantized_input,
            &mut arena,
            &mut output_f32,
        )
        .unwrap();
        if output_f32 != expected_f32 {
            mismatches += 1;
            println!(
                "F32 MISMATCH: got {:?} expected {:?}",
                output_f32, expected_f32
            );
        }
    }
    println!("{}/{} exact matches", total - mismatches, total);
    if mismatches > 0 {
        std::process::exit(1);
    }
}
"#;

#[test]
fn real_tflite_model_predict_matches_reference_interpreter_output() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixtures = manifest_dir.join("fixtures");

    let bytes = fs::read(fixtures.join("dense_mlp.tflite")).expect("fixture .tflite must exist");
    let graph = embedded_nn_tflite::import_tflite(&bytes).expect("import should succeed");
    let code = embedded_nn_codegen::RustCodeGenerator::new("DenseMlpNet").generate(&graph);

    // Deliberately placed outside the workspace tree (not under `target/`): a nested Cargo
    // manifest inside the workspace confuses cargo's workspace resolution ("believes it's in a
    // workspace when it's not").
    let scratch = std::env::temp_dir().join("embedded_nn_tflite_real_model_accuracy_check");
    let _ = fs::remove_dir_all(&scratch);
    fs::create_dir_all(scratch.join("src")).unwrap();

    let embedded_nn_path = manifest_dir
        .parent()
        .unwrap()
        .join("embedded-nn")
        .canonicalize()
        .unwrap();

    fs::write(
        scratch.join("Cargo.toml"),
        format!(
            "[package]\nname = \"real_model_accuracy_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\nembedded-nn = {{ path = {:?} }}\n\n[[bin]]\nname = \"runner\"\npath = \"src/main.rs\"\n",
            embedded_nn_path
        ),
    )
    .unwrap();

    fs::write(scratch.join("src").join("model.rs"), &code).unwrap();
    fs::write(scratch.join("src").join("main.rs"), RUNNER_MAIN).unwrap();
    fs::copy(
        fixtures.join("test_vectors.txt"),
        scratch.join("test_vectors.txt"),
    )
    .unwrap();

    let output = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(scratch.join("Cargo.toml"))
        .output()
        .expect("failed to invoke cargo for the standalone runner crate");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "standalone runner crate failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("6/6 exact matches"),
        "expected bit-exact match against real TFLite reference output, got:\n{stdout}"
    );

    let _ = fs::remove_dir_all(&scratch);
}
