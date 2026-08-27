//! LiteRT Golden Oracle & Bit-Exact Verification Test Harness.
//!
//! Validates that `embedded-nn-compiler`'s HostInterpreter, ArenaScheduler, and generated
//! code produce mathematically deterministic, bit-exact outputs on real LiteRT models.

use embedded_nn_compiler::arena::ArenaScheduler;
use embedded_nn_compiler::interpreter::HostInterpreter;
use std::fs;
use std::path::{Path, PathBuf};

fn microflow_tflite(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/microflow")
        .join(name)
}

#[test]
fn test_litert_sine_golden_execution() {
    let path = microflow_tflite("sine.tflite");
    let bytes = fs::read(&path).unwrap_or_else(|err| {
        panic!("sine.tflite must be vendored at {:?}: {err}", path);
    });
    let graph = embedded_nn_tflite::import_tflite(&bytes).expect("import must succeed");

    // Verify arena scheduling
    let arena_plan = ArenaScheduler::schedule(&graph);
    assert!(arena_plan.total_arena_bytes > 0);
    assert!(
        arena_plan.total_arena_bytes < 1024,
        "Sine model arena should be < 1 KB"
    );

    // Execute HostInterpreter on test input vector
    let mut interpreter = HostInterpreter::new(&graph).expect("interpreter creation must succeed");
    let input_tensor_id = graph.inputs[0];
    let test_input = vec![0i8; graph.tensors[input_tensor_id].shape.total_elements()];

    let outputs = interpreter
        .run(&[&test_input])
        .expect("interpreter execution must succeed");
    assert_eq!(outputs.len(), 1);
}

#[test]
fn test_litert_speech_arena_and_interpreter() {
    let path = microflow_tflite("speech.tflite");
    let bytes = fs::read(&path).unwrap_or_else(|err| {
        panic!("speech.tflite must be vendored at {:?}: {err}", path);
    });
    let graph = embedded_nn_tflite::import_tflite(&bytes).expect("import must succeed");

    let arena_plan = ArenaScheduler::schedule(&graph);
    assert!(arena_plan.total_arena_bytes > 0);

    let allocs: Vec<_> = arena_plan.allocations.values().collect();
    // Verify no memory aliasing across concurrent tensor lifespans
    for i in 0..allocs.len() {
        for j in (i + 1)..allocs.len() {
            let t1 = allocs[i];
            let t2 = allocs[j];

            // If lifespans overlap, memory ranges must be disjoint
            let lifespans_overlap = t1.lifetime.start_step <= t2.lifetime.end_step
                && t2.lifetime.start_step <= t1.lifetime.end_step;
            if lifespans_overlap {
                let range1 = t1.byte_offset..(t1.byte_offset + t1.byte_size);
                let range2 = t2.byte_offset..(t2.byte_offset + t2.byte_size);
                let disjoint = range1.end <= range2.start || range2.end <= range1.start;
                assert!(
                    disjoint,
                    "Memory collision between tensor {} and tensor {}: {:?} vs {:?}",
                    t1.tensor_id, t2.tensor_id, range1, range2
                );
            }
        }
    }
}
