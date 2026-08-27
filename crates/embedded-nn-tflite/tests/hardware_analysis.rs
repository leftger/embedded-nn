//! Host-side hardware-analysis numbers for MicroFlow's public models.
//!
//! The checked-in `analysis/hardware.csv` is the comparable TFLM-style table
//! (accuracy / flash weights / SRAM arena). QEMU firmware size is filled from
//! `examples/qemu-lm3s6965`; STM32 rows stay methodology-only until a board run.

use embedded_nn::{dequantize_s8_to_f32, quantize_f32_to_s8};
use embedded_nn_compiler::arena::ArenaScheduler;
use embedded_nn_compiler::interpreter::HostInterpreter;
use std::fs;
use std::path::{Path, PathBuf};

fn microflow_tflite(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/microflow")
        .join(name)
}

fn import(name: &str) -> embedded_nn_compiler::ir::ModelGraph {
    let path = microflow_tflite(name);
    let bytes = fs::read(&path).unwrap_or_else(|err| {
        panic!("{name} must be vendored at {:?}: {err}", path);
    });
    embedded_nn_tflite::import_tflite(&bytes).unwrap_or_else(|err| {
        panic!("importing {name} must succeed: {err}");
    })
}

fn sine_mae(graph: &embedded_nn_compiler::ir::ModelGraph) -> f32 {
    let input = &graph.tensors[graph.inputs[0]];
    let output = &graph.tensors[graph.outputs[0]];
    let mut interpreter = HostInterpreter::new(graph).unwrap();
    let mut abs_err = 0.0f32;
    let samples = 100u32;
    for i in 0..samples {
        let x = (i as f32) * (core::f32::consts::PI * 2.0) / (samples as f32);
        let q_in = quantize_f32_to_s8(x, input.quant.scale, input.quant.zero_point);
        let outs = interpreter.run(&[&[q_in]]).unwrap();
        let y = dequantize_s8_to_f32(outs[0][0], output.quant.scale, output.quant.zero_point);
        abs_err += (y - x.sin()).abs();
    }
    abs_err / samples as f32
}

#[test]
fn microflow_models_have_scheduled_arenas_and_sine_tracks_sin() {
    let sine = import("sine.tflite");
    let constructed = {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/constructed/sine_fc_int8.tflite");
        let bytes = fs::read(&path).unwrap();
        embedded_nn_tflite::import_tflite(&bytes).unwrap()
    };
    let constructed_arena = ArenaScheduler::schedule(&constructed).total_arena_bytes;
    let speech = import("speech.tflite");
    let person = import("person_detect.tflite");

    let sine_arena = ArenaScheduler::schedule(&sine).total_arena_bytes;
    let speech_arena = ArenaScheduler::schedule(&speech).total_arena_bytes;
    let person_arena = ArenaScheduler::schedule(&person).total_arena_bytes;
    let mae = sine_mae(&sine);

    assert!(sine_arena > 0 && sine_arena < 1024);
    assert!(speech_arena > 0);
    assert!(person_arena > 0);
    assert!(
        mae < 0.2,
        "sine MAE vs sin(x) on [0, 2π] should stay under 0.2, got {mae}"
    );

    let csv_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../analysis/hardware.csv");
    let csv = fs::read_to_string(&csv_path).unwrap_or_else(|err| {
        panic!("analysis/hardware.csv must exist at {:?}: {err}", csv_path);
    });
    assert_csv_host_row(
        &csv,
        "sine.tflite",
        sine.total_weights_size_bytes(),
        sine_arena,
        Some(mae),
    );
    assert_csv_host_row(
        &csv,
        "speech.tflite",
        speech.total_weights_size_bytes(),
        speech_arena,
        None,
    );
    assert_csv_host_row(
        &csv,
        "person_detect.tflite",
        person.total_weights_size_bytes(),
        person_arena,
        None,
    );
    let qemu = csv
        .lines()
        .find(|line| line.starts_with("sine_fc_int8.tflite,qemu-lm3s6965evb,"))
        .expect("missing QEMU row");
    let qemu_cols: Vec<&str> = qemu.split(',').collect();
    assert_eq!(
        qemu_cols[4].parse::<usize>().unwrap(),
        constructed.total_weights_size_bytes()
    );
    assert_eq!(qemu_cols[5].parse::<usize>().unwrap(), constructed_arena);
}

fn assert_csv_host_row(csv: &str, model: &str, weights: usize, arena: usize, mae: Option<f32>) {
    let row = csv
        .lines()
        .find(|line| line.starts_with(&format!("{model},host-interpreter,")))
        .unwrap_or_else(|| panic!("missing host-interpreter row for {model}"));
    let cols: Vec<&str> = row.split(',').collect();
    assert_eq!(
        cols[4].parse::<usize>().unwrap(),
        weights,
        "{model} weights"
    );
    assert_eq!(cols[5].parse::<usize>().unwrap(), arena, "{model} arena");
    if let Some(expected_mae) = mae {
        let recorded = cols[6].parse::<f32>().unwrap();
        assert!(
            (recorded - expected_mae).abs() < 1e-4,
            "{model} MAE {recorded} vs {expected_mae}"
        );
    }
}
