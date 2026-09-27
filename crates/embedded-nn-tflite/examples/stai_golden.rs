//! Host-side golden-reference generator for ST Edge AI / STM32N6 models.
//!
//! Imports a quantized `.tflite` with `embedded-nn-tflite` and runs it on the
//! host integer interpreter for a fixed, deterministic input, printing the
//! int8 logits of the last `FULLY_CONNECTED` layer. Those are exactly the values
//! an STM32N6 Neural-ART (ATON) execution of the same model writes to its
//! logits buffer, so they make a hardware-independent oracle for the NPU.
//!
//! ```text
//! cargo run -p embedded-nn-tflite --example stai_golden -- <model.tflite> [input.rgb]
//! ```
//!
//! With no `input.rgb` a deterministic synthetic pattern (`(i * 7) % 251` over
//! `1x96x96x3` uint8) is used. Pass a raw `RGB8` file of the model's input
//! dimensions (e.g. `assets/mona_lisa_96.rgb`) to evaluate a real image instead.

use embedded_nn_compiler::interpreter::HostInterpreter;
use embedded_nn_compiler::ir::ModelGraph;

/// Deterministic synthetic input (`1x96x96x3` uint8, NHWC).
fn synthetic_input(len: usize) -> Vec<i8> {
    (0..len).map(|i| (((i * 7) % 251) as u8) as i8).collect()
}

/// Load a raw `RGB8` file and convert it to the interpreter's int8 domain.
///
/// The importer rewrites UINT8 tensors to INT8 by subtracting 128 from *both*
/// the constant values and the zero-points (see `embedded-nn-tflite`'s module
/// docs). Activations are not stored in the model, so the caller has to perform
/// the same shift: a uint8 pixel `v` becomes `v - 128`.
///
/// Note this is exactly equivalent to the raw uint8 arithmetic an accelerator
/// performs: `(v - 128) - (zp - 128) == v - zp`. So the file itself stays raw
/// uint8 (what a device like STM32N6's ATON consumes) and only the host
/// interpreter sees the centred form.
fn load_raw_rgb(path: &str) -> Vec<i8> {
    std::fs::read(path)
        .expect("read raw RGB")
        .into_iter()
        .map(|b| (b as i16 - 128) as i8)
        .collect()
}

fn run(graph: &ModelGraph, input: &[i8], tag: &str) -> Option<Vec<i8>> {
    let mut interpreter = HostInterpreter::new(graph).ok()?;
    match interpreter.run(&[input]) {
        Ok(outputs) => {
            let first = outputs.into_iter().next().unwrap_or_default();
            println!("{tag}: {:?}", first);
            Some(first)
        }
        Err(err) => {
            println!("{tag}: inference failed: {err:?}");
            None
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: stai_golden <model.tflite> [input.rgb]");
    let input = match args.next() {
        Some(rgb) => {
            println!("input: {rgb} (raw RGB8)");
            load_raw_rgb(&rgb)
        }
        None => {
            println!("input: synthetic (i * 7) % 251");
            synthetic_input(96 * 96 * 3)
        }
    };

    let bytes = std::fs::read(&path).expect("read model");
    let mut graph = embedded_nn_tflite::import_tflite(&bytes).expect("import tflite");

    run(&graph, &input, "full model output");

    // Truncate at the last FullyConnected so the reference matches the NPU's
    // pre-softmax logits buffer rather than the dequantized graph output.
    if let Some(fc) = graph
        .layers
        .iter()
        .rev()
        .find(|layer| format!("{:?}", layer.op).contains("FullyConnected"))
    {
        let outputs = fc.outputs.clone();
        println!("truncating graph at layer outputs {outputs:?}");
        graph.outputs = outputs;
        if let Some(logits) = run(&graph, &input, "pre-softmax logits (golden)") {
            let argmax = logits
                .iter()
                .enumerate()
                .max_by_key(|(_, v)| **v)
                .map(|(i, _)| i)
                .unwrap_or(0);
            let min = logits.iter().copied().min().unwrap_or(0);
            let max = logits.iter().copied().max().unwrap_or(0);
            println!(
                "argmax = {argmax} (spread {min}..{max}, range {})",
                max as i32 - min as i32
            );
        }
    } else {
        println!("no FullyConnected layer found; graph is already pre-softmax");
    }
}
