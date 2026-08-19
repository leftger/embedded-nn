# embedded-nn

[![crates.io](https://img.shields.io/crates/v/embedded-nn.svg)](https://crates.io/crates/embedded-nn)
[![docs.rs](https://img.shields.io/docsrs/embedded-nn)](https://docs.rs/embedded-nn)
[![CI](https://github.com/leftger/embedded-nn/actions/workflows/ci.yml/badge.svg)](https://github.com/leftger/embedded-nn/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

A pure Rust, `#![no_std]` neural network inference runtime, compiler, and TinyML platform for microcontrollers and embedded targets, ported from and inspired by ARM's **CMSIS-NN**, **TensorFlow Lite Micro**, and **MicroFlow**.

📖 **[Read the End-to-End TinyML Guide (Data Collection -> Training -> Deployment)](docs/END_TO_END_TINYML_GUIDE.md)**

---

## Workspace Crates

| Crate | Description |
| :--- | :--- |
| [`crates/embedded-nn`](crates/embedded-nn) | Core `#![no_std]` runtime & quantized neural kernels (`s8`, `s16`, `s4` sub-byte). |
| [`crates/embedded-nn-compiler`](crates/embedded-nn-compiler) | Model graph IR, Ahead-of-Time static SRAM arena scheduler, and PTQ/QAT quantizer. |
| [`crates/embedded-nn-codegen`](crates/embedded-nn-codegen) | Zero-allocation `#![no_std]` Rust code & static const weight array emitter. |
| [`crates/embedded-nn-macros`](crates/embedded-nn-macros) | Procedural macro `#[embedded_nn_model("...")]` for compile-time model embedding. |
| [`crates/embedded-nn-live`](crates/embedded-nn-live) | USB (`nusb`) streaming telemetry & Hardware-in-the-Loop (HIL) verification protocol. |
| [`crates/embedded-nn-cli`](crates/embedded-nn-cli) | `enn` CLI tool for static memory profiling, codegen, and device discovery. |
| [`crates/embedded-nn-studio`](crates/embedded-nn-studio) | Interactive `eframe`/`egui` desktop TinyML Studio (Ingest -> DSP -> Train -> Arena -> Deploy). |

---

## Key Features

- **`#![no_std]` Bare-Metal Support**: Built for bare-metal targets (ARM Cortex-M, RISC-V, ESP32) with **zero dynamic heap allocations (`alloc`)**.
- **Static Arena Memory Scheduler**: Computes tensor birth/death intervals to reuse SRAM memory buffers ahead of time, minimizing peak SRAM footprint.
- **4-Bit Sub-Byte (`s4`) Quantization**: Pack two signed 4-bit weights into a single byte for 50% Flash savings (`fully_connected_s4`, `convolve_s4`).
- **Target SIMD Hooks**: Vectorized 4-way and 8-way dot-product abstractions (`vec_dot_s8`, `vec_dot_s16`) optimized for compiler auto-vectorization and hardware SIMD acceleration.
- **Core Neural Operators**:
  - **Convolution**: 2D Conv (`s8`, `s4`, `f32`), 1x1 Fast Conv, Depthwise Conv.
  - **Dense / Fully Connected**: Matrix multiplication (`s8`, `s16`, `s4`, `f32`).
  - **Activations**: ReLU, ReLU6, LeakyReLU, Sigmoid, Tanh, `FusedActivation` enum.
  - **Pooling**: Max Pooling 2D, Average Pooling 2D (`s8`, `s16`).
  - **Softmax**: Fixed-point exponential Softmax (`s8`, `s16`) & float Softmax (`f32`).
  - **Recurrent**: Unidirectional `LSTM` cell (`lstm_step_s8_s16`) and `SVDF` layer.

---

## Quickstart

### 1. Compile-Time Model Embedding
```rust
use embedded_nn_macros::embedded_nn_model;

#[embedded_nn_model("models/gesture_classifier.json")]
pub struct GestureClassifier;

fn run_inference() {
    let mut arena = [0u8; GestureClassifier::ARENA_SIZE];
    let sensor_features = [12i8, -4, 30, 2];
    
    let logits = GestureClassifier::predict(&sensor_features, &mut arena).unwrap();
}
```

### 2. Static Memory Profiling (CLI)
```bash
cargo run -p embedded-nn-cli -- profile --model models/gesture_classifier.json
```

### 3. Launching the TinyML Studio
```bash
cargo run -p embedded-nn-studio
```

---

## License

Dual-licensed under either of:
- MIT License ([`LICENSE-MIT`](./LICENSE-MIT))
- Apache License, Version 2.0 ([`LICENSE-APACHE`](./LICENSE-APACHE))
