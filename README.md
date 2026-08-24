# embedded-nn

<p align="center">
  <img src="assets/aztec_rustacean.png" alt="embedded-nn" width="100%">
</p>

[![crates.io](https://img.shields.io/crates/v/embedded-nn.svg)](https://crates.io/crates/embedded-nn)
[![docs.rs](https://img.shields.io/docsrs/embedded-nn)](https://docs.rs/embedded-nn)
[![CI](https://github.com/leftger/embedded-nn/actions/workflows/ci.yml/badge.svg)](https://github.com/leftger/embedded-nn/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

A pure Rust, `#![no_std]` neural network inference runtime, ahead-of-time (AOT) compiler, and TinyML platform for microcontrollers and edge silicon. Designed with zero dynamic allocations, static interval-colored SRAM memory reuse, sub-byte LUT quantization, and functional safety integrity checks.

Inspired by and synergized with ARM's **CMSIS-NN**, Google's **LiteRT / TensorFlow Lite Micro**, and **MicroFlow**.

📖 **[Read the End-to-End TinyML Guide (Data Collection -> Training -> Deployment)](docs/END_TO_END_TINYML_GUIDE.md)**  
📖 **[Read the PyTorch -> LiteRT -> embedded-nn Export Workflow](docs/LITERT_PYTORCH_WORKFLOW.md)**  
📖 **[Read the Live Hardware-in-the-Loop Protocol Specification](docs/LIVE_PROTOCOL.md)**  

---

## Workspace Crates

| Crate | Description |
| :--- | :--- |
| [`crates/embedded-nn`](crates/embedded-nn) | Core `#![no_std]` runtime: SIMD/DSP-unrolled quantized kernels (`s8`, `s16`, `s4`), nonlinear Codebook LUT quantization, ISO 26262 Flash CRC32 & arena canaries (`safety`), and TinyML anomaly detectors (`anomaly`). |
| [`crates/embedded-nn-compiler`](crates/embedded-nn-compiler) | Model graph IR, ahead-of-time interval-colored SRAM arena scheduler, quantization math, and host integer interpreter. |
| [`crates/embedded-nn-codegen`](crates/embedded-nn-codegen) | Standalone `#![no_std]` Rust code emitter, C99 standalone header generator (`model.h`), and 1-Click CMake / Makefile project bundler (`bundle`). |
| [`crates/embedded-nn-litert-plugin`](crates/embedded-nn-litert-plugin) | Google LiteRT C ABI Compiler Plugin (`libLiteRtCompilerPlugin_embedded_nn.so`) for native LiteRT model compilation. |
| [`crates/embedded-nn-tflite`](crates/embedded-nn-tflite) | Upstream TFLite & LiteRT v2 importer with support for MLPerf / TFLM benchmark models (`keyword_scrambled_8bit.tflite`, `person_detect.tflite`, etc.). |
| [`crates/embedded-nn-train`](crates/embedded-nn-train) | Host Burn QAT/PTQ trainer, SpecAugment, and Auto-TinyML Pareto Frontier Optimizer (`pareto`). |
| [`crates/embedded-nn-studio`](crates/embedded-nn-studio) | Interactive Desktop & WebAssembly (WASM) TinyML Studio: 3D gesture visualizer, Mel DSP, Burn training, Pareto trade-off explorer, static vs dynamic arena comparator, and USB-HS live inspector. |
| [`crates/embedded-nn-live`](crates/embedded-nn-live) | Binary USB-HS / UART HIL streaming protocol (`0xE6 0x4E` frames, CRC-16, vendor bulk `1209:e612`) and multi-modal 6-DOF / 9-DOF dataset parser. |
| [`crates/embedded-nn-cli`](crates/embedded-nn-cli) | `enn` CLI for memory profiling, codegen, TFLite ingest, dataset validation, and HIL test runner. |
| [`crates/embedded-nn-macros`](crates/embedded-nn-macros) | Procedural macro `#[embedded_nn_model("...")]` for compile-time model embedding and zero-allocation execution. |

---

## Architecture & End-to-End Pipeline

```mermaid
flowchart LR
    subgraph Authoring ["1. Authoring & Training"]
        PyTorch["PyTorch / LiteRT / Burn"] --> Model["Quantized Model (.tflite / .json)"]
    end

    subgraph Compiler ["2. embedded-nn AOT Engine"]
        Model --> IR["ModelGraph IR"]
        IR --> Pareto["Auto-TinyML Pareto Search"]
        IR --> Arena["Static Interval Arena Scheduler"]
    end

    subgraph Targets ["3. Deployment Targets"]
        Arena --> RustCodegen["Rust #![no_std] Crate"]
        Arena --> CCodegen["C99 Standalone Header (.h)"]
        Arena --> Plugin["LiteRT Compiler Plugin (.so)"]
    end

    subgraph Hardware ["4. Silicon Execution"]
        RustCodegen --> STM32["STM32WBA65RI / Cortex-M33 / M4 / M7 / ESP32 / RP2040"]
        CCodegen --> STM32
        STM32 --> HIL["USB-HS / UART Live Inspector Telemetry"]
    end
```

---

## Key Features

### 1. Zero-Allocation Bare-Metal Execution (`#![no_std]`)
Built from the ground up for bare-metal targets (ARM Cortex-M, RISC-V, ESP32, Xtensa) with **zero dynamic heap allocations (`alloc`)**. All memory is statically scheduled ahead of time.

### 2. Static Interval-Colored SRAM Arena Scheduler
Computes exact tensor birth and death lifetimes across topological execution steps to reuse physical SRAM buffers. Reduces peak RAM footprint by **15% to 35%** compared to runtime dynamic allocators.

### 3. SIMD & ARMv8-M DSP Kernel Acceleration
Vectorized 4-way and 8-way dot-product accumulation (`dot_product_s8_accum`, `vec_dot_s8`, `vec_dot_s16`) utilizing ARM Cortex-M DSP assembly (`SMLAD`) and auto-vectorization.

### 4. Sub-Byte 4-Bit (`s4`) & Nonlinear Codebook LUT Quantization
- **Linear `s4`**: Packs two signed 4-bit weights per byte for a **50% Flash memory reduction**.
- **Nonlinear LUT `s4_lut`**: 4-bit indices indexing into a 16-entry codebook table, enabling nonlinear K-Means weight clustering with minimal accuracy loss.

### 5. Industrial Safety & Flash Integrity Protection
- **Flash CRC32 Bitflip Check**: `verify_weights_integrity(&weights, expected_crc)` validates Flash weight tables at boot time to prevent silent memory decay.
- **Arena Guard Canaries**: `verify_arena_integrity(arena, required_bytes, guard_canary)` with `0xDEAD_CAFE` guard constants catches buffer overruns and stack collisions.

### 6. Tiny Anomaly Detection & Condition Monitoring
- **`ReconstructionAnomalyDetector`**: Integer `i8` and `f32` mean-squared reconstruction error scoring for unsupervised autoencoders (bearing wear, motor vibration).
- **`MahalanobisAnomalyDetector`**: Multivariate distance scoring against baseline multi-channel sensor distributions.

### 7. Google LiteRT & TFLite-Micro Synergy
- **LiteRT Compiler Plugin**: Native C-ABI plugin (`crates/embedded-nn-litert-plugin`) for the new LiteRT runtime.
- **TFLM Benchmark Zoo**: Proven ingestion of official Google MLPerf Tiny models (`keyword_scrambled_8bit.tflite`, `person_detect.tflite`).

### 8. 1-Click Multi-Target Firmware Exporter
- **C99 CMake Pack**: Generates `include/embedded_model.h`, `src/main.c`, `CMakeLists.txt`, and `Makefile`.
- **Rust `#![no_std]` Crate Pack**: Generates a standalone `Cargo.toml` and `src/lib.rs` ready for embedded firmware.

---

## Quickstart

### 1. Compile-Time Model Embedding (Rust)
```rust
use embedded_nn_macros::embedded_nn_model;

// Imports ModelGraph or .tflite directly at compile time
#[embedded_nn_model("models/gesture_classifier.json")]
pub struct GestureClassifier;

fn main() {
    let mut arena = [0u8; GestureClassifier::ARENA_SIZE];
    let sensor_features = [12i8, -4, 30, 2];
    
    let logits = GestureClassifier::predict(&sensor_features, &mut arena).unwrap();
    let top_class = GestureClassifier::predict_class(&sensor_features, &mut arena).unwrap();
}
```

### 2. Standalone C99 Header-Only Inference
```c
#include "gesture_model.h"

static uint8_t g_arena[GESTURE_MODEL_ARENA_SIZE_BYTES];
static int8_t g_input[GESTURE_MODEL_INPUT_DIM];
static int8_t g_output[GESTURE_MODEL_OUTPUT_DIM];

void run_inference(void) {
    // Zero dynamic allocations, pure C99 fixed-point math
    int status = gesture_model_predict(g_input, g_output, g_arena);
    if (status == 0) {
        // Output logits available in g_output
    }
}
```

### 3. Launching TinyML Studio (Desktop & WebAssembly)
```bash
# Launch native desktop application
cargo run -p embedded-nn-studio

# Or serve WebAssembly Studio in browser via Trunk
cd crates/embedded-nn-studio && trunk serve
```

### 4. Hardware-in-the-Loop (HIL) CLI
```bash
# Enumerate connected USB-HS TinyML hardware
enn devices

# Ping device agent & run live inference benchmark
enn hil ping
enn hil infer --input 64
```

---

## License

Dual-licensed under either of:
- MIT License ([`LICENSE-MIT`](./LICENSE-MIT))
- Apache License, Version 2.0 ([`LICENSE-APACHE`](./LICENSE-APACHE))
