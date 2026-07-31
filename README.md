# embedded-nn

[![crates.io](https://img.shields.io/crates/v/embedded-nn.svg)](https://crates.io/crates/embedded-nn)
[![docs.rs](https://img.shields.io/docsrs/embedded-nn)](https://docs.rs/embedded-nn)
[![CI](https://github.com/leftger/embedded-nn/actions/workflows/ci.yml/badge.svg)](https://github.com/leftger/embedded-nn/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

A pure Rust, `#![no_std]` neural network inference library for microcontrollers and embedded targets, ported from and inspired by ARM's **CMSIS-NN** and **TensorFlow Lite Micro**.

## Features

- **`#![no_std]` Bare-Metal Support**: Built for bare-metal targets (ARM Cortex-M, RISC-V, ESP32, etc.) with zero required heap allocations (`alloc`).
- **Target SIMD Hooks**: Vectorized 4-way and 8-way dot-product abstractions (`vec_dot_s8`, `vec_dot_s16`) optimized for compiler auto-vectorization and hardware SIMD acceleration.
- **Quantization Support**:
  - `int8` (s8) and `int16` (s16) per-tensor and per-channel fixed-point quantization (`requantize`, `quantize_f32_to_s8`, `dequantize_s8_to_f32`).
  - `int4` (`s4`) 4-bit sub-byte quantization (packed nibbles for sub-byte weight compression).
- **Recurrent Operators**: Unidirectional `LSTM` cell (`lstm_step_s8_s16`) and `SVDF` (Singular Value Decomposition Filter) layer.
- **Float Fallback Operations**: IEEE-754 `f16` half-precision conversions (`f16_to_f32`, `f32_to_f16`), `f32` convolution, dense, and softmax layers.
- **Core Neural Operators**:
  - **Convolution**: 2D Conv (`s8`, `s4`, `f32`), 1x1 Fast Conv, Depthwise Conv.
  - **Dense / Fully Connected**: Matrix multiplication (`s8`, `s16`, `s4`, `f32`).
  - **Activations**: ReLU, ReLU6, LeakyReLU, Sigmoid, Tanh, `FusedActivation` enum.
  - **Pooling**: Max Pooling 2D, Average Pooling 2D (`s8`, `s16`).
  - **Softmax**: Fixed-point exponential Softmax (`s8`, `s16`) & float Softmax (`f32`).
  - **Utilities**: `TensorView` spatial windowing with `TensorViewPadding` (`Same`, `Valid`), Depthwise concatenation, Padding, Transposition, Reshaping.

---

## Quick Example: 4-Bit Sub-Byte (`s4`) Fully Connected Layer

```rust
use embedded_nn::{
    fully_connected_s4, pack_s4_pair, Activation, Dims, FcParams, PerTensorQuantParams,
};

fn run_s4_inference() {
    let fc_params = FcParams {
        input_offset: 0,
        filter_offset: 0,
        output_offset: 0,
        activation: Activation::int8_unconstrained(),
    };

    let quant_params = PerTensorQuantParams::new(1073741824, 0); // 0.5 multiplier

    let input_dims = Dims::new(1, 1, 1, 2);
    let input = [4i8, 6i8];

    let filter_dims = Dims::new(2, 1, 1, 1);
    // Pack two 4-bit signed weights (-8..7) into a single byte: w0 = 2, w1 = 3
    let packed_kernel = [pack_s4_pair(2i8, 3i8)];
    
    let output_dims = Dims::new(1, 1, 1, 1);
    let mut output = [0i8; 1];

    fully_connected_s4(
        &fc_params,
        &quant_params,
        &input_dims,
        &input,
        &filter_dims,
        &packed_kernel,
        None,
        &output_dims,
        &mut output,
    ).unwrap();

    // output[0] = 13
}
```

---

## Module Overview

| Module | Description |
| :--- | :--- |
| [`types`](src/types.rs) | Tensor dimensions (`Dims`), kernel shapes (`Tile`), `TensorView` spatial windowing (`TensorViewPadding`), quantization parameters, `FusedActivation`, layer config structs. |
| [`support`](src/support.rs) | Requantization math (`requantize`, `doubling_high_mult_no_sat`, `divide_by_power_of_two`), float quantization (`quantize_f32_to_s8`, `dequantize_s8_to_f32`). |
| [`simd`](src/simd.rs) | Vectorized 4-way/8-way dot product routines (`vec_dot_s8`, `vec_dot_s16`). |
| [`subbyte`](src/subbyte.rs) | 4-bit (`s4`) sub-byte packing/unpacking and quantized layers (`fully_connected_s4`, `convolve_s4`). |
| [`recurrent`](src/recurrent.rs) | Recurrent neural network layers (`lstm_step_s8_s16`, `svdf_s8`). |
| [`float_ops`](src/float_ops.rs) | IEEE-754 `f16` half-precision conversions and `f32` fallback operations (`convolve_f32`, `softmax_f32`). |
| [`activations`](src/activations.rs) | ReLU, ReLU6, LeakyReLU, Sigmoid, Tanh, Fused activations. |
| [`basic_math`](src/basic_math.rs) | Elementwise Add, Subtract, Multiply. |
| [`convolution`](src/convolution.rs) | 2D Convolution, Depthwise Convolution (per-tensor & per-channel). |
| [`fully_connected`](src/fully_connected.rs) | Dense / Linear layer (per-tensor & per-channel). |
| [`pooling`](src/pooling.rs) | Max Pool 2D, Average Pool 2D. |
| [`softmax`](src/softmax.rs) | Fixed-point exponential Softmax. |
| [`concat`](src/concat.rs) | Tensor depthwise concatenation. |
| [`pad`](src/pad.rs) | Tensor padding. |
| [`transpose`](src/transpose.rs) | Matrix and spatial transposition. |
| [`reshape`](src/reshape.rs) | Tensor reshaping. |

---

## License

The contents of this repository are dual-licensed under the _MIT OR Apache 2.0_
License. That means you can choose either the MIT license or the Apache 2.0
license when you re-use this code. See [`LICENSE-MIT`](./LICENSE-MIT) or
[`LICENSE-APACHE`](./LICENSE-APACHE) for more information on each specific
license.
