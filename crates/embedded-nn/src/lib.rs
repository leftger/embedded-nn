//! # embedded-nn
//!
//! `embedded-nn` is a pure Rust, `#![no_std]` neural network inference library for microcontrollers and embedded targets,
//! inspired by ARM's CMSIS-NN and TensorFlow Lite Micro.
//!
//! ## Modules
//! - [`types`]: Dimensional shapes, parameter structures, error types.
//! - [`support`]: Fixed-point quantization math, rounding division, bit operations.
//! - [`activations`]: ReLU, ReLU6, LeakyReLU, Sigmoid, Tanh.
//! - [`basic_math`]: Elementwise addition, subtraction, and multiplication.
//! - [`convolution`]: 2D Convolution, 1x1 Convolution, Depthwise Convolution, Transposed Conv, 1D Temporal Conv.
//! - [`fully_connected`]: Fully Connected (Linear / Dense) layers and Batch Matrix Multiplication (`BatchMatMul`).
//! - [`pooling`]: Max Pooling, Average Pooling.
//! - [`softmax`]: Softmax activation.
//! - [`mod@concat`]: Depthwise concatenation.
//! - [`pad`]: Tensor padding.
//! - [`transpose`]: Matrix and spatial transposition.
//! - [`reshape`]: Reshaping operations.
//! - [`simd`]: Target SIMD vectorization abstractions, ARM DSP SMLAD assembly, and dot-product acceleration.
//! - [`subbyte`]: Sub-byte 4-bit (`s4`) quantization layers and packing helpers.
//! - [`recurrent`]: Recurrent neural network layers (LSTM cell, SVDF filter with 8-bit & 16-bit state).
//! - [`float_ops`]: Floating-point (`f32` & IEEE-754 `f16`) fallback layers.

#![no_std]
#![deny(missing_docs)]
#![allow(
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::identity_op,
    clippy::erasing_op,
    clippy::manual_div_ceil,
    clippy::manual_clamp,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::manual_is_multiple_of,
    clippy::explicit_counter_loop
)]

pub mod activations;
pub mod basic_math;
pub mod concat;
pub mod convolution;
#[cfg(feature = "dsp")]
pub mod feature_dsp;
pub mod float_ops;
pub mod fully_connected;
#[cfg(feature = "libm")]
pub mod ml;
pub mod pad;
pub mod pooling;
pub mod recurrent;
pub mod reshape;
pub mod simd;
pub mod slice;
pub mod softmax;
pub mod subbyte;
pub mod support;
pub mod transpose;
pub mod types;

#[cfg(feature = "libm")]
pub use ml::{
    GaussianNaiveBayesInstanceF32, SvmInstanceF32, SvmKernelType, hz_to_mel, mel_filterbank_f32,
    mel_to_hz, mfcc_f32,
};

pub use support::{
    clamp, dequantize_s8_to_f32, dequantize_s16_to_f32, divide_by_power_of_two,
    doubling_high_mult_no_sat, pack_q15x2_32x1, pack_s8x4_32x1, quantize_f32_to_s8,
    quantize_f32_to_s16, requantize, requantize_s64,
};
pub use types::{
    Activation, Context, ConvParams, Dims, DwConvParams, Error, FcParams, FusedActivation,
    Padding2D, PerChannelQuantParams, PerTensorQuantParams, PoolParams, QuantParams, Result,
    SoftmaxParams, TensorView, TensorViewPadding, Tile,
};

pub use basic_math::{
    ElementwiseAddParams, ElementwiseMulParams, elementwise_add_s8, elementwise_mul_s8,
};
pub use concat::concatenation_s8;
pub use convolution::{
    convolve_1_x_n_s8, convolve_per_channel_s8, convolve_s8, depthwise_conv_per_channel_s8,
    transpose_conv_s8,
};
pub use float_ops::{f16_to_f32, f32_to_f16};
pub use fully_connected::{
    batch_matmul_s8, batch_matmul_s16, fully_connected_per_channel_s8, fully_connected_s8,
};
pub use pad::{pad_s8, reduce_mean_s8};
pub use pooling::{avg_pool_s8, max_pool_s8};
pub use recurrent::{LstmGateParams, lstm_step_s8_s16, lstm_step_s16, svdf_s8, svdf_state_s16_s8};
pub use simd::{vec_dot_s8, vec_dot_s16};
pub use slice::strided_slice_s8;
pub use softmax::softmax_s8;
pub use subbyte::{convolve_s4, fully_connected_s4, pack_s4_pair, unpack_s4_pair};
pub use transpose::{transpose_2d_s8, transpose_nd_s8, transpose_spatial_s8};
