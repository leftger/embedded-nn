//! Host-side Burn training that emits an integer [`ModelGraph`].
//!
//! Burn stays on the host. Device inference continues to use CMSIS-NN-style s8 kernels.

mod conv_svdf;
mod mlp;
mod quantize;

pub use conv_svdf::{compare_quant_paths, train_model};
pub use mlp::{
    QuantCompare, TrainArch, TrainConfig, TrainMode, TrainReport, dequant_outputs, train_dense_mlp,
};
pub use quantize::ptq_dense_mlp;
