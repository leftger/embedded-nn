//! Host-side Burn training that emits an integer [`ModelGraph`].
//!
//! Burn stays on the host. Device inference continues to use CMSIS-NN-style s8 kernels.

mod mlp;
mod quantize;

pub use mlp::{TrainConfig, TrainMode, TrainReport, dequant_outputs, train_dense_mlp};
pub use quantize::ptq_dense_mlp;
