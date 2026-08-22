//! Host-side Burn training that emits an integer [`ModelGraph`].
//!
//! Burn stays on the host. Device inference continues to use CMSIS-NN-style s8 kernels.

pub mod augment;
mod conv_svdf;
mod mlp;
pub mod pareto;
mod quantize;

pub use augment::{
    AugmentConfig, apply_frequency_mask, apply_noise, apply_scaling, apply_time_mask,
};
pub use conv_svdf::{compare_quant_paths, train_model};
pub use mlp::{
    QuantCompare, TrainArch, TrainConfig, TrainMode, TrainReport, dequant_outputs, train_dense_mlp,
};
pub use pareto::{ParetoCandidate, evaluate_pareto_candidates, mark_pareto_frontier};
pub use quantize::ptq_dense_mlp;
