#![allow(
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::manual_div_ceil,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::identity_op,
    clippy::erasing_op
)]

pub mod arena;
pub mod builder;
pub mod ir;
pub mod quant;

pub use arena::{ArenaPlan, ArenaScheduler, TensorAllocation};
pub use builder::ModelBuilder;
pub use ir::*;
pub use quant::*;
