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
pub mod dsp_contract;
pub mod interpreter;
pub mod ir;
pub mod prune;
pub mod quant;

pub use arena::{ArenaPlan, ArenaScheduler, TensorAllocation};
pub use builder::ModelBuilder;
pub use dsp_contract::DspContract;
pub use interpreter::{HostInterpreter, InterpreterError};
pub use ir::*;
pub use prune::{
    PruningReport, compute_fc_neuron_l1_importances, find_lightest_fc_neuron,
    prune_fc_hidden_neuron, prune_graph_l1,
};
pub use quant::*;
