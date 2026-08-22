#![allow(
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::manual_div_ceil,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::identity_op,
    clippy::erasing_op,
    clippy::single_match,
    clippy::useless_format
)]

pub mod bundle;
pub mod emit_c;
pub mod emit_rust;

pub use bundle::{BundleFile, generate_c_project_bundle, generate_rust_crate_bundle};
pub use emit_c::CCodeGenerator;
pub use emit_rust::RustCodeGenerator;
