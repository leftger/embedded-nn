#![allow(
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::identity_op,
    clippy::erasing_op,
    clippy::manual_div_ceil,
    clippy::manual_clamp,
    clippy::needless_range_loop
)]

mod app;
mod state;
mod syntax;
mod theme;
mod views;

fn main() -> eframe::Result<()> {
    app::run_studio()
}
