mod app;
mod state;
mod syntax;
mod theme;
mod views;

fn main() -> eframe::Result<()> {
    app::run_studio()
}
