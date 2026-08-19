mod app;
mod views;

use app::EmbeddedNnStudioApp;
use eframe::egui;

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 720.0])
            .with_min_inner_size([800.0, 500.0])
            .with_title("embedded-nn Studio - TinyML Development Platform"),
        ..Default::default()
    };

    eframe::run_native(
        "embedded-nn Studio",
        native_options,
        Box::new(|_cc| Ok(Box::new(EmbeddedNnStudioApp::default()))),
    )
}
