use crate::state::{ModelImportStatus, StudioState};
use crate::theme::configure_theme;
use crate::views::arena::ArenaView;
use crate::views::codegen::CodegenView;
use crate::views::dsp::DspView;
use crate::views::ingest::IngestView;
use crate::views::train::TrainView;
use eframe::egui;
use embedded_nn_live::host::DeviceLink;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudioTab {
    Ingest,
    Dsp,
    Train,
    Arena,
    Codegen,
}

pub struct EmbeddedNnStudioApp {
    pub current_tab: StudioTab,
    pub state: StudioState,
    pub ingest_view: IngestView,
    pub dsp_view: DspView,
    pub train_view: TrainView,
    pub arena_view: ArenaView,
    pub codegen_view: CodegenView,
    pub device_link: Option<DeviceLink>,
}

impl Default for EmbeddedNnStudioApp {
    fn default() -> Self {
        Self {
            current_tab: StudioTab::Ingest,
            state: StudioState::default(),
            ingest_view: IngestView::new(),
            dsp_view: DspView::new(),
            train_view: TrainView::new(),
            arena_view: ArenaView::new(),
            codegen_view: CodegenView::new(),
            device_link: None,
        }
    }
}

impl eframe::App for EmbeddedNnStudioApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("studio_top_header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("⚡ embedded-nn studio");
                ui.label("• Embedded TinyML Platform");

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    egui::widgets::global_theme_preference_buttons(ui);
                });
            });

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(format!(
                    "Model source: {}",
                    self.state.model_source.display_name()
                ));
                if ui.button("Open .tflite").clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .add_filter("TensorFlow Lite", &["tflite"])
                        .pick_file()
                {
                    let _ = self.state.import_tflite_path(path);
                }
                if ui.button("Open ModelGraph .json").clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .add_filter("embedded-nn ModelGraph JSON", &["json"])
                        .pick_file()
                {
                    let _ = self.state.import_json_path(path);
                }
                match &self.state.model_import_status {
                    ModelImportStatus::Idle => {}
                    ModelImportStatus::Imported(message) => {
                        ui.colored_label(egui::Color32::from_rgb(100, 220, 140), message);
                    }
                    ModelImportStatus::Error(message) => {
                        ui.colored_label(egui::Color32::from_rgb(240, 90, 90), message);
                    }
                }
            });

            // Step-by-Step Pipeline Navigation Bar with Live Status Badges
            ui.horizontal(|ui| {
                let sample_count = self.state.samples.len();
                let mel_bins = self.state.dsp.num_mel_bins;
                let latest_acc = self.state.val_acc_history.last().copied().unwrap_or(0.0);
                let arena_bytes = self
                    .state
                    .arena_plan
                    .as_ref()
                    .map(|p| p.total_arena_bytes)
                    .unwrap_or(0);

                let tab_1_label = format!("1. Ingest ({} samples)", sample_count);
                let tab_2_label = format!("2. DSP ({} bins)", mel_bins);
                let tab_3_label = format!("3. Train ({:.0}% acc)", latest_acc);
                let tab_4_label = format!("4. Arena ({} B)", arena_bytes);
                let tab_5_label = "5. Rust Codegen & HIL".to_string();

                ui.selectable_value(&mut self.current_tab, StudioTab::Ingest, tab_1_label);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Dsp, tab_2_label);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Train, tab_3_label);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Arena, tab_4_label);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Codegen, tab_5_label);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.current_tab {
            StudioTab::Ingest => {
                self.ingest_view
                    .show(ui, &mut self.state, &mut self.device_link)
            }
            StudioTab::Dsp => self.dsp_view.show(ui, &mut self.state),
            StudioTab::Train => self.train_view.show(ui, &mut self.state),
            StudioTab::Arena => self.arena_view.show(ui, &mut self.state),
            StudioTab::Codegen => self
                .codegen_view
                .show(ui, &mut self.state, self.device_link.as_ref()),
        });

        // Request continuous repaint for smooth 60 FPS live oscilloscope stream and training progress
        if self.current_tab == StudioTab::Ingest
            || self.current_tab == StudioTab::Codegen
            || self.state.is_training
            || self.ingest_view.is_recording
            || self.device_link.as_ref().is_some_and(|link| link.is_alive())
        {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
    }
}

pub fn run_studio() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1150.0, 720.0])
            .with_min_inner_size([850.0, 520.0])
            .with_title("embedded-nn Studio - TinyML Development Platform"),
        ..Default::default()
    };

    eframe::run_native(
        "embedded-nn Studio",
        native_options,
        Box::new(|cc| {
            configure_theme(&cc.egui_ctx);
            Ok(Box::new(EmbeddedNnStudioApp::default()))
        }),
    )
}
