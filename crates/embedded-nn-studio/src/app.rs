use crate::state::{ModelImportStatus, StudioState};
use crate::theme::configure_theme;
use crate::views::arena::ArenaView;
use crate::views::codegen::CodegenView;
use crate::views::dsp::DspView;
use crate::views::gesture_3d::Gesture3DView;
use crate::views::ingest::IngestView;
use crate::views::live_inspector::LiveInspectorView;
use crate::views::train::TrainView;
use eframe::egui;
use embedded_nn_live::host::DeviceLink;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudioTab {
    Ingest,
    Gesture3D,
    Dsp,
    Train,
    Pareto,
    Arena,
    Codegen,
    Inspector,
}

pub struct EmbeddedNnStudioApp {
    pub current_tab: StudioTab,
    pub state: StudioState,
    pub ingest_view: IngestView,
    pub gesture_view: Gesture3DView,
    pub dsp_view: DspView,
    pub train_view: TrainView,
    pub arena_view: ArenaView,
    pub codegen_view: CodegenView,
    pub inspector_view: LiveInspectorView,
    pub device_link: Option<DeviceLink>,
}

impl Default for EmbeddedNnStudioApp {
    fn default() -> Self {
        Self {
            current_tab: StudioTab::Ingest,
            state: StudioState::default(),
            ingest_view: IngestView::new(),
            gesture_view: Gesture3DView::new(),
            dsp_view: DspView::new(),
            train_view: TrainView::new(),
            arena_view: ArenaView::new(),
            codegen_view: CodegenView::new(),
            inspector_view: LiveInspectorView::new(),
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
                if ui.button("💾 Save Project (.ennproj)").clicked() {
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(path) = rfd::FileDialog::new()
                        .set_file_name("project.ennproj")
                        .add_filter("embedded-nn Studio Project", &["ennproj", "json"])
                        .save_file()
                    {
                        let msg = self.state.save_project_file(&path);
                        self.ingest_view.import_status = match msg {
                            Ok(s) => s,
                            Err(e) => format!("Save error: {e}"),
                        };
                    }
                }
                if ui.button("📂 Open Project (.ennproj)").clicked() {
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("embedded-nn Studio Project", &["ennproj", "json"])
                        .pick_file()
                    {
                        let msg = self.state.load_project_file(&path);
                        self.ingest_view.import_status = match msg {
                            Ok(s) => s,
                            Err(e) => format!("Load error: {e}"),
                        };
                    }
                }
                if ui
                    .button("⚡ Showcase End-to-End Gesture Pipeline")
                    .clicked()
                {
                    self.state.reset_showcase_pipeline();
                }
                egui::ComboBox::from_id_salt("top_model_zoo_presets_combo")
                    .selected_text("🏛️ Model Zoo Presets")
                    .show_ui(ui, |ui| {
                        for preset in crate::state::ModelZooPreset::ALL {
                            if ui
                                .selectable_label(
                                    matches!(
                                        &self.state.model_source,
                                        crate::state::ModelSource::ZooPreset(name) if name == preset.title()
                                    ),
                                    preset.title(),
                                )
                                .on_hover_text(preset.description())
                                .clicked()
                            {
                                let _ = self.state.load_zoo_preset(preset);
                            }
                        }
                    });
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
                if ui.button("Open Dataset (.jsonl/csv)").clicked()
                    && let Some(paths) = rfd::FileDialog::new()
                        .add_filter(
                            "Dataset Interchange (.jsonl, .csv, .json)",
                            &["jsonl", "ndjson", "json", "csv", "tsv"],
                        )
                        .pick_files()
                {
                    match self.state.import_dataset_paths(&paths) {
                        Ok(n) => {
                            self.ingest_view.import_status = format!("Imported {n} sample(s).");
                            self.current_tab = StudioTab::Ingest;
                        }
                        Err(e) => {
                            self.ingest_view.import_status = format!("Import error: {e}");
                        }
                    }
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
                let tab_2_label = "2. 🌐 3D Gesture".to_string();
                let tab_3_label = format!("3. DSP ({} bins)", mel_bins);
                let tab_4_label = format!("4. Train ({:.0}% acc)", latest_acc);
                let tab_pareto_label = "5. ⚡ Pareto".to_string();
                let tab_5_label = format!("6. Arena ({} B)", arena_bytes);
                let tab_6_label = "7. Codegen".to_string();
                let tab_7_label = "8. 🔬 Live Inspector".to_string();

                ui.selectable_value(&mut self.current_tab, StudioTab::Ingest, tab_1_label);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Gesture3D, tab_2_label);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Dsp, tab_3_label);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Train, tab_4_label);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Pareto, tab_pareto_label);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Arena, tab_5_label);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Codegen, tab_6_label);
                ui.label("➔");
                ui.selectable_value(&mut self.current_tab, StudioTab::Inspector, tab_7_label);
            });
        });

        // Handle dropped files (drag & drop import for datasets and models)
        ctx.input(|i| {
            if !i.raw.dropped_files.is_empty() {
                let paths: Vec<std::path::PathBuf> = i
                    .raw
                    .dropped_files
                    .iter()
                    .filter_map(|f| f.path.clone())
                    .collect();
                if !paths.is_empty() {
                    let mut dataset_paths = Vec::new();
                    for path in paths {
                        let ext = path
                            .extension()
                            .and_then(|s| s.to_str())
                            .unwrap_or("")
                            .to_ascii_lowercase();
                        if ext == "tflite" {
                            let _ = self.state.import_tflite_path(path);
                        } else if ext == "jsonl"
                            || ext == "ndjson"
                            || ext == "csv"
                            || ext == "tsv"
                            || (ext == "json" && self.state.import_json_path(path.clone()).is_err())
                        {
                            dataset_paths.push(path);
                        }
                    }
                    if !dataset_paths.is_empty() {
                        match self.state.import_dataset_paths(&dataset_paths) {
                            Ok(n) => {
                                self.ingest_view.import_status =
                                    format!("Imported {n} sample(s) from drag-and-drop.");
                                self.current_tab = StudioTab::Ingest;
                            }
                            Err(e) => {
                                self.ingest_view.import_status =
                                    format!("Drag-and-drop import error: {e}");
                            }
                        }
                    }
                }
            }
        });

        // Drain the device on every frame, not just while Ingest is visible,
        // so the 3D gesture view sees a live trajectory too.
        self.ingest_view.poll_device(&self.device_link);

        egui::CentralPanel::default().show(ctx, |ui| match self.current_tab {
            StudioTab::Ingest => self
                .ingest_view
                .show(ui, &mut self.state, &mut self.device_link),
            StudioTab::Gesture3D => {
                let live = self.ingest_view.live_trajectory();
                self.gesture_view.ui(ui, &self.state, live.as_deref());
            }
            StudioTab::Dsp => self.dsp_view.show(ui, &mut self.state),
            StudioTab::Train => self.train_view.show(ui, &mut self.state),
            StudioTab::Pareto => crate::views::pareto::render(ui, &mut self.state),
            StudioTab::Arena => self.arena_view.show(ui, &mut self.state),
            StudioTab::Codegen => {
                self.codegen_view
                    .show(ui, &mut self.state, self.device_link.as_ref())
            }
            StudioTab::Inspector => {
                self.inspector_view.ui(ui, &mut self.device_link);
            }
        });

        // Request continuous repaint for smooth 60 FPS live oscilloscope stream and training progress
        if self.current_tab == StudioTab::Ingest
            || self.current_tab == StudioTab::Gesture3D
            || self.current_tab == StudioTab::Codegen
            || self.current_tab == StudioTab::Inspector
            || self.state.is_training
            || self.ingest_view.is_recording
            || self
                .device_link
                .as_ref()
                .is_some_and(|link| link.is_alive())
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

#[cfg(target_arch = "wasm32")]
pub async fn run_studio_web(canvas_id: &str) -> Result<(), eframe::wasm_bindgen::JsValue> {
    let web_options = eframe::WebOptions::default();
    eframe::WebRunner::new()
        .start(
            canvas_id,
            web_options,
            Box::new(|cc| {
                configure_theme(&cc.egui_ctx);
                Ok(Box::new(EmbeddedNnStudioApp::default()))
            }),
        )
        .await
}
