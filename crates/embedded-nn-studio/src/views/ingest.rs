use crate::state::{DatasetSample, StudioState};
use eframe::egui;
use embedded_nn_live::decode_f32_le;
use embedded_nn_live::host::{DeviceLink, OwnedMsg, UsbBridge};
use std::path::{Path, PathBuf};

fn file_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .into_owned()
}

/// Label assigned to imported records that carry no label of their own.
const UNLABELED_IMPORT: &str = "unlabeled_import";

#[derive(Default)]
pub struct IngestView {
    pub selected_port: String,
    pub sample_rate_hz: u32,
    pub current_label: String,
    pub new_class_name: String,
    pub is_recording: bool,
    pub live_sensor_history: Vec<f32>,
    pub live_time_counter: f32,
    pub import_status: String,
    pub available_agents: Vec<String>,
    pub link_status: String,
}

impl IngestView {
    pub fn new() -> Self {
        Self {
            selected_port: "Simulated IMU Source".into(),
            sample_rate_hz: 100,
            current_label: "wave_left".into(),
            new_class_name: String::new(),
            is_recording: false,
            live_sensor_history: (0..100).map(|i| ((i as f32) * 0.1).sin() * 0.7).collect(),
            live_time_counter: 0.0,
            import_status: String::new(),
            available_agents: Vec::new(),
            link_status: "Disconnected".into(),
        }
    }

    fn import_dataset_files(&mut self, state: &mut StudioState, paths: &[PathBuf]) {
        let mut imported = 0usize;
        let mut errors: Vec<String> = Vec::new();

        for path in paths {
            match std::fs::read_to_string(path).map_err(|e| e.to_string()) {
                Ok(contents) => match embedded_nn_live::parse_jsonl(&contents) {
                    Ok(records) => {
                        for record in records {
                            let label = record.label_or(UNLABELED_IMPORT);
                            let class_idx = state.class_index_or_insert(&label);
                            let id = state.next_sample_id;
                            state.next_sample_id += 1;
                            state.samples.push(DatasetSample {
                                id,
                                label,
                                class_idx,
                                raw_waveform: record.scalar_channel(),
                                frames: Vec::new(),
                                quantized_frames: Vec::new(),
                            });
                            imported += 1;
                        }
                    }
                    Err(e) => errors.push(format!("{}: {}", file_name(path), e)),
                },
                Err(e) => errors.push(format!("{}: {}", file_name(path), e)),
            }
        }

        if imported > 0 {
            state.recompute_all_frames();
            state.reset_training();
            state.rebuild_model_graph_and_codegen();
        }

        self.import_status = if errors.is_empty() {
            format!("Imported {} sample(s).", imported)
        } else {
            format!("Imported {} sample(s). {}", imported, errors.join("; "))
        };
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        state: &mut StudioState,
        device_link: &mut Option<DeviceLink>,
    ) {
        ui.horizontal(|ui| {
            ui.heading("📊 1. Data Ingestion, Telemetry & Dataset Tagging");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("🔄 Reset to Demo Dataset").clicked() {
                    state.load_demo_dataset();
                    state.recompute_all_frames();
                    state.use_demo_trainer();
                }
                ui.label(format!("Total Dataset Samples: {}", state.samples.len()));
            });
        });

        ui.add_space(4.0);
        ui.label(
            "Record real-time physical sensor time-series (IMU accelerometer, gyroscopes, audio, PPG) or simulate streaming data, annotate class labels, and maintain balanced dataset splits.",
        );
        ui.add_space(8.0);

        // Top Control Bar
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Sensor Source:");
                egui::ComboBox::from_id_salt("sensor_source_combo")
                    .selected_text(&self.selected_port)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.selected_port,
                            "Simulated IMU Source".into(),
                            "Simulated IMU Source",
                        );
                        for agent in &self.available_agents {
                            ui.selectable_value(
                                &mut self.selected_port,
                                agent.clone(),
                                format!("USB-HS {agent}"),
                            );
                        }
                    });
                if ui.button("Refresh USB").clicked() {
                    self.available_agents = UsbBridge::list_devices();
                    self.link_status = format!("{} agent(s)", self.available_agents.len());
                }
                if ui.button("Connect").clicked() {
                    *device_link = None;
                    if self.selected_port == "Simulated IMU Source" {
                        self.link_status = "Using simulated IMU (no USB).".into();
                    } else {
                        match DeviceLink::connect(&self.selected_port) {
                            Ok(link) => {
                                self.link_status = format!("Connecting {}", link.device_id());
                                *device_link = Some(link);
                            }
                            Err(error) => self.link_status = error,
                        }
                    }
                }
                if ui.button("Disconnect").clicked() {
                    *device_link = None;
                    self.link_status = "Disconnected".into();
                }
                if let Some(link) = device_link.as_ref() {
                    if link.is_handshaked() {
                        self.link_status = format!("Ready {}", link.device_id());
                    }
                    if ui.button("Ping").clicked() {
                        link.ping();
                    }
                    if let Some(OwnedMsg::SensorFrame {
                        timestamp_ms,
                        channel_count,
                        values,
                    }) = link.take_sensor()
                    {
                        let mut samples = vec![0.0f32; values.len() / 4];
                        if let Ok(n) = decode_f32_le(&values, &mut samples) {
                            self.live_sensor_history.extend(&samples[..n]);
                            if self.live_sensor_history.len() > 400 {
                                let extra = self.live_sensor_history.len() - 400;
                                self.live_sensor_history.drain(..extra);
                            }
                            self.link_status =
                                format!("t={timestamp_ms} ms ch={channel_count} samples={n}");
                        }
                    }
                    if let Some(error) = link.take_error() {
                        self.link_status = error;
                    }
                }
                ui.label(&self.link_status);

                ui.separator();

                ui.label("Sampling Rate:");
                ui.add(egui::DragValue::new(&mut self.sample_rate_hz).suffix(" Hz"));

                ui.separator();

                ui.label("Target Class Tag:");
                if !state.classes.is_empty() {
                    egui::ComboBox::from_id_salt("class_select_combo")
                        .selected_text(&self.current_label)
                        .show_ui(ui, |ui| {
                            for class_name in &state.classes {
                                ui.selectable_value(
                                    &mut self.current_label,
                                    class_name.clone(),
                                    class_name,
                                );
                            }
                        });
                }

                let record_btn = if self.is_recording {
                    egui::Button::new("⏹ Stop Recording").fill(egui::Color32::from_rgb(180, 50, 50))
                } else {
                    egui::Button::new("⏺ Record Sample").fill(egui::Color32::from_rgb(40, 140, 70))
                };

                if ui.add(record_btn).clicked() {
                    self.is_recording = !self.is_recording;
                    if !self.is_recording {
                        // Capture recorded buffer into dataset
                        let class_idx = state
                            .classes
                            .iter()
                            .position(|c| c == &self.current_label)
                            .unwrap_or(0);
                        let id = state.next_sample_id;
                        state.next_sample_id += 1;

                        state.samples.push(crate::state::DatasetSample {
                            id,
                            label: self.current_label.clone(),
                            class_idx,
                            raw_waveform: self.live_sensor_history.clone(),
                            frames: Vec::new(),
                            quantized_frames: Vec::new(),
                        });
                        state.recompute_all_frames();
                        state.rebuild_model_graph_and_codegen();
                    }
                }

                ui.separator();

                if ui.button("📂 Import Dataset File(s)").clicked()
                    && let Some(paths) = rfd::FileDialog::new()
                        .add_filter("JSON Lines dataset", &["jsonl", "ndjson"])
                        .pick_files()
                {
                    self.import_dataset_files(state, &paths);
                }
            });

            if !self.import_status.is_empty() {
                ui.label(&self.import_status);
            }
        });

        ui.add_space(8.0);

        // Update live waveform stream
        self.live_time_counter += 0.05;
        let t = self.live_time_counter;
        let new_sample = (t * 2.5).sin() * 0.7 + (t * 7.0).cos() * 0.2;
        self.live_sensor_history.remove(0);
        self.live_sensor_history.push(new_sample);

        // Live Oscilloscope & Class Balance Layout
        ui.columns(2, |cols| {
            // Left Column: Live Oscilloscope
            cols[0].group(|ui| {
                ui.horizontal(|ui| {
                    ui.label("📈 Live Oscilloscope Stream (Active Sensor Channel)");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if self.is_recording {
                            ui.colored_label(egui::Color32::from_rgb(255, 80, 80), "● RECORDING");
                        } else {
                            ui.colored_label(
                                egui::Color32::from_rgb(80, 200, 120),
                                "● LIVE STREAM",
                            );
                        }
                    });
                });

                let (rect, _resp) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), 130.0),
                    egui::Sense::hover(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(14, 17, 24));

                // Draw center grid line
                let mid_y = rect.center().y;
                painter.line_segment(
                    [
                        egui::pos2(rect.left(), mid_y),
                        egui::pos2(rect.right(), mid_y),
                    ],
                    egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(30, 40, 55)),
                );

                let n = self.live_sensor_history.len();
                if n > 1 {
                    let dx = rect.width() / (n - 1) as f32;
                    let scale_y = 50.0;
                    for i in 0..n - 1 {
                        let p1 = egui::pos2(
                            rect.left() + (i as f32) * dx,
                            mid_y - self.live_sensor_history[i] * scale_y,
                        );
                        let p2 = egui::pos2(
                            rect.left() + ((i + 1) as f32) * dx,
                            mid_y - self.live_sensor_history[i + 1] * scale_y,
                        );
                        let stroke_color = if self.is_recording {
                            egui::Color32::from_rgb(255, 90, 90)
                        } else {
                            egui::Color32::from_rgb(60, 210, 130)
                        };
                        painter.line_segment([p1, p2], egui::Stroke::new(2.0_f32, stroke_color));
                    }
                }
            });

            // Right Column: Class Distribution & Label Manager
            cols[1].group(|ui| {
                ui.label("🏷️ Class Distribution & Balance Monitor");
                ui.add_space(4.0);

                let mut class_counts = vec![0usize; state.classes.len()];
                for sample in &state.samples {
                    if sample.class_idx < class_counts.len() {
                        class_counts[sample.class_idx] += 1;
                    }
                }

                let total = state.samples.len().max(1);
                for (i, class_name) in state.classes.iter().enumerate() {
                    let count = class_counts.get(i).copied().unwrap_or(0);
                    let ratio = count as f32 / total as f32;
                    ui.horizontal(|ui| {
                        ui.label(format!("{}:", class_name));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(format!("{} samples ({:.0}%)", count, ratio * 100.0));
                            ui.add(egui::ProgressBar::new(ratio).desired_width(120.0));
                        });
                    });
                }

                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Add Class:");
                    ui.text_edit_singleline(&mut self.new_class_name);
                    if ui.button("+ Add").clicked() && !self.new_class_name.trim().is_empty() {
                        let name = self.new_class_name.trim().to_string();
                        if !state.classes.contains(&name) {
                            state.classes.push(name);
                            self.new_class_name.clear();
                            state.reset_training();
                            state.rebuild_model_graph_and_codegen();
                        }
                    }
                });
            });
        });

        ui.add_space(8.0);

        // Bottom: Recorded Dataset Browser Table
        ui.group(|ui| {
            ui.label("📁 Dataset Samples Explorer");
            ui.separator();

            egui::ScrollArea::vertical()
                .id_salt("ingest_samples_scroll_area")
                .max_height(160.0)
                .show(ui, |ui| {
                    egui::Grid::new("dataset_samples_grid")
                        .striped(true)
                        .min_col_width(80.0)
                        .show(ui, |ui| {
                            ui.label("ID");
                            ui.label("Class Tag");
                            ui.label("Signal Length");
                            ui.label("Feature Vector");
                            ui.label("Actions");
                            ui.end_row();

                            let mut delete_id = None;
                            let mut relabel = None;
                            for sample in state.samples.iter().rev().take(30) {
                                ui.label(format!("#{:03}", sample.id));

                                ui.push_id(sample.id, |ui| {
                                    egui::ComboBox::from_id_salt("sample_relabel_combo")
                                        .selected_text(&sample.label)
                                        .show_ui(ui, |ui| {
                                            for (idx, class_name) in
                                                state.classes.iter().enumerate()
                                            {
                                                if ui
                                                    .selectable_label(
                                                        sample.label == *class_name,
                                                        class_name,
                                                    )
                                                    .clicked()
                                                {
                                                    relabel =
                                                        Some((sample.id, idx, class_name.clone()));
                                                }
                                            }
                                        });
                                });

                                ui.label(format!("{} samples", sample.raw_waveform.len()));
                                let num_bins = sample.frames.first().map(|f| f.len()).unwrap_or(0);
                                ui.label(format!(
                                    "{} Mel bins × {} frames",
                                    num_bins,
                                    sample.frames.len()
                                ));

                                ui.push_id(sample.id, |ui| {
                                    if ui.button("🗑 Delete").clicked() {
                                        delete_id = Some(sample.id);
                                    }
                                });
                                ui.end_row();
                            }

                            if let Some((id, class_idx, label)) = relabel
                                && let Some(sample) = state.samples.iter_mut().find(|s| s.id == id)
                            {
                                sample.label = label;
                                sample.class_idx = class_idx;
                                state.rebuild_model_graph_and_codegen();
                            }

                            if let Some(id) = delete_id {
                                state.samples.retain(|s| s.id != id);
                                state.rebuild_model_graph_and_codegen();
                            }
                        });
                });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = concat!(
        r#"{"sample_id":"b1","label":null,"sample_rate_hz":400.0,"channel_names":["x","y","z"],"waveform":[[3.0,4.0,0.0],[0.0,0.0,2.0]]}"#,
        "\n",
        r#"{"sample_id":"b2","label":"recoil_anomaly","sample_rate_hz":400.0,"channel_names":["value"],"waveform":[[0.5],[-0.5]]}"#,
        "\n"
    );

    fn collect_text<'a>(shapes: impl Iterator<Item = &'a egui::Shape>, out: &mut String) {
        for shape in shapes {
            match shape {
                egui::Shape::Text(text) => out.push_str(text.galley.text()),
                egui::Shape::Vec(inner) => collect_text(inner.iter(), out),
                _ => {}
            }
        }
    }

    fn write_fixture(name: &str, contents: &str) -> PathBuf {
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn import_adds_samples_collapses_channels_and_registers_new_classes() {
        let mut view = IngestView::new();
        let mut state = StudioState::default();
        let before = state.samples.len();
        let path = write_fixture("enn_ingest_import_ok.jsonl", FIXTURE);

        view.import_dataset_files(&mut state, std::slice::from_ref(&path));

        assert_eq!(state.samples.len(), before + 2);
        let imported = &state.samples[before..];
        assert_eq!(imported[0].label, UNLABELED_IMPORT);
        assert_eq!(imported[0].raw_waveform, vec![5.0, 2.0]);
        assert_eq!(imported[1].label, "recoil_anomaly");
        assert_eq!(imported[1].raw_waveform, vec![0.5, -0.5]);
        assert!(state.classes.contains(&UNLABELED_IMPORT.to_string()));
        assert_eq!(state.classes[imported[1].class_idx], "recoil_anomaly");

        // Features and codegen must be refreshed for the newly imported samples.
        assert_eq!(
            imported[0].frames.len(),
            crate::state::StudioState::num_frames_for_config(&state.dsp)
        );
        assert_eq!(imported[0].frames[0].len(), state.dsp.num_mel_bins);
        assert!(!state.generated_rust_code.is_empty());
        assert_eq!(state.bias_fc2.len(), state.classes.len());

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn view_paints_import_button_and_imported_sample_labels() {
        let mut view = IngestView::new();
        let mut state = StudioState::default();
        let path = write_fixture("enn_ingest_render.jsonl", FIXTURE);
        view.import_dataset_files(&mut state, std::slice::from_ref(&path));

        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1400.0, 1000.0),
            )),
            ..Default::default()
        };
        let output = ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let mut device_link = None;
                view.show(ui, &mut state, &mut device_link)
            });
        });

        let mut painted = String::new();
        collect_text(output.shapes.iter().map(|c| &c.shape), &mut painted);

        assert!(painted.contains("Import Dataset File(s)"));
        assert!(painted.contains("Imported 2 sample(s)."));
        // Newest samples render first in the explorer, each with a relabel combo.
        assert!(painted.contains("recoil_anomaly"));
        assert!(painted.contains(UNLABELED_IMPORT));

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn import_reports_malformed_file_without_adding_samples() {
        let mut view = IngestView::new();
        let mut state = StudioState::default();
        let before = state.samples.len();
        let path = write_fixture("enn_ingest_import_bad.jsonl", "not json\n");

        view.import_dataset_files(&mut state, std::slice::from_ref(&path));

        assert_eq!(state.samples.len(), before);
        assert!(view.import_status.contains("enn_ingest_import_bad.jsonl"));
        assert!(view.import_status.contains("line 1"));

        std::fs::remove_file(&path).unwrap();
    }
}
