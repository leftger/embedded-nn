use crate::state::StudioState;
use eframe::egui;

#[derive(Default)]
pub struct IngestView {
    pub selected_port: String,
    pub sample_rate_hz: u32,
    pub current_label: String,
    pub new_class_name: String,
    pub is_recording: bool,
    pub live_sensor_history: Vec<f32>,
    pub live_time_counter: f32,
}

impl IngestView {
    pub fn new() -> Self {
        Self {
            selected_port: "USB-CDC (ACM0)".into(),
            sample_rate_hz: 100,
            current_label: "wave_left".into(),
            new_class_name: String::new(),
            is_recording: false,
            live_sensor_history: (0..100).map(|i| ((i as f32) * 0.1).sin() * 0.7).collect(),
            live_time_counter: 0.0,
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut StudioState) {
        ui.horizontal(|ui| {
            ui.heading("📊 1. Data Ingestion, Telemetry & Dataset Tagging");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("🔄 Reset to Demo Dataset").clicked() {
                    state.load_demo_dataset();
                    state.recompute_all_features();
                    state.rebuild_model_graph_and_codegen();
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
                            "USB-CDC (ACM0)".into(),
                            "🔌 USB-CDC (ACM0)",
                        );
                        ui.selectable_value(
                            &mut self.selected_port,
                            "ST-Link V3 Telemetry".into(),
                            "⚡ ST-Link V3 Telemetry",
                        );
                        ui.selectable_value(
                            &mut self.selected_port,
                            "Simulated IMU Source".into(),
                            "💻 Simulated IMU Source",
                        );
                    });

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

                        let feats = state.extract_features_from_waveform(&self.live_sensor_history);
                        let q_feats: Vec<i8> = feats
                            .iter()
                            .map(|&f| ((f * 127.0).round().clamp(-128.0, 127.0)) as i8)
                            .collect();

                        state.samples.push(crate::state::DatasetSample {
                            id,
                            label: self.current_label.clone(),
                            class_idx,
                            raw_waveform: self.live_sensor_history.clone(),
                            features: feats,
                            quantized_features: q_feats,
                        });
                        state.rebuild_model_graph_and_codegen();
                    }
                }
            });
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
                            for sample in state.samples.iter().rev().take(30) {
                                ui.label(format!("#{:03}", sample.id));
                                ui.colored_label(
                                    egui::Color32::from_rgb(100, 180, 255),
                                    &sample.label,
                                );
                                ui.label(format!("{} samples", sample.raw_waveform.len()));
                                ui.label(format!("{} Mel bins", sample.features.len()));

                                if ui.button("🗑 Delete").clicked() {
                                    delete_id = Some(sample.id);
                                }
                                ui.end_row();
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
