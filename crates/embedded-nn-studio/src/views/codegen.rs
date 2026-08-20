use crate::state::StudioState;
use crate::syntax::highlight_rust;
use eframe::egui;

/// Inner/outer margins of an `egui` group frame, which shrink the space usable by its contents.
const GROUP_VERTICAL_PADDING: f32 = 16.0;

#[derive(Default)]
pub struct CodegenView {
    pub copy_status: Option<String>,
    pub selected_test_sample_idx: usize,
}

impl CodegenView {
    pub fn new() -> Self {
        Self {
            copy_status: None,
            selected_test_sample_idx: 0,
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut StudioState) {
        ui.horizontal(|ui| {
            ui.heading("⚡ 5. Zero-Allocation #![no_std] Rust Code Generator & HIL Playground");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("💾 Export to src/model.rs").clicked() {
                    let _ = std::fs::write("model.rs", &state.generated_rust_code);
                    self.copy_status = Some("Saved model to ./model.rs".into());
                }

                if ui.button("📋 Copy Code").clicked() {
                    ui.ctx().copy_text(state.generated_rust_code.clone());
                    self.copy_status = Some("Rust code copied to clipboard!".into());
                }

                if let Some(status) = &self.copy_status {
                    ui.colored_label(egui::Color32::from_rgb(100, 220, 140), status);
                }
            });
        });

        ui.add_space(4.0);
        ui.label(
            "Directly compiles the trained network weights, quantization parameters, and static SRAM arena offsets into a standalone #![no_std] Rust module calling embedded-nn CMSIS-NN/s4 kernels with zero heap allocation.",
        );
        ui.add_space(8.0);

        // Split Layout: Left is Interactive Live Inference Playground, Right is Generated Rust Code
        let column_height = ui.available_height();
        ui.columns(2, |cols| {
            // Left Column: Interactive Inference Playground
            cols[0].set_min_height(column_height);
            cols[0].group(|ui| {
                ui.set_min_height(column_height - GROUP_VERTICAL_PADDING);
                ui.horizontal(|ui| {
                    ui.label("🎮 Live Virtual Inference Playground");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("⚡ Run Predict").clicked() {
                            state.run_test_inference();
                        }
                    });
                });

                ui.add_space(4.0);

                if !state.samples.is_empty() {
                    ui.horizontal(|ui| {
                        ui.label("Load Dataset Vector:");
                        self.selected_test_sample_idx =
                            self.selected_test_sample_idx.min(state.samples.len() - 1);
                        let prev_idx = self.selected_test_sample_idx;
                        egui::ComboBox::from_id_salt("codegen_test_sample_combo")
                            .selected_text(format!(
                                "Sample #{:03} ({})",
                                state.samples[self.selected_test_sample_idx].id,
                                state.samples[self.selected_test_sample_idx].label
                            ))
                            .show_ui(ui, |ui| {
                                for (idx, s) in state.samples.iter().enumerate().take(30) {
                                    ui.selectable_value(
                                        &mut self.selected_test_sample_idx,
                                        idx,
                                        format!("#{:03} - {}", s.id, s.label),
                                    );
                                }
                            });

                        if self.selected_test_sample_idx != prev_idx {
                            let num_mel_bins = state.dsp.num_mel_bins;
                            let sample = &state.samples[self.selected_test_sample_idx];
                            state.test_input_vector = StudioState::test_input_vector_for(
                                state.model_config.arch,
                                num_mel_bins,
                                sample,
                            );
                            state.run_test_inference();
                        }
                    });
                }

                ui.separator();
                ui.label("Predicted Class Probabilities (Softmax):");
                ui.add_space(4.0);

                let mut highest_idx = 0;
                let mut highest_prob = -1.0;
                for (i, &prob) in state.test_probabilities.iter().enumerate() {
                    if prob > highest_prob {
                        highest_prob = prob;
                        highest_idx = i;
                    }
                }

                for (i, class_name) in state.classes.iter().enumerate() {
                    let prob = state.test_probabilities.get(i).copied().unwrap_or(0.0);
                    let is_winner = i == highest_idx;

                    ui.horizontal(|ui| {
                        if is_winner {
                            ui.colored_label(
                                egui::Color32::from_rgb(80, 220, 120),
                                format!("▶ {}:", class_name),
                            );
                        } else {
                            ui.label(format!("  {}:", class_name));
                        }

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(format!("{:.1}%", prob * 100.0));
                            ui.add(egui::ProgressBar::new(prob).desired_width(120.0));
                        });
                    });
                }

                ui.separator();
                ui.label("Input Feature Sliders (Int8 quantized):");
                egui::ScrollArea::vertical()
                    .id_salt("codegen_feature_sliders_scroll")
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
                        let mut changed = false;
                        for (i, val) in state.test_input_vector.iter_mut().enumerate() {
                            ui.push_id(i, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(format!("Bin {:02}:", i));
                                    if ui.add(egui::Slider::new(val, -128..=127)).changed() {
                                        changed = true;
                                    }
                                });
                            });
                        }
                        if changed {
                            state.run_test_inference();
                        }
                    });
            });

            // Right Column: Generated #![no_std] Rust Source Code with Syntax Highlighting
            cols[1].set_min_height(column_height);
            cols[1].group(|ui| {
                ui.set_min_height(column_height - GROUP_VERTICAL_PADDING);
                ui.label("📄 Generated #![no_std] Rust Module");
                ui.separator();

                let mut layouter = |ui: &egui::Ui, text: &str, wrap_width: f32| {
                    let mut layout_job = highlight_rust(ui.ctx(), text);
                    layout_job.wrap.max_width = wrap_width;
                    ui.fonts(|f| f.layout_job(layout_job))
                };

                egui::ScrollArea::vertical()
                    .id_salt("codegen_rust_code_scroll")
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut state.generated_rust_code)
                                .font(egui::TextStyle::Monospace)
                                .layouter(&mut layouter)
                                .lock_focus(true)
                                .desired_width(f32::INFINITY),
                        );
                    });
            });
        });
    }
}
