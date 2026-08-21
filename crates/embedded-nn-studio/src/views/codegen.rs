use crate::state::StudioState;
use crate::syntax::highlight_rust;
use eframe::egui;
use embedded_nn_live::host::{DeviceLink, OwnedMsg};
use std::path::Path;

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

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        state: &mut StudioState,
        device_link: Option<&DeviceLink>,
    ) {
        if let Some(link) = device_link {
            match link.take_result() {
                Some(OwnedMsg::InferenceResult {
                    execution_cycles,
                    logits,
                    ..
                }) => {
                    state.apply_device_inference(execution_cycles, &logits);
                }
                Some(OwnedMsg::Pong) | None => {}
                Some(other) => {
                    state.golden_status = Some(format!("Device message: {other:?}"));
                }
            }
            if let Some(error) = link.take_error() {
                state.golden_status = Some(format!("HIL: {error}"));
            }
        }

        ui.horizontal(|ui| {
            ui.heading("⚡ 5. Zero-Allocation #![no_std] Rust Code Generator & HIL Playground");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(
                        state.export_enabled(),
                        egui::Button::new("💾 Export model.rs"),
                    )
                    .clicked()
                {
                    self.copy_status =
                        Some(match state.export_model_bundle(Path::new("model.rs")) {
                            Ok(message) => message,
                            Err(error) => error,
                        });
                }

                if ui
                    .add_enabled(state.export_enabled(), egui::Button::new("📋 Copy Code"))
                    .clicked()
                {
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
            "Generates a standalone #![no_std] Rust module from the active ModelGraph, using embedded-nn integer kernels and static arena offsets.",
        );
        ui.label(format!("Source: {}", state.model_source.display_name()));
        if !state.production_export_eligible() {
            ui.colored_label(
                egui::Color32::from_rgb(245, 170, 60),
                "WARNING: Demo trainer output is educational and is not production-qualified.",
            );
            ui.checkbox(
                &mut state.allow_demo_export,
                "I understand; enable copy/export of this demo model",
            );
        }
        if ui
            .add_enabled(
                matches!(
                    state.model_source,
                    crate::state::ModelSource::ImportedTflite(_)
                ),
                egui::Button::new("Compare TFLite golden"),
            )
            .clicked()
        {
            state.compare_imported_tflite_golden();
        }
        if let Some(status) = &state.golden_status {
            ui.colored_label(egui::Color32::from_rgb(180, 210, 255), status);
        }
        if let Some(graph) = &state.compiled_graph {
            for (index, tensor_id) in graph.inputs.iter().enumerate() {
                if let Some(tensor) = graph.tensors.iter().find(|tensor| tensor.id == *tensor_id) {
                    ui.label(format!(
                        "Input {index} '{}': [{}, {}, {}, {}], {:?}, scale {}, zero-point {}",
                        tensor.name,
                        tensor.shape.batches,
                        tensor.shape.height,
                        tensor.shape.width,
                        tensor.shape.channels,
                        tensor.dtype,
                        tensor.quant.scale,
                        tensor.quant.zero_point
                    ));
                }
            }
            for (index, tensor_id) in graph.outputs.iter().enumerate() {
                if let Some(tensor) = graph.tensors.iter().find(|tensor| tensor.id == *tensor_id) {
                    ui.label(format!(
                        "Output {index} '{}': {} values, scale {}, zero-point {}",
                        tensor.name,
                        tensor.shape.total_elements(),
                        tensor.quant.scale,
                        tensor.quant.zero_point
                    ));
                }
            }
        }
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
                        let device_ready = device_link.is_some_and(|link| link.is_handshaked());
                        if ui
                            .add_enabled(device_ready, egui::Button::new("Run on device"))
                            .clicked()
                        {
                            state.run_test_inference();
                            if let Some(link) = device_link {
                                let input: Vec<u8> = state
                                    .test_input_vector
                                    .iter()
                                    .map(|&value| value as u8)
                                    .collect();
                                link.infer(1, 0, input);
                            }
                        }
                    });
                });

                ui.add_space(4.0);

                if !state.model_source.is_imported() && !state.samples.is_empty() {
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
                ui.label(if state.model_source.is_imported() {
                    "Integer graph output (dequantized for probability display):"
                } else {
                    "Demo float preview probabilities:"
                });
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

                if let Some(cycles) = state.last_device_cycles {
                    ui.label(format!(
                        "Device: {cycles} DWT cycles, logits {:?}",
                        state.last_device_logits
                    ));
                    if !state.last_device_logits.is_empty()
                        && state.last_device_logits == state.test_output_logits
                    {
                        ui.colored_label(
                            egui::Color32::from_rgb(80, 220, 120),
                            "Device logits match host interpreter.",
                        );
                    }
                }

                ui.separator();
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
                let mut additional_changed = false;
                for (input_index, values) in
                    state.test_additional_input_vectors.iter_mut().enumerate()
                {
                    ui.separator();
                    ui.label(format!(
                        "Input Tensor {} (Int8 quantized):",
                        input_index + 1
                    ));
                    let mut changed = false;
                    for (value_index, value) in values.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(format!("Value {value_index}:"));
                            if ui.add(egui::Slider::new(value, -128..=127)).changed() {
                                changed = true;
                            }
                        });
                    }
                    additional_changed |= changed;
                }
                if additional_changed {
                    state.run_test_inference();
                }
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
