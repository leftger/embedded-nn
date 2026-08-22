use crate::state::{ModelArchitecture, QuantizationMode, StudioState};
use eframe::egui;

#[derive(Default)]
pub struct TrainView;

impl TrainView {
    pub fn new() -> Self {
        Self
    }

    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut StudioState) {
        ui.horizontal(|ui| {
            ui.heading("🧠 3. Trainer");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let latest_acc = state.val_acc_history.last().copied().unwrap_or(0.0);
                ui.colored_label(
                    if latest_acc > 90.0 {
                        egui::Color32::from_rgb(80, 220, 120)
                    } else {
                        egui::Color32::from_rgb(240, 180, 60)
                    },
                    format!(
                        "Accuracy: {:.1}% | Epoch: {}/{}",
                        latest_acc, state.current_epoch, state.model_config.epochs
                    ),
                );
            });
        });

        ui.add_space(4.0);
        ui.label(
            "Integrated Production QAT/PTQ Trainer with Adam optimizer, SpecAugment, and fake-quant straight-through estimators for Dense MLP, TinyConv1D, and Recurrent SVDF.",
        );
        ui.label(format!(
            "Active model source: {}",
            state.model_source.display_name()
        ));
        ui.add_space(8.0);

        if state.model_source.is_imported() {
            ui.colored_label(
                egui::Color32::from_rgb(100, 200, 240),
                "Training controls are disabled for imported models; the imported graph remains the source of truth.",
            );
            if ui.button("Switch to Studio Trainer").clicked() {
                state.use_demo_trainer();
            }
            return;
        }

        let mut config_changed = false;

        // Model Config Controls
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Architecture:");
                let prev_arch = state.model_config.arch;
                egui::ComboBox::from_id_salt("train_arch_type_combo")
                    .selected_text(match state.model_config.arch {
                        ModelArchitecture::DenseMLP => "Dense MLP (Tabular/Sensor)",
                        ModelArchitecture::TinyConv1D => "TinyConv1D (Temporal Patterns)",
                        ModelArchitecture::RecurrentSVDF => "Recurrent SVDF",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.model_config.arch,
                            ModelArchitecture::DenseMLP,
                            "Dense MLP (Tabular/Sensor)",
                        );
                        ui.selectable_value(
                            &mut state.model_config.arch,
                            ModelArchitecture::TinyConv1D,
                            "TinyConv1D (Temporal Patterns)",
                        );
                        ui.selectable_value(
                            &mut state.model_config.arch,
                            ModelArchitecture::RecurrentSVDF,
                            "Recurrent SVDF",
                        );
                    });
                if state.model_config.arch != prev_arch {
                    config_changed = true;
                }

                ui.separator();

                ui.label("Hidden Units:");
                let prev_hidden = state.model_config.hidden_units;
                egui::ComboBox::from_id_salt("train_hidden_units_combo")
                    .selected_text(format!("{} units", state.model_config.hidden_units))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.model_config.hidden_units,
                            8,
                            "8 units (Ultra-lean)",
                        );
                        ui.selectable_value(
                            &mut state.model_config.hidden_units,
                            16,
                            "16 units (Balanced)",
                        );
                        ui.selectable_value(
                            &mut state.model_config.hidden_units,
                            32,
                            "32 units (Higher Capacity)",
                        );
                    });
                if state.model_config.hidden_units != prev_hidden {
                    config_changed = true;
                }

                ui.separator();

                ui.label("Target Quantization:");
                let prev_q = state.model_config.quant_mode;
                egui::ComboBox::from_id_salt("train_quant_type_combo")
                    .selected_text(match state.model_config.quant_mode {
                        QuantizationMode::Int4SubByte => "s4 (4-bit sub-byte packed, 50% Flash)",
                        QuantizationMode::Int8FixedPoint => "s8 (8-bit fixed-point CMSIS-NN)",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.model_config.quant_mode,
                            QuantizationMode::Int4SubByte,
                            "s4 (4-bit sub-byte packed, 50% Flash)",
                        );
                        ui.selectable_value(
                            &mut state.model_config.quant_mode,
                            QuantizationMode::Int8FixedPoint,
                            "s8 (8-bit fixed-point CMSIS-NN)",
                        );
                    });
                if state.model_config.quant_mode != prev_q {
                    config_changed = true;
                }
            });

            ui.add_space(4.0);

            ui.horizontal(|ui| {
                ui.label("Epochs:");
                ui.add(egui::DragValue::new(&mut state.model_config.epochs).range(10..=300));

                ui.separator();

                ui.label("Learning Rate:");
                ui.add(
                    egui::DragValue::new(&mut state.model_config.learning_rate)
                        .speed(0.002)
                        .range(0.001..=0.1),
                );

                ui.separator();

                let train_btn = if state.is_training {
                    egui::Button::new("⏸ Pause").fill(egui::Color32::from_rgb(160, 100, 40))
                } else {
                    egui::Button::new("▶ Run Training").fill(egui::Color32::from_rgb(40, 130, 70))
                };
                if ui.add(train_btn).clicked() {
                    state.is_training = !state.is_training;
                }

                if ui.button("⏭ Step 10 Epochs").clicked() {
                    state.run_simulated_training(10);
                    state.rebuild_model_graph_and_codegen();
                }

                if ui.button("🔥 Burn PTQ").clicked() {
                    state.run_burn_training(false);
                }
                if ui.button("🔥 Burn QAT").clicked() {
                    state.run_burn_training(true);
                }

                if ui.button("🔄 Reset Weights").clicked() {
                    state.reset_training();
                    state.rebuild_model_graph_and_codegen();
                }
            });

            ui.add_space(6.0);

            // Data Augmentation & SpecAugment Sub-panel
            ui.horizontal(|ui| {
                ui.checkbox(
                    &mut state.model_config.enable_augmentation,
                    "⚡ Data Augmentation & SpecAugment",
                );
                if state.model_config.enable_augmentation {
                    ui.separator();
                    ui.label("Noise σ:");
                    ui.add(
                        egui::DragValue::new(&mut state.model_config.augment_config.noise_std_dev)
                            .speed(0.005)
                            .range(0.0..=0.1)
                            .suffix(" g"),
                    );

                    ui.separator();
                    ui.label("SpecAug Freq Mask:");
                    ui.add(
                        egui::DragValue::new(
                            &mut state.model_config.augment_config.max_freq_mask_channels,
                        )
                        .range(0..=4)
                        .suffix(" bins"),
                    );

                    ui.separator();
                    ui.label("Time Mask:");
                    ui.add(
                        egui::DragValue::new(
                            &mut state.model_config.augment_config.max_time_mask_frames,
                        )
                        .range(0..=4)
                        .suffix(" frames"),
                    );
                }
            });
        });

        if config_changed {
            state.reset_training();
            state.run_simulated_training(30);
            state.rebuild_model_graph_and_codegen();
        }

        // Advance training if active
        if state.is_training {
            state.step_training_epoch();
            state.rebuild_model_graph_and_codegen();
        }

        ui.add_space(8.0);

        // Loss/Accuracy Plots & Confusion Matrix
        ui.columns(2, |cols| {
            // Left Column: Loss & Accuracy Curves
            cols[0].group(|ui| {
                ui.label("📉 Convergence Metrics (Training Loss & Accuracy)");

                let (rect, _resp) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), 160.0),
                    egui::Sense::hover(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(16, 20, 28));

                // Draw background grid lines
                for y_ratio in [0.25, 0.5, 0.75] {
                    let y = rect.top() + y_ratio * rect.height();
                    painter.line_segment(
                        [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                        egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(26, 32, 44)),
                    );
                }

                // Plot Loss Curve (Orange)
                let n_loss = state.train_loss_history.len();
                if n_loss > 1 {
                    let max_loss = state
                        .train_loss_history
                        .first()
                        .copied()
                        .unwrap_or(1.0)
                        .max(1.0);
                    let dx = rect.width() / (n_loss - 1) as f32;
                    for i in 0..n_loss - 1 {
                        let l1 = (state.train_loss_history[i] / max_loss).clamp(0.0, 1.0);
                        let l2 = (state.train_loss_history[i + 1] / max_loss).clamp(0.0, 1.0);
                        let p1 = egui::pos2(
                            rect.left() + i as f32 * dx,
                            rect.bottom() - 10.0 - l1 * (rect.height() - 20.0),
                        );
                        let p2 = egui::pos2(
                            rect.left() + (i + 1) as f32 * dx,
                            rect.bottom() - 10.0 - l2 * (rect.height() - 20.0),
                        );
                        painter.line_segment(
                            [p1, p2],
                            egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(240, 160, 50)),
                        );
                    }
                }

                // Plot Accuracy Curve (Green)
                let n_acc = state.val_acc_history.len();
                if n_acc > 1 {
                    let dx = rect.width() / (n_acc - 1) as f32;
                    for i in 0..n_acc - 1 {
                        let a1 = state.val_acc_history[i] / 100.0;
                        let a2 = state.val_acc_history[i + 1] / 100.0;
                        let p1 = egui::pos2(
                            rect.left() + i as f32 * dx,
                            rect.bottom() - 10.0 - a1 * (rect.height() - 20.0),
                        );
                        let p2 = egui::pos2(
                            rect.left() + (i + 1) as f32 * dx,
                            rect.bottom() - 10.0 - a2 * (rect.height() - 20.0),
                        );
                        painter.line_segment(
                            [p1, p2],
                            egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(70, 210, 120)),
                        );
                    }
                }

                ui.horizontal(|ui| {
                    ui.colored_label(egui::Color32::from_rgb(240, 160, 50), "— Loss");
                    ui.colored_label(egui::Color32::from_rgb(70, 210, 120), "— Accuracy");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let loss_val = state.train_loss_history.last().copied().unwrap_or(0.0);
                        ui.label(format!("Final Loss: {:.4}", loss_val));
                    });
                });
            });

            // Right Column: Confusion Matrix Heatmap
            cols[1].group(|ui| {
                ui.label("🎯 Multi-Class Confusion Matrix");

                let num_classes = state.classes.len();
                let (rect, _resp) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), 160.0),
                    egui::Sense::hover(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(16, 20, 28));

                if num_classes > 0 {
                    let cell_w = rect.width() / num_classes as f32;
                    let cell_h = rect.height() / num_classes as f32;

                    for r in 0..num_classes {
                        for c in 0..num_classes {
                            let count = state
                                .confusion_matrix
                                .get(r)
                                .and_then(|row| row.get(c))
                                .copied()
                                .unwrap_or(0);
                            let cell_rect = egui::Rect::from_min_size(
                                egui::pos2(
                                    rect.left() + c as f32 * cell_w + 1.0,
                                    rect.top() + r as f32 * cell_h + 1.0,
                                ),
                                egui::vec2(cell_w - 2.0, cell_h - 2.0),
                            );

                            let col = if r == c {
                                // Diagonal: correct predictions (green)
                                let intensity = ((count as f32 * 12.0).clamp(40.0, 220.0)) as u8;
                                egui::Color32::from_rgb(20, intensity, 40)
                            } else {
                                // Off-diagonal: errors (red / neutral)
                                if count > 0 {
                                    egui::Color32::from_rgb(160, 40, 40)
                                } else {
                                    egui::Color32::from_rgb(22, 28, 38)
                                }
                            };

                            painter.rect_filled(cell_rect, 2.0, col);
                            painter.text(
                                cell_rect.center(),
                                egui::Align2::CENTER_CENTER,
                                format!("{}", count),
                                egui::FontId::proportional(12.0),
                                egui::Color32::WHITE,
                            );
                        }
                    }
                }

                ui.horizontal(|ui| {
                    ui.label("Rows = Ground Truth | Columns = Predicted");
                });
            });
        });
    }
}
