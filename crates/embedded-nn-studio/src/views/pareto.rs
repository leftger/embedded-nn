//! Interactive Auto-TinyML Pareto Frontier Explorer View.
//!
//! Visualizes the trade-off space across Accuracy, SRAM Arena, Flash Memory, and Latency,
//! allowing the developer to select optimal configurations with one click.

use crate::state::{ModelArchitecture, QuantizationMode, StudioState};
use eframe::egui;
use embedded_nn_train::pareto::evaluate_pareto_candidates;

pub fn render(ui: &mut egui::Ui, state: &mut StudioState) {
    ui.heading("⚡ Auto-TinyML & Pareto Frontier Explorer");
    ui.label(
        "Explore multi-objective trade-offs between on-device Accuracy, SRAM Arena, Flash Footprint, and Latency.",
    );

    ui.add_space(8.0);

    let candidates = evaluate_pareto_candidates(state.dsp.num_mel_bins, state.classes.len().max(2));

    ui.horizontal(|ui| {
        ui.label(format!("Evaluated Configurations: {}", candidates.len()));
        ui.separator();
        let optimal_count = candidates.iter().filter(|c| c.is_pareto_optimal).count();
        ui.label(
            egui::RichText::new(format!("★ {} Pareto-Optimal Candidates", optimal_count))
                .color(egui::Color32::from_rgb(60, 200, 100)),
        );
    });

    ui.add_space(8.0);

    // Render interactive 2D trade-off canvas
    egui::Frame::canvas(ui.style()).show(ui, |ui| {
        let (rect, _resp) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), 260.0),
            egui::Sense::hover(),
        );
        let painter = ui.painter_at(rect);

        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(20, 24, 30));
        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(45, 55, 70)),
            egui::StrokeKind::Inside,
        );

        // Draw grid lines
        for step in 1..5 {
            let y = rect.top() + (rect.height() / 5.0) * step as f32;
            painter.line_segment(
                [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                egui::Stroke::new(0.5_f32, egui::Color32::from_rgb(35, 45, 60)),
            );
        }

        // Draw axis labels
        painter.text(
            egui::pos2(rect.left() + 10.0, rect.top() + 10.0),
            egui::Align2::LEFT_TOP,
            "Accuracy (Higher is Better)",
            egui::FontId::proportional(11.0),
            egui::Color32::from_rgb(160, 175, 195),
        );
        painter.text(
            egui::pos2(rect.right() - 10.0, rect.bottom() - 10.0),
            egui::Align2::RIGHT_BOTTOM,
            "SRAM Arena + Flash (Lower is Better) →",
            egui::FontId::proportional(11.0),
            egui::Color32::from_rgb(160, 175, 195),
        );

        // Plot candidate points
        for c in &candidates {
            // Map SRAM (32..128) to X, Accuracy (0.85..1.0) to Y
            let norm_x = ((c.sram_arena_bytes as f32 - 20.0) / 120.0).clamp(0.1, 0.9);
            let norm_y = ((c.accuracy - 0.85) / 0.15).clamp(0.1, 0.9);

            let pt_x = rect.left() + rect.width() * norm_x;
            let pt_y = rect.bottom() - rect.height() * norm_y;
            let pt = egui::pos2(pt_x, pt_y);

            let (color, radius) = if c.is_pareto_optimal {
                (egui::Color32::from_rgb(50, 220, 120), 7.0)
            } else {
                (egui::Color32::from_rgb(120, 130, 145), 4.0)
            };

            painter.circle_filled(pt, radius, color);
            painter.text(
                egui::pos2(pt.x + 10.0, pt.y - 4.0),
                egui::Align2::LEFT_CENTER,
                &c.name,
                egui::FontId::proportional(11.0),
                if c.is_pareto_optimal {
                    egui::Color32::WHITE
                } else {
                    egui::Color32::from_rgb(150, 160, 175)
                },
            );
        }
    });

    ui.add_space(10.0);

    // Candidates Table
    egui::Grid::new("pareto_table")
        .striped(true)
        .min_col_width(90.0)
        .show(ui, |ui| {
            ui.strong("Configuration");
            ui.strong("Architecture");
            ui.strong("Precision");
            ui.strong("Accuracy");
            ui.strong("Flash");
            ui.strong("SRAM Arena");
            ui.strong("Est. Cycles");
            ui.strong("Pareto?");
            ui.strong("Action");
            ui.end_row();

            for c in &candidates {
                ui.label(&c.name);
                ui.label(&c.arch_name);
                ui.label(format!("{}-bit", c.quant_bits));
                ui.label(format!("{:.1}%", c.accuracy * 100.0));
                ui.label(format!("{} B", c.flash_bytes));
                ui.label(format!("{} B", c.sram_arena_bytes));
                ui.label(format!("{} cyc", c.estimated_cycles));
                if c.is_pareto_optimal {
                    ui.colored_label(egui::Color32::from_rgb(60, 200, 100), "★ Optimal");
                } else {
                    ui.label("Dominated");
                }

                if ui.button("Apply").clicked() {
                    match c.arch_name.as_str() {
                        "DenseMLP" => state.model_config.arch = ModelArchitecture::DenseMLP,
                        "TinyConv1D" => state.model_config.arch = ModelArchitecture::TinyConv1D,
                        "RecurrentSVDF" => {
                            state.model_config.arch = ModelArchitecture::RecurrentSVDF
                        }
                        _ => {}
                    }
                    state.model_config.hidden_units = c.hidden_units;
                    state.model_config.quant_mode = if c.quant_bits == 4 {
                        QuantizationMode::Int4SubByte
                    } else {
                        QuantizationMode::Int8FixedPoint
                    };
                    state.reset_training();
                    state.rebuild_model_graph_and_codegen();
                }
                ui.end_row();
            }
        });
}
