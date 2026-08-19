use crate::state::StudioState;
use eframe::egui;

#[derive(Debug, Clone)]
struct McuSpec {
    name: &'static str,
    arch: &'static str,
    freq_mhz: u32,
    sram_kb: usize,
    flash_kb: usize,
}

const MCU_TARGETS: &[McuSpec] = &[
    McuSpec {
        name: "STM32F401RE",
        arch: "ARM Cortex-M4F (FPU)",
        freq_mhz: 84,
        sram_kb: 64,
        flash_kb: 512,
    },
    McuSpec {
        name: "ESP32-S3",
        arch: "Xtensa 32-bit LX7 Dual",
        freq_mhz: 240,
        sram_kb: 512,
        flash_kb: 8192,
    },
    McuSpec {
        name: "RP2040",
        arch: "Dual ARM Cortex-M0+",
        freq_mhz: 133,
        sram_kb: 264,
        flash_kb: 2048,
    },
    McuSpec {
        name: "RP2350",
        arch: "Dual ARM Cortex-M33",
        freq_mhz: 150,
        sram_kb: 520,
        flash_kb: 4096,
    },
    McuSpec {
        name: "nRF52840",
        arch: "ARM Cortex-M4F (BLE)",
        freq_mhz: 64,
        sram_kb: 256,
        flash_kb: 1024,
    },
];

#[derive(Default)]
pub struct ArenaView {
    pub selected_mcu_idx: usize,
}

impl ArenaView {
    pub fn new() -> Self {
        Self {
            selected_mcu_idx: 0,
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut StudioState) {
        ui.horizontal(|ui| {
            ui.heading("🔬 4. Static Memory Arena Scheduler & MCU Silicon Profiler");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if let Some(plan) = &state.arena_plan {
                    ui.colored_label(
                        egui::Color32::from_rgb(100, 200, 255),
                        format!(
                            "Static Arena: {} Bytes (Zero Heap Allocation)",
                            plan.total_arena_bytes
                        ),
                    );
                }
            });
        });

        ui.add_space(4.0);
        ui.label(
            "Ahead-of-Time static lifetime scheduler analyzes tensor birth and death intervals, allocating physical memory offsets in a single continuous SRAM arena buffer. Zero heap fragmentation, zero runtime malloc.",
        );
        ui.add_space(8.0);

        let mcu = &MCU_TARGETS[self.selected_mcu_idx];

        // Target Hardware Selector
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Target Microcontroller:");
                egui::ComboBox::from_id_salt("mcu_select_combo")
                    .selected_text(format!("{} - {}", mcu.name, mcu.arch))
                    .show_ui(ui, |ui| {
                        for (idx, target) in MCU_TARGETS.iter().enumerate() {
                            ui.selectable_value(
                                &mut self.selected_mcu_idx,
                                idx,
                                format!("{} - {}", target.name, target.arch),
                            );
                        }
                    });

                ui.separator();
                ui.label(format!("Clock: {} MHz", mcu.freq_mhz));
                ui.separator();
                ui.label(format!("SRAM: {} KB", mcu.sram_kb));
                ui.separator();
                ui.label(format!("Flash: {} KB", mcu.flash_kb));
            });
        });

        ui.add_space(8.0);

        let flash_weights = state
            .compiled_graph
            .as_ref()
            .map(|g| g.total_weights_size_bytes())
            .unwrap_or(0);
        let sram_arena = state
            .arena_plan
            .as_ref()
            .map(|p| p.total_arena_bytes)
            .unwrap_or(0);

        let sram_budget_bytes = mcu.sram_kb * 1024;
        let flash_budget_bytes = mcu.flash_kb * 1024;

        let sram_ratio = (sram_arena as f32 / sram_budget_bytes as f32).clamp(0.0, 1.0);
        let flash_ratio = (flash_weights as f32 / flash_budget_bytes as f32).clamp(0.0, 1.0);

        // Hardware Resource Gauges
        ui.columns(3, |cols| {
            cols[0].group(|ui| {
                ui.label("⚡ Flash (ROM) Footprint");
                ui.heading(format!("{:.2} KB", flash_weights as f32 / 1024.0));
                ui.label(format!("{} bytes const weights & biases", flash_weights));
                ui.add(
                    egui::ProgressBar::new(flash_ratio)
                        .text(format!("{:.3}% of Target Flash", flash_ratio * 100.0)),
                );
            });

            cols[1].group(|ui| {
                ui.label("🧠 SRAM Activation Arena");
                ui.heading(format!("{} Bytes", sram_arena));
                ui.label("100% compile-time planned workspace");
                ui.add(
                    egui::ProgressBar::new(sram_ratio)
                        .text(format!("{:.3}% of Target SRAM", sram_ratio * 100.0)),
                );
            });

            cols[2].group(|ui| {
                ui.label("⏱ Latency Estimate");
                let estimated_macs = flash_weights.max(1);
                let cycles_per_mac = 2; // Average SIMD 4-way MAC on Cortex-M
                let total_cycles = (estimated_macs * cycles_per_mac) as u32;
                let latency_us = total_cycles as f32 / mcu.freq_mhz as f32;

                ui.heading(format!("{:.1} µs", latency_us));
                ui.label(format!(
                    "~{} CPU cycles @ {}MHz",
                    total_cycles, mcu.freq_mhz
                ));
                ui.colored_label(
                    egui::Color32::from_rgb(80, 220, 120),
                    "Ultra Low Power Budget",
                );
            });
        });

        ui.add_space(8.0);

        // Interactive Tensor Lifetime Waterfall Map
        ui.group(|ui| {
            ui.label("📊 SRAM Buffer Lifetime Allocation Waterfall");
            ui.label("Visualizing tensor birth/death execution steps and physical buffer reuse inside ARENA_SIZE_BYTES:");

            let (rect, _resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 160.0), egui::Sense::hover());
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(16, 20, 28));

            if let (Some(graph), Some(plan)) = (&state.compiled_graph, &state.arena_plan) {
                let num_steps = graph.layers.len().max(1) + 1;
                let arena_total = plan.total_arena_bytes.max(1);

                let dx = rect.width() / num_steps as f32;

                // Color palette for tensor blocks
                let colors = [
                    egui::Color32::from_rgb(60, 130, 220),
                    egui::Color32::from_rgb(220, 100, 60),
                    egui::Color32::from_rgb(70, 190, 120),
                    egui::Color32::from_rgb(180, 70, 200),
                    egui::Color32::from_rgb(230, 180, 40),
                ];

                for (idx, (t_id, alloc)) in plan.allocations.iter().enumerate() {
                    let col = colors[idx % colors.len()];

                    let x1 = rect.left() + (alloc.lifetime.start_step as f32) * dx;
                    let x2 = (rect.left() + ((alloc.lifetime.end_step + 1) as f32) * dx).min(rect.right());

                    let y_ratio_start = alloc.byte_offset as f32 / arena_total as f32;
                    let y_ratio_height = alloc.byte_size as f32 / arena_total as f32;

                    let y1 = rect.top() + y_ratio_start * (rect.height() - 20.0) + 10.0;
                    let y2 = (y1 + y_ratio_height * (rect.height() - 20.0)).max(y1 + 14.0);

                    let block_rect = egui::Rect::from_min_max(egui::pos2(x1 + 2.0, y1), egui::pos2(x2 - 2.0, y2));
                    painter.rect_filled(block_rect, 3.0, col);

                    let label = format!("T{:02} [{}] ({} B)", t_id, alloc.name, alloc.byte_size);
                    painter.text(
                        block_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        label,
                        egui::FontId::proportional(11.0),
                        egui::Color32::WHITE,
                    );
                }
            }
        });
    }
}
