use eframe::egui;

#[derive(Default)]
pub struct ArenaView {
    pub flash_bytes: usize,
    pub sram_arena_bytes: usize,
    pub target_mcu: String,
    pub estimated_cycles: u32,
}

impl ArenaView {
    pub fn new() -> Self {
        Self {
            flash_bytes: 4280,
            sram_arena_bytes: 512,
            target_mcu: "STM32F401 (Cortex-M4F @ 84MHz)".into(),
            estimated_cycles: 12400,
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        ui.heading("4. Static Memory Arena & Target MCU Profiler");
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.label("Target Architecture:");
            egui::ComboBox::from_id_salt("target_mcu_combo")
                .selected_text(&self.target_mcu)
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.target_mcu,
                        "STM32F401 (Cortex-M4F @ 84MHz)".into(),
                        "STM32F401 (Cortex-M4F @ 84MHz)",
                    );
                    ui.selectable_value(
                        &mut self.target_mcu,
                        "ESP32-S3 (Xtensa Dual @ 240MHz)".into(),
                        "ESP32-S3 (Xtensa Dual @ 240MHz)",
                    );
                    ui.selectable_value(
                        &mut self.target_mcu,
                        "RP2040 / RP2350 (Cortex-M0+/M33)".into(),
                        "RP2040 / RP2350 (Cortex-M0+/M33)",
                    );
                    ui.selectable_value(
                        &mut self.target_mcu,
                        "nRF52840 (Cortex-M4F @ 64MHz)".into(),
                        "nRF52840 (Cortex-M4F @ 64MHz)",
                    );
                });
        });

        ui.add_space(10.0);
        ui.separator();

        ui.columns(3, |cols| {
            cols[0].group(|ui| {
                ui.label("⚡ Flash Footprint (Weights & Code)");
                ui.heading(format!("{:.2} KB", self.flash_bytes as f32 / 1024.0));
                ui.label(format!("{} bytes static consts", self.flash_bytes));
            });
            cols[1].group(|ui| {
                ui.label("🧠 Peak SRAM Arena (Zero Alloc)");
                ui.heading(format!("{} Bytes", self.sram_arena_bytes));
                ui.label("100% compile-time planned");
            });
            cols[2].group(|ui| {
                ui.label("⏱ Estimated Latency");
                ui.heading(format!(
                    "{:.2} ms",
                    (self.estimated_cycles as f32) / 84000.0
                ));
                ui.label(format!("{} CPU cycles", self.estimated_cycles));
            });
        });

        ui.add_space(12.0);
        ui.label("Arena Buffer Lifetime Allocation Map:");

        let (rect, _response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), 120.0),
            egui::Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(20, 24, 30));

        // Draw allocated blocks
        let blocks = [
            (
                "Input Buffer",
                0.0,
                0.25,
                0.0,
                0.4,
                egui::Color32::from_rgb(60, 120, 200),
            ),
            (
                "Conv1 Scratch",
                0.2,
                0.6,
                0.35,
                0.7,
                egui::Color32::from_rgb(200, 100, 60),
            ),
            (
                "FC Activation",
                0.5,
                0.9,
                0.1,
                0.5,
                egui::Color32::from_rgb(80, 180, 100),
            ),
            (
                "Output Logits",
                0.8,
                1.0,
                0.6,
                0.9,
                egui::Color32::from_rgb(180, 60, 180),
            ),
        ];

        for (name, x1, x2, y1, y2, col) in blocks {
            let block_rect = egui::Rect::from_min_max(
                egui::pos2(
                    rect.left() + x1 * rect.width(),
                    rect.top() + y1 * rect.height(),
                ),
                egui::pos2(
                    rect.left() + x2 * rect.width(),
                    rect.top() + y2 * rect.height(),
                ),
            );
            painter.rect_filled(block_rect, 2.0, col);
            painter.text(
                block_rect.center(),
                egui::Align2::CENTER_CENTER,
                name,
                egui::FontId::proportional(12.0),
                egui::Color32::WHITE,
            );
        }
    }
}
