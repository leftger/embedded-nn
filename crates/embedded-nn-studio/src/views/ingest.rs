use eframe::egui;

#[derive(Default)]
pub struct IngestView {
    pub selected_port: String,
    pub sample_rate_hz: u32,
    pub current_label: String,
    pub is_recording: bool,
    pub recorded_samples_count: usize,
    pub live_sensor_history: Vec<f32>,
}

impl IngestView {
    pub fn new() -> Self {
        Self {
            selected_port: "USB-CDC (ACM0)".into(),
            sample_rate_hz: 100,
            current_label: "motion_idle".into(),
            is_recording: false,
            recorded_samples_count: 142,
            live_sensor_history: (0..120).map(|i| ((i as f32) * 0.1).sin() * 0.8).collect(),
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        ui.heading("1. Live Data Ingestion & Sensor Scope");
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.label("Sensor Device:");
            egui::ComboBox::from_id_salt("sensor_port_combo")
                .selected_text(&self.selected_port)
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.selected_port,
                        "USB-CDC (ACM0)".into(),
                        "USB-CDC (ACM0)",
                    );
                    ui.selectable_value(
                        &mut self.selected_port,
                        "ST-Link V3 Telemetry".into(),
                        "ST-Link V3 Telemetry",
                    );
                    ui.selectable_value(
                        &mut self.selected_port,
                        "Simulated IMU Source".into(),
                        "Simulated IMU Source",
                    );
                });

            ui.label("Rate:");
            ui.add(egui::DragValue::new(&mut self.sample_rate_hz).suffix(" Hz"));
        });

        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.label("Sample Label:");
            ui.text_edit_singleline(&mut self.current_label);

            let record_btn_text = if self.is_recording {
                "⏹ Stop Recording"
            } else {
                "⏺ Record Sample"
            };
            if ui.button(record_btn_text).clicked() {
                self.is_recording = !self.is_recording;
                if !self.is_recording {
                    self.recorded_samples_count += 1;
                }
            }

            ui.label(format!(
                "Total Dataset Samples: {}",
                self.recorded_samples_count
            ));
        });

        ui.add_space(12.0);
        ui.separator();
        ui.label("Live Waveform Monitor:");

        // Draw live waveform visualization
        let (rect, _response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), 160.0),
            egui::Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(20, 24, 30));

        let num_points = self.live_sensor_history.len();
        if num_points > 1 {
            let dx = rect.width() / (num_points - 1) as f32;
            let mid_y = rect.center().y;
            let scale_y = 60.0;

            for i in 0..num_points - 1 {
                let p1 = egui::pos2(
                    rect.left() + (i as f32) * dx,
                    mid_y - self.live_sensor_history[i] * scale_y,
                );
                let p2 = egui::pos2(
                    rect.left() + ((i + 1) as f32) * dx,
                    mid_y - self.live_sensor_history[i + 1] * scale_y,
                );
                painter.line_segment(
                    [p1, p2],
                    egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(80, 200, 120)),
                );
            }
        }
    }
}
