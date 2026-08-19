use eframe::egui;

#[derive(Default)]
pub struct DspView {
    pub fft_length: usize,
    pub window_type: String,
    pub num_mel_bins: usize,
    pub high_pass_cutoff: f32,
}

impl DspView {
    pub fn new() -> Self {
        Self {
            fft_length: 256,
            window_type: "Hann".into(),
            num_mel_bins: 16,
            high_pass_cutoff: 20.0,
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        ui.heading("2. DSP Feature Extraction & Preprocessing");
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.label("Windowing:");
            egui::ComboBox::from_id_salt("dsp_window_combo")
                .selected_text(&self.window_type)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.window_type, "Hann".into(), "Hann");
                    ui.selectable_value(&mut self.window_type, "Hamming".into(), "Hamming");
                    ui.selectable_value(&mut self.window_type, "Rectangular".into(), "Rectangular");
                });

            ui.label("FFT Size:");
            egui::ComboBox::from_id_salt("dsp_fft_combo")
                .selected_text(format!("{}", self.fft_length))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.fft_length, 128, "128");
                    ui.selectable_value(&mut self.fft_length, 256, "256");
                    ui.selectable_value(&mut self.fft_length, 512, "512");
                });

            ui.label("Mel Bins:");
            ui.add(egui::DragValue::new(&mut self.num_mel_bins).range(8..=64));

            ui.label("High-Pass Cutoff:");
            ui.add(egui::DragValue::new(&mut self.high_pass_cutoff).suffix(" Hz"));
        });

        ui.add_space(12.0);
        ui.separator();
        ui.label("Feature Heatmap / Spectrogram Preview:");

        let (rect, _response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), 160.0),
            egui::Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(15, 18, 24));

        let cols = 32;
        let rows = self.num_mel_bins.clamp(4, 32);
        let cell_w = rect.width() / cols as f32;
        let cell_h = rect.height() / rows as f32;

        for r in 0..rows {
            for c in 0..cols {
                let intensity = (((r * 7 + c * 13) % 255) as u8).saturating_add(40);
                let col = egui::Color32::from_rgb(intensity / 4, intensity / 2, intensity);
                let cell_rect = egui::Rect::from_min_size(
                    egui::pos2(
                        rect.left() + c as f32 * cell_w,
                        rect.top() + r as f32 * cell_h,
                    ),
                    egui::vec2(cell_w - 1.0, cell_h - 1.0),
                );
                painter.rect_filled(cell_rect, 1.0, col);
            }
        }
    }
}
