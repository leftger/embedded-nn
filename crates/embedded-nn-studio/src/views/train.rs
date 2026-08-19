use eframe::egui;

#[derive(Default)]
pub struct TrainView {
    pub model_arch: String,
    pub quant_mode: String,
    pub epochs: u32,
    pub learning_rate: f32,
    pub is_training: bool,
    pub current_epoch: u32,
    pub current_loss: f32,
    pub current_acc: f32,
    pub loss_history: Vec<f32>,
}

impl TrainView {
    pub fn new() -> Self {
        Self {
            model_arch: "TinyConv1D (IMU Gestures)".into(),
            quant_mode: "s4 (4-bit sub-byte packed)".into(),
            epochs: 40,
            learning_rate: 0.005,
            is_training: false,
            current_epoch: 40,
            current_loss: 0.042,
            current_acc: 98.4,
            loss_history: vec![0.85, 0.62, 0.45, 0.31, 0.22, 0.15, 0.09, 0.06, 0.042],
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        ui.heading("3. Model Design, Burn Training & QAT Quantization");
        ui.add_space(8.0);

        ui.horizontal(|ui| {
            ui.label("Architecture:");
            egui::ComboBox::from_id_salt("train_arch_combo")
                .selected_text(&self.model_arch)
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.model_arch,
                        "TinyConv1D (IMU Gestures)".into(),
                        "TinyConv1D (IMU Gestures)",
                    );
                    ui.selectable_value(
                        &mut self.model_arch,
                        "DenseMLP (Tabular/Sensor)".into(),
                        "DenseMLP (Tabular/Sensor)",
                    );
                    ui.selectable_value(
                        &mut self.model_arch,
                        "TinyLSTM (Time-Series)".into(),
                        "TinyLSTM (Time-Series)",
                    );
                });

            ui.label("Target Quantization:");
            egui::ComboBox::from_id_salt("train_quant_combo")
                .selected_text(&self.quant_mode)
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.quant_mode,
                        "s4 (4-bit sub-byte packed)".into(),
                        "s4 (4-bit sub-byte packed)",
                    );
                    ui.selectable_value(
                        &mut self.quant_mode,
                        "s8 (8-bit fixed-point)".into(),
                        "s8 (8-bit fixed-point)",
                    );
                    ui.selectable_value(
                        &mut self.quant_mode,
                        "s16 (16-bit high-precision)".into(),
                        "s16 (16-bit high-precision)",
                    );
                });
        });

        ui.add_space(6.0);

        ui.horizontal(|ui| {
            ui.label("Epochs:");
            ui.add(egui::DragValue::new(&mut self.epochs).range(1..=500));

            ui.label("Learning Rate:");
            ui.add(egui::DragValue::new(&mut self.learning_rate).speed(0.001));

            let train_text = if self.is_training {
                "⏹ Stop Training"
            } else {
                "▶ Start Burn QAT Training"
            };
            if ui.button(train_text).clicked() {
                self.is_training = !self.is_training;
            }
        });

        ui.add_space(10.0);
        ui.separator();

        ui.horizontal(|ui| {
            ui.label(format!(
                "Status: Epoch {}/{}",
                self.current_epoch, self.epochs
            ));
            ui.label(format!("Loss: {:.4}", self.current_loss));
            ui.label(format!("Val Accuracy: {:.1}%", self.current_acc));
        });

        ui.add_space(8.0);
        ui.label("Training Loss Convergence:");

        let (rect, _response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), 140.0),
            egui::Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(20, 24, 30));

        let num_points = self.loss_history.len();
        if num_points > 1 {
            let dx = rect.width() / (num_points - 1) as f32;
            for i in 0..num_points - 1 {
                let y1 = rect.bottom() - 10.0 - self.loss_history[i] * (rect.height() - 20.0);
                let y2 = rect.bottom() - 10.0 - self.loss_history[i + 1] * (rect.height() - 20.0);
                let p1 = egui::pos2(rect.left() + (i as f32) * dx, y1);
                let p2 = egui::pos2(rect.left() + ((i + 1) as f32) * dx, y2);
                painter.line_segment(
                    [p1, p2],
                    egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(240, 160, 60)),
                );
            }
        }
    }
}
