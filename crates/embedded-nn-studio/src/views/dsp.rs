use crate::state::{StudioState, WindowFunction};
use eframe::egui;

#[derive(Default)]
pub struct DspView {
    pub selected_sample_idx: usize,
    pub selected_frame_idx: usize,
}

impl DspView {
    pub fn new() -> Self {
        Self {
            selected_sample_idx: 0,
            selected_frame_idx: 0,
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut StudioState) {
        ui.horizontal(|ui| {
            ui.heading("🎛️ 2. DSP Preprocessing & Feature Extraction Pipeline");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(format!(
                    "Dataset Samples Processed: {}",
                    state.samples.len()
                ));
            });
        });

        ui.add_space(4.0);
        ui.label(
            "Transform raw noisy time-series sensor signals into compact frequency-domain feature matrices (FFT / Mel Filterbanks). Ensuring identical bit-level preprocessing between host training and target MCU runtime.",
        );
        ui.add_space(8.0);

        let mut dsp_changed = false;

        // Top Control Parameters
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Window Function:");
                let prev_win = state.dsp.window_type;
                egui::ComboBox::from_id_salt("dsp_window_type_combo")
                    .selected_text(match state.dsp.window_type {
                        WindowFunction::Hann => "Hann Window (Smooth)",
                        WindowFunction::Hamming => "Hamming Window",
                        WindowFunction::Rectangular => "Rectangular (None)",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.dsp.window_type,
                            WindowFunction::Hann,
                            "Hann Window (Smooth)",
                        );
                        ui.selectable_value(
                            &mut state.dsp.window_type,
                            WindowFunction::Hamming,
                            "Hamming Window",
                        );
                        ui.selectable_value(
                            &mut state.dsp.window_type,
                            WindowFunction::Rectangular,
                            "Rectangular (None)",
                        );
                    });
                if state.dsp.window_type != prev_win {
                    dsp_changed = true;
                }

                ui.separator();

                ui.label("Window Size:");
                let prev_size = state.dsp.window_size;
                egui::ComboBox::from_id_salt("dsp_window_size_combo")
                    .selected_text(format!("{} samples", state.dsp.window_size))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut state.dsp.window_size, 32, "32 samples");
                        ui.selectable_value(&mut state.dsp.window_size, 64, "64 samples");
                        ui.selectable_value(&mut state.dsp.window_size, 128, "128 samples");
                    });
                if state.dsp.window_size != prev_size {
                    dsp_changed = true;
                }

                ui.separator();

                ui.label("Mel Filter Bins:");
                if ui
                    .add(egui::DragValue::new(&mut state.dsp.num_mel_bins).range(8..=32))
                    .changed()
                {
                    dsp_changed = true;
                }

                ui.separator();

                ui.label("High-Pass Cutoff:");
                if ui
                    .add(
                        egui::DragValue::new(&mut state.dsp.high_pass_cutoff)
                            .suffix(" Hz")
                            .range(1.0..=100.0),
                    )
                    .changed()
                {
                    dsp_changed = true;
                }
            });

            ui.add_space(4.0);

            ui.horizontal(|ui| {
                ui.label("Sample Rate:");
                if ui
                    .add(
                        egui::DragValue::new(&mut state.dsp.sample_rate)
                            .suffix(" Hz")
                            .range(10.0..=2000.0),
                    )
                    .changed()
                {
                    dsp_changed = true;
                }

                ui.separator();

                ui.label("Frame Hop Size:");
                if ui
                    .add(
                        egui::DragValue::new(&mut state.dsp.frame_hop_size)
                            .suffix(" samples")
                            .range(1..=512),
                    )
                    .changed()
                {
                    dsp_changed = true;
                }

                ui.separator();

                ui.label("Capture Window:");
                let min_capture = state.dsp.window_size as u64;
                if ui
                    .add(
                        egui::DragValue::new(&mut state.dsp.capture_samples)
                            .suffix(" samples")
                            .range(min_capture..=4096),
                    )
                    .changed()
                {
                    dsp_changed = true;
                }

                ui.separator();

                ui.label("Mel energy floor:");
                if ui
                    .add(
                        egui::DragValue::new(&mut state.dsp.mel_energy_floor)
                            .range(0.001..=1.0)
                            .speed(0.001),
                    )
                    .changed()
                {
                    dsp_changed = true;
                }

                ui.separator();

                let num_frames = StudioState::num_frames_for_config(&state.dsp);
                ui.label(format!("→ {} frames/sample", num_frames));
            });
        });

        if dsp_changed {
            state.recompute_all_frames();
            state.reset_training();
            state.rebuild_model_graph_and_codegen();
        }

        ui.add_space(8.0);

        // Inspect Active Sample
        if !state.samples.is_empty() {
            ui.horizontal(|ui| {
                ui.label("Preview Sample:");
                self.selected_sample_idx = self.selected_sample_idx.min(state.samples.len() - 1);
                egui::ComboBox::from_id_salt("dsp_preview_sample_combo")
                    .selected_text(format!(
                        "Sample #{:03} [Class: {}]",
                        state.samples[self.selected_sample_idx].id,
                        state.samples[self.selected_sample_idx].label
                    ))
                    .show_ui(ui, |ui| {
                        for (idx, s) in state.samples.iter().enumerate() {
                            ui.selectable_value(
                                &mut self.selected_sample_idx,
                                idx,
                                format!("#{:03} - {}", s.id, s.label),
                            );
                        }
                    });
            });
        }

        ui.add_space(8.0);

        let active_sample = state.samples.get(self.selected_sample_idx).cloned();

        if let Some(s) = &active_sample
            && s.frames.len() > 1
        {
            self.selected_frame_idx = self.selected_frame_idx.min(s.frames.len() - 1);
            ui.horizontal(|ui| {
                ui.label("Preview Frame:");
                ui.add(
                    egui::Slider::new(&mut self.selected_frame_idx, 0..=s.frames.len() - 1)
                        .text("frame index"),
                );
            });
            ui.add_space(8.0);
        }

        // 3-Stage Signal Transformation Flow Visualization
        ui.columns(3, |cols| {
            // Stage 1: Raw Time-Domain Signal
            cols[0].group(|ui| {
                ui.label("Stage 1: Raw Time-Domain Sensor");
                ui.label("Sensor values captured over time window");

                let (rect, _resp) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), 130.0),
                    egui::Sense::hover(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(16, 20, 28));

                let mid_y = rect.center().y;
                if let Some(s) = &active_sample {
                    let n = s.raw_waveform.len();
                    if n > 1 {
                        let dx = rect.width() / (n - 1) as f32;
                        for i in 0..n - 1 {
                            let p1 = egui::pos2(
                                rect.left() + (i as f32) * dx,
                                mid_y - s.raw_waveform[i] * 50.0,
                            );
                            let p2 = egui::pos2(
                                rect.left() + ((i + 1) as f32) * dx,
                                mid_y - s.raw_waveform[i + 1] * 50.0,
                            );
                            painter.line_segment(
                                [p1, p2],
                                egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(100, 170, 255)),
                            );
                        }
                    }
                }
            });

            // Stage 2: FFT / Mel Spectrogram Energy Bins
            cols[1].group(|ui| {
                ui.label("Stage 2: FFT & Mel Filterbank");
                ui.label("Continuous spectral power distribution");

                let (rect, _resp) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), 130.0),
                    egui::Sense::hover(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(16, 20, 28));

                if let Some(s) = &active_sample
                    && let Some(frame) = s.frames.get(self.selected_frame_idx)
                {
                    let num_bins = frame.len();
                    if num_bins > 0 {
                        let col_w = rect.width() / num_bins as f32;
                        for (i, &energy) in frame.iter().enumerate() {
                            let bar_h =
                                (energy * (rect.height() - 10.0)).clamp(2.0, rect.height() - 10.0);
                            let bar_rect = egui::Rect::from_min_size(
                                egui::pos2(
                                    rect.left() + i as f32 * col_w + 1.0,
                                    rect.bottom() - bar_h - 4.0,
                                ),
                                egui::vec2(col_w - 2.0, bar_h),
                            );
                            let intensity = ((energy * 255.0) as u8).saturating_add(60);
                            painter.rect_filled(
                                bar_rect,
                                2.0,
                                egui::Color32::from_rgb(intensity / 3, intensity, intensity / 2),
                            );
                        }
                    }
                }
            });

            // Stage 3: Quantized Feature Vector (Int8)
            cols[2].group(|ui| {
                ui.label("Stage 3: Quantized NN Input Vector");
                ui.label("Fixed-point s8 values fed to embedded-nn");

                let (rect, _resp) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), 130.0),
                    egui::Sense::hover(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(16, 20, 28));

                if let Some(s) = &active_sample
                    && let Some(frame) = s.quantized_frames.get(self.selected_frame_idx)
                {
                    let num_bins = frame.len();
                    if num_bins > 0 {
                        let col_w = rect.width() / num_bins as f32;
                        let mid_y = rect.center().y;
                        for (i, &q_val) in frame.iter().enumerate() {
                            let h = (q_val as f32 / 127.0) * 50.0;
                            let bar_rect = if h >= 0.0 {
                                egui::Rect::from_min_max(
                                    egui::pos2(rect.left() + i as f32 * col_w + 1.0, mid_y - h),
                                    egui::pos2(rect.left() + (i + 1) as f32 * col_w - 1.0, mid_y),
                                )
                            } else {
                                egui::Rect::from_min_max(
                                    egui::pos2(rect.left() + i as f32 * col_w + 1.0, mid_y),
                                    egui::pos2(
                                        rect.left() + (i + 1) as f32 * col_w - 1.0,
                                        mid_y - h,
                                    ),
                                )
                            };
                            painter.rect_filled(
                                bar_rect,
                                2.0,
                                egui::Color32::from_rgb(240, 160, 60),
                            );
                        }
                    }
                }
            });
        });

        ui.add_space(8.0);

        // 2D Temporal Spectrogram Waterfall Heatmap
        ui.group(|ui| {
            ui.label(egui::RichText::new("🌊 2D Temporal Mel-Spectrogram Heatmap Matrix [Time × Frequency]").strong());
            ui.label("Visualizes the complete temporal-spectral feature matrix fed directly to 2D CNN and Temporal ResNet architectures:");

            if let Some(s) = &active_sample && !s.frames.is_empty() {
                let num_frames = s.frames.len();
                let num_bins = s.frames[0].len();

                let (rect, _resp) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), 130.0),
                    egui::Sense::hover(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(12, 14, 20));

                let cell_w = rect.width() / num_frames as f32;
                let cell_h = rect.height() / num_bins as f32;

                for (t, frame) in s.frames.iter().enumerate() {
                    for (f, &val) in frame.iter().enumerate() {
                        let cell_rect = egui::Rect::from_min_size(
                            egui::pos2(
                                rect.left() + (t as f32) * cell_w,
                                rect.bottom() - ((f + 1) as f32) * cell_h,
                            ),
                            egui::vec2(cell_w.max(1.0), cell_h.max(1.0)),
                        );

                        // Magma/Turbo gradient: dark purple -> magenta -> orange -> bright yellow
                        let norm = val.clamp(0.0, 1.0);
                        let r = ((norm * 2.2).clamp(0.0, 1.0) * 255.0) as u8;
                        let g = ((norm * 1.6 - 0.2).clamp(0.0, 1.0) * 255.0) as u8;
                        let b = (((1.0 - norm) * 1.4).clamp(0.0, 1.0) * 255.0) as u8;

                        painter.rect_filled(
                            cell_rect,
                            0.0,
                            egui::Color32::from_rgb(r, g, b),
                        );
                    }
                }
            } else {
                ui.label("No active sample or frames extracted.");
            }
        });
    }
}
