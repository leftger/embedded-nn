#![allow(dead_code)]

use crate::state::StudioState;
use eframe::egui::{self, Color32, Pos2, Stroke, Vec2};
use std::f32::consts::PI;

/// Fraction of the trajectory revealed per frame while playing back, roughly a
/// four-second sweep at 60 FPS regardless of how long the capture is.
const PLAYBACK_STEP: f32 = 1.0 / 240.0;

pub struct Gesture3DView {
    pub azimuth_deg: f32,
    pub elevation_deg: f32,
    pub zoom: f32,
    pub auto_rotate: bool,
    pub playback_progress: f32,
    pub is_playing: bool,
    pub selected_sample_idx: usize,
}

impl Default for Gesture3DView {
    fn default() -> Self {
        Self {
            azimuth_deg: 45.0,
            elevation_deg: 30.0,
            zoom: 1.0,
            auto_rotate: false,
            playback_progress: 1.0,
            is_playing: false,
            selected_sample_idx: 0,
        }
    }
}

impl Gesture3DView {
    pub fn new() -> Self {
        Self::default()
    }

    /// Trajectory to draw plus a label describing where it came from. A live
    /// stream wins over the dataset; a scalar-only capture falls back to a
    /// synthetic helix so the sample is at least visible.
    pub fn resolve_source(
        &self,
        state: &StudioState,
        live: Option<&[[f32; 3]]>,
    ) -> (Vec<[f32; 3]>, String) {
        if let Some(points) = live
            && !points.is_empty()
        {
            return (points.to_vec(), "Live device stream".into());
        }

        let Some(sample) = state.samples.get(self.selected_sample_idx) else {
            return (Vec::new(), "No samples".into());
        };
        let tag = format!("Sample #{:03} [{}]", sample.id, sample.label);

        if !sample.trajectory.is_empty() {
            return (sample.trajectory.clone(), tag);
        }
        let synthetic = sample
            .raw_waveform
            .iter()
            .enumerate()
            .map(|(i, &m)| {
                let phase = i as f32 * 0.1;
                [m * phase.cos(), m * phase.sin(), m]
            })
            .collect();
        (synthetic, format!("{tag} — scalar only, synthetic path"))
    }

    /// Renders the 3D gesture trajectory viewport.
    pub fn ui(&mut self, ui: &mut egui::Ui, state: &StudioState, live: Option<&[[f32; 3]]>) {
        let is_live = live.is_some_and(|points| !points.is_empty());
        if !state.samples.is_empty() {
            self.selected_sample_idx = self.selected_sample_idx.min(state.samples.len() - 1);
        }
        let (raw_samples, source_label) = self.resolve_source(state, live);
        let raw_samples = raw_samples.as_slice();

        // A live stream is always drawn whole; scrubbing only makes sense for a
        // finished capture.
        if is_live {
            self.is_playing = false;
            self.playback_progress = 1.0;
        } else if self.is_playing {
            self.playback_progress += PLAYBACK_STEP;
            if self.playback_progress >= 1.0 {
                self.playback_progress = 1.0;
                self.is_playing = false;
            }
            ui.ctx().request_repaint();
        }

        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.heading("🌐 3D Gesture Trajectory Visualizer");
                ui.label(format!(
                    "• {source_label} • {} trajectory points",
                    raw_samples.len()
                ));

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("↺ Reset View").clicked() {
                        self.azimuth_deg = 45.0;
                        self.elevation_deg = 30.0;
                        self.zoom = 1.0;
                    }
                    ui.toggle_value(&mut self.auto_rotate, "⟳ Auto-Rotate");
                });
            });

            ui.add_space(4.0);

            ui.horizontal(|ui| {
                ui.add_enabled_ui(!is_live && !state.samples.is_empty(), |ui| {
                    ui.label("Captured Sample:");
                    egui::ComboBox::from_id_salt("gesture3d_sample_combo")
                        .selected_text(match state.samples.get(self.selected_sample_idx) {
                            Some(s) => format!("#{:03} - {}", s.id, s.label),
                            None => "None".into(),
                        })
                        .show_ui(ui, |ui| {
                            for (idx, s) in state.samples.iter().enumerate() {
                                let has_xyz = if s.trajectory.is_empty() { "" } else { " ▪" };
                                ui.selectable_value(
                                    &mut self.selected_sample_idx,
                                    idx,
                                    format!("#{:03} - {}{}", s.id, s.label, has_xyz),
                                );
                            }
                        });

                    ui.separator();

                    let play_label = if self.is_playing {
                        "⏸ Pause"
                    } else {
                        "▶ Replay"
                    };
                    if ui.button(play_label).clicked() {
                        self.is_playing = !self.is_playing;
                        // Restarting from a finished sweep should replay it.
                        if self.is_playing && self.playback_progress >= 1.0 {
                            self.playback_progress = 0.0;
                        }
                    }
                    ui.add(
                        egui::Slider::new(&mut self.playback_progress, 0.0..=1.0)
                            .show_value(false)
                            .text("playback"),
                    );
                });
            });

            ui.add_space(4.0);

            // Viewport controls toolbar
            ui.horizontal(|ui| {
                ui.label("Azimuth:");
                ui.add(egui::Slider::new(&mut self.azimuth_deg, -180.0..=180.0).suffix("°"));
                ui.label("Elevation:");
                ui.add(egui::Slider::new(&mut self.elevation_deg, -85.0..=85.0).suffix("°"));
                ui.label("Zoom:");
                ui.add(egui::Slider::new(&mut self.zoom, 0.3..=3.0));
            });

            if self.auto_rotate {
                self.azimuth_deg = (self.azimuth_deg + 0.5) % 360.0;
                ui.ctx().request_repaint();
            }

            ui.add_space(6.0);

            // Canvas for 3D projection
            let desired_size = Vec2::new(ui.available_width(), 320.0);
            let (response, painter) = ui.allocate_painter(desired_size, egui::Sense::drag());
            let rect = response.rect;

            // Handle drag interactions for orbit rotation
            if response.dragged() {
                self.azimuth_deg += response.drag_delta().x * 0.5;
                self.elevation_deg = (self.elevation_deg - response.drag_delta().y * 0.5).clamp(-85.0, 85.0);
            }

            // Draw dark viewport background & grid
            painter.rect_filled(rect, 8.0, Color32::from_rgb(18, 20, 24));
            painter.rect_stroke(rect, 8.0, Stroke::new(1.0_f32, Color32::from_rgb(45, 50, 60)), egui::StrokeKind::Inside);

            let center = rect.center();
            let base_scale = (rect.width().min(rect.height()) * 0.35) * self.zoom;

            // Precompute rotation matrices
            let rad_az = self.azimuth_deg * PI / 180.0;
            let rad_el = self.elevation_deg * PI / 180.0;
            let (sin_az, cos_az) = rad_az.sin_cos();
            let (sin_el, cos_el) = rad_el.sin_cos();

            let project = |x: f32, y: f32, z: f32| -> Pos2 {
                // 3D rotation: Azimuth around Z, Elevation around X
                let x1 = x * cos_az - y * sin_az;
                let y1 = x * sin_az + y * cos_az;
                let z1 = z;

                let y2 = y1 * cos_el - z1 * sin_el;
                let z2 = y1 * sin_el + z1 * cos_el;

                // Orthographic projection to 2D screen coordinates
                Pos2::new(center.x + x1 * base_scale, center.y - (z2 * base_scale * 0.8 + y2 * 0.2))
            };

            // Draw 3D Reference Axis Gizmo (X: Red, Y: Green, Z: Blue)
            let origin = project(0.0, 0.0, 0.0);
            let x_axis = project(0.5, 0.0, 0.0);
            let y_axis = project(0.0, 0.5, 0.0);
            let z_axis = project(0.0, 0.0, 0.5);

            painter.line_segment([origin, x_axis], Stroke::new(2.0_f32, Color32::from_rgb(230, 80, 80)));
            painter.line_segment([origin, y_axis], Stroke::new(2.0_f32, Color32::from_rgb(80, 220, 100)));
            painter.line_segment([origin, z_axis], Stroke::new(2.0_f32, Color32::from_rgb(80, 140, 240)));

            painter.text(x_axis, egui::Align2::CENTER_CENTER, "X (Pitch)", egui::FontId::proportional(11.0), Color32::from_rgb(230, 80, 80));
            painter.text(y_axis, egui::Align2::CENTER_CENTER, "Y (Roll)", egui::FontId::proportional(11.0), Color32::from_rgb(80, 220, 100));
            painter.text(z_axis, egui::Align2::CENTER_CENTER, "Z (Yaw)", egui::FontId::proportional(11.0), Color32::from_rgb(80, 140, 240));

            // Integrate 3-axis acceleration into a 3D trajectory path
            if !raw_samples.is_empty() {
                let mut trajectory: Vec<[f32; 3]> = Vec::with_capacity(raw_samples.len());
                let mut px = 0.0f32;
                let mut py = 0.0f32;
                let mut pz = 0.0f32;
                let mut vx = 0.0f32;
                let mut vy = 0.0f32;
                let mut vz = 0.0f32;

                let dt = 0.01; // 100 Hz sampling interval (10 ms)
                let damping = 0.96; // High-pass velocity damping to prevent integration runaway

                for s in raw_samples {
                    // Remove static 1g gravity bias approximately along dominant axis
                    let ax = s[0];
                    let ay = s[1];
                    let az = s[2] - 1.0;

                    vx = (vx + ax * dt) * damping;
                    vy = (vy + ay * dt) * damping;
                    vz = (vz + az * dt) * damping;

                    px += vx * dt;
                    py += vy * dt;
                    pz += vz * dt;

                    trajectory.push([px, py, pz]);
                }

                // Render 3D Trajectory Ribbon / Line Segments
                let count = (trajectory.len() as f32 * self.playback_progress).round() as usize;
                for i in 1..count.min(trajectory.len()) {
                    let p0 = trajectory[i - 1];
                    let p1 = trajectory[i];

                    let pt0 = project(p0[0] * 5.0, p0[1] * 5.0, p0[2] * 5.0);
                    let pt1 = project(p1[0] * 5.0, p1[1] * 5.0, p1[2] * 5.0);

                    // Color gradient along time / magnitude
                    let t = i as f32 / trajectory.len() as f32;
                    let color = Color32::from_rgb(
                        (80.0 + 175.0 * t) as u8,
                        (200.0 - 80.0 * t) as u8,
                        (240.0 - 160.0 * t) as u8,
                    );

                    painter.line_segment([pt0, pt1], Stroke::new(2.5_f32, color));
                }

                // Draw head cursor
                if count > 0 && count <= trajectory.len() {
                    let head = trajectory[count - 1];
                    let pt_head = project(head[0] * 5.0, head[1] * 5.0, head[2] * 5.0);
                    painter.circle_filled(pt_head, 5.0, Color32::from_rgb(255, 220, 60));
                }
            } else {
                painter.text(
                    center,
                    egui::Align2::CENTER_CENTER,
                    "No 3-axis trajectory loaded. Capture or import a .jsonl dataset sample to view 3D motion path.",
                    egui::FontId::proportional(13.0),
                    Color32::from_rgb(140, 145, 160),
                );
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_stream_outranks_the_selected_capture() {
        let mut state = StudioState::default();
        state.load_demo_dataset();
        let view = Gesture3DView::new();
        let live = [[1.0, 2.0, 3.0]];

        let (points, label) = view.resolve_source(&state, Some(&live));
        assert_eq!(points, vec![[1.0, 2.0, 3.0]]);
        assert_eq!(label, "Live device stream");

        // An empty live buffer is not a stream; fall through to the capture.
        let (points, label) = view.resolve_source(&state, Some(&[]));
        assert_eq!(points, state.samples[0].trajectory);
        assert!(label.starts_with("Sample #001"));
    }

    #[test]
    fn scalar_only_capture_is_flagged_as_a_synthetic_path() {
        let mut state = StudioState::default();
        state.load_demo_dataset();
        state.samples[0].trajectory.clear();

        let view = Gesture3DView::new();
        let (points, label) = view.resolve_source(&state, None);

        assert_eq!(points.len(), state.samples[0].raw_waveform.len());
        assert!(label.contains("synthetic"), "got {label}");
    }

    #[test]
    fn empty_dataset_or_stale_selection_yields_no_trajectory() {
        let mut state = StudioState::default();
        state.samples.clear();
        let mut view = Gesture3DView::new();

        let (points, label) = view.resolve_source(&state, None);
        assert!(points.is_empty());
        assert_eq!(label, "No samples");

        // A selection left over from a larger dataset must not panic.
        view.selected_sample_idx = usize::MAX;
        assert!(view.resolve_source(&state, None).0.is_empty());
    }
}
