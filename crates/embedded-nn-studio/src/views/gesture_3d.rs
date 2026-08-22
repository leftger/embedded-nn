#![allow(dead_code)]

use eframe::egui::{self, Color32, Pos2, Stroke, Vec2};
use std::f32::consts::PI;

pub struct Gesture3DView {
    pub azimuth_deg: f32,
    pub elevation_deg: f32,
    pub zoom: f32,
    pub auto_rotate: bool,
    pub playback_progress: f32,
    pub is_playing: bool,
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
        }
    }
}

impl Gesture3DView {
    pub fn new() -> Self {
        Self::default()
    }

    /// Renders the 3D gesture trajectory viewport.
    pub fn ui(&mut self, ui: &mut egui::Ui, raw_samples: &[[f32; 3]]) {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.heading("🌐 3D Gesture Trajectory Visualizer");
                ui.label(format!("• {} trajectory points", raw_samples.len()));

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
