#![allow(dead_code)]

use eframe::egui::{self, Color32, ProgressBar, Rect, Stroke, Vec2};
use embedded_nn_live::host::{DeviceLink, UsbBridge};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct LayerBenchmark {
    pub name: String,
    pub cycles: u32,
    pub time_us: f32,
    pub output_shape: [usize; 4],
    pub activations: Vec<i8>,
}

pub struct LiveInspectorView {
    pub selected_device: Option<String>,
    pub available_devices: Vec<UsbBridge>,
    pub is_connected: bool,
    pub last_poll: Instant,
    pub live_logits: Vec<f32>,
    pub class_names: Vec<String>,
    pub layer_benchmarks: Vec<LayerBenchmark>,
    pub total_cycles: u32,
    pub total_time_us: f32,
    pub selected_layer_idx: usize,
    pub live_accel: [f32; 3],
    pub is_simulated: bool,
}

impl Default for LiveInspectorView {
    fn default() -> Self {
        // Pre-populate realistic layer benchmarks for immediate inspection / simulation
        let demo_layers = vec![
            LayerBenchmark {
                name: "Layer 0: Conv1D (k=3, ch=4)".into(),
                cycles: 3850,
                time_us: 38.5,
                output_shape: [1, 16, 4, 1],
                activations: vec![
                    12, -45, 89, 102, -15, 34, -78, 110, -2, 67, 45, -89, 95, -112, 33, 48, -56,
                    72, -18, 90, 105, -30, 44, -80, 65, -92, 115, 20, -40, 85, -15, 60,
                ],
            },
            LayerBenchmark {
                name: "Layer 1: MaxPool1D (stride=2)".into(),
                cycles: 820,
                time_us: 8.2,
                output_shape: [1, 8, 4, 1],
                activations: vec![
                    89, 102, 34, 110, 67, 45, 95, 48, 72, 90, 105, 44, 65, 115, 85, 60,
                ],
            },
            LayerBenchmark {
                name: "Layer 2: FullyConnected (32 -> 16)".into(),
                cycles: 1420,
                time_us: 14.2,
                output_shape: [1, 16, 1, 1],
                activations: vec![
                    10, -25, 48, 95, -60, 82, -10, 35, 70, -85, 110, -40, 25, 60, -90, 105,
                ],
            },
            LayerBenchmark {
                name: "Layer 3: FullyConnected (16 -> 2)".into(),
                cycles: 580,
                time_us: 5.8,
                output_shape: [1, 2, 1, 1],
                activations: vec![84, -92],
            },
            LayerBenchmark {
                name: "Layer 4: Softmax".into(),
                cycles: 320,
                time_us: 3.2,
                output_shape: [1, 2, 1, 1],
                activations: vec![122, 5],
            },
        ];

        let total_c: u32 = demo_layers.iter().map(|l| l.cycles).sum();
        let total_t: f32 = demo_layers.iter().map(|l| l.time_us).sum();

        Self {
            selected_device: None,
            available_devices: Vec::new(),
            is_connected: true, // Defaults to ready simulated state
            last_poll: Instant::now(),
            live_logits: vec![0.92, 0.08],
            class_names: vec!["Gesture: Swipe Left".into(), "Gesture: Swipe Right".into()],
            layer_benchmarks: demo_layers,
            total_cycles: total_c,
            total_time_us: total_t,
            selected_layer_idx: 0,
            live_accel: [0.02, 0.05, 0.98],
            is_simulated: true,
        }
    }
}

impl LiveInspectorView {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, device_link: &mut Option<DeviceLink>) {
        if let Some(link) = device_link.as_mut() {
            while let Some(msg) = link.take_result() {
                match msg {
                    embedded_nn_live::host::OwnedMsg::InferenceResult {
                        execution_cycles,
                        logits,
                        ..
                    } => {
                        self.is_simulated = false;
                        self.total_cycles = execution_cycles;
                        self.total_time_us = execution_cycles as f32 / 100.0;
                        if !logits.is_empty() {
                            let logits_i8: Vec<i8> = logits.iter().map(|&b| b as i8).collect();
                            let max_l = logits_i8.iter().copied().fold(i8::MIN, i8::max) as f32;
                            let exp_sum: f32 =
                                logits_i8.iter().map(|&l| ((l as f32) - max_l).exp()).sum();
                            self.live_logits = logits_i8
                                .iter()
                                .map(|&l| ((l as f32) - max_l).exp() / exp_sum.max(1e-6))
                                .collect();
                        }
                    }
                    embedded_nn_live::host::OwnedMsg::LayerProfile {
                        layer_idx,
                        execution_cycles,
                        activations,
                        ..
                    } => {
                        let idx = layer_idx as usize;
                        if let Some(bench) = self.layer_benchmarks.get_mut(idx) {
                            bench.cycles = execution_cycles;
                            bench.time_us = execution_cycles as f32 / 100.0;
                            bench.activations = activations.iter().map(|&b| b as i8).collect();
                        }
                    }
                    _ => {}
                }
            }
        }

        ui.vertical(|ui| {
            // Header & Device Connection Bar
            ui.horizontal(|ui| {
                ui.heading("🔬 Live On-Device Activation & Profiling Inspector");
                ui.label("• Guts of Inference & Kernel Profiling");

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("🔄 Refresh USB").clicked() {
                        if let Ok(devs) = UsbBridge::enumerate_devices() {
                            self.available_devices = devs;
                        }
                    }

                    if self.is_simulated {
                        ui.colored_label(
                            egui::Color32::from_rgb(100, 200, 255),
                            "● Mode: Simulated Cortex-M33 Profile",
                        );
                    } else if device_link.is_some() {
                        ui.colored_label(
                            egui::Color32::from_rgb(100, 220, 140),
                            "● Connected: NUCLEO-WBA65RI (USB-HS)",
                        );
                    } else {
                        ui.colored_label(
                            egui::Color32::from_rgb(240, 140, 60),
                            "○ Standby: No Hardware Attached",
                        );
                    }
                });
            });

            ui.add_space(6.0);

            // Top Status Panel: Live Softmax Confidence HUD & Latency Summary
            ui.columns(2, |cols| {
                // Column 1: Live Softmax Confidence HUD
                cols[0].group(|ui| {
                    ui.label(egui::RichText::new("🎯 Live Softmax Confidence HUD").strong());
                    ui.add_space(4.0);

                    for (i, prob) in self.live_logits.iter().enumerate() {
                        let name = self
                            .class_names
                            .get(i)
                            .cloned()
                            .unwrap_or_else(|| format!("Class {}", i));
                        ui.horizontal(|ui| {
                            ui.label(format!("{}:", name));
                            ui.add(ProgressBar::new(*prob).show_percentage().animate(true));
                        });
                    }
                });

                // Column 2: Total Hardware Execution Summary
                cols[1].group(|ui| {
                    ui.label(egui::RichText::new("⚡ Hardware Timing & Arena Footprint").strong());
                    ui.add_space(4.0);

                    ui.horizontal(|ui| {
                        ui.label("Total Inference Latency:");
                        ui.colored_label(
                            Color32::from_rgb(100, 220, 140),
                            format!(
                                "{:.2} µs ({} DWT Cycles)",
                                self.total_time_us, self.total_cycles
                            ),
                        );
                    });

                    ui.horizontal(|ui| {
                        ui.label("Cortex-M33 Clock:");
                        ui.label("100 MHz (PLL1)");
                        ui.label("• Peak Frame Rate:");
                        let max_fps = if self.total_time_us > 0.0 {
                            1_000_000.0 / self.total_time_us
                        } else {
                            0.0
                        };
                        ui.colored_label(
                            Color32::from_rgb(100, 200, 255),
                            format!("{:.0} FPS", max_fps),
                        );
                    });
                });
            });

            ui.add_space(8.0);

            // Main Section: Layer Latency Waterfall Chart
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("📊 Layer-by-Layer Latency Waterfall").strong());
                    ui.label("• Exact per-kernel execution duration on STM32WBA65RI");
                });

                ui.add_space(6.0);

                let total_us = self.total_time_us.max(0.1);
                for (idx, layer) in self.layer_benchmarks.iter().enumerate() {
                    let fraction = (layer.time_us / total_us).clamp(0.0, 1.0);
                    let is_selected = self.selected_layer_idx == idx;

                    ui.horizontal(|ui| {
                        let btn_label = if is_selected {
                            format!("▶ {}", layer.name)
                        } else {
                            format!("  {}", layer.name)
                        };

                        if ui.selectable_label(is_selected, btn_label).clicked() {
                            self.selected_layer_idx = idx;
                        }

                        let bar_color = match idx % 4 {
                            0 => Color32::from_rgb(70, 140, 240),
                            1 => Color32::from_rgb(60, 200, 160),
                            2 => Color32::from_rgb(230, 170, 60),
                            _ => Color32::from_rgb(220, 90, 120),
                        };

                        let (rect, _) = ui.allocate_exact_size(
                            Vec2::new(ui.available_width() - 140.0, 18.0),
                            egui::Sense::hover(),
                        );
                        let painter = ui.painter();
                        painter.rect_filled(rect, 4.0, Color32::from_rgb(30, 34, 42));

                        let fill_rect = Rect::from_min_size(
                            rect.min,
                            Vec2::new(rect.width() * fraction, rect.height()),
                        );
                        painter.rect_filled(fill_rect, 4.0, bar_color);

                        ui.label(format!(
                            "{:.1} µs ({:.1}%)",
                            layer.time_us,
                            fraction * 100.0
                        ));
                    });
                }
            });

            ui.add_space(8.0);

            // Layer Activation Heatmap & Dynamic Range Matrix
            if let Some(selected_layer) = self.layer_benchmarks.get(self.selected_layer_idx) {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "🔥 Activation Heatmap: {}",
                                selected_layer.name
                            ))
                            .strong(),
                        );
                        ui.label(format!(
                            "• Shape: {:?} • {} activation elements",
                            selected_layer.output_shape,
                            selected_layer.activations.len()
                        ));
                    });

                    ui.add_space(6.0);

                    // Dynamic Range Saturation Meter
                    let sat_count = selected_layer
                        .activations
                        .iter()
                        .filter(|&&v| v == i8::MAX || v == i8::MIN)
                        .count();
                    let zero_count = selected_layer
                        .activations
                        .iter()
                        .filter(|&&v| v == 0)
                        .count();
                    let total = selected_layer.activations.len().max(1);

                    ui.horizontal(|ui| {
                        ui.label(format!(
                            "Zero elements (ReLU sparsity): {:.1}%",
                            (zero_count as f32 / total as f32) * 100.0
                        ));
                        ui.label("•");
                        let sat_pct = (sat_count as f32 / total as f32) * 100.0;
                        if sat_pct > 5.0 {
                            ui.colored_label(
                                Color32::from_rgb(240, 90, 90),
                                format!("⚠ Saturated: {:.1}% (Quantization clipping)", sat_pct),
                            );
                        } else {
                            ui.colored_label(
                                Color32::from_rgb(100, 220, 140),
                                "✓ Dynamic Range: Optimal (No Clipping)",
                            );
                        }
                    });

                    ui.add_space(6.0);

                    // Activation Grid Heatmap (8 columns)
                    let cols = 8;
                    egui::Grid::new("activation_heatmap_grid")
                        .spacing([4.0, 4.0])
                        .show(ui, |ui| {
                            for (i, &val) in selected_layer.activations.iter().enumerate() {
                                // Normalize i8 [-128..127] to color gradient
                                let norm = (val as f32 + 128.0) / 255.0;
                                let color = if val == 0 {
                                    Color32::from_rgb(24, 28, 36) // Sparsity / zero
                                } else if val > 0 {
                                    Color32::from_rgb(
                                        (40.0 + 215.0 * norm) as u8,
                                        (80.0 + 140.0 * norm) as u8,
                                        (180.0 * (1.0 - norm)) as u8,
                                    )
                                } else {
                                    Color32::from_rgb(
                                        (160.0 * (1.0 - norm)) as u8,
                                        (50.0 + 60.0 * norm) as u8,
                                        (220.0 * (1.0 - norm)) as u8,
                                    )
                                };

                                let (rect, _) = ui.allocate_exact_size(
                                    Vec2::new(32.0, 24.0),
                                    egui::Sense::hover(),
                                );
                                let painter = ui.painter();
                                painter.rect_filled(rect, 3.0, color);
                                painter.rect_stroke(
                                    rect,
                                    3.0,
                                    Stroke::new(1.0_f32, Color32::from_rgb(50, 55, 65)),
                                    egui::StrokeKind::Inside,
                                );
                                painter.text(
                                    rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    format!("{}", val),
                                    egui::FontId::monospace(10.0),
                                    Color32::WHITE,
                                );

                                if (i + 1) % cols == 0 {
                                    ui.end_row();
                                }
                            }
                        });
                });
            }
        });
    }
}
