use crate::state::StudioState;
use eframe::egui;

#[derive(Debug, Clone)]
struct McuSpec {
    name: &'static str,
    arch: &'static str,
    freq_mhz: u32,
    sram_kb: usize,
    flash_kb: usize,
    /// Rust target triple the generated `#![no_std]` model is compiled for on this silicon.
    rust_target: &'static str,
    /// Extra `-C target-feature` flags this silicon supports beyond the triple's baseline.
    /// Empty when the triple alone describes the target.
    target_features: &'static str,
    /// SRAM held back by a radio/protocol stack before any of it can be handed to the
    /// activation arena. `None` means this profile reserves nothing (a 0 KB reserve) and
    /// exposes no reserve control; `Some(kb)` is an editable starting point sized for a
    /// typical stack configuration, not a datasheet constant.
    stack_reserve_kb: Option<usize>,
    /// Active core current consumption at full clock frequency in mA.
    active_ma: f32,
    /// Sleep / standby current consumption in mA.
    sleep_ma: f32,
    /// Estimated clock cycles required per single INT8 multiply-accumulate on this architecture.
    cycles_per_mac: f32,
}

impl McuSpec {
    fn default_reserve_kb(&self) -> usize {
        self.stack_reserve_kb.unwrap_or(0)
    }

    fn reserve_is_configurable(&self) -> bool {
        self.stack_reserve_kb.is_some()
    }
}

const MCU_TARGETS: &[McuSpec] = &[
    McuSpec {
        name: "STM32WBA65RI",
        arch: "ARM Cortex-M33 (FPU + DSP)",
        freq_mhz: 100,
        sram_kb: 512,
        flash_kb: 2048,
        rust_target: "thumbv8m.main-none-eabihf",
        target_features: "+dsp",
        stack_reserve_kb: Some(192),
        active_ma: 8.5,
        sleep_ma: 0.004,
        cycles_per_mac: 1.0,
    },
    McuSpec {
        name: "STM32H743ZI",
        arch: "ARM Cortex-M7 Dual-Issue (480MHz)",
        freq_mhz: 480,
        sram_kb: 1024,
        flash_kb: 2048,
        rust_target: "thumbv7em-none-eabihf",
        target_features: "+fp64",
        stack_reserve_kb: None,
        active_ma: 110.0,
        sleep_ma: 0.040,
        cycles_per_mac: 0.5,
    },
    McuSpec {
        name: "nRF5340",
        arch: "Dual ARM Cortex-M33 (BLE 5.3 Audio)",
        freq_mhz: 128,
        sram_kb: 512,
        flash_kb: 1024,
        rust_target: "thumbv8m.main-none-eabihf",
        target_features: "+dsp",
        stack_reserve_kb: Some(128),
        active_ma: 6.2,
        sleep_ma: 0.003,
        cycles_per_mac: 1.0,
    },
    McuSpec {
        name: "ESP32-S3",
        arch: "Xtensa 32-bit LX7 Dual (Vector SIMD)",
        freq_mhz: 240,
        sram_kb: 512,
        flash_kb: 8192,
        rust_target: "xtensa-esp32s3-none-elf",
        target_features: "",
        stack_reserve_kb: None,
        active_ma: 45.0,
        sleep_ma: 0.015,
        cycles_per_mac: 0.8,
    },
    McuSpec {
        name: "RP2350",
        arch: "Dual ARM Cortex-M33 (Pico 2)",
        freq_mhz: 150,
        sram_kb: 520,
        flash_kb: 4096,
        rust_target: "thumbv8m.main-none-eabihf",
        target_features: "",
        stack_reserve_kb: None,
        active_ma: 14.0,
        sleep_ma: 0.008,
        cycles_per_mac: 1.2,
    },
    McuSpec {
        name: "Cortex-M55 / Ethos-U55",
        arch: "Armv8.1-M Helium + MicroNPU",
        freq_mhz: 200,
        sram_kb: 512,
        flash_kb: 2048,
        rust_target: "thumbv8.1m.main-none-eabihf",
        target_features: "+helium,+dsp",
        stack_reserve_kb: None,
        active_ma: 18.0,
        sleep_ma: 0.006,
        cycles_per_mac: 0.2,
    },
    McuSpec {
        name: "STM32F401RE",
        arch: "ARM Cortex-M4F (FPU)",
        freq_mhz: 84,
        sram_kb: 64,
        flash_kb: 512,
        rust_target: "thumbv7em-none-eabihf",
        target_features: "",
        stack_reserve_kb: None,
        active_ma: 12.0,
        sleep_ma: 0.010,
        cycles_per_mac: 1.5,
    },
    McuSpec {
        name: "RP2040",
        arch: "Dual ARM Cortex-M0+",
        freq_mhz: 133,
        sram_kb: 264,
        flash_kb: 2048,
        rust_target: "thumbv6m-none-eabi",
        target_features: "",
        stack_reserve_kb: None,
        active_ma: 18.0,
        sleep_ma: 0.020,
        cycles_per_mac: 4.0,
    },
    McuSpec {
        name: "nRF52840",
        arch: "ARM Cortex-M4F (BLE)",
        freq_mhz: 64,
        sram_kb: 256,
        flash_kb: 1024,
        rust_target: "thumbv7em-none-eabihf",
        target_features: "",
        stack_reserve_kb: Some(64),
        active_ma: 4.8,
        sleep_ma: 0.002,
        cycles_per_mac: 1.5,
    },
];

/// How the selected target's SRAM splits three ways: what the silicon has in total, what a
/// radio/protocol stack holds back, and what is therefore actually available to the activation
/// arena. The arena is always judged against `available_bytes`, never against `total_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SramBudget {
    total_bytes: usize,
    reserved_bytes: usize,
    available_bytes: usize,
}

impl SramBudget {
    fn new(total_kb: usize, reserved_kb: usize) -> Self {
        let total_bytes = total_kb.saturating_mul(1024);
        let reserved_bytes = reserved_kb.saturating_mul(1024).min(total_bytes);
        Self {
            total_bytes,
            reserved_bytes,
            available_bytes: total_bytes - reserved_bytes,
        }
    }

    fn fits(&self, arena_bytes: usize) -> bool {
        arena_bytes <= self.available_bytes
    }

    /// Signed, so an over-budget arena reports how far it overshoots rather than saturating.
    fn headroom_bytes(&self, arena_bytes: usize) -> i64 {
        self.available_bytes as i64 - arena_bytes as i64
    }

    /// Fraction of *available* SRAM the arena occupies, clamped to `[0, 1]` for gauge display.
    fn utilization(&self, arena_bytes: usize) -> f32 {
        if self.available_bytes == 0 {
            return if arena_bytes == 0 { 0.0 } else { 1.0 };
        }
        (arena_bytes as f32 / self.available_bytes as f32).clamp(0.0, 1.0)
    }
}

fn bytes_to_kb(bytes: usize) -> f32 {
    bytes as f32 / 1024.0
}

pub struct ArenaView {
    pub selected_mcu_idx: usize,
    /// Radio/protocol-stack SRAM reserve in KB, stored per target so editing one target's
    /// reserve and switching away preserves it (and leaves every other target untouched).
    stack_reserve_kb: Vec<usize>,
    /// Configurable target inference rate in Hz for battery life and power simulation.
    pub inference_rate_hz: f32,
    /// Selected battery capacity in mAh (e.g. 220 for CR2032, 500 for LiPo).
    pub battery_capacity_mah: f32,
}

impl Default for ArenaView {
    fn default() -> Self {
        Self::new()
    }
}

impl ArenaView {
    pub fn new() -> Self {
        Self {
            selected_mcu_idx: 0,
            stack_reserve_kb: MCU_TARGETS
                .iter()
                .map(McuSpec::default_reserve_kb)
                .collect(),
            inference_rate_hz: 10.0,
            battery_capacity_mah: 220.0,
        }
    }

    fn reserve_kb(&self, mcu_idx: usize) -> usize {
        self.stack_reserve_kb.get(mcu_idx).copied().unwrap_or(0)
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

        let selected_idx = self.selected_mcu_idx;
        let mcu = &MCU_TARGETS[selected_idx];

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
                ui.label(format!("SRAM: {} KB total", mcu.sram_kb));
                ui.separator();
                ui.label(format!("Flash: {} KB total", mcu.flash_kb));
            });

            ui.horizontal(|ui| {
                ui.label(format!("Rust target: {}", mcu.rust_target));
                if !mcu.target_features.is_empty() {
                    ui.separator();
                    ui.label(format!(
                        "RUSTFLAGS: -C target-feature={}",
                        mcu.target_features
                    ));
                }
            });

            if !mcu.target_features.is_empty() {
                ui.colored_label(
                    egui::Color32::from_rgb(245, 170, 60),
                    format!(
                        "Advisory: `-C target-feature={}` is accepted on stable Rust but warns \
                         \"unstable feature specified for `-Ctarget-feature`\" -- the flag is not \
                         stability-guaranteed. The generated kernels are portable and build \
                         without it.",
                        mcu.target_features
                    ),
                );
            }

            if mcu.reserve_is_configurable() {
                ui.horizontal(|ui| {
                    ui.label("Radio / protocol stack SRAM reserve:");
                    if let Some(reserve_kb) = self.stack_reserve_kb.get_mut(selected_idx) {
                        ui.add(
                            egui::DragValue::new(reserve_kb)
                                .range(0..=mcu.sram_kb)
                                .suffix(" KB"),
                        );
                    }
                    ui.separator();
                    ui.label(
                        "Held back for the BLE controller/host and its buffers. Adjust to match \
                         your stack's linker budget; the arena is sized from what remains.",
                    );
                });
            }
        });

        let sram = SramBudget::new(mcu.sram_kb, self.reserve_kb(selected_idx));

        ui.add_space(4.0);
        ui.label(format!(
            "SRAM: {:.0} KB total - {:.0} KB reserved = {:.0} KB available to the arena",
            bytes_to_kb(sram.total_bytes),
            bytes_to_kb(sram.reserved_bytes),
            bytes_to_kb(sram.available_bytes),
        ));

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

        let flash_budget_bytes = mcu.flash_kb * 1024;

        let sram_ratio = sram.utilization(sram_arena);
        let sram_fits = sram.fits(sram_arena);
        let sram_headroom = sram.headroom_bytes(sram_arena);
        let flash_ratio = (flash_weights as f32 / flash_budget_bytes as f32).clamp(0.0, 1.0);

        // Hardware Resource Gauges
        ui.columns(3, |cols| {
            cols[0].group(|ui| {
                ui.label("⚡ Flash (ROM) Footprint");
                ui.heading(format!("{:.2} KB", flash_weights as f32 / 1024.0));
                ui.label(format!("{} bytes const weights & biases", flash_weights));
                ui.add(egui::ProgressBar::new(flash_ratio).text(format!(
                    "{:.3}% of {} KB total Flash",
                    flash_ratio * 100.0,
                    mcu.flash_kb
                )));
            });

            cols[1].group(|ui| {
                ui.label("🧠 SRAM Activation Arena");
                ui.heading(format!("{} Bytes", sram_arena));
                ui.label(format!(
                    "{:.0} KB available of {:.0} KB total ({:.0} KB reserved)",
                    bytes_to_kb(sram.available_bytes),
                    bytes_to_kb(sram.total_bytes),
                    bytes_to_kb(sram.reserved_bytes),
                ));
                ui.add(
                    egui::ProgressBar::new(sram_ratio)
                        .fill(if sram_fits {
                            egui::Color32::from_rgb(60, 130, 220)
                        } else {
                            egui::Color32::from_rgb(210, 70, 70)
                        })
                        .text(format!("{:.3}% of available SRAM", sram_ratio * 100.0)),
                );
                if sram_fits {
                    ui.colored_label(
                        egui::Color32::from_rgb(80, 220, 120),
                        format!(
                            "PASS: fits available SRAM with {:.2} KB headroom",
                            sram_headroom as f32 / 1024.0
                        ),
                    );
                } else {
                    ui.colored_label(
                        egui::Color32::from_rgb(240, 100, 100),
                        format!(
                            "FAIL: exceeds available SRAM by {} bytes ({:.2} KB)",
                            -sram_headroom,
                            -sram_headroom as f32 / 1024.0
                        ),
                    );
                }
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

        // Memory Efficiency Comparison: embedded-nn vs. tflite-micro
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("⚖ Memory Efficiency vs. tflite-micro:").strong());
                let tensor_count = state
                    .compiled_graph
                    .as_ref()
                    .map(|g| g.tensors.len())
                    .unwrap_or(4);
                let tflm_metadata_overhead = tensor_count * 96 + 512;
                let tflm_estimated_arena = sram_arena + tflm_metadata_overhead;
                let sram_saved_pct =
                    (tflm_metadata_overhead as f32 / tflm_estimated_arena as f32) * 100.0;

                ui.label(format!(
                    "embedded-nn static arena: {} B  vs.  tflite-micro runtime allocator: ~{} B",
                    sram_arena, tflm_estimated_arena
                ));
                ui.separator();
                ui.colored_label(
                    egui::Color32::from_rgb(60, 220, 120),
                    format!(
                        "★ Saves ~{} B ({:.1}% less SRAM overhead)",
                        tflm_metadata_overhead, sram_saved_pct
                    ),
                );
            });
        });

        ui.add_space(8.0);

        // Hardware Power, Energy & Battery Life Estimator
        ui.group(|ui| {
            ui.label(
                egui::RichText::new("🔋 Silicon Power, Energy & Battery Runtime Estimator")
                    .strong(),
            );
            let total_macs = state
                .compiled_graph
                .as_ref()
                .map(calculate_model_macs)
                .unwrap_or(flash_weights.max(1));
            let total_cycles = ((total_macs as f32) * mcu.cycles_per_mac) as u32;
            let latency_us = total_cycles as f32 / mcu.freq_mhz as f32;
            let latency_ms = latency_us / 1000.0;
            let energy_uj = mcu.active_ma * 3.3 * latency_ms;
            let max_fps = 1000.0 / latency_ms.max(0.001);

            let duty_cycle = (latency_ms / 1000.0 * self.inference_rate_hz).clamp(0.0, 1.0);
            let avg_current_ma = (duty_cycle * mcu.active_ma) + ((1.0 - duty_cycle) * mcu.sleep_ma);
            let battery_hours = self.battery_capacity_mah / avg_current_ma.max(0.0001);
            let battery_days = battery_hours / 24.0;

            ui.horizontal(|ui| {
                ui.label("Inference Rate:");
                ui.add(
                    egui::Slider::new(&mut self.inference_rate_hz, 0.1..=100.0)
                        .suffix(" Hz")
                        .logarithmic(true),
                );
                ui.separator();
                ui.label("Battery Type:");
                egui::ComboBox::from_id_salt("arena_battery_type_combo")
                    .selected_text(format!("{:.0} mAh", self.battery_capacity_mah))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.battery_capacity_mah,
                            220.0,
                            "CR2032 Coin Cell (220 mAh)",
                        );
                        ui.selectable_value(&mut self.battery_capacity_mah, 500.0, "LiPo 500 mAh");
                        ui.selectable_value(
                            &mut self.battery_capacity_mah,
                            1000.0,
                            "AAA x2 Battery (1000 mAh)",
                        );
                        ui.selectable_value(
                            &mut self.battery_capacity_mah,
                            2500.0,
                            "18650 Li-Ion (2500 mAh)",
                        );
                    });
            });

            ui.separator();
            ui.columns(4, |pcols| {
                pcols[0].label(format!("Total Complexity:\n{} MACs", total_macs));
                pcols[1].label(format!(
                    "Latency / Max FPS:\n{:.2} ms (~{:.0} FPS)",
                    latency_ms, max_fps
                ));
                pcols[2].label(format!("Energy / Inference:\n{:.2} µJ @ 3.3V", energy_uj));
                if battery_days >= 365.0 {
                    pcols[3].colored_label(
                        egui::Color32::from_rgb(80, 220, 120),
                        format!(
                            "Est. Battery Life:\n{:.1} Years ({:.0} Days)",
                            battery_days / 365.0,
                            battery_days
                        ),
                    );
                } else if battery_days >= 1.0 {
                    pcols[3].colored_label(
                        egui::Color32::from_rgb(80, 220, 120),
                        format!(
                            "Est. Battery Life:\n{:.0} Days ({:.1} Months)",
                            battery_days,
                            battery_days / 30.0
                        ),
                    );
                } else {
                    pcols[3].colored_label(
                        egui::Color32::from_rgb(240, 180, 60),
                        format!("Est. Battery Life:\n{:.1} Hours", battery_hours),
                    );
                }
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

pub fn calculate_model_macs(graph: &embedded_nn_compiler::ir::ModelGraph) -> usize {
    let mut total_macs = 0;
    for layer in &graph.layers {
        match &layer.op {
            embedded_nn_compiler::ir::OpPayload::FullyConnected { weights, .. } => {
                total_macs += weights.len();
            }
            embedded_nn_compiler::ir::OpPayload::Conv1D {
                kernel_w, weights, ..
            } => {
                if let (Some(in_t), Some(out_t)) = (
                    layer
                        .inputs
                        .first()
                        .and_then(|id| graph.tensors.iter().find(|t| t.id == *id)),
                    layer
                        .outputs
                        .first()
                        .and_then(|id| graph.tensors.iter().find(|t| t.id == *id)),
                ) {
                    let out_channels =
                        weights.len() / (kernel_w.max(&1) * in_t.shape.channels.max(1));
                    total_macs +=
                        out_t.shape.width * out_channels * (kernel_w * in_t.shape.channels);
                } else {
                    total_macs += weights.len();
                }
            }
            embedded_nn_compiler::ir::OpPayload::Conv2D {
                kernel_h,
                kernel_w,
                weights,
                ..
            } => {
                if let (Some(in_t), Some(out_t)) = (
                    layer
                        .inputs
                        .first()
                        .and_then(|id| graph.tensors.iter().find(|t| t.id == *id)),
                    layer
                        .outputs
                        .first()
                        .and_then(|id| graph.tensors.iter().find(|t| t.id == *id)),
                ) {
                    let out_channels = weights.len()
                        / (kernel_h.max(&1) * kernel_w.max(&1) * in_t.shape.channels.max(1));
                    total_macs += out_t.shape.height
                        * out_t.shape.width
                        * out_channels
                        * (kernel_h * kernel_w * in_t.shape.channels);
                } else {
                    total_macs += weights.len();
                }
            }
            embedded_nn_compiler::ir::OpPayload::DepthwiseConv2D {
                kernel_h,
                kernel_w,
                ch_mult,
                ..
            } => {
                if let (Some(in_t), Some(out_t)) = (
                    layer
                        .inputs
                        .first()
                        .and_then(|id| graph.tensors.iter().find(|t| t.id == *id)),
                    layer
                        .outputs
                        .first()
                        .and_then(|id| graph.tensors.iter().find(|t| t.id == *id)),
                ) {
                    total_macs += out_t.shape.height
                        * out_t.shape.width
                        * in_t.shape.channels
                        * ch_mult
                        * (kernel_h * kernel_w);
                }
            }
            embedded_nn_compiler::ir::OpPayload::Svdf {
                rank,
                memory_size,
                weights_feature,
                ..
            } => {
                if let Some(in_t) = layer
                    .inputs
                    .first()
                    .and_then(|id| graph.tensors.iter().find(|t| t.id == *id))
                {
                    let units = weights_feature.len() / (rank * in_t.shape.total_elements().max(1));
                    total_macs += units * (rank * in_t.shape.total_elements() + rank * memory_size);
                } else {
                    total_macs += weights_feature.len();
                }
            }
            _ => {}
        }
    }
    total_macs
}

#[cfg(test)]
mod tests {
    use super::*;

    const KB: usize = 1024;

    fn spec(name: &str) -> &'static McuSpec {
        MCU_TARGETS
            .iter()
            .find(|target| target.name == name)
            .unwrap_or_else(|| panic!("{name} is not a known MCU target"))
    }

    fn index_of(name: &str) -> usize {
        MCU_TARGETS
            .iter()
            .position(|target| target.name == name)
            .unwrap_or_else(|| panic!("{name} is not a known MCU target"))
    }

    #[test]
    fn wba65ri_production_profile_matches_silicon_and_toolchain() {
        let wba = spec("STM32WBA65RI");
        assert_eq!(wba.arch, "ARM Cortex-M33 (FPU + DSP)");
        assert_eq!(wba.freq_mhz, 100);
        assert_eq!(wba.flash_kb, 2048);
        assert_eq!(wba.sram_kb, 512);
        assert_eq!(wba.rust_target, "thumbv8m.main-none-eabihf");
        assert_eq!(wba.target_features, "+dsp");
        assert_eq!(wba.default_reserve_kb(), 192);
        assert!(wba.reserve_is_configurable());
    }

    #[test]
    fn targets_without_a_radio_stack_reserve_nothing() {
        for target in MCU_TARGETS.iter().filter(|t| t.stack_reserve_kb.is_none()) {
            assert_eq!(
                target.default_reserve_kb(),
                0,
                "{} should not reserve SRAM",
                target.name
            );
            assert!(
                !target.reserve_is_configurable(),
                "{} should not expose a reserve control",
                target.name
            );

            let budget = SramBudget::new(target.sram_kb, target.default_reserve_kb());
            assert_eq!(budget.reserved_bytes, 0);
            assert_eq!(budget.available_bytes, budget.total_bytes);
        }
    }

    #[test]
    fn budget_distinguishes_total_from_available_sram() {
        let wba = spec("STM32WBA65RI");
        let budget = SramBudget::new(wba.sram_kb, wba.default_reserve_kb());

        assert_eq!(budget.total_bytes, 512 * KB);
        assert_eq!(budget.reserved_bytes, 192 * KB);
        assert_eq!(budget.available_bytes, 320 * KB);
    }

    #[test]
    fn reserve_larger_than_sram_clamps_to_no_available_memory() {
        let budget = SramBudget::new(512, 1024);

        assert_eq!(budget.reserved_bytes, budget.total_bytes);
        assert_eq!(budget.available_bytes, 0);
        assert!(budget.fits(0));
        assert!(!budget.fits(1));
        assert_eq!(budget.utilization(0), 0.0);
        assert_eq!(budget.utilization(1), 1.0);
    }

    #[test]
    fn arena_pass_fail_is_judged_against_available_not_total_sram() {
        let budget = SramBudget::new(512, 192);

        // Fits the 512 KB of physical SRAM, but not the 320 KB left after the stack reserve.
        assert!(!budget.fits(400 * KB));
        assert_eq!(budget.headroom_bytes(400 * KB), -(80 * KB as i64));

        assert!(budget.fits(320 * KB));
        assert_eq!(budget.headroom_bytes(320 * KB), 0);

        assert!(budget.fits(319 * KB));
        assert_eq!(budget.headroom_bytes(319 * KB), KB as i64);
    }

    #[test]
    fn utilization_is_a_clamped_fraction_of_available_sram() {
        let budget = SramBudget::new(512, 192);

        assert_eq!(budget.utilization(0), 0.0);
        assert!((budget.utilization(160 * KB) - 0.5).abs() < 1e-6);
        assert_eq!(budget.utilization(320 * KB), 1.0);
        assert_eq!(budget.utilization(usize::MAX / 2), 1.0);
    }

    #[test]
    fn reserve_defaults_come_from_specs_and_are_preserved_per_target() {
        let mut view = ArenaView::new();
        assert_eq!(view.stack_reserve_kb.len(), MCU_TARGETS.len());
        for (idx, target) in MCU_TARGETS.iter().enumerate() {
            assert_eq!(view.reserve_kb(idx), target.default_reserve_kb());
        }

        let wba_idx = index_of("STM32WBA65RI");
        view.stack_reserve_kb[wba_idx] = 64;

        // Switching targets and back must not reset the edited value, and must not leak it.
        view.selected_mcu_idx = index_of("RP2350");
        assert_eq!(view.reserve_kb(view.selected_mcu_idx), 0);
        view.selected_mcu_idx = wba_idx;
        assert_eq!(view.reserve_kb(wba_idx), 64);

        let budget = SramBudget::new(spec("STM32WBA65RI").sram_kb, view.reserve_kb(wba_idx));
        assert_eq!(budget.available_bytes, 448 * KB);
    }

    #[test]
    fn unknown_target_index_reserves_nothing() {
        let view = ArenaView::new();
        assert_eq!(view.reserve_kb(MCU_TARGETS.len()), 0);
    }
}
