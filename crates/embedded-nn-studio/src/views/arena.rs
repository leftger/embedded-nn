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
        name: "STM32F401RE",
        arch: "ARM Cortex-M4F (FPU)",
        freq_mhz: 84,
        sram_kb: 64,
        flash_kb: 512,
        rust_target: "thumbv7em-none-eabihf",
        target_features: "",
        stack_reserve_kb: None,
    },
    McuSpec {
        name: "ESP32-S3",
        arch: "Xtensa 32-bit LX7 Dual",
        freq_mhz: 240,
        sram_kb: 512,
        flash_kb: 8192,
        rust_target: "xtensa-esp32s3-none-elf",
        target_features: "",
        stack_reserve_kb: None,
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
    },
    McuSpec {
        name: "RP2350",
        arch: "Dual ARM Cortex-M33",
        freq_mhz: 150,
        sram_kb: 520,
        flash_kb: 4096,
        rust_target: "thumbv8m.main-none-eabihf",
        target_features: "",
        stack_reserve_kb: None,
    },
    McuSpec {
        name: "nRF52840",
        arch: "ARM Cortex-M4F (BLE)",
        freq_mhz: 64,
        sram_kb: 256,
        flash_kb: 1024,
        rust_target: "thumbv7em-none-eabihf",
        target_features: "",
        stack_reserve_kb: None,
    },
    McuSpec {
        name: "STM32WBA65RI",
        arch: "ARM Cortex-M33 (FPU + DSP)",
        freq_mhz: 100,
        sram_kb: 512,
        flash_kb: 2048,
        rust_target: "thumbv8m.main-none-eabihf",
        target_features: "+dsp",
        stack_reserve_kb: Some(192),
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
        for target in MCU_TARGETS.iter().filter(|t| t.name != "STM32WBA65RI") {
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
