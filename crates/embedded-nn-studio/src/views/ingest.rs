use crate::state::StudioState;
use eframe::egui;
use embedded_nn_live::decode_f32_le;
use embedded_nn_live::host::{DeviceLink, OwnedMsg, UsbBridge};
use std::path::PathBuf;

/// Points retained per channel in the live oscilloscope ring buffer.
const SCOPE_HISTORY: usize = 400;

/// Names the first three channels X/Y/Z, matching the 3-DOF accelerometer the
/// HIL agent streams; anything beyond that is shown by index.
fn channel_label(channel: usize) -> String {
    match channel {
        0 => "X".into(),
        1 => "Y".into(),
        2 => "Z".into(),
        other => format!("ch{other}"),
    }
}

/// Per-channel trace colour, cycling for channel counts beyond 3.
fn channel_color(channel: usize) -> egui::Color32 {
    const PALETTE: [egui::Color32; 4] = [
        egui::Color32::from_rgb(255, 105, 97),
        egui::Color32::from_rgb(60, 210, 130),
        egui::Color32::from_rgb(90, 165, 255),
        egui::Color32::from_rgb(230, 190, 90),
    ];
    PALETTE[channel % PALETTE.len()]
}

#[derive(Default)]
pub struct IngestView {
    pub selected_port: String,
    pub sample_rate_hz: u32,
    pub current_label: String,
    pub new_class_name: String,
    pub is_recording: bool,
    pub is_burst_mode: bool,
    pub burst_length: usize,
    /// Dedicated buffer accumulating samples during an active recording burst
    pub active_burst_channels: Vec<Vec<f32>>,
    /// One ring buffer per sensor channel, de-interleaved from `SensorFrame`.
    /// The scope draws every channel; recordings collapse them to a scalar.
    pub live_channels: Vec<Vec<f32>>,
    pub live_time_counter: f32,
    pub import_status: String,
    pub available_agents: Vec<String>,
    pub link_status: String,
}

impl IngestView {
    pub fn new() -> Self {
        Self {
            selected_port: "Simulated IMU Source".into(),
            sample_rate_hz: 100,
            current_label: "wave_left".into(),
            new_class_name: String::new(),
            is_recording: false,
            is_burst_mode: true,
            burst_length: 128,
            active_burst_channels: Vec::new(),
            live_channels: vec![(0..100).map(|i| ((i as f32) * 0.1).sin() * 0.7).collect()],
            live_time_counter: 0.0,
            import_status: String::new(),
            available_agents: Vec::new(),
            link_status: "Disconnected".into(),
        }
    }

    /// Appends one de-interleaved `SensorFrame` payload, reshaping the buffers
    /// if the device changed its channel count.
    fn push_sensor_samples(&mut self, samples: &[f32], channel_count: usize) {
        let stride = channel_count.max(1);
        if self.live_channels.len() != stride {
            self.live_channels = vec![Vec::new(); stride];
        }
        for step in samples.chunks_exact(stride) {
            for (channel, value) in self.live_channels.iter_mut().zip(step) {
                channel.push(*value);
                if channel.len() > SCOPE_HISTORY {
                    let extra = channel.len() - SCOPE_HISTORY;
                    channel.drain(..extra);
                }
            }
        }
        if self.is_recording {
            if self.active_burst_channels.len() != stride {
                self.active_burst_channels = vec![Vec::new(); stride];
            }
            for step in samples.chunks_exact(stride) {
                for (channel, value) in self.active_burst_channels.iter_mut().zip(step) {
                    channel.push(*value);
                }
            }
        }
    }

    /// Number of samples collected in the current active recording buffer.
    pub fn recorded_samples_count(&self) -> usize {
        self.active_burst_channels
            .iter()
            .map(Vec::len)
            .min()
            .unwrap_or(0)
    }

    /// Starts a fresh recording burst from the current moment.
    pub fn start_recording(&mut self) {
        self.is_recording = true;
        let stride = self.live_channels.len().max(1);
        self.active_burst_channels = vec![Vec::new(); stride];
    }

    /// Commits the active recording buffer into the dataset with the current class tag.
    pub fn commit_recording(&mut self, state: &mut StudioState) {
        if self.active_burst_channels.is_empty() {
            self.is_recording = false;
            return;
        }
        let total_samples = self.recorded_samples_count();
        let target_len = if self.is_burst_mode && self.burst_length > 0 {
            total_samples.min(self.burst_length)
        } else {
            total_samples
        };

        if target_len == 0 {
            self.is_recording = false;
            self.active_burst_channels.clear();
            return;
        }

        let raw_waveform = self.scalar_history_from(&self.active_burst_channels, target_len);
        let trajectory = self.trajectory_from(&self.active_burst_channels, target_len);

        let prev_classes_len = state.classes.len();
        let class_idx = state.class_index_or_insert(&self.current_label);
        let id = state.next_sample_id;
        state.next_sample_id += 1;

        state.samples.push(crate::state::DatasetSample {
            id,
            label: self.current_label.clone(),
            class_idx,
            raw_waveform,
            trajectory,
            frames: Vec::new(),
            quantized_frames: Vec::new(),
        });
        state.recompute_all_frames();
        if state.classes.len() != prev_classes_len {
            state.reset_training();
        }
        state.rebuild_model_graph_and_codegen();

        self.is_recording = false;
        self.active_burst_channels.clear();
        self.import_status = format!(
            "Captured sample #{id} ({} samples) tagged '{}'.",
            target_len, self.current_label
        );
    }

    fn scalar_history_from(&self, channels: &[Vec<f32>], max_len: usize) -> Vec<f32> {
        let len = channels
            .iter()
            .map(Vec::len)
            .min()
            .unwrap_or(0)
            .min(max_len);
        match channels {
            [] => Vec::new(),
            [single] => single[..len].to_vec(),
            multi => (0..len)
                .map(|i| multi.iter().map(|c| c[i] * c[i]).sum::<f32>().sqrt())
                .collect(),
        }
    }

    fn trajectory_from(&self, channels: &[Vec<f32>], max_len: usize) -> Vec<[f32; 3]> {
        let [x, y, z] = channels else {
            return Vec::new();
        };
        let len = x.len().min(y.len()).min(z.len()).min(max_len);
        (0..len).map(|i| [x[i], y[i], z[i]]).collect()
    }

    /// Drains the device into the live channel buffers, or advances the
    /// simulated source when no device is handshaked. The app shell calls this
    /// every frame rather than `show` doing it, so the stream keeps flowing
    /// while a tab other than Ingest is on screen.
    pub fn poll_device(&mut self, device_link: &Option<DeviceLink>) {
        let Some(link) = device_link.as_ref() else {
            self.advance_simulated_source();
            return;
        };

        if link.is_handshaked() {
            self.link_status = format!("Ready {}", link.device_id());
        }
        if let Some(OwnedMsg::SensorFrame {
            timestamp_ms,
            channel_count,
            values,
        }) = link.take_sensor()
        {
            let mut samples = vec![0.0f32; values.len() / 4];
            if let Ok(n) = decode_f32_le(&values, &mut samples) {
                // `values` is channel-interleaved (x,y,z,x,y,z,...).
                self.push_sensor_samples(&samples[..n], usize::from(channel_count));
                self.link_status = format!("t={timestamp_ms} ms ch={channel_count} samples={n}");
            }
        }
        if let Some(error) = link.take_error() {
            self.link_status = error;
        }
        if !link.is_handshaked() {
            self.advance_simulated_source();
        }
    }

    fn advance_simulated_source(&mut self) {
        self.live_time_counter += 0.05;
        let t = self.live_time_counter;
        let new_sample = (t * 2.5).sin() * 0.7 + (t * 7.0).cos() * 0.2;
        self.push_sensor_samples(&[new_sample], 1);
    }

    /// Re-interleaves the live buffers into XYZ points for the 3D gesture
    /// view. `None` unless the source is exactly three channels, since a
    /// scalar or 6-DOF stream has no meaningful 3D trajectory.
    pub fn live_trajectory(&self) -> Option<Vec<[f32; 3]>> {
        let [x, y, z] = self.live_channels.as_slice() else {
            return None;
        };
        let len = x.len().min(y.len()).min(z.len());
        Some((0..len).map(|i| [x[i], y[i], z[i]]).collect())
    }

    fn import_dataset_files(&mut self, state: &mut StudioState, paths: &[PathBuf]) {
        match state.import_dataset_paths(paths) {
            Ok(count) => {
                self.import_status = format!("Imported {} sample(s).", count);
            }
            Err(e) => {
                self.import_status = format!("Import error: {e}");
            }
        }
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        state: &mut StudioState,
        device_link: &mut Option<DeviceLink>,
    ) {
        // Spacebar shortcut to start/stop burst capture when no text input is active
        let space_pressed = ui.input(|i| i.key_pressed(egui::Key::Space));
        let wants_keyboard = ui.ctx().wants_keyboard_input();
        if space_pressed && !wants_keyboard {
            if self.is_recording {
                self.commit_recording(state);
            } else {
                self.start_recording();
            }
        }

        // Auto-finalize if in burst mode and enough samples accumulated
        if self.is_recording
            && self.is_burst_mode
            && self.recorded_samples_count() >= self.burst_length
        {
            self.commit_recording(state);
        }

        ui.horizontal(|ui| {
            ui.heading("📊 1. Data Ingestion, Telemetry & Dataset Tagging");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("🔄 Reset to Demo Dataset").clicked() {
                    state.load_demo_dataset();
                    state.recompute_all_frames();
                    state.use_demo_trainer();
                }
                ui.label(format!("Total Dataset Samples: {}", state.samples.len()));
            });
        });

        ui.add_space(4.0);
        ui.label(
            "Record real-time physical sensor time-series (IMU accelerometer, gyroscopes, audio, PPG) or simulate streaming data, annotate class labels, and maintain balanced dataset splits.",
        );
        ui.add_space(8.0);

        // Top Control Bar
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("Sensor Source:");
                egui::ComboBox::from_id_salt("sensor_source_combo")
                    .selected_text(&self.selected_port)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.selected_port,
                            "Simulated IMU Source".into(),
                            "Simulated IMU Source",
                        );
                        for agent in &self.available_agents {
                            ui.selectable_value(
                                &mut self.selected_port,
                                agent.clone(),
                                format!("USB-HS {agent}"),
                            );
                        }
                    });
                if ui.button("Refresh USB").clicked() {
                    self.available_agents = UsbBridge::list_devices();
                    self.link_status = format!("{} agent(s)", self.available_agents.len());
                }
                if ui.button("Connect").clicked() {
                    *device_link = None;
                    if self.selected_port == "Simulated IMU Source" {
                        self.link_status = "Using simulated IMU (no USB).".into();
                    } else {
                        match DeviceLink::connect(&self.selected_port) {
                            Ok(link) => {
                                self.link_status = format!("Connecting {}", link.device_id());
                                *device_link = Some(link);
                            }
                            Err(error) => self.link_status = error,
                        }
                    }
                }
                if ui.button("Disconnect").clicked() {
                    *device_link = None;
                    self.link_status = "Disconnected".into();
                }
                if let Some(link) = device_link.as_ref()
                    && ui.button("Ping").clicked()
                {
                    link.ping();
                }
                ui.label(&self.link_status);

                ui.separator();

                ui.label("Sampling Rate:");
                ui.add(egui::DragValue::new(&mut self.sample_rate_hz).suffix(" Hz"));

                ui.separator();

                ui.label("Target Class Tag:");
                if !state.classes.is_empty() {
                    egui::ComboBox::from_id_salt("class_select_combo")
                        .selected_text(&self.current_label)
                        .show_ui(ui, |ui| {
                            for class_name in &state.classes {
                                ui.selectable_value(
                                    &mut self.current_label,
                                    class_name.clone(),
                                    class_name,
                                );
                            }
                        });
                }

                ui.separator();

                ui.checkbox(&mut self.is_burst_mode, "Burst Mode");
                if self.is_burst_mode {
                    ui.label("Length:");
                    ui.add(
                        egui::DragValue::new(&mut self.burst_length)
                            .range(16..=1024)
                            .suffix(" samples"),
                    );
                }

                let count = self.recorded_samples_count();
                let record_btn = if self.is_recording {
                    if self.is_burst_mode {
                        egui::Button::new(format!(
                            "⏹ Stop ({}/{}) [Space]",
                            count, self.burst_length
                        ))
                        .fill(egui::Color32::from_rgb(180, 50, 50))
                    } else {
                        egui::Button::new(format!("⏹ Stop ({count}) [Space]"))
                            .fill(egui::Color32::from_rgb(180, 50, 50))
                    }
                } else {
                    egui::Button::new("⏺ Record Sample [Space]")
                        .fill(egui::Color32::from_rgb(40, 140, 70))
                };

                if ui.add(record_btn).clicked() {
                    if self.is_recording {
                        self.commit_recording(state);
                    } else {
                        self.start_recording();
                    }
                }

                ui.separator();

                if ui.button("📂 Import Dataset File(s)").clicked()
                    && let Some(paths) = rfd::FileDialog::new()
                        .add_filter(
                            "Dataset Interchange (.jsonl, .csv, .json)",
                            &["jsonl", "ndjson", "json", "csv", "tsv"],
                        )
                        .pick_files()
                {
                    self.import_dataset_files(state, &paths);
                }

                if ui.button("💾 Export Dataset (.jsonl)").clicked() {
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(path) = rfd::FileDialog::new()
                        .set_file_name("dataset.jsonl")
                        .add_filter("JSON Lines dataset", &["jsonl", "ndjson"])
                        .save_file()
                    {
                        self.import_status = match state.export_dataset_jsonl(&path) {
                            Ok(msg) => msg,
                            Err(err) => err,
                        };
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        self.import_status =
                            match state.export_dataset_jsonl(Path::new("dataset.jsonl")) {
                                Ok(msg) => msg,
                                Err(err) => err,
                            };
                    }
                }
            });

            if !self.import_status.is_empty() {
                ui.label(&self.import_status);
            }
        });

        ui.add_space(8.0);

        // Live Oscilloscope & Class Balance Layout
        ui.columns(2, |cols| {
            // Left Column: Live Oscilloscope
            cols[0].group(|ui| {
                ui.horizontal(|ui| {
                    ui.label("📈 Live Oscilloscope Stream");
                    for channel in 0..self.live_channels.len() {
                        ui.colored_label(channel_color(channel), channel_label(channel));
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if self.is_recording {
                            let count = self.recorded_samples_count();
                            let label = if self.is_burst_mode {
                                let pct = (count as f32 / self.burst_length.max(1) as f32)
                                    .clamp(0.0, 1.0);
                                format!(
                                    "● RECORDING: {}/{} ({:.0}%)",
                                    count,
                                    self.burst_length,
                                    pct * 100.0
                                )
                            } else {
                                format!("● RECORDING: {} samples", count)
                            };
                            ui.colored_label(egui::Color32::from_rgb(255, 80, 80), label);
                        } else {
                            ui.colored_label(
                                egui::Color32::from_rgb(80, 200, 120),
                                "● LIVE STREAM",
                            );
                        }
                    });
                });

                let (rect, _resp) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), 130.0),
                    egui::Sense::hover(),
                );
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(14, 17, 24));

                // Draw center grid line
                let mid_y = rect.center().y;
                painter.line_segment(
                    [
                        egui::pos2(rect.left(), mid_y),
                        egui::pos2(rect.right(), mid_y),
                    ],
                    egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(30, 40, 55)),
                );

                let scale_y = 50.0;
                for (channel, history) in self.live_channels.iter().enumerate() {
                    let n = history.len();
                    if n < 2 {
                        continue;
                    }
                    let dx = rect.width() / (n - 1) as f32;
                    let stroke = egui::Stroke::new(
                        if self.is_recording { 2.0_f32 } else { 1.5_f32 },
                        channel_color(channel),
                    );
                    for i in 0..n - 1 {
                        let p1 =
                            egui::pos2(rect.left() + (i as f32) * dx, mid_y - history[i] * scale_y);
                        let p2 = egui::pos2(
                            rect.left() + ((i + 1) as f32) * dx,
                            mid_y - history[i + 1] * scale_y,
                        );
                        painter.line_segment([p1, p2], stroke);
                    }
                }
            });

            // Right Column: Class Distribution & Label Manager
            cols[1].group(|ui| {
                ui.label("🏷️ Class Distribution & Balance Monitor");
                ui.add_space(4.0);

                let mut class_counts = vec![0usize; state.classes.len()];
                for sample in &state.samples {
                    if sample.class_idx < class_counts.len() {
                        class_counts[sample.class_idx] += 1;
                    }
                }

                let total = state.samples.len().max(1);
                for (i, class_name) in state.classes.iter().enumerate() {
                    let count = class_counts.get(i).copied().unwrap_or(0);
                    let ratio = count as f32 / total as f32;
                    ui.horizontal(|ui| {
                        ui.label(format!("{}:", class_name));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(format!("{} samples ({:.0}%)", count, ratio * 100.0));
                            ui.add(egui::ProgressBar::new(ratio).desired_width(120.0));
                        });
                    });
                }

                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Add Class:");
                    ui.text_edit_singleline(&mut self.new_class_name);
                    if ui.button("+ Add").clicked() && !self.new_class_name.trim().is_empty() {
                        let name = self.new_class_name.trim().to_string();
                        if !state.classes.contains(&name) {
                            state.classes.push(name);
                            self.new_class_name.clear();
                            state.reset_training();
                            state.rebuild_model_graph_and_codegen();
                        }
                    }
                });
            });
        });

        ui.add_space(8.0);

        // Bottom: Recorded Dataset Browser Table
        ui.group(|ui| {
            ui.label("📁 Dataset Samples Explorer");
            ui.separator();

            egui::ScrollArea::vertical()
                .id_salt("ingest_samples_scroll_area")
                .max_height(160.0)
                .show(ui, |ui| {
                    egui::Grid::new("dataset_samples_grid")
                        .striped(true)
                        .min_col_width(80.0)
                        .show(ui, |ui| {
                            ui.label("ID");
                            ui.label("Class Tag");
                            ui.label("Signal Length");
                            ui.label("Feature Vector");
                            ui.label("Actions");
                            ui.end_row();

                            let mut delete_id = None;
                            let mut relabel = None;
                            for sample in state.samples.iter().rev().take(30) {
                                ui.label(format!("#{:03}", sample.id));

                                ui.push_id(sample.id, |ui| {
                                    egui::ComboBox::from_id_salt("sample_relabel_combo")
                                        .selected_text(&sample.label)
                                        .show_ui(ui, |ui| {
                                            for (idx, class_name) in
                                                state.classes.iter().enumerate()
                                            {
                                                if ui
                                                    .selectable_label(
                                                        sample.label == *class_name,
                                                        class_name,
                                                    )
                                                    .clicked()
                                                {
                                                    relabel =
                                                        Some((sample.id, idx, class_name.clone()));
                                                }
                                            }
                                        });
                                });

                                ui.label(format!("{} samples", sample.raw_waveform.len()));
                                let num_bins = sample.frames.first().map(|f| f.len()).unwrap_or(0);
                                ui.label(format!(
                                    "{} Mel bins × {} frames",
                                    num_bins,
                                    sample.frames.len()
                                ));

                                ui.push_id(sample.id, |ui| {
                                    if ui.button("🗑 Delete").clicked() {
                                        delete_id = Some(sample.id);
                                    }
                                });
                                ui.end_row();
                            }

                            if let Some((id, class_idx, label)) = relabel
                                && let Some(sample) = state.samples.iter_mut().find(|s| s.id == id)
                            {
                                sample.label = label;
                                sample.class_idx = class_idx;
                                state.rebuild_model_graph_and_codegen();
                            }

                            if let Some(id) = delete_id {
                                state.samples.retain(|s| s.id != id);
                                state.rebuild_model_graph_and_codegen();
                            }
                        });
                });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::UNLABELED_IMPORT;

    const FIXTURE: &str = concat!(
        r#"{"sample_id":"b1","label":null,"sample_rate_hz":400.0,"channel_names":["x","y","z"],"waveform":[[3.0,4.0,0.0],[0.0,0.0,2.0]]}"#,
        "\n",
        r#"{"sample_id":"b2","label":"recoil_anomaly","sample_rate_hz":400.0,"channel_names":["value"],"waveform":[[0.5],[-0.5]]}"#,
        "\n"
    );

    fn collect_text<'a>(shapes: impl Iterator<Item = &'a egui::Shape>, out: &mut String) {
        for shape in shapes {
            match shape {
                egui::Shape::Text(text) => out.push_str(text.galley.text()),
                egui::Shape::Vec(inner) => collect_text(inner.iter(), out),
                _ => {}
            }
        }
    }

    fn write_fixture(name: &str, contents: &str) -> PathBuf {
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn import_adds_samples_collapses_channels_and_registers_new_classes() {
        let mut view = IngestView::new();
        let mut state = StudioState::default();
        let before = state.samples.len();
        let path = write_fixture("enn_ingest_import_ok.jsonl", FIXTURE);

        view.import_dataset_files(&mut state, std::slice::from_ref(&path));

        assert_eq!(state.samples.len(), before + 2);
        let imported = &state.samples[before..];
        assert_eq!(imported[0].label, UNLABELED_IMPORT);
        assert_eq!(imported[0].raw_waveform, vec![5.0, 2.0]);
        assert_eq!(imported[1].label, "recoil_anomaly");
        assert_eq!(imported[1].raw_waveform, vec![0.5, -0.5]);

        // The 3-axis record keeps its motion path for the 3D view; the scalar
        // one has none to keep.
        assert_eq!(
            imported[0].trajectory,
            vec![[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]
        );
        assert!(imported[1].trajectory.is_empty());
        assert!(state.classes.contains(&UNLABELED_IMPORT.to_string()));
        assert_eq!(state.classes[imported[1].class_idx], "recoil_anomaly");

        // Features and codegen must be refreshed for the newly imported samples.
        assert_eq!(
            imported[0].frames.len(),
            crate::state::StudioState::num_frames_for_config(&state.dsp)
        );
        assert_eq!(imported[0].frames[0].len(), state.dsp.num_mel_bins);
        assert!(!state.generated_rust_code.is_empty());
        assert_eq!(state.bias_fc2.len(), state.classes.len());

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn live_trajectory_rebuilds_xyz_points_only_for_three_channel_streams() {
        let mut view = IngestView::new();

        // The default simulated source is scalar: no 3D trajectory.
        assert!(view.live_trajectory().is_none());

        view.push_sensor_samples(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3);
        assert_eq!(
            view.live_trajectory().unwrap(),
            vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        );

        // A channel-count change reshapes the buffers and drops the trajectory.
        view.push_sensor_samples(&[0.25], 1);
        assert!(view.live_trajectory().is_none());
    }

    #[test]
    fn view_paints_import_button_and_imported_sample_labels() {
        let mut view = IngestView::new();
        let mut state = StudioState::default();
        let path = write_fixture("enn_ingest_render.jsonl", FIXTURE);
        view.import_dataset_files(&mut state, std::slice::from_ref(&path));

        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1400.0, 1000.0),
            )),
            ..Default::default()
        };
        let output = ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let mut device_link = None;
                view.show(ui, &mut state, &mut device_link)
            });
        });

        let mut painted = String::new();
        collect_text(output.shapes.iter().map(|c| &c.shape), &mut painted);

        assert!(painted.contains("Import Dataset File(s)"));
        assert!(painted.contains("Imported 2 sample(s)."));
        // Newest samples render first in the explorer, each with a relabel combo.
        assert!(painted.contains("recoil_anomaly"));
        assert!(painted.contains(UNLABELED_IMPORT));

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn import_reports_malformed_file_without_adding_samples() {
        let mut view = IngestView::new();
        let mut state = StudioState::default();
        let before = state.samples.len();
        let path = write_fixture("enn_ingest_import_bad.jsonl", "not json\n");

        view.import_dataset_files(&mut state, std::slice::from_ref(&path));

        assert_eq!(state.samples.len(), before);
        assert!(view.import_status.contains("enn_ingest_import_bad.jsonl"));
        assert!(view.import_status.contains("line 1"));

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn burst_recording_accumulates_exact_samples_and_tags_class() {
        let mut view = IngestView::new();
        let mut state = StudioState::default();
        let initial_samples = state.samples.len();

        view.current_label = "custom_gesture".into();
        view.is_burst_mode = true;
        view.burst_length = 64;

        view.start_recording();
        assert!(view.is_recording);
        assert_eq!(view.recorded_samples_count(), 0);

        // Push 64 3-axis samples (simulating 64 sensor ticks)
        for _ in 0..64 {
            view.push_sensor_samples(&[0.1, 0.2, 0.9], 3);
        }

        assert_eq!(view.recorded_samples_count(), 64);

        view.commit_recording(&mut state);
        assert!(!view.is_recording);
        assert_eq!(state.samples.len(), initial_samples + 1);

        let sample = state.samples.last().unwrap();
        assert_eq!(sample.label, "custom_gesture");
        assert_eq!(sample.raw_waveform.len(), 64);
        assert_eq!(sample.trajectory.len(), 64);
    }

    #[test]
    fn export_dataset_jsonl_roundtrips_records() {
        let state = StudioState::default();
        let path = std::env::temp_dir().join("enn_test_export.jsonl");

        let res = state.export_dataset_jsonl(&path);
        assert!(res.is_ok());

        let contents = std::fs::read_to_string(&path).unwrap();
        let parsed = embedded_nn_live::parse_jsonl(&contents).unwrap();
        assert_eq!(parsed.len(), state.samples.len());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_burst_recording_auto_finalizes_in_show() {
        let mut view = IngestView::new();
        let mut state = StudioState::default();
        let initial_samples = state.samples.len();

        view.current_label = "auto_burst_class".into();
        view.is_burst_mode = true;
        view.burst_length = 32;

        view.start_recording();
        assert!(view.is_recording);

        for _ in 0..32 {
            view.push_sensor_samples(&[0.5, -0.5, 1.0], 3);
        }

        // Run an egui show frame to trigger auto-finalization
        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1400.0, 1000.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let mut device_link = None;
                view.show(ui, &mut state, &mut device_link);
            });
        });

        assert!(
            !view.is_recording,
            "Burst must auto-finalize once target samples reached"
        );
        assert_eq!(state.samples.len(), initial_samples + 1);
        assert_eq!(state.samples.last().unwrap().label, "auto_burst_class");
    }

    #[test]
    fn test_view_renders_export_and_burst_ui() {
        let mut view = IngestView::new();
        let mut state = StudioState::default();

        let ctx = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1400.0, 1000.0),
            )),
            ..Default::default()
        };
        let output = ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let mut device_link = None;
                view.show(ui, &mut state, &mut device_link);
            });
        });

        let mut painted = String::new();
        collect_text(output.shapes.iter().map(|c| &c.shape), &mut painted);

        assert!(painted.contains("Burst Mode"));
        assert!(painted.contains("Record Sample [Space]"));
        assert!(painted.contains("Export Dataset"));
    }
}
