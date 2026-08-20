use embedded_dsp::complex_math::cmplx_mag_f32;
use embedded_dsp::filter_design::biquad_highpass_coeffs;
use embedded_dsp::filtering::{BiquadCascadeInstanceF32, biquad_cascade_df1_f32};
use embedded_dsp::transform::rfft_f32;
use embedded_dsp::window::{apply_window_f32, hamming_f32, hanning_f32};
use embedded_nn::mel_filterbank_f32;
use embedded_nn_codegen::RustCodeGenerator;
use embedded_nn_compiler::arena::{ArenaPlan, ArenaScheduler};
use embedded_nn_compiler::builder::ModelBuilder;
use embedded_nn_compiler::ir::*;
use embedded_nn_compiler::quant::{
    calculate_asymmetric_quant_s8, calculate_output_requant_multiplier,
    calculate_symmetric_quant_s8, quantize_and_pack_weights_s4, quantize_weights_s8,
};
use std::f32::consts::PI;

/// Number of output channels for the `TinyConv1D` temporal convolution frontend.
const CONV1D_OUT_CHANNELS: usize = 4;
/// Kernel width (in frames) for the `TinyConv1D` temporal convolution frontend.
const CONV1D_KERNEL_W: usize = 3;
/// SVDF rank (number of filters per unit) for the `RecurrentSVDF` architecture.
const SVDF_RANK: usize = 1;
/// SVDF delay-line memory depth (time steps) for the `RecurrentSVDF` architecture.
const SVDF_MEMORY_SIZE: usize = 4;
/// Fixed symmetric scale used to quantize the raw Mel-energy sensor features (matches
/// `quantize_frame`'s `(f * 127.0)` convention) -- the "input" end of every quantization chain.
const INPUT_FEATURE_SCALE: f32 = 1.0 / 127.0;
/// Fixed symmetric scale for SVDF's internal i8 delay-line state. This is an
/// implementation-internal fixed-point representation with no externally observed float range,
/// so its scale is a convention (matching `INPUT_FEATURE_SCALE`) rather than calibrated.
const SVDF_STATE_SCALE: f32 = 1.0 / 127.0;

#[derive(Debug, Clone)]
pub struct DatasetSample {
    pub id: usize,
    pub label: String,
    pub class_idx: usize,
    pub raw_waveform: Vec<f32>,
    /// Per-frame Mel-energy feature sequence (the real temporal axis), each frame
    /// length `dsp.num_mel_bins`. Every sample has exactly `num_frames_for_config(&dsp)` frames.
    /// This is the single source of truth; pooled/flattened views are derived on demand.
    pub frames: Vec<Vec<f32>>,
    pub quantized_frames: Vec<Vec<i8>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunction {
    Hann,
    Hamming,
    Rectangular,
}

#[derive(Debug, Clone)]
pub struct DspConfig {
    /// FFT analysis window length in samples. Must be a power of two (embedded-dsp FFT constraint).
    pub window_size: usize,
    pub window_type: WindowFunction,
    pub num_mel_bins: usize,
    /// High-pass filter cutoff in Hz, applied to the raw waveform before windowing; also used as
    /// the Mel filterbank's lower frequency bound. `<= 0.0` disables the filter.
    pub high_pass_cutoff: f32,
    /// Sensor sampling rate in Hz, used for FFT bin -> Hz mapping and filter design.
    pub sample_rate: f32,
    /// Stride between successive analysis frames, in samples.
    pub frame_hop_size: usize,
    /// Fixed capture-window length every raw waveform is truncated/zero-padded to before framing,
    /// so every sample yields the same number of frames regardless of its original recording length.
    pub capture_samples: usize,
}

impl Default for DspConfig {
    fn default() -> Self {
        Self {
            window_size: 64,
            window_type: WindowFunction::Hann,
            num_mel_bins: 16,
            high_pass_cutoff: 10.0,
            sample_rate: 100.0,
            frame_hop_size: 32,
            capture_samples: 256,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    DenseMLP,
    TinyConv1D,
    RecurrentSVDF,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    Int4SubByte,
    Int8FixedPoint,
    Int16HighPrecision,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arch: ModelArchitecture,
    pub hidden_units: usize,
    pub quant_mode: QuantizationMode,
    pub epochs: usize,
    pub learning_rate: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            arch: ModelArchitecture::DenseMLP,
            hidden_units: 16,
            quant_mode: QuantizationMode::Int4SubByte,
            epochs: 50,
            learning_rate: 0.02,
        }
    }
}

pub struct StudioState {
    // Dataset & Classes
    pub classes: Vec<String>,
    pub samples: Vec<DatasetSample>,
    pub next_sample_id: usize,

    // DSP Pipeline
    pub dsp: DspConfig,

    // Training Config & Progress
    pub model_config: ModelConfig,
    pub is_training: bool,
    pub current_epoch: usize,
    pub train_loss_history: Vec<f32>,
    pub val_acc_history: Vec<f32>,
    pub confusion_matrix: Vec<Vec<usize>>, // [true_class][pred_class]

    // Trained Weights & Model Graph (DenseMLP head, also reused as the classifier head
    // for TinyConv1D and RecurrentSVDF)
    pub weights_fc1: Vec<f32>, // shape: (hidden_units, dense_head_input_dim)
    pub bias_fc1: Vec<f32>,
    pub weights_fc2: Vec<f32>, // shape: (num_classes, dense_head_input_dim2)
    pub bias_fc2: Vec<f32>,

    // TinyConv1D: real, trained temporal convolution frontend (kernel slides over frames/time;
    // channels = Mel bins). weights shape: (CONV1D_OUT_CHANNELS, kernel_w, num_mel_bins)
    pub conv1d_weights: Vec<f32>,
    pub conv1d_bias: Vec<f32>,

    // RecurrentSVDF: real, trained reservoir + delay-line filter, updated via direct (non-BPTT)
    // gradients -- see step_training_epoch_svdf for why no recursive unroll is needed.
    pub svdf_weights_feature: Vec<f32>, // shape: (units*rank, num_mel_bins)
    pub svdf_weights_time: Vec<f32>,    // shape: (units*rank, SVDF_MEMORY_SIZE)
    pub svdf_bias: Vec<f32>,            // shape: (units)

    // Compiled Graph, Arena Plan & Codegen
    pub compiled_graph: Option<ModelGraph>,
    pub arena_plan: Option<ArenaPlan>,
    pub generated_rust_code: String,

    // Live Sensor / HIL Test Vector
    pub test_input_vector: Vec<i8>,
    pub test_output_logits: Vec<i8>,
    pub test_probabilities: Vec<f32>,
}

impl Default for StudioState {
    fn default() -> Self {
        let mut state = Self {
            classes: vec![
                "idle".into(),
                "wave_left".into(),
                "wave_right".into(),
                "shake".into(),
            ],
            samples: Vec::new(),
            next_sample_id: 1,

            dsp: DspConfig::default(),
            model_config: ModelConfig::default(),

            is_training: false,
            current_epoch: 0,
            train_loss_history: Vec::new(),
            val_acc_history: Vec::new(),
            confusion_matrix: vec![vec![0; 4]; 4],

            weights_fc1: Vec::new(),
            bias_fc1: Vec::new(),
            weights_fc2: Vec::new(),
            bias_fc2: Vec::new(),

            conv1d_weights: Vec::new(),
            conv1d_bias: Vec::new(),

            svdf_weights_feature: Vec::new(),
            svdf_weights_time: Vec::new(),
            svdf_bias: Vec::new(),

            compiled_graph: None,
            arena_plan: None,
            generated_rust_code: String::new(),

            test_input_vector: vec![0; 16],
            test_output_logits: vec![0; 4],
            test_probabilities: vec![0.25; 4],
        };

        // Populate with rich starter demo dataset
        state.load_demo_dataset();
        state.recompute_all_frames();
        state.reset_training();
        state.run_simulated_training(40);
        state.rebuild_model_graph_and_codegen();
        state
    }
}

impl StudioState {
    /// Loads a multi-class synthetic IMU gesture dataset for instant interactive experimentation
    pub fn load_demo_dataset(&mut self) {
        self.samples.clear();
        self.next_sample_id = 1;

        let num_samples_per_class = 20;
        let signal_length = 256;

        for (class_idx, class_name) in self.classes.iter().enumerate() {
            for s in 0..num_samples_per_class {
                let mut waveform = Vec::with_capacity(signal_length);
                let phase_shift = (s as f32) * 0.15;
                let noise_level = 0.08;

                for t in 0..signal_length {
                    let time = (t as f32) / (signal_length as f32);
                    let noise = (((t * 13 + s * 7) % 100) as f32 / 100.0 - 0.5) * noise_level;

                    let val = match class_idx {
                        0 => 0.05 * (time * 2.0 * PI).sin() + noise, // idle: low amplitude drift
                        1 => (2.0 * PI * (time * 1.5 + phase_shift)).sin() * 0.9 + noise, // wave_left: low-freq sweep
                        2 => (2.0 * PI * (time * 3.5 + phase_shift)).sin() * 0.85 + noise, // wave_right: mid-freq sweep
                        3 => (2.0 * PI * (time * 7.0 + phase_shift)).sin() * 0.95 + noise * 2.0, // shake: high-freq burst
                        _ => 0.0,
                    };
                    waveform.push(val);
                }

                let id = self.next_sample_id;
                self.next_sample_id += 1;
                self.samples.push(DatasetSample {
                    id,
                    label: class_name.clone(),
                    class_idx,
                    raw_waveform: waveform,
                    frames: Vec::new(),
                    quantized_frames: Vec::new(),
                });
            }
        }
    }

    /// Index of `label` in `classes`, appending it as a new class if unseen
    pub fn class_index_or_insert(&mut self, label: &str) -> usize {
        match self.classes.iter().position(|c| c == label) {
            Some(idx) => idx,
            None => {
                self.classes.push(label.to_string());
                self.classes.len() - 1
            }
        }
    }

    /// Number of frames a raw waveform normalized to `capture_samples` yields under the current
    /// window/hop configuration. Deterministic and identical across all samples, so every
    /// architecture's weight shapes stay fixed regardless of an individual recording's length.
    pub(crate) fn num_frames_for_config(dsp: &DspConfig) -> usize {
        let capture = dsp.capture_samples.max(dsp.window_size);
        let hop = dsp.frame_hop_size.max(1);
        (capture - dsp.window_size) / hop + 1
    }

    /// Effective TinyConv1D kernel width and resulting output width for the current frame count.
    /// The kernel shrinks gracefully (never underflows) when there are fewer frames than the
    /// nominal `CONV1D_KERNEL_W`.
    fn conv1d_shape_for_config(dsp: &DspConfig) -> (usize, usize) {
        let num_frames = Self::num_frames_for_config(dsp).max(1);
        let kernel_w = CONV1D_KERNEL_W.min(num_frames);
        let out_width = num_frames - kernel_w + 1;
        (kernel_w, out_width)
    }

    /// Truncates or zero-pads `raw` to exactly `capture_samples`, so every sample yields the same
    /// number of frames regardless of how long the original recording was.
    fn normalize_raw_to_capture_length(raw: &[f32], capture_samples: usize) -> Vec<f32> {
        let mut buf = vec![0.0f32; capture_samples];
        let n = raw.len().min(capture_samples);
        buf[..n].copy_from_slice(&raw[..n]);
        buf
    }

    /// Applies the (optional) high-pass biquad filter over the full continuous signal, before
    /// framing -- filtering per-frame independently would reset filter state at every frame
    /// boundary and corrupt the response, especially with overlapping frames.
    fn apply_high_pass_filter(dsp: &DspConfig, raw: &[f32]) -> Vec<f32> {
        if dsp.high_pass_cutoff <= 0.0 {
            return raw.to_vec();
        }
        let coeffs = biquad_highpass_coeffs(dsp.high_pass_cutoff, dsp.sample_rate, 0.7071);
        let mut hp_state = [0.0f32; 4];
        let mut instance = BiquadCascadeInstanceF32::init(1, &coeffs, &mut hp_state);
        let mut filtered = vec![0.0f32; raw.len()];
        biquad_cascade_df1_f32(&mut instance, raw, &mut filtered);
        filtered
    }

    /// Real Window -> FFT -> magnitude -> Mel-filterbank feature extraction for a single frame.
    /// `frame.len()` must equal `dsp.window_size`; `window_size` must be a power of two
    /// (embedded-dsp's FFT constraint -- Studio's UI only offers power-of-two sizes).
    fn extract_frame_features(dsp: &DspConfig, frame: &[f32]) -> Vec<f32> {
        let n = dsp.window_size;
        let mut windowed = frame.to_vec();
        windowed.resize(n, 0.0);

        match dsp.window_type {
            WindowFunction::Hann => {
                let mut win = vec![0.0f32; n];
                hanning_f32(&mut win);
                apply_window_f32(&mut windowed, &win);
            }
            WindowFunction::Hamming => {
                let mut win = vec![0.0f32; n];
                hamming_f32(&mut win);
                apply_window_f32(&mut windowed, &win);
            }
            WindowFunction::Rectangular => {}
        }

        let mut spectrum = vec![0.0f32; 2 * n];
        rfft_f32(&windowed, &mut spectrum, n, 0);

        let mut mag = vec![0.0f32; n];
        cmplx_mag_f32(&spectrum, &mut mag);

        let half = n / 2;
        let min_freq = dsp.high_pass_cutoff.max(1.0);
        let max_freq = (dsp.sample_rate / 2.0).max(min_freq + 1.0);

        let mut mel_energies = vec![0.0f32; dsp.num_mel_bins];
        mel_filterbank_f32(
            &mag[..half],
            dsp.sample_rate,
            min_freq,
            max_freq,
            &mut mel_energies,
        );

        // Normalize into a [0, 1] range so downstream quantization (`* 127.0`) and the QAT
        // clamp ranges used during training stay meaningful regardless of signal amplitude.
        let max_e = mel_energies.iter().cloned().fold(1e-6f32, f32::max);
        for e in &mut mel_energies {
            *e = (*e / max_e).clamp(0.0, 1.0);
        }
        mel_energies
    }

    /// Real multi-frame DSP pipeline: normalize -> high-pass filter -> sliding window -> per-frame
    /// Window/FFT/Mel extraction. Every sample yields exactly `num_frames_for_config(dsp)` frames.
    pub fn extract_frame_sequence_with_dsp(dsp: &DspConfig, raw: &[f32]) -> Vec<Vec<f32>> {
        let capture = dsp.capture_samples.max(dsp.window_size);
        let normalized = Self::normalize_raw_to_capture_length(raw, capture);
        let filtered = Self::apply_high_pass_filter(dsp, &normalized);

        let num_frames = Self::num_frames_for_config(dsp);
        let hop = dsp.frame_hop_size.max(1);

        (0..num_frames)
            .map(|f| {
                let start = f * hop;
                let end = (start + dsp.window_size).min(filtered.len());
                let mut frame_buf = vec![0.0f32; dsp.window_size];
                if end > start {
                    frame_buf[..end - start].copy_from_slice(&filtered[start..end]);
                }
                Self::extract_frame_features(dsp, &frame_buf)
            })
            .collect()
    }

    fn mean_pool_frames(frames: &[Vec<f32>], num_mel_bins: usize) -> Vec<f32> {
        if frames.is_empty() {
            return vec![0.0; num_mel_bins];
        }
        let mut pooled = vec![0.0f32; num_mel_bins];
        for frame in frames {
            for (p, &v) in pooled.iter_mut().zip(frame.iter()) {
                *p += v;
            }
        }
        for p in &mut pooled {
            *p /= frames.len() as f32;
        }
        pooled
    }

    fn quantize_frame(frame: &[f32]) -> Vec<i8> {
        frame
            .iter()
            .map(|&f| ((f * 127.0).round().clamp(-128.0, 127.0)) as i8)
            .collect()
    }

    /// The HIL playground's flat i8 test vector, per architecture: `DenseMLP`/`RecurrentSVDF`
    /// use the pooled per-sample feature vector (derived on demand from `frames`); `TinyConv1D`'s
    /// deployed `predict()` takes a full multi-frame window in one call, so its test vector is
    /// the flattened frame sequence.
    pub(crate) fn test_input_vector_for(
        arch: ModelArchitecture,
        num_mel_bins: usize,
        sample: &DatasetSample,
    ) -> Vec<i8> {
        match arch {
            ModelArchitecture::DenseMLP | ModelArchitecture::RecurrentSVDF => {
                let pooled = Self::mean_pool_frames(&sample.frames, num_mel_bins);
                Self::quantize_frame(&pooled)
            }
            ModelArchitecture::TinyConv1D => sample
                .quantized_frames
                .iter()
                .flat_map(|f| f.iter().copied())
                .collect(),
        }
    }

    /// Recompute the per-frame Mel feature sequence for all samples when DSP parameters change
    pub fn recompute_all_frames(&mut self) {
        let dsp = self.dsp.clone();
        for sample in &mut self.samples {
            let frame_seq = Self::extract_frame_sequence_with_dsp(&dsp, &sample.raw_waveform);
            let quant_frame_seq: Vec<Vec<i8>> =
                frame_seq.iter().map(|f| Self::quantize_frame(f)).collect();

            sample.frames = frame_seq;
            sample.quantized_frames = quant_frame_seq;
        }

        let arch = self.model_config.arch;
        if let Some(first) = self.samples.first() {
            self.test_input_vector = Self::test_input_vector_for(arch, dsp.num_mel_bins, first);
        }
    }

    /// Deterministic pseudo-random weight generator (same LCG-ish scheme used throughout Studio)
    /// so architecture switches remain reproducible without pulling in a `rand` dependency.
    fn pseudo_random_weight(index: usize, mult: usize, add: usize, scale: f32) -> f32 {
        (((index * mult + add) % 100) as f32 / 100.0 - 0.5) * 2.0 * scale
    }

    /// Initialize training weights
    pub fn reset_training(&mut self) {
        self.current_epoch = 0;
        self.is_training = false;
        self.train_loss_history.clear();
        self.val_acc_history.clear();

        let num_inputs = self.dsp.num_mel_bins;
        let num_classes = self.classes.len();

        match self.model_config.arch {
            ModelArchitecture::DenseMLP => {
                let num_hidden = self.model_config.hidden_units;
                let scale1 = (2.0 / (num_inputs + num_hidden) as f32).sqrt();
                let scale2 = (2.0 / (num_hidden + num_classes) as f32).sqrt();

                self.weights_fc1 = (0..num_hidden * num_inputs)
                    .map(|i| Self::pseudo_random_weight(i, 17, 5, scale1))
                    .collect();
                self.bias_fc1 = vec![0.0; num_hidden];

                self.weights_fc2 = (0..num_classes * num_hidden)
                    .map(|i| Self::pseudo_random_weight(i, 31, 11, scale2))
                    .collect();
                self.bias_fc2 = vec![0.0; num_classes];

                self.conv1d_weights.clear();
                self.conv1d_bias.clear();
                self.svdf_weights_feature.clear();
                self.svdf_weights_time.clear();
                self.svdf_bias.clear();
            }
            ModelArchitecture::TinyConv1D => {
                let num_hidden = self.model_config.hidden_units;
                let (kernel_w, out_width) = Self::conv1d_shape_for_config(&self.dsp);
                let conv_out_len = CONV1D_OUT_CHANNELS * out_width;

                let conv_scale =
                    (2.0 / (kernel_w * num_inputs + CONV1D_OUT_CHANNELS) as f32).sqrt();
                self.conv1d_weights = (0..CONV1D_OUT_CHANNELS * kernel_w * num_inputs)
                    .map(|i| Self::pseudo_random_weight(i, 13, 3, conv_scale))
                    .collect();
                self.conv1d_bias = vec![0.0; CONV1D_OUT_CHANNELS];

                let scale1 = (2.0 / (conv_out_len + num_hidden) as f32).sqrt();
                let scale2 = (2.0 / (num_hidden + num_classes) as f32).sqrt();

                self.weights_fc1 = (0..num_hidden * conv_out_len)
                    .map(|i| Self::pseudo_random_weight(i, 17, 5, scale1))
                    .collect();
                self.bias_fc1 = vec![0.0; num_hidden];

                self.weights_fc2 = (0..num_classes * num_hidden)
                    .map(|i| Self::pseudo_random_weight(i, 31, 11, scale2))
                    .collect();
                self.bias_fc2 = vec![0.0; num_classes];

                self.svdf_weights_feature.clear();
                self.svdf_weights_time.clear();
                self.svdf_bias.clear();
            }
            ModelArchitecture::RecurrentSVDF => {
                let units = self.model_config.hidden_units;
                let feature_dim = units * SVDF_RANK;

                let reservoir_scale = (1.0 / num_inputs as f32).sqrt();
                self.svdf_weights_feature = (0..feature_dim * num_inputs)
                    .map(|i| Self::pseudo_random_weight(i, 19, 7, reservoir_scale))
                    .collect();
                self.svdf_weights_time = (0..feature_dim * SVDF_MEMORY_SIZE)
                    .map(|i| Self::pseudo_random_weight(i, 23, 9, reservoir_scale))
                    .collect();
                self.svdf_bias = vec![0.0; units];

                let scale2 = (2.0 / (units + num_classes) as f32).sqrt();
                self.weights_fc2 = (0..num_classes * units)
                    .map(|i| Self::pseudo_random_weight(i, 31, 11, scale2))
                    .collect();
                self.bias_fc2 = vec![0.0; num_classes];

                self.weights_fc1.clear();
                self.bias_fc1.clear();
                self.conv1d_weights.clear();
                self.conv1d_bias.clear();
            }
        }

        self.confusion_matrix = vec![vec![0; num_classes]; num_classes];
    }

    /// Run single epoch of training (SGD + Backpropagation + QAT simulation)
    pub fn step_training_epoch(&mut self) {
        if self.samples.is_empty() {
            return;
        }

        match self.model_config.arch {
            ModelArchitecture::DenseMLP => self.step_training_epoch_dense(),
            ModelArchitecture::TinyConv1D => self.step_training_epoch_conv1d(),
            ModelArchitecture::RecurrentSVDF => self.step_training_epoch_svdf(),
        }
    }

    /// Records loss/accuracy/confusion-matrix bookkeeping shared by all architectures
    fn finish_epoch(
        &mut self,
        total_loss: f32,
        correct_predictions: usize,
        conf_matrix: Vec<Vec<usize>>,
    ) {
        self.current_epoch += 1;
        let avg_loss = total_loss / self.samples.len() as f32;
        let accuracy = (correct_predictions as f32 / self.samples.len() as f32) * 100.0;

        self.train_loss_history.push(avg_loss);
        self.val_acc_history.push(accuracy);
        self.confusion_matrix = conf_matrix;

        if self.current_epoch >= self.model_config.epochs {
            self.is_training = false;
        }
    }

    /// DenseMLP: Dense(hidden, ReLU) -> Dense(classes) -> Softmax, over the pooled feature vector.
    /// DenseMLP forward pass: Dense(hidden, ReLU) -> Dense(classes). Shared by training,
    /// inference, and activation-range calibration so all three stay numerically identical.
    fn forward_dense(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let num_inputs = self.dsp.num_mel_bins;
        let num_hidden = self.model_config.hidden_units;
        let num_classes = self.classes.len();

        let mut hidden = vec![0.0f32; num_hidden];
        for h in 0..num_hidden {
            let mut sum = self.bias_fc1[h];
            for i in 0..num_inputs {
                sum += x[i] * self.weights_fc1[h * num_inputs + i];
            }
            // Simulated QAT clamp / ReLU
            hidden[h] = sum.max(0.0).min(1.0);
        }

        let mut logits = vec![0.0f32; num_classes];
        for c in 0..num_classes {
            let mut sum = self.bias_fc2[c];
            for h in 0..num_hidden {
                sum += hidden[h] * self.weights_fc2[c * num_hidden + h];
            }
            logits[c] = sum;
        }

        (hidden, logits)
    }

    fn step_training_epoch_dense(&mut self) {
        let num_inputs = self.dsp.num_mel_bins;
        let num_hidden = self.model_config.hidden_units;
        let num_classes = self.classes.len();
        let lr = self.model_config.learning_rate;

        let mut total_loss = 0.0;
        let mut correct_predictions = 0;
        let mut conf_matrix = vec![vec![0; num_classes]; num_classes];

        for sample in &self.samples {
            if sample.frames.is_empty() {
                continue;
            }
            let x = Self::mean_pool_frames(&sample.frames, num_inputs);
            let (hidden, logits) = self.forward_dense(&x);

            let (probs, pred_c, sample_loss) =
                Self::softmax_cross_entropy(&logits, sample.class_idx);
            total_loss += sample_loss;
            if pred_c == sample.class_idx {
                correct_predictions += 1;
            }
            conf_matrix[sample.class_idx][pred_c] += 1;

            // Backward Pass & Weight Updates (SGD)
            let mut d_logits = probs.clone();
            d_logits[sample.class_idx] -= 1.0;

            let mut d_hidden = vec![0.0f32; num_hidden];
            for c in 0..num_classes {
                let d_out = d_logits[c];
                self.bias_fc2[c] -= lr * d_out;
                for h in 0..num_hidden {
                    d_hidden[h] += d_out * self.weights_fc2[c * num_hidden + h];
                    self.weights_fc2[c * num_hidden + h] -= lr * d_out * hidden[h];
                }
            }

            for h in 0..num_hidden {
                if hidden[h] > 0.0 && hidden[h] < 1.0 {
                    let d_h = d_hidden[h];
                    self.bias_fc1[h] -= lr * d_h;
                    for i in 0..num_inputs {
                        self.weights_fc1[h * num_inputs + i] -= lr * d_h * x[i];
                    }
                }
            }
        }

        self.finish_epoch(total_loss, correct_predictions, conf_matrix);
    }

    /// TinyConv1D forward pass: temporal Conv1D + ReLU (kernel slides over frames/time; channels
    /// = Mel bins) -> flatten -> Dense(hidden, ReLU) -> Dense(classes). Shared by training,
    /// inference, and activation-range calibration so all three stay numerically identical.
    fn forward_conv1d(&self, frames: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let num_inputs = self.dsp.num_mel_bins;
        let num_hidden = self.model_config.hidden_units;
        let num_classes = self.classes.len();
        let (kernel_w, out_width) = Self::conv1d_shape_for_config(&self.dsp);
        let out_channels = CONV1D_OUT_CHANNELS;
        let conv_out_len = out_channels * out_width;

        let mut conv_act = vec![0.0f32; conv_out_len];
        for oc in 0..out_channels {
            for ow in 0..out_width {
                let mut sum = self.conv1d_bias[oc];
                for k in 0..kernel_w {
                    let frame = &frames[ow + k];
                    for ic in 0..num_inputs {
                        sum +=
                            frame[ic] * self.conv1d_weights[(oc * kernel_w + k) * num_inputs + ic];
                    }
                }
                conv_act[oc * out_width + ow] = sum.max(0.0).min(1.0);
            }
        }

        let mut hidden = vec![0.0f32; num_hidden];
        for h in 0..num_hidden {
            let mut sum = self.bias_fc1[h];
            for i in 0..conv_out_len {
                sum += conv_act[i] * self.weights_fc1[h * conv_out_len + i];
            }
            hidden[h] = sum.max(0.0).min(1.0);
        }

        let mut logits = vec![0.0f32; num_classes];
        for c in 0..num_classes {
            let mut sum = self.bias_fc2[c];
            for h in 0..num_hidden {
                sum += hidden[h] * self.weights_fc2[c * num_hidden + h];
            }
            logits[c] = sum;
        }

        (conv_act, hidden, logits)
    }

    /// TinyConv1D: real temporal Conv1D (kernel slides over frames/time; channels = Mel bins,
    /// ReLU) -> flatten -> Dense(hidden, ReLU) -> Dense(classes) -> Softmax.
    fn step_training_epoch_conv1d(&mut self) {
        let num_inputs = self.dsp.num_mel_bins;
        let num_hidden = self.model_config.hidden_units;
        let num_classes = self.classes.len();
        let lr = self.model_config.learning_rate;

        let (kernel_w, out_width) = Self::conv1d_shape_for_config(&self.dsp);
        let out_channels = CONV1D_OUT_CHANNELS;
        let conv_out_len = out_channels * out_width;
        let expected_frames = Self::num_frames_for_config(&self.dsp);

        let mut total_loss = 0.0;
        let mut correct_predictions = 0;
        let mut conf_matrix = vec![vec![0; num_classes]; num_classes];

        for sample in &self.samples {
            let frames = &sample.frames;
            if frames.len() != expected_frames {
                continue;
            }

            let (conv_act, hidden, logits) = self.forward_conv1d(frames);

            let (probs, pred_c, sample_loss) =
                Self::softmax_cross_entropy(&logits, sample.class_idx);
            total_loss += sample_loss;
            if pred_c == sample.class_idx {
                correct_predictions += 1;
            }
            conf_matrix[sample.class_idx][pred_c] += 1;

            // Backward Pass & Weight Updates (SGD)
            let mut d_logits = probs.clone();
            d_logits[sample.class_idx] -= 1.0;

            let mut d_hidden = vec![0.0f32; num_hidden];
            for c in 0..num_classes {
                let d_out = d_logits[c];
                self.bias_fc2[c] -= lr * d_out;
                for h in 0..num_hidden {
                    d_hidden[h] += d_out * self.weights_fc2[c * num_hidden + h];
                    self.weights_fc2[c * num_hidden + h] -= lr * d_out * hidden[h];
                }
            }

            let mut d_conv_act = vec![0.0f32; conv_out_len];
            for h in 0..num_hidden {
                if hidden[h] > 0.0 && hidden[h] < 1.0 {
                    let d_h = d_hidden[h];
                    self.bias_fc1[h] -= lr * d_h;
                    for i in 0..conv_out_len {
                        d_conv_act[i] += d_h * self.weights_fc1[h * conv_out_len + i];
                        self.weights_fc1[h * conv_out_len + i] -= lr * d_h * conv_act[i];
                    }
                }
            }

            // 5. Backward through Conv1D (standard 1D conv weight/bias gradients)
            for oc in 0..out_channels {
                for ow in 0..out_width {
                    let act = conv_act[oc * out_width + ow];
                    if act > 0.0 && act < 1.0 {
                        let d_act = d_conv_act[oc * out_width + ow];
                        self.conv1d_bias[oc] -= lr * d_act;
                        for k in 0..kernel_w {
                            let frame = &frames[ow + k];
                            for ic in 0..num_inputs {
                                self.conv1d_weights[(oc * kernel_w + k) * num_inputs + ic] -=
                                    lr * d_act * frame[ic];
                            }
                        }
                    }
                }
            }
        }

        self.finish_epoch(total_loss, correct_predictions, conf_matrix);
    }

    /// RecurrentSVDF forward pass: fixed lookback-window delay-line filter -> Dense(classes).
    /// Shared by training, inference, and activation-range calibration so all three stay
    /// numerically identical. Returns `(raw_feature, lookback_start, svdf_out, logits)` --
    /// `raw_feature`/`lookback_start` are exposed so training's backward pass can reuse them
    /// without recomputing the forward projection.
    fn forward_svdf(&self, frames: &[Vec<f32>]) -> (Vec<Vec<f32>>, usize, Vec<f32>, Vec<f32>) {
        let num_inputs = self.dsp.num_mel_bins;
        let units = self.model_config.hidden_units;
        let num_classes = self.classes.len();
        let rank = SVDF_RANK;
        let memory_size = SVDF_MEMORY_SIZE;
        let feature_dim = units * rank;

        let num_frames = frames.len();
        let lookback_start = num_frames.saturating_sub(memory_size);
        let lookback_len = num_frames - lookback_start;

        // Per-feature projection, only for frames inside the lookback window.
        let mut raw_feature = vec![vec![0.0f32; feature_dim]; lookback_len];
        for (li, t) in (lookback_start..num_frames).enumerate() {
            for f in 0..feature_dim {
                let mut acc = 0.0f32;
                for i in 0..num_inputs {
                    acc += frames[t][i] * self.svdf_weights_feature[f * num_inputs + i];
                }
                raw_feature[li][f] = acc;
            }
        }

        let mut svdf_out = vec![0.0f32; units];
        for u in 0..units {
            let mut acc = self.svdf_bias[u];
            for r in 0..rank {
                let f = u * rank + r;
                for m in 0..memory_size {
                    // Tap `m` corresponds to time `t`; taps before frame 0 are the delay
                    // line's zero-initial state (cold start, matching the real MCU kernel).
                    let t = num_frames as isize - memory_size as isize + m as isize;
                    if t >= lookback_start as isize {
                        let li = (t - lookback_start as isize) as usize;
                        acc += raw_feature[li][f] * self.svdf_weights_time[f * memory_size + m];
                    }
                }
            }
            svdf_out[u] = acc;
        }

        let mut logits = vec![0.0f32; num_classes];
        for c in 0..num_classes {
            let mut sum = self.bias_fc2[c];
            for u in 0..units {
                sum += svdf_out[u] * self.weights_fc2[c * units + u];
            }
            logits[c] = sum;
        }

        (raw_feature, lookback_start, svdf_out, logits)
    }

    /// RecurrentSVDF: real trained reservoir + delay-line filter -> Dense(classes) -> Softmax.
    ///
    /// The delay line is a fixed lookback *window* (a FIFO shift register), not a nonlinear
    /// recurrent hidden state, so each output tap maps directly to exactly one historical frame's
    /// feature projection. That means gradients flow straight from the output to each contributing
    /// frame without needing an unrolled backprop-through-time recursion -- this is the exact
    /// computation `svdf_s8` performs at runtime, just replayed in float over real frame history
    /// instead of the steady-state single-frame approximation used before a frame axis existed.
    fn step_training_epoch_svdf(&mut self) {
        let num_inputs = self.dsp.num_mel_bins;
        let units = self.model_config.hidden_units;
        let num_classes = self.classes.len();
        let lr = self.model_config.learning_rate;
        let rank = SVDF_RANK;
        let memory_size = SVDF_MEMORY_SIZE;
        let feature_dim = units * rank;
        let expected_frames = Self::num_frames_for_config(&self.dsp);

        let mut total_loss = 0.0;
        let mut correct_predictions = 0;
        let mut conf_matrix = vec![vec![0; num_classes]; num_classes];

        for sample in &self.samples {
            let frames = &sample.frames;
            if frames.len() != expected_frames || frames.is_empty() {
                continue;
            }
            let num_frames = frames.len();

            let (raw_feature, lookback_start, svdf_out, logits) = self.forward_svdf(frames);
            let lookback_len = num_frames - lookback_start;

            let (probs, pred_c, sample_loss) =
                Self::softmax_cross_entropy(&logits, sample.class_idx);
            total_loss += sample_loss;
            if pred_c == sample.class_idx {
                correct_predictions += 1;
            }
            conf_matrix[sample.class_idx][pred_c] += 1;

            // Backward: dense head, then direct (non-recursive) gradients into the reservoir.
            let mut d_logits = probs.clone();
            d_logits[sample.class_idx] -= 1.0;

            let mut d_svdf_out = vec![0.0f32; units];
            for c in 0..num_classes {
                let d_out = d_logits[c];
                self.bias_fc2[c] -= lr * d_out;
                for u in 0..units {
                    d_svdf_out[u] += d_out * self.weights_fc2[c * units + u];
                    self.weights_fc2[c * units + u] -= lr * d_out * svdf_out[u];
                }
            }

            let mut d_raw_feature = vec![vec![0.0f32; feature_dim]; lookback_len];
            for u in 0..units {
                let d_out = d_svdf_out[u];
                self.svdf_bias[u] -= lr * d_out;
                for r in 0..rank {
                    let f = u * rank + r;
                    for m in 0..memory_size {
                        let t = num_frames as isize - memory_size as isize + m as isize;
                        if t >= lookback_start as isize {
                            let li = (t - lookback_start as isize) as usize;
                            let idx = f * memory_size + m;
                            let wt_old = self.svdf_weights_time[idx];
                            d_raw_feature[li][f] += d_out * wt_old;
                            self.svdf_weights_time[idx] -= lr * d_out * raw_feature[li][f];
                        }
                    }
                }
            }

            for (li, t) in (lookback_start..num_frames).enumerate() {
                for f in 0..feature_dim {
                    let d_rf = d_raw_feature[li][f];
                    for i in 0..num_inputs {
                        self.svdf_weights_feature[f * num_inputs + i] -= lr * d_rf * frames[t][i];
                    }
                }
            }
        }

        self.finish_epoch(total_loss, correct_predictions, conf_matrix);
    }

    /// Shared softmax + cross-entropy + argmax used by all per-architecture training loops
    fn softmax_cross_entropy(logits: &[f32], true_class: usize) -> (Vec<f32>, usize, f32) {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|&l| (l - max_logit).exp() / exp_sum)
            .collect();

        let sample_loss = -(probs[true_class].max(1e-7)).ln();

        let mut pred_c = 0;
        let mut max_prob = -1.0;
        for (c, &p) in probs.iter().enumerate() {
            if p > max_prob {
                max_prob = p;
                pred_c = c;
            }
        }

        (probs, pred_c, sample_loss)
    }

    pub fn run_simulated_training(&mut self, epochs: usize) {
        for _ in 0..epochs {
            self.step_training_epoch();
        }
    }

    /// Quantizes `weights` per the currently selected [`QuantizationMode`], returning (s8, s4)
    fn quantize_head_weights(&self, weights: &[f32]) -> (Vec<i8>, Option<Vec<i8>>, f32) {
        let max_w = weights.iter().map(|w| w.abs()).fold(0.1f32, f32::max);
        let quant_p = calculate_symmetric_quant_s8(max_w);
        match self.model_config.quant_mode {
            QuantizationMode::Int4SubByte => {
                let packed = quantize_and_pack_weights_s4(weights, quant_p.scale);
                (Vec::new(), Some(packed), quant_p.scale)
            }
            _ => {
                let s8 = quantize_weights_s8(weights, quant_p.scale);
                (s8, None, quant_p.scale)
            }
        }
    }

    /// Combines a calibrated float activation range with the preceding input/weight scales into
    /// the `QuantParams` a layer's output tensor needs for correct fixed-point requantization
    /// (`input_scale * weight_scale / output_scale`, the standard CMSIS-NN/TFLite convention).
    fn calibrated_output_quant(
        input_scale: f32,
        weight_scale: f32,
        min: f32,
        max: f32,
    ) -> QuantParams {
        let range = calculate_asymmetric_quant_s8(min, max);
        let (multiplier, shift) =
            calculate_output_requant_multiplier(input_scale, weight_scale, range.scale);
        QuantParams {
            multiplier,
            shift,
            zero_point: range.zero_point,
            scale: range.scale,
        }
    }

    /// Combines input/weight scales into the `QuantParams` needed to requantize into a *fixed*
    /// target scale (rather than a calibrated range) -- used for SVDF's raw input tensor, whose
    /// combined multiplier must target the internal state's fixed scale convention, since the
    /// state itself has no externally observed float range to calibrate against.
    fn fixed_target_quant(input_scale: f32, weight_scale: f32, target_scale: f32) -> QuantParams {
        let (multiplier, shift) =
            calculate_output_requant_multiplier(input_scale, weight_scale, target_scale);
        QuantParams {
            multiplier,
            shift,
            zero_point: 0,
            scale: target_scale,
        }
    }

    /// Real activation-range calibration for asymmetric output quantization: scans the full
    /// sample set through the current architecture's forward math (no gradient) and returns the
    /// (min, max) float range of the final pre-softmax logits, and -- for RecurrentSVDF only --
    /// the SVDF layer's raw output. Both are unclamped, unlike the ReLU-clamped hidden/conv
    /// layers, which use the fixed simulated QAT range `[0, 1]` instead and don't need scanning.
    fn calibrate_activation_ranges(&self) -> ((f32, f32), Option<(f32, f32)>) {
        let num_inputs = self.dsp.num_mel_bins;
        let expected_frames = Self::num_frames_for_config(&self.dsp);

        let mut logits_lo = f32::INFINITY;
        let mut logits_hi = f32::NEG_INFINITY;
        let mut svdf_lo = f32::INFINITY;
        let mut svdf_hi = f32::NEG_INFINITY;

        for sample in &self.samples {
            if sample.frames.len() != expected_frames {
                continue;
            }
            let logits = match self.model_config.arch {
                ModelArchitecture::DenseMLP => {
                    let x = Self::mean_pool_frames(&sample.frames, num_inputs);
                    self.forward_dense(&x).1
                }
                ModelArchitecture::TinyConv1D => self.forward_conv1d(&sample.frames).2,
                ModelArchitecture::RecurrentSVDF => {
                    let (_, _, svdf_out, logits) = self.forward_svdf(&sample.frames);
                    for &v in &svdf_out {
                        svdf_lo = svdf_lo.min(v);
                        svdf_hi = svdf_hi.max(v);
                    }
                    logits
                }
            };
            for &l in &logits {
                logits_lo = logits_lo.min(l);
                logits_hi = logits_hi.max(l);
            }
        }

        let logits_range = if logits_lo.is_finite() && logits_hi > logits_lo {
            (logits_lo, logits_hi)
        } else {
            (-1.0, 1.0)
        };
        let svdf_range = if self.model_config.arch == ModelArchitecture::RecurrentSVDF {
            Some(if svdf_lo.is_finite() && svdf_hi > svdf_lo {
                (svdf_lo, svdf_hi)
            } else {
                (-1.0, 1.0)
            })
        } else {
            None
        };

        (logits_range, svdf_range)
    }

    /// Rebuilds the formal ModelGraph, schedules the static memory arena, and emits Rust code
    pub fn rebuild_model_graph_and_codegen(&mut self) {
        let num_inputs = self.dsp.num_mel_bins;
        let num_classes = self.classes.len();

        // Keep the HIL test vector sized for whichever architecture is now selected.
        let expected_len = match self.model_config.arch {
            ModelArchitecture::DenseMLP | ModelArchitecture::RecurrentSVDF => num_inputs,
            ModelArchitecture::TinyConv1D => Self::num_frames_for_config(&self.dsp) * num_inputs,
        };
        if self.test_input_vector.len() != expected_len {
            let arch = self.model_config.arch;
            self.test_input_vector = self
                .samples
                .first()
                .map(|s| Self::test_input_vector_for(arch, num_inputs, s))
                .unwrap_or_else(|| vec![0; expected_len]);
        }

        // Real activation-range calibration (asymmetric quantization): scans the dataset through
        // the current architecture's forward math to find the true float range of unclamped
        // layers (logits, and the SVDF output), rather than the fake QuantParams::default()
        // placeholder used before. ReLU-clamped layers use the fixed [0, 1] QAT range directly.
        let (logits_range, svdf_range) = self.calibrate_activation_ranges();
        let hidden_range = (0.0f32, 1.0f32);

        let mut builder = ModelBuilder::new("GestureNeuralNet");

        // `head_input_scale` tracks the float scale of whatever tensor feeds the FC head next
        // (the raw input for DenseMLP, Conv1D's calibrated output for TinyConv1D, or SVDF's
        // calibrated output for RecurrentSVDF), so the FC head's own requantization chain can
        // combine it with its own weight scale correctly.
        let (head_input_id, head_input_scale) = match self.model_config.arch {
            ModelArchitecture::DenseMLP => {
                let in_id = builder.add_input(
                    "sensor_features",
                    TensorShape::new_1d(num_inputs),
                    DataType::Int8,
                    None,
                );
                (in_id, INPUT_FEATURE_SCALE)
            }
            ModelArchitecture::TinyConv1D => {
                let num_frames = Self::num_frames_for_config(&self.dsp);
                let (kernel_w, _) = Self::conv1d_shape_for_config(&self.dsp);

                let in_id = builder.add_input(
                    "sensor_features",
                    TensorShape::new_4d(1, 1, num_frames, num_inputs),
                    DataType::Int8,
                    None,
                );

                // Conv1D's IR has no s4 payload variant (deliberately, v1 scope) -- always
                // quantize to s8 regardless of the globally selected QuantizationMode, unlike
                // the FC head below which does respect it.
                let max_wc = self
                    .conv1d_weights
                    .iter()
                    .map(|w| w.abs())
                    .fold(0.1f32, f32::max);
                let quant_pc = calculate_symmetric_quant_s8(max_wc);
                let conv_s8 = quantize_weights_s8(&self.conv1d_weights, quant_pc.scale);
                let conv_bias_s32: Vec<i32> = self
                    .conv1d_bias
                    .iter()
                    .map(|&b| (b * 100.0) as i32)
                    .collect();

                let conv_output_quant = Self::calibrated_output_quant(
                    INPUT_FEATURE_SCALE,
                    quant_pc.scale,
                    hidden_range.0,
                    hidden_range.1,
                );
                let conv_scale = conv_output_quant.scale;

                let conv_id = builder.add_conv1d_layer(
                    "conv1",
                    in_id,
                    CONV1D_OUT_CHANNELS,
                    kernel_w,
                    1,
                    0,
                    1,
                    conv_s8,
                    Some(conv_bias_s32),
                    ActivationType::Relu,
                    Some(conv_output_quant),
                );
                (conv_id, conv_scale)
            }
            ModelArchitecture::RecurrentSVDF => {
                let max_wf = self
                    .svdf_weights_feature
                    .iter()
                    .map(|w| w.abs())
                    .fold(0.1f32, f32::max);
                let quant_pf = calculate_symmetric_quant_s8(max_wf);
                let svdf_feat_s8 = quantize_weights_s8(&self.svdf_weights_feature, quant_pf.scale);

                let max_wt = self
                    .svdf_weights_time
                    .iter()
                    .map(|w| w.abs())
                    .fold(0.1f32, f32::max);
                let quant_pt = calculate_symmetric_quant_s8(max_wt);
                let svdf_time_s8 = quantize_weights_s8(&self.svdf_weights_time, quant_pt.scale);

                let svdf_bias_s32: Vec<i32> =
                    self.svdf_bias.iter().map(|&b| (b * 100.0) as i32).collect();

                // The SVDF kernel's internal delay-line state is an implementation-internal i8
                // representation with no externally observed float range, so its scale is fixed
                // by convention (SVDF_STATE_SCALE) rather than calibrated.
                let input_quant =
                    Self::fixed_target_quant(INPUT_FEATURE_SCALE, quant_pf.scale, SVDF_STATE_SCALE);
                let in_id = builder.add_input(
                    "sensor_features",
                    TensorShape::new_1d(num_inputs),
                    DataType::Int8,
                    Some(input_quant),
                );

                let (svdf_lo, svdf_hi) = svdf_range.unwrap_or((-1.0, 1.0));
                let output_quant = Self::calibrated_output_quant(
                    SVDF_STATE_SCALE,
                    quant_pt.scale,
                    svdf_lo,
                    svdf_hi,
                );
                let svdf_scale = output_quant.scale;

                let units = self.model_config.hidden_units;
                let svdf_id = builder.add_svdf_layer(
                    "svdf1",
                    in_id,
                    units,
                    SVDF_RANK,
                    SVDF_MEMORY_SIZE,
                    svdf_feat_s8,
                    svdf_time_s8,
                    Some(svdf_bias_s32),
                    ActivationType::None,
                    Some(output_quant),
                );
                (svdf_id, svdf_scale)
            }
        };

        // DenseMLP and TinyConv1D route through the shared FC1(hidden, ReLU) -> FC2(classes) head.
        // RecurrentSVDF skips FC1 (its "hidden layer" is the SVDF output itself).
        let pre_softmax_id = if self.model_config.arch == ModelArchitecture::RecurrentSVDF {
            let (fc2_s8, fc2_s4, weight2_scale) = self.quantize_head_weights(&self.weights_fc2);
            let bias2_s32: Vec<i32> = self.bias_fc2.iter().map(|&b| (b * 100.0) as i32).collect();
            let logits_quant = Self::calibrated_output_quant(
                head_input_scale,
                weight2_scale,
                logits_range.0,
                logits_range.1,
            );
            builder.add_dense_layer(
                "dense_output",
                head_input_id,
                num_classes,
                fc2_s8,
                fc2_s4,
                Some(bias2_s32),
                ActivationType::None,
                None,
                Some(logits_quant),
            )
        } else {
            let num_hidden = self.model_config.hidden_units;

            let (fc1_s8, fc1_s4, weight1_scale) = self.quantize_head_weights(&self.weights_fc1);
            let bias1_s32: Vec<i32> = self.bias_fc1.iter().map(|&b| (b * 100.0) as i32).collect();
            let fc1_output_quant = Self::calibrated_output_quant(
                head_input_scale,
                weight1_scale,
                hidden_range.0,
                hidden_range.1,
            );
            let fc1_scale = fc1_output_quant.scale;
            let fc1_id = builder.add_dense_layer(
                "dense_layer1",
                head_input_id,
                num_hidden,
                fc1_s8,
                fc1_s4,
                Some(bias1_s32),
                ActivationType::Relu,
                None,
                Some(fc1_output_quant),
            );

            let (fc2_s8, fc2_s4, weight2_scale) = self.quantize_head_weights(&self.weights_fc2);
            let bias2_s32: Vec<i32> = self.bias_fc2.iter().map(|&b| (b * 100.0) as i32).collect();
            let logits_quant = Self::calibrated_output_quant(
                fc1_scale,
                weight2_scale,
                logits_range.0,
                logits_range.1,
            );
            builder.add_dense_layer(
                "dense_output",
                fc1_id,
                num_classes,
                fc2_s8,
                fc2_s4,
                Some(bias2_s32),
                ActivationType::None,
                None,
                Some(logits_quant),
            )
        };

        let softmax_id = builder.add_softmax("softmax_output", pre_softmax_id);
        builder.mark_output(softmax_id);

        let graph = builder.build();
        let plan = ArenaScheduler::schedule(&graph);

        let codegen = RustCodeGenerator::new("GestureNeuralNet");
        let generated_code = codegen.generate(&graph);

        self.compiled_graph = Some(graph);
        self.arena_plan = Some(plan);
        self.generated_rust_code = generated_code;

        self.run_test_inference();
    }

    /// Unflattens a flat i8 test vector into `num_frames` frames of `num_mel_bins` f32 values.
    fn unflatten_test_frames(&self, num_frames: usize) -> Vec<Vec<f32>> {
        let num_inputs = self.dsp.num_mel_bins;
        (0..num_frames)
            .map(|t| {
                (0..num_inputs)
                    .map(|i| {
                        self.test_input_vector
                            .get(t * num_inputs + i)
                            .map(|&v| v as f32 / 127.0)
                            .unwrap_or(0.0)
                    })
                    .collect()
            })
            .collect()
    }

    /// Evaluates current test vector through forward pass
    pub fn run_test_inference(&mut self) {
        let num_inputs = self.dsp.num_mel_bins;

        let logits = match self.model_config.arch {
            ModelArchitecture::DenseMLP => {
                let x: Vec<f32> = (0..num_inputs)
                    .map(|i| {
                        self.test_input_vector
                            .get(i)
                            .map(|&v| v as f32 / 127.0)
                            .unwrap_or(0.0)
                    })
                    .collect();
                self.forward_dense(&x).1
            }
            ModelArchitecture::TinyConv1D => {
                let num_frames = Self::num_frames_for_config(&self.dsp);
                let frames = self.unflatten_test_frames(num_frames);
                self.forward_conv1d(&frames).2
            }
            ModelArchitecture::RecurrentSVDF => {
                // Replay the full sample history through the delay line (same math as training),
                // rather than a cold single-frame start, so the preview matches training fidelity.
                let num_frames = Self::num_frames_for_config(&self.dsp).max(SVDF_MEMORY_SIZE);
                let frames = self.unflatten_test_frames(num_frames);
                self.forward_svdf(&frames).3
            }
        };

        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
        self.test_probabilities = logits
            .iter()
            .map(|&l| (l - max_l).exp() / exp_sum)
            .collect();

        self.test_output_logits = logits
            .iter()
            .map(|&l| ((l * 20.0).round().clamp(-128.0, 127.0)) as i8)
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set_arch(state: &mut StudioState, arch: ModelArchitecture) {
        state.model_config.arch = arch;
        state.reset_training();
        state.run_simulated_training(5);
        state.rebuild_model_graph_and_codegen();
    }

    #[test]
    fn test_dense_mlp_generates_dense_only_code() {
        let mut state = StudioState::default();
        set_arch(&mut state, ModelArchitecture::DenseMLP);
        assert!(state.generated_rust_code.contains("fully_connected_s8"));
        assert!(!state.generated_rust_code.contains("convolve_1_x_n_s8"));
        assert!(!state.generated_rust_code.contains("svdf_s8"));
    }

    #[test]
    fn test_tiny_conv1d_generates_conv_code_and_differs_from_dense() {
        let mut dense_state = StudioState::default();
        set_arch(&mut dense_state, ModelArchitecture::DenseMLP);

        let mut conv_state = StudioState::default();
        set_arch(&mut conv_state, ModelArchitecture::TinyConv1D);

        assert!(conv_state.generated_rust_code.contains("convolve_1_x_n_s8"));
        assert_ne!(
            conv_state.generated_rust_code,
            dense_state.generated_rust_code
        );
        assert!(conv_state.test_probabilities.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_tiny_conv1d_weights_nonempty_under_every_quant_mode() {
        // Regression test: Conv1D's IR has no s4 payload variant, so its weights must always be
        // quantized to s8 regardless of the globally selected QuantizationMode. Routing them
        // through the FC head's quant_mode-aware helper previously left them silently empty
        // under Int4SubByte (the default mode), producing `static CONV1_WEIGHTS_S8: [i8; 0]`.
        for mode in [
            QuantizationMode::Int4SubByte,
            QuantizationMode::Int8FixedPoint,
            QuantizationMode::Int16HighPrecision,
        ] {
            let mut state = StudioState::default();
            state.model_config.quant_mode = mode;
            set_arch(&mut state, ModelArchitecture::TinyConv1D);
            assert!(!state.conv1d_weights.is_empty());
            assert!(
                !state
                    .generated_rust_code
                    .contains("CONV1_WEIGHTS_S8: [i8; 0]"),
                "conv1d weights were empty under {:?}",
                mode
            );
        }
    }

    #[test]
    fn test_recurrent_svdf_generates_svdf_code_with_state_param() {
        let mut state = StudioState::default();
        set_arch(&mut state, ModelArchitecture::RecurrentSVDF);

        assert!(state.generated_rust_code.contains("svdf_s8"));
        assert!(state.generated_rust_code.contains("SVDF_STATE_BYTES"));
        assert!(
            state
                .generated_rust_code
                .contains("svdf_state: &mut [i8; SVDF_STATE_BYTES]")
        );
        assert!(state.test_probabilities.iter().all(|p| p.is_finite()));
        assert!(state.test_output_logits.iter().any(|&l| l != 0));
    }

    #[test]
    fn test_architecture_switch_reshapes_weights_without_panicking() {
        let mut state = StudioState::default();
        for arch in [
            ModelArchitecture::TinyConv1D,
            ModelArchitecture::RecurrentSVDF,
            ModelArchitecture::DenseMLP,
        ] {
            set_arch(&mut state, arch);
        }
    }

    #[test]
    fn test_multi_frame_extraction_yields_deterministic_frame_count() {
        let dsp = DspConfig::default();
        let expected = StudioState::num_frames_for_config(&dsp);
        assert!(expected > 1, "default config should yield multiple frames");

        // Short recording (shorter than capture_samples) must still yield the same frame count.
        let short_raw: Vec<f32> = (0..30).map(|i| (i as f32 * 0.3).sin()).collect();
        let frames_short = StudioState::extract_frame_sequence_with_dsp(&dsp, &short_raw);
        assert_eq!(frames_short.len(), expected);

        // Longer recording (longer than capture_samples) must also yield the same frame count.
        let long_raw: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.05).sin()).collect();
        let frames_long = StudioState::extract_frame_sequence_with_dsp(&dsp, &long_raw);
        assert_eq!(frames_long.len(), expected);

        for frame in frames_short.iter().chain(frames_long.iter()) {
            assert_eq!(frame.len(), dsp.num_mel_bins);
            assert!(frame.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_dense_head_test_vector_is_pooled_frame_mean() {
        let mut state = StudioState::default();
        state.recompute_all_frames();
        let sample = state.samples.first().expect("demo dataset is non-empty");
        assert!(!sample.frames.is_empty());

        let pooled_i8 = StudioState::test_input_vector_for(
            ModelArchitecture::DenseMLP,
            state.dsp.num_mel_bins,
            sample,
        );
        assert_eq!(pooled_i8.len(), state.dsp.num_mel_bins);

        let manual_mean = StudioState::mean_pool_frames(&sample.frames, state.dsp.num_mel_bins);
        let manual_quant = StudioState::quantize_frame(&manual_mean);
        assert_eq!(pooled_i8, manual_quant);
    }

    #[test]
    fn test_calibrated_quant_replaces_fake_default_for_every_architecture() {
        // Before real calibration, every layer's output quant was `QuantParams::default()`
        // (multiplier 1073741824, shift 0, zero_point 0) regardless of actual layer statistics.
        // After calibration, the FC1 (or Conv1D) hidden-layer quant should reflect the real
        // input_scale * weight_scale / output_scale combination, which will essentially never
        // coincide exactly with the old placeholder for a real (non-trivial) weight scale.
        for arch in [
            ModelArchitecture::DenseMLP,
            ModelArchitecture::TinyConv1D,
            ModelArchitecture::RecurrentSVDF,
        ] {
            let mut state = StudioState::default();
            set_arch(&mut state, arch);

            let graph = state.compiled_graph.as_ref().expect("graph compiled");
            let hidden_or_output_layer = graph
                .layers
                .iter()
                .find(|l| {
                    matches!(
                        l.op,
                        OpPayload::FullyConnected { .. }
                            | OpPayload::Conv1D { .. }
                            | OpPayload::Svdf { .. }
                    )
                })
                .expect("at least one quantized layer");
            let out_id = hidden_or_output_layer.outputs[0];
            let out_tensor = graph.tensors.iter().find(|t| t.id == out_id).unwrap();

            assert_ne!(
                (out_tensor.quant.multiplier, out_tensor.quant.shift),
                (1073741824, 0),
                "arch {:?} still has the fake default quant params",
                arch
            );
        }
    }

    #[test]
    fn test_calibrate_activation_ranges_is_finite_and_sane() {
        let state = StudioState::default();
        let (logits_range, svdf_range) = state.calibrate_activation_ranges();
        assert!(logits_range.0.is_finite() && logits_range.1.is_finite());
        assert!(logits_range.1 > logits_range.0);
        assert!(svdf_range.is_none()); // default arch is DenseMLP
    }
}
