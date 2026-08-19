use embedded_nn_codegen::RustCodeGenerator;
use embedded_nn_compiler::arena::{ArenaPlan, ArenaScheduler};
use embedded_nn_compiler::builder::ModelBuilder;
use embedded_nn_compiler::ir::*;
use embedded_nn_compiler::quant::{
    calculate_symmetric_quant_s8, quantize_and_pack_weights_s4, quantize_weights_s8,
};
use std::f32::consts::PI;

#[derive(Debug, Clone)]
pub struct DatasetSample {
    pub id: usize,
    pub label: String,
    pub class_idx: usize,
    pub raw_waveform: Vec<f32>,
    pub features: Vec<f32>,
    pub quantized_features: Vec<i8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunction {
    Hann,
    Hamming,
    Rectangular,
}

#[derive(Debug, Clone)]
pub struct DspConfig {
    pub window_size: usize,
    pub window_type: WindowFunction,
    pub num_mel_bins: usize,
    pub high_pass_cutoff: f32,
}

impl Default for DspConfig {
    fn default() -> Self {
        Self {
            window_size: 64,
            window_type: WindowFunction::Hann,
            num_mel_bins: 16,
            high_pass_cutoff: 10.0,
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

    // Trained Weights & Model Graph
    pub weights_fc1: Vec<f32>, // shape: (hidden_units, num_features)
    pub bias_fc1: Vec<f32>,
    pub weights_fc2: Vec<f32>, // shape: (num_classes, hidden_units)
    pub bias_fc2: Vec<f32>,

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

            compiled_graph: None,
            arena_plan: None,
            generated_rust_code: String::new(),

            test_input_vector: vec![0; 16],
            test_output_logits: vec![0; 4],
            test_probabilities: vec![0.25; 4],
        };

        // Populate with rich starter demo dataset
        state.load_demo_dataset();
        state.recompute_all_features();
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
        let signal_length = 64;

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
                    features: Vec::new(),
                    quantized_features: Vec::new(),
                });
            }
        }
    }

    /// Extract frequency-domain energy bins from raw time-series using DSP settings
    pub fn extract_features_with_dsp(dsp: &DspConfig, raw: &[f32]) -> Vec<f32> {
        let num_bins = dsp.num_mel_bins;
        let n = raw.len().min(dsp.window_size);
        if n == 0 {
            return vec![0.0; num_bins];
        }

        // Apply window function
        let mut windowed = Vec::with_capacity(n);
        for (i, &x) in raw.iter().take(n).enumerate() {
            let w = match dsp.window_type {
                WindowFunction::Hann => {
                    0.5 * (1.0 - (2.0 * PI * i as f32 / (n as f32 - 1.0)).cos())
                }
                WindowFunction::Hamming => {
                    0.54 - 0.46 * (2.0 * PI * i as f32 / (n as f32 - 1.0)).cos()
                }
                WindowFunction::Rectangular => 1.0,
            };
            windowed.push(x * w);
        }

        // Simplified discrete Fourier energy bin accumulator for visualization and training
        let mut energies = vec![0.0f32; num_bins];
        for k in 0..num_bins {
            let freq = (k + 1) as f32 * 0.5;
            let mut re = 0.0;
            let mut im = 0.0;
            for (t, &val) in windowed.iter().enumerate() {
                let angle = 2.0 * PI * freq * (t as f32) / (n as f32);
                re += val * angle.cos();
                im -= val * angle.sin();
            }
            let mag = (re * re + im * im).sqrt() / (n as f32);
            energies[k] = (mag * 4.0).clamp(0.0, 1.0);
        }

        energies
    }

    pub fn extract_features_from_waveform(&self, raw: &[f32]) -> Vec<f32> {
        Self::extract_features_with_dsp(&self.dsp, raw)
    }

    /// Recompute features for all samples when DSP parameters change
    pub fn recompute_all_features(&mut self) {
        let dsp = self.dsp.clone();
        for sample in &mut self.samples {
            let feats = Self::extract_features_with_dsp(&dsp, &sample.raw_waveform);
            let quant_feats: Vec<i8> = feats
                .iter()
                .map(|&f| ((f * 127.0).round().clamp(-128.0, 127.0)) as i8)
                .collect();
            sample.features = feats;
            sample.quantized_features = quant_feats;
        }

        if let Some(first) = self.samples.first() {
            self.test_input_vector = first.quantized_features.clone();
        }
    }

    /// Initialize training weights
    pub fn reset_training(&mut self) {
        self.current_epoch = 0;
        self.is_training = false;
        self.train_loss_history.clear();
        self.val_acc_history.clear();

        let num_inputs = self.dsp.num_mel_bins;
        let num_hidden = self.model_config.hidden_units;
        let num_classes = self.classes.len();

        // Xavier / Glorot initialization
        let scale1 = (2.0 / (num_inputs + num_hidden) as f32).sqrt();
        let scale2 = (2.0 / (num_hidden + num_classes) as f32).sqrt();

        self.weights_fc1 = (0..num_hidden * num_inputs)
            .map(|i| (((i * 17 + 5) % 100) as f32 / 100.0 - 0.5) * 2.0 * scale1)
            .collect();
        self.bias_fc1 = vec![0.0; num_hidden];

        self.weights_fc2 = (0..num_classes * num_hidden)
            .map(|i| (((i * 31 + 11) % 100) as f32 / 100.0 - 0.5) * 2.0 * scale2)
            .collect();
        self.bias_fc2 = vec![0.0; num_classes];

        self.confusion_matrix = vec![vec![0; num_classes]; num_classes];
    }

    /// Run single epoch of training (SGD + Backpropagation + QAT simulation)
    pub fn step_training_epoch(&mut self) {
        if self.samples.is_empty() {
            return;
        }

        let num_inputs = self.dsp.num_mel_bins;
        let num_hidden = self.model_config.hidden_units;
        let num_classes = self.classes.len();
        let lr = self.model_config.learning_rate;

        let mut total_loss = 0.0;
        let mut correct_predictions = 0;

        let mut conf_matrix = vec![vec![0; num_classes]; num_classes];

        for sample in &self.samples {
            let x = &sample.features;
            if x.len() != num_inputs {
                continue;
            }

            // 1. Forward Pass Layer 1: Dense + ReLU
            let mut hidden = vec![0.0f32; num_hidden];
            for h in 0..num_hidden {
                let mut sum = self.bias_fc1[h];
                for i in 0..num_inputs {
                    sum += x[i] * self.weights_fc1[h * num_inputs + i];
                }
                // Simulated QAT clamp / ReLU
                hidden[h] = sum.max(0.0).min(1.0);
            }

            // 2. Forward Pass Layer 2: Dense + Softmax
            let mut logits = vec![0.0f32; num_classes];
            for c in 0..num_classes {
                let mut sum = self.bias_fc2[c];
                for h in 0..num_hidden {
                    sum += hidden[h] * self.weights_fc2[c * num_hidden + h];
                }
                logits[c] = sum;
            }

            // Softmax
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
            let probs: Vec<f32> = logits
                .iter()
                .map(|&l| (l - max_logit).exp() / exp_sum)
                .collect();

            // Cross-Entropy Loss
            let true_c = sample.class_idx;
            let sample_loss = -(probs[true_c].max(1e-7)).ln();
            total_loss += sample_loss;

            // Prediction
            let mut pred_c = 0;
            let mut max_prob = -1.0;
            for (c, &p) in probs.iter().enumerate() {
                if p > max_prob {
                    max_prob = p;
                    pred_c = c;
                }
            }
            if pred_c == true_c {
                correct_predictions += 1;
            }
            conf_matrix[true_c][pred_c] += 1;

            // 3. Backward Pass & Weight Updates (SGD)
            let mut d_logits = probs.clone();
            d_logits[true_c] -= 1.0; // Gradient of cross-entropy with softmax

            // Update FC2
            let mut d_hidden = vec![0.0f32; num_hidden];
            for c in 0..num_classes {
                let d_out = d_logits[c];
                self.bias_fc2[c] -= lr * d_out;
                for h in 0..num_hidden {
                    d_hidden[h] += d_out * self.weights_fc2[c * num_hidden + h];
                    self.weights_fc2[c * num_hidden + h] -= lr * d_out * hidden[h];
                }
            }

            // Update FC1 (ReLU derivative)
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

    pub fn run_simulated_training(&mut self, epochs: usize) {
        for _ in 0..epochs {
            self.step_training_epoch();
        }
    }

    /// Rebuilds the formal ModelGraph, schedules the static memory arena, and emits Rust code
    pub fn rebuild_model_graph_and_codegen(&mut self) {
        let num_inputs = self.dsp.num_mel_bins;
        let num_hidden = self.model_config.hidden_units;
        let num_classes = self.classes.len();

        let mut builder = ModelBuilder::new("GestureNeuralNet");
        let in_id = builder.add_input(
            "sensor_features",
            TensorShape::new_1d(num_inputs),
            DataType::Int8,
        );

        // Quantize weights for FC1
        let max_w1 = self
            .weights_fc1
            .iter()
            .map(|w| w.abs())
            .fold(0.1f32, f32::max);
        let quant_p1 = calculate_symmetric_quant_s8(max_w1);

        let (fc1_s8, fc1_s4) = match self.model_config.quant_mode {
            QuantizationMode::Int4SubByte => {
                let packed = quantize_and_pack_weights_s4(&self.weights_fc1, quant_p1.scale);
                (Vec::new(), Some(packed))
            }
            _ => {
                let s8 = quantize_weights_s8(&self.weights_fc1, quant_p1.scale);
                (s8, None)
            }
        };

        let bias1_s32: Vec<i32> = self.bias_fc1.iter().map(|&b| (b * 100.0) as i32).collect();

        let fc1_id = builder.add_dense_layer(
            "dense_layer1",
            in_id,
            num_hidden,
            fc1_s8,
            fc1_s4,
            Some(bias1_s32),
            ActivationType::Relu,
        );

        // Quantize weights for FC2
        let max_w2 = self
            .weights_fc2
            .iter()
            .map(|w| w.abs())
            .fold(0.1f32, f32::max);
        let quant_p2 = calculate_symmetric_quant_s8(max_w2);
        let (fc2_s8, fc2_s4) = match self.model_config.quant_mode {
            QuantizationMode::Int4SubByte => {
                let packed = quantize_and_pack_weights_s4(&self.weights_fc2, quant_p2.scale);
                (Vec::new(), Some(packed))
            }
            _ => {
                let s8 = quantize_weights_s8(&self.weights_fc2, quant_p2.scale);
                (s8, None)
            }
        };
        let bias2_s32: Vec<i32> = self.bias_fc2.iter().map(|&b| (b * 100.0) as i32).collect();

        let fc2_id = builder.add_dense_layer(
            "dense_output",
            fc1_id,
            num_classes,
            fc2_s8,
            fc2_s4,
            Some(bias2_s32),
            ActivationType::None,
        );

        let softmax_id = builder.add_softmax("softmax_output", fc2_id);
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

    /// Evaluates current test vector through forward pass
    pub fn run_test_inference(&mut self) {
        let num_inputs = self.dsp.num_mel_bins;
        let num_hidden = self.model_config.hidden_units;
        let num_classes = self.classes.len();

        let mut hidden = vec![0.0f32; num_hidden];
        for h in 0..num_hidden {
            let mut sum = self.bias_fc1[h];
            for i in 0..num_inputs.min(self.test_input_vector.len()) {
                sum += (self.test_input_vector[i] as f32 / 127.0)
                    * self.weights_fc1[h * num_inputs + i];
            }
            hidden[h] = sum.max(0.0);
        }

        let mut logits = vec![0.0f32; num_classes];
        for c in 0..num_classes {
            let mut sum = self.bias_fc2[c];
            for h in 0..num_hidden {
                sum += hidden[h] * self.weights_fc2[c * num_hidden + h];
            }
            logits[c] = sum;
        }

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
