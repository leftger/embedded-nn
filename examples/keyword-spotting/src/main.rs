//! # Keyword Spotting (KWS) Audio Example
//!
//! Demonstrates an end-to-end TinyML keyword spotting pipeline for microcontrollers:
//! 1. Audio stream buffering (16 kHz audio)
//! 2. Zero-allocation on-device Mel filterbank DSP extraction
//! 3. Static SRAM arena execution with quantized INT8 neural network layers
//! 4. Softmax probability output and keyword detection ("silence", "unknown", "yes", "no")

use embedded_nn::activations::relu_s8;
use embedded_nn::feature_dsp::{FeatureDspConfig, WindowKind, extract_mel_sequence};
use embedded_nn::fully_connected::fully_connected_s8;
use embedded_nn::softmax::softmax_s8;
use embedded_nn::support::quantize_f32_to_s8;
use embedded_nn::types::{Activation, Dims, FcParams, PerTensorQuantParams};

/// Keyword classes recognized by the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keyword {
    Silence,
    Unknown,
    Yes,
    No,
}

impl Keyword {
    pub fn from_index(index: usize) -> Self {
        match index {
            0 => Keyword::Silence,
            1 => Keyword::Unknown,
            2 => Keyword::Yes,
            3 => Keyword::No,
            _ => Keyword::Unknown,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Keyword::Silence => "silence",
            Keyword::Unknown => "unknown",
            Keyword::Yes => "yes",
            Keyword::No => "no",
        }
    }
}

/// DSP configuration matching a 16 kHz sample rate microphone stream.
const DSP_CONFIG: FeatureDspConfig = FeatureDspConfig {
    window_size: 64,                    // 4ms window @ 16 kHz
    window_kind: WindowKind::Hann,      // Hann windowing
    num_mel_bins: 16,                   // 16 Mel frequency channels
    high_pass_cutoff_hz: 80.0,          // Filter out low-frequency microphone rumble
    sample_rate_hz: 16000.0,            // 16 kHz audio
    frame_hop_size: 32,                 // 50% frame overlap
    capture_samples: 256,               // 16ms audio capture window
    input_scale: 1.0 / 127.0,           // Symmetric s8 normalization scale
};

const NUM_FRAMES: usize = 7; // (256 - 64) / 32 + 1 = 7 frames
const NUM_MEL_BINS: usize = 16;
const INPUT_FEATURES: usize = NUM_FRAMES * NUM_MEL_BINS; // 112 features
const HIDDEN_UNITS: usize = 32;
const NUM_CLASSES: usize = 4;

/// Static weights for Layer 1 (Hidden Dense: 112 -> 32 = 3584 weights)
static LAYER1_WEIGHTS: [i8; INPUT_FEATURES * HIDDEN_UNITS] = [
    1; INPUT_FEATURES * HIDDEN_UNITS
];

/// Static weights for Layer 2 (Output Dense: 32 -> 4 = 128 weights)
static LAYER2_WEIGHTS: [i8; HIDDEN_UNITS * NUM_CLASSES] = [
    -4, 1, 6, -3, 2, -5, 4, 1, -3, 5, 2, -4, 6, 1, -2, 5,
    -3, 4, 2, -5, 1, 6, -2, 4, -3, 5, 1, -4, 6, 2, -3, 5,
    1, -4, 5, -2, 6, 2, -3, 4, 1, -5, 6, 3, -2, 4, 1, -5,
    6, 2, -4, 3, 1, -5, 4, 2, 6, -3, 1, 5, -2, 4, 6, -1,
    2, 5, -3, 4, 1, -5, 6, 2, 3, -4, 5, 1, -2, 6, 4, -3,
    5, 1, -4, 6, 2, -3, 5, 1, -2, 4, 6, -5, 3, 1, -4, 2,
    -2, 6, 1, -4, 5, 3, -2, 6, 4, 1, -5, 3, 2, -4, 6, 1,
    5, -2, 4, 1, -3, 6, 2, -5, 4, 1, -2, 6, 3, 5, -4, 1,
];

fn main() {
    println!("=== embedded-nn Keyword Spotting (KWS) Demo ===");
    println!("Architecture: Audio Stream (16kHz) -> Mel DSP -> 112 -> 32 (ReLU) -> 4 (Softmax)");

    // 1. Synthesize a 16ms audio capture burst (256 samples @ 16kHz) containing a ~1kHz formant tone
    let mut audio_capture = [0.0f32; DSP_CONFIG.capture_samples];
    for (i, sample) in audio_capture.iter_mut().enumerate() {
        let t = i as f32 / DSP_CONFIG.sample_rate_hz;
        // 1 kHz acoustic frequency
        *sample = libm::sinf(2.0 * core::f32::consts::PI * 1000.0 * t) * 0.8;
    }

    // 2. Perform zero-allocation on-device Mel feature extraction
    let mut mel_energies = [0.0f32; INPUT_FEATURES];
    let frames_extracted = extract_mel_sequence(&DSP_CONFIG, &audio_capture, &mut mel_energies)
        .expect("Mel feature extraction failed");

    println!(
        "Extracted {} frames of {} Mel filterbank bins (total features: {})",
        frames_extracted, DSP_CONFIG.num_mel_bins, INPUT_FEATURES
    );

    // 3. Quantize Mel features from float to INT8
    let mut input_quantized = [0i8; INPUT_FEATURES];
    for i in 0..INPUT_FEATURES {
        input_quantized[i] = quantize_f32_to_s8(mel_energies[i], DSP_CONFIG.input_scale, 0);
    }

    // 4. Static memory arena allocation (Zero heap allocation!)
    let mut arena_hidden = [0i8; HIDDEN_UNITS];
    let mut arena_logits = [0i8; NUM_CLASSES];
    let mut probabilities = [0i8; NUM_CLASSES];

    // Layer 1: Fully Connected (112 -> 32)
    let fc1_params = FcParams {
        input_offset: 0,
        filter_offset: 0,
        output_offset: 0,
        activation: Activation::int8_unconstrained(),
    };
    let fc1_quant = PerTensorQuantParams {
        multiplier: 1073741824, // Q30 scale ~ 1.0
        shift: 0,
    };
    let in_dims = Dims { n: 1, h: 1, w: 1, c: INPUT_FEATURES as i32 };
    let filter1_dims = Dims { n: INPUT_FEATURES as i32, h: 1, w: 1, c: HIDDEN_UNITS as i32 };
    let hidden_dims = Dims { n: 1, h: 1, w: 1, c: HIDDEN_UNITS as i32 };

    fully_connected_s8(
        &fc1_params,
        &fc1_quant,
        &in_dims,
        &input_quantized,
        &filter1_dims,
        &LAYER1_WEIGHTS,
        None,
        &hidden_dims,
        &mut arena_hidden,
    )
    .expect("FC1 layer failed");

    // Activation: ReLU
    relu_s8(&mut arena_hidden);

    // Layer 2: Fully Connected (32 -> 4)
    let fc2_params = FcParams {
        input_offset: 0,
        filter_offset: 0,
        output_offset: 0,
        activation: Activation::int8_unconstrained(),
    };
    let fc2_quant = PerTensorQuantParams {
        multiplier: 1073741824,
        shift: 0,
    };
    let filter2_dims = Dims { n: HIDDEN_UNITS as i32, h: 1, w: 1, c: NUM_CLASSES as i32 };
    let out_dims = Dims { n: 1, h: 1, w: 1, c: NUM_CLASSES as i32 };

    fully_connected_s8(
        &fc2_params,
        &fc2_quant,
        &hidden_dims,
        &arena_hidden,
        &filter2_dims,
        &LAYER2_WEIGHTS,
        None,
        &out_dims,
        &mut arena_logits,
    )
    .expect("FC2 layer failed");

    // Output Softmax Activation
    softmax_s8(&arena_logits, 1, NUM_CLASSES, 1073741824, 0, -128, &mut probabilities)
        .expect("Softmax failed");

    // 5. Determine top predicted keyword class
    let mut top_idx = 0;
    let mut top_val = probabilities[0];
    for (i, &val) in probabilities.iter().enumerate() {
        if val > top_val {
            top_val = val;
            top_idx = i;
        }
    }

    let detected_keyword = Keyword::from_index(top_idx);
    println!("\n--- Inference Results ---");
    println!("Logits:        {:?}", arena_logits);
    println!("Probabilities: {:?}", probabilities);
    println!("Detected Word: \"{}\" (index: {})", detected_keyword.as_str(), top_idx);
    println!("Memory Footprint: Zero dynamic heap allocations! (Stack arena < 256 bytes)");
}
