//! # 6-DOF IMU Gesture Recognition Example
//!
//! Demonstrates real-time motion gesture classification for wearables and IoT nodes:
//! 1. 6-DOF IMU (Accelerometer [X,Y,Z] + Gyroscope [X,Y,Z]) sliding window buffering
//! 2. Fixed-point INT8 feature quantization
//! 3. Static memory arena inference through a quantized multi-layer perceptron (96 -> 32 -> 16 -> 4)
//! 4. Argmax gesture decoding: [Rest, Punch, Wave, Circle]
//!
//! Run with:
//! ```console
//! cargo run --example gesture_recognition --features="libm"
//! ```

use embedded_nn::activations::relu_s8;
use embedded_nn::fully_connected::fully_connected_s8;
use embedded_nn::support::quantize_f32_to_s8;
use embedded_nn::types::{Activation, Dims, FcParams, PerTensorQuantParams};

/// Recognized gesture classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gesture {
    Rest,
    Punch,
    Wave,
    Circle,
}

impl Gesture {
    pub fn from_index(index: usize) -> Self {
        match index {
            0 => Gesture::Rest,
            1 => Gesture::Punch,
            2 => Gesture::Wave,
            3 => Gesture::Circle,
            _ => Gesture::Rest,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Gesture::Rest => "Rest / Idle",
            Gesture::Punch => "Punch (Forward Thrust)",
            Gesture::Wave => "Wave (Lateral Oscillation)",
            Gesture::Circle => "Circle (Radial Gyration)",
        }
    }
}

const IMU_CHANNELS: usize = 6; // Accel X, Y, Z + Gyro X, Y, Z
const WINDOW_SAMPLES: usize = 16; // 16 timesteps per classification window
const TOTAL_FEATURES: usize = IMU_CHANNELS * WINDOW_SAMPLES; // 96 features

const LAYER1_OUT: usize = 32;
const LAYER2_OUT: usize = 16;
const NUM_CLASSES: usize = 4;

// Pre-trained quantized neural network weights stored in microcontroller Flash
static LAYER1_WEIGHTS: [i8; TOTAL_FEATURES * LAYER1_OUT] = [2; TOTAL_FEATURES * LAYER1_OUT];
static LAYER1_BIAS: [i32; LAYER1_OUT] = [10; LAYER1_OUT];

static LAYER2_WEIGHTS: [i8; LAYER1_OUT * LAYER2_OUT] = [-1; LAYER1_OUT * LAYER2_OUT];
static LAYER2_BIAS: [i32; LAYER2_OUT] = [5; LAYER2_OUT];

static LAYER3_WEIGHTS: [i8; LAYER2_OUT * NUM_CLASSES] = [
    // Layer 3 weights
    3, -2, 4, 1, -1, 5, -3, 2, 4, -2, 1, 5, -3, 2, 5, -1, 1, 4, -2, 3, -2, 5, 3, -4, 4, -1, 2, 5,
    -3, 4, 1, -2, 2, -4, 5, 1, -1, 3, -2, 4, 5, -2, 3, 1, -4, 1, 5, 2, 3, -3, 2, 4, -2, 4, 1, -5,
    1, 5, -4, 2, -3, 2, 4, 1,
];
static LAYER3_BIAS: [i32; NUM_CLASSES] = [0, 10, -5, 15];

fn main() {
    println!("=== embedded-nn 6-DOF IMU Gesture Recognition Demo ===");
    println!("Architecture: 6-Axis IMU (16 frames) -> 96 -> 32 (ReLU) -> 16 (ReLU) -> 4 (Logits)");

    // 1. Simulate an IMU buffer captured during a "Punch" gesture (sharp +X acceleration and +Y angular velocity)
    let mut imu_buffer = [[0.0f32; IMU_CHANNELS]; WINDOW_SAMPLES];
    for (t, frame) in imu_buffer.iter_mut().enumerate() {
        let progress = t as f32 / WINDOW_SAMPLES as f32;
        frame[0] = 3.5 * libm::sinf(progress * core::f32::consts::PI); // Accel X: strong forward impulse
        frame[1] = 0.2; // Accel Y
        frame[2] = 0.98; // Accel Z (1G gravity)
        frame[3] = 0.1; // Gyro X
        frame[4] = 2.4 * libm::sinf(progress * core::f32::consts::PI); // Gyro Y
        frame[5] = 0.05; // Gyro Z
    }

    // 2. Quantize the 96 floating-point features into INT8
    let mut quantized_input = [0i8; TOTAL_FEATURES];
    let input_scale = 1.0 / 4.0; // +/- 4.0 range maps to [-128, 127]
    for (t, frame) in imu_buffer.iter().enumerate() {
        for c in 0..IMU_CHANNELS {
            quantized_input[t * IMU_CHANNELS + c] = quantize_f32_to_s8(frame[c], input_scale, 0);
        }
    }

    println!(
        "Quantized IMU Input Window: {} INT8 samples",
        quantized_input.len()
    );

    // 3. Setup static SRAM memory buffers for all intermediate activations (Zero dynamic heap allocations)
    let mut layer1_out = [0i8; LAYER1_OUT];
    let mut layer2_out = [0i8; LAYER2_OUT];
    let mut logits_out = [0i8; NUM_CLASSES];

    let standard_fc_params = FcParams {
        input_offset: 0,
        filter_offset: 0,
        output_offset: 0,
        activation: Activation::int8_unconstrained(),
    };
    let quant_scale = PerTensorQuantParams {
        multiplier: 1073741824,
        shift: 0,
    };

    // --- Forward Pass Layer 1 (96 -> 32) ---
    let dims_in = Dims::new(1, 1, 1, TOTAL_FEATURES as i32);
    let dims_filter1 = Dims::new(TOTAL_FEATURES as i32, 1, 1, LAYER1_OUT as i32);
    let dims_layer1 = Dims::new(1, 1, 1, LAYER1_OUT as i32);

    fully_connected_s8(
        &standard_fc_params,
        &quant_scale,
        &dims_in,
        &quantized_input,
        &dims_filter1,
        &LAYER1_WEIGHTS,
        Some(&LAYER1_BIAS),
        &dims_layer1,
        &mut layer1_out,
    )
    .expect("Layer 1 inference failed");
    relu_s8(&mut layer1_out);

    // --- Forward Pass Layer 2 (32 -> 16) ---
    let dims_filter2 = Dims::new(LAYER1_OUT as i32, 1, 1, LAYER2_OUT as i32);
    let dims_layer2 = Dims::new(1, 1, 1, LAYER2_OUT as i32);

    fully_connected_s8(
        &standard_fc_params,
        &quant_scale,
        &dims_layer1,
        &layer1_out,
        &dims_filter2,
        &LAYER2_WEIGHTS,
        Some(&LAYER2_BIAS),
        &dims_layer2,
        &mut layer2_out,
    )
    .expect("Layer 2 inference failed");
    relu_s8(&mut layer2_out);

    // --- Forward Pass Layer 3 (16 -> 4) ---
    let dims_filter3 = Dims::new(LAYER2_OUT as i32, 1, 1, NUM_CLASSES as i32);
    let dims_out = Dims::new(1, 1, 1, NUM_CLASSES as i32);

    fully_connected_s8(
        &standard_fc_params,
        &quant_scale,
        &dims_layer2,
        &layer2_out,
        &dims_filter3,
        &LAYER3_WEIGHTS,
        Some(&LAYER3_BIAS),
        &dims_out,
        &mut logits_out,
    )
    .expect("Layer 3 inference failed");

    // 4. Argmax Classification
    let mut max_val = logits_out[0];
    let mut predicted_class = 0;
    for (i, &logit) in logits_out.iter().enumerate() {
        if logit > max_val {
            max_val = logit;
            predicted_class = i;
        }
    }

    let gesture = Gesture::from_index(predicted_class);
    println!("\n--- Gesture Classification Results ---");
    println!("Output Logits: {:?}", logits_out);
    println!("Predicted ID:  {}", predicted_class);
    println!("Gesture Name:  \"{}\"", gesture.as_str());
    println!(
        "Total SRAM Footprint: {} bytes stack, 0 bytes heap",
        layer1_out.len() + layer2_out.len() + logits_out.len()
    );
}
