//! # 6-DOF IMU Gesture Recognition Example (Compile-Time Embedding)
//!
//! Demonstrates compile-time model embedding using the `#[embedded_nn_model]` macro
//! for zero-allocation `#![no_std]` execution on edge silicon.

use embedded_nn_macros::embedded_nn_model;

#[embedded_nn_model("models/gesture_model.json")]
pub struct GestureClassifier;

/// Gesture classes recognized by the model.
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
            Gesture::Punch => "Punch (Forward Impulse)",
            Gesture::Wave => "Wave (Lateral Oscillation)",
            Gesture::Circle => "Circle (Radial Gyration)",
        }
    }
}

fn main() {
    println!("=== embedded-nn IMU Gesture Recognition Demo (Procedural Macro) ===");
    println!("Compiled Model Constants:");
    println!("  INPUT_DIM:  {} INT8 elements", GestureClassifier::INPUT_DIM);
    println!("  OUTPUT_DIM: {} INT8 elements", GestureClassifier::OUTPUT_DIM);
    println!("  ARENA_SIZE: {} bytes SRAM", GestureClassifier::ARENA_SIZE);

    // 1. Prepare static SRAM activation arena
    let mut arena = [0u8; GestureClassifier::ARENA_SIZE];

    // 2. Synthesize a 16-element IMU gesture window
    let imu_sample = [12i8, -4, 28, 5, -15, 30, -8, 14, -20, 18, 10, -12, 25, -9, 7, -2];

    // 3. Execute zero-allocation neural network forward pass
    let logits = GestureClassifier::predict(&imu_sample, &mut arena).expect("Inference failed");

    // 4. Retrieve top predicted class
    let mut top_idx = 0;
    let mut top_val = logits[0];
    for (i, &val) in logits.iter().enumerate() {
        if val > top_val {
            top_val = val;
            top_idx = i;
        }
    }

    let gesture = Gesture::from_index(top_idx);
    println!("\n--- Inference Output ---");
    println!("Input Features: {:?}", imu_sample);
    println!("Output Logits:  {:?}", logits);
    println!("Predicted Class: {} ({})", top_idx, gesture.as_str());
    println!("Memory Allocation: Exactly 0 bytes heap allocated!");
}
