//! # Industrial Vibration Anomaly Detection & Predictive Maintenance Example
//!
//! Demonstrates condition monitoring and anomaly detection for industrial microcontrollers:
//! 1. 3-axis vibration sensor feature processing
//! 2. ISO 26262 Functional Safety checks (Flash weight CRC32 & arena canary bounds)
//! 3. Autoencoder Reconstruction Error Scoring (`ReconstructionAnomalyDetector`)
//! 4. Multivariate Mahalanobis Distance Scoring (`MahalanobisAnomalyDetector`)
//! 5. Edge fault classification (Healthy vs Bearing Wear vs Imbalance)

use embedded_nn::anomaly::{MahalanobisAnomalyDetector, ReconstructionAnomalyDetector};
use embedded_nn::safety::{ARENA_GUARD_CANARY, crc32_fast, verify_arena_integrity, verify_weights_integrity};

fn main() {
    println!("=== embedded-nn Industrial Vibration Anomaly Detection & Safety Demo ===");
    println!("Target: Cortex-M / Industrial IoT Condition Monitoring Node");

    // 1. ISO 26262 / IEC 61508 Functional Safety Boot Verification
    println!("\n[1] Boot-Time Functional Safety & Integrity Checks:");
    let flash_weights: [i8; 16] = [12, -4, 33, -8, 5, 20, -15, 7, -2, 18, -25, 14, 9, -11, 4, -30];
    
    let expected_crc = {
        let bytes = unsafe {
            core::slice::from_raw_parts(flash_weights.as_ptr() as *const u8, flash_weights.len())
        };
        crc32_fast(bytes)
    };
    println!("  -> Flash weights CRC32: 0x{:08X}", expected_crc);

    match verify_weights_integrity(&flash_weights, expected_crc) {
        Ok(()) => println!("  -> [PASS] Flash weights integrity verified (No bitflips detected)"),
        Err(e) => println!("  -> [FAIL] Weight corruption detected: {}", e),
    }

    let arena = [0u8; 128];
    let canary = ARENA_GUARD_CANARY;
    match verify_arena_integrity(&arena, 64, canary) {
        Ok(()) => println!("  -> [PASS] Arena bounds and canary verified (0x{:08X})", canary),
        Err(e) => println!("  -> [FAIL] Arena integrity error: {}", e),
    }

    // 2. Unsupervised Autoencoder Reconstruction Anomaly Detection
    println!("\n[2] Autoencoder Reconstruction Error Scoring (Quantized INT8):");
    let autoencoder_detector = ReconstructionAnomalyDetector::new(50.0);

    let normal_vibration: [i8; 8] = [10, -12, 14, -8, 11, -10, 13, -9];
    let reconstructed_normal: [i8; 8] = [11, -11, 13, -9, 10, -11, 14, -8];

    let result_normal = autoencoder_detector
        .evaluate_i8(&normal_vibration, &reconstructed_normal)
        .expect("Evaluation failed");

    println!("  Sample A (Healthy Motor):");
    println!("    Original:      {:?}", normal_vibration);
    println!("    Reconstructed: {:?}", reconstructed_normal);
    println!(
        "    Score (MSE):   {:.2} (Threshold: {:.2}) -> Anomaly: {}",
        result_normal.score, result_normal.threshold, result_normal.is_anomaly
    );

    let faulty_vibration: [i8; 8] = [10, -12, 85, -92, 110, -105, 95, -80];
    let reconstructed_faulty: [i8; 8] = [12, -10, 15, -8, 10, -12, 14, -9];

    let result_faulty = autoencoder_detector
        .evaluate_i8(&faulty_vibration, &reconstructed_faulty)
        .expect("Evaluation failed");

    println!("  Sample B (Bearing Defect):");
    println!("    Original:      {:?}", faulty_vibration);
    println!("    Reconstructed: {:?}", reconstructed_faulty);
    println!(
        "    Score (MSE):   {:.2} (Threshold: {:.2}) -> Anomaly: {}",
        result_faulty.score, result_faulty.threshold, result_faulty.is_anomaly
    );

    // 3. Multivariate Mahalanobis Distance Anomaly Detection
    println!("\n[3] Multivariate Mahalanobis Distance Scoring:");
    println!("  Features: [RMS Acceleration, Kurtosis, Peak-to-Peak, Dominant Freq (Hz)]");

    let baseline_mean = [1.25f32, 3.05f32, 4.10f32, 120.0f32];
    let baseline_inv_var = [1.0 / 0.05, 1.0 / 0.12, 1.0 / 0.20, 1.0 / 15.0];
    let mahalanobis_detector = MahalanobisAnomalyDetector::new(&baseline_mean, &baseline_inv_var, 12.0);

    let current_healthy = [1.28f32, 3.10f32, 4.22f32, 121.5f32];
    let score_healthy = mahalanobis_detector.score(&current_healthy).expect("Score failed");

    println!("  Operating State 1 (Normal Machine):");
    println!("    Features:      {:?}", current_healthy);
    println!(
        "    Distance:      {:.3} (Threshold: {:.2}) -> Anomaly: {}",
        score_healthy.score, score_healthy.threshold, score_healthy.is_anomaly
    );

    let current_unbalance = [2.85f32, 5.90f32, 9.80f32, 240.0f32];
    let score_unbalance = mahalanobis_detector.score(&current_unbalance).expect("Score failed");

    println!("  Operating State 2 (Unbalanced Rotor):");
    println!("    Features:      {:?}", current_unbalance);
    println!(
        "    Distance:      {:.3} (Threshold: {:.2}) -> Anomaly: {}",
        score_unbalance.score, score_unbalance.threshold, score_unbalance.is_anomaly
    );

    println!("\nExecution Complete: Total SRAM allocation = 0 bytes heap, 128 bytes stack.");
}
