//! Edge-case coverage for on-device calibration, ML feature helpers, and host
//! interpreter error paths.

use embedded_nn::anomaly::{
    DistanceMetric, FewShotPrototypeMatcherS8, cosine_similarity_s8, euclidean_distance_s8,
    manhattan_distance_s8,
};
use embedded_nn::calibration::{OutputLayerCalibratorF32, OutputLayerCalibratorS8};
use embedded_nn::ml::{hz_to_mel, mel_filterbank_f32, mel_to_hz, mfcc_f32};
use embedded_nn::safety::{
    ARENA_GUARD_CANARY, crc32_fast, verify_arena_integrity, verify_weights_integrity,
};

#[test]
fn calibrator_f32_constructor_errors_and_cross_entropy() {
    let mut weights = [0.0f32; 6];
    let mut bias = [0.0f32; 3];
    assert!(OutputLayerCalibratorF32::new(0, 2, &mut weights, &mut bias).is_err());
    assert!(OutputLayerCalibratorF32::new(3, 0, &mut weights, &mut bias).is_err());
    assert!(OutputLayerCalibratorF32::new(3, 2, &mut weights, &mut bias).is_ok());
    assert!(OutputLayerCalibratorF32::new(2, 2, &mut weights, &mut bias).is_err());

    let mut weights = [0.1f32, 0.2, -0.3, 0.4, 0.5, -0.6];
    let mut bias = [0.0f32; 3];
    let mut cal = OutputLayerCalibratorF32::new(3, 2, &mut weights, &mut bias).unwrap();
    let mut logits = [0.0f32; 3];
    cal.predict(&[1.0, -1.0], &mut logits);
    assert!(logits.iter().any(|v| *v != 0.0));

    let mut scratch = [0.0f32; 3];
    let _ = cal.train_step_mse(&[1.0, -1.0], &[1.0, 0.0, 0.0], 0.01, &mut scratch);
    let _ = cal.train_step_cross_entropy(&[1.0, -1.0], 0, 0.01, &mut scratch);
}

#[test]
fn calibrator_s8_constructor_errors_and_predict() {
    let mut weights = [0i8; 6];
    let mut bias = [0i32; 3];
    assert!(OutputLayerCalibratorS8::new(0, 2, &mut weights, &mut bias).is_err());
    assert!(OutputLayerCalibratorS8::new(3, 0, &mut weights, &mut bias).is_err());
    assert!(OutputLayerCalibratorS8::new(3, 2, &mut weights, &mut bias).is_ok());

    let mut weights = [1i8, 0, 0, 1, -1, -2];
    let mut bias = [1i32, 0, -1];
    let mut cal = OutputLayerCalibratorS8::new(3, 2, &mut weights, &mut bias).unwrap();
    let mut logits = [0i32; 3];
    assert_eq!(cal.predict(&[1, 1], &mut logits), 0);
    cal.train_step_sgd(&[1, -1], 0, 1, &mut logits);
}

#[test]
fn safety_helpers_and_ml_features() {
    let text = b"123456789";
    assert_eq!(crc32_fast(text), 0xCBF43926);
    let mut weights = [1i8, 2, 3];
    let bytes =
        unsafe { core::slice::from_raw_parts(weights.as_ptr() as *const u8, weights.len()) };
    let crc = crc32_fast(bytes);
    assert!(verify_weights_integrity(&weights, crc).is_ok());
    weights[0] ^= 1;
    let corruption = verify_weights_integrity(&weights, crc)
        .unwrap_err()
        .to_string();
    assert!(corruption.contains("Flash weight corruption"));
    let arena = [0u8; 64];
    assert!(verify_arena_integrity(&arena, 32, ARENA_GUARD_CANARY).is_ok());
    let overflow = verify_arena_integrity(&arena, 128, ARENA_GUARD_CANARY)
        .unwrap_err()
        .to_string();
    assert!(overflow.contains("Arena memory overflow"));
    let canary = verify_arena_integrity(&arena, 32, 0)
        .unwrap_err()
        .to_string();
    assert!(canary.contains("guard canary corrupted"));

    assert!(hz_to_mel(0.0) >= 0.0);
    assert!((mel_to_hz(hz_to_mel(1000.0)) - 1000.0).abs() < 1.0);

    let mut filter = [0.0f32; 8];
    let mags = [1.0f32; 256];
    mel_filterbank_f32(&mags, 8000.0, 0.0, 4000.0, &mut filter);
    let mut mfccs = [0.0f32; 4];
    mfcc_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &mut mfccs);
}

#[test]
fn quantized_distance_and_prototype_matcher_s8() {
    let a = [1i8, 2, 3];
    let b = [1i8, 2, 3];
    let c = [4i8, 5, 6];
    assert_eq!(euclidean_distance_s8(&a, &b), 0);
    assert!(euclidean_distance_s8(&a, &c) > 0);
    assert_eq!(manhattan_distance_s8(&a, &b), 0);
    assert!(manhattan_distance_s8(&a, &c) > 0);
    assert!(cosine_similarity_s8(&a, &b) > 0);
    assert_eq!(cosine_similarity_s8(&[], &[]), 0);
    assert_eq!(cosine_similarity_s8(&[0, 0], &[1, 1]), 0);

    let prototypes: &[i8] = &[1, 2, 9, 9];
    let matcher =
        FewShotPrototypeMatcherS8::new(2, 2, prototypes, DistanceMetric::CosineSimilarity).unwrap();
    assert!(
        FewShotPrototypeMatcherS8::new(0, 2, prototypes, DistanceMetric::EuclideanDistance)
            .is_err()
    );
    assert!(matcher.predict(&[1]).is_err());
    let (class, _score) = matcher.predict(&[1, 2]).unwrap();
    assert_eq!(class, 0);

    for metric in [
        DistanceMetric::EuclideanDistance,
        DistanceMetric::ManhattanDistance,
    ] {
        let m = FewShotPrototypeMatcherS8::new(2, 2, prototypes, metric).unwrap();
        let (c, _score) = m.predict(&[1, 2]).unwrap();
        assert_eq!(c, 0);
    }
}
