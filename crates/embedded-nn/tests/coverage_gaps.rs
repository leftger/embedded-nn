//! Coverage-gap tests for lower-covered inference utilities.

use embedded_nn::anomaly::{
    DistanceMetric, FewShotPrototypeMatcherF32, MahalanobisAnomalyDetector,
    ReconstructionAnomalyDetector, cosine_similarity_f32, euclidean_distance_f32,
    manhattan_distance_f32,
};
use embedded_nn::early_exit::{EarlyExitDecision, EarlyExitGateF32, EarlyExitGateS8};

#[test]
fn reconstruction_and_mahalanobis_anomaly_detection() {
    let detector = ReconstructionAnomalyDetector::new(4.0);

    let normal = detector.evaluate_i8(&[0, 10, 20], &[0, 10, 20]).unwrap();
    assert!(!normal.is_anomaly);

    let anomaly = detector.evaluate_i8(&[0, 10, 20], &[0, 60, 20]).unwrap();
    assert!(anomaly.is_anomaly);

    assert!(detector.evaluate_i8(&[], &[]).is_err());
    assert!(detector.evaluate_i8(&[1], &[1, 2]).is_err());

    let f = detector.evaluate_f32(&[1.0, 2.0], &[1.0, 2.0]).unwrap();
    assert!(!f.is_anomaly);

    let maha = MahalanobisAnomalyDetector::new(&[1.0, 2.0], &[1.0, 1.0], 1.0);
    assert_eq!(maha.score(&[1.0, 2.0]).unwrap().score, 0.0);
    assert!(maha.score(&[10.0, 2.0]).unwrap().is_anomaly);
    assert!(maha.score(&[1.0]).is_err());
}

#[test]
fn distance_metrics_and_prototype_matcher() {
    let a = [1.0, 2.0, 3.0];
    let b = [1.0, 2.0, 3.0];
    let c = [4.0, 5.0, 6.0];

    assert_eq!(euclidean_distance_f32(&a, &b), 0.0);
    assert!(euclidean_distance_f32(&a, &c) > 0.0);
    assert_eq!(manhattan_distance_f32(&a, &b), 0.0);
    assert!(manhattan_distance_f32(&a, &c) > 0.0);
    assert!((cosine_similarity_f32(&a, &b) - 1.0).abs() < 1e-5);

    let prototypes: &[f32] = &[1.0, 2.0, 9.0, 9.0];
    let matcher =
        FewShotPrototypeMatcherF32::new(2, 2, prototypes, DistanceMetric::EuclideanDistance)
            .unwrap();
    let (class, _score) = matcher.predict(&[1.1, 2.1]).unwrap();
    assert_eq!(class, 0);

    assert_eq!(DistanceMetric::CosineSimilarity as u8, 0);
    assert_eq!(DistanceMetric::EuclideanDistance as u8, 1);
    assert_eq!(DistanceMetric::ManhattanDistance as u8, 2);
}

#[test]
fn early_exit_gates_skip_and_trigger() {
    let s8 = EarlyExitGateS8::new(2, &[1, 1], 0, 5).unwrap();
    let skipped = s8.evaluate(&[1, 1], || 99);
    assert!(matches!(skipped, EarlyExitDecision::Skipped { .. }));
    assert!(!skipped.was_triggered());
    assert_eq!(skipped.gate_score(), 2.0);

    let triggered = s8.evaluate(&[3, 3], || 42);
    assert_eq!(
        triggered,
        EarlyExitDecision::Triggered {
            gate_score: 6.0,
            output: 42,
        }
    );
    assert!(triggered.was_triggered());
    assert!(EarlyExitGateS8::new(0, &[], 0, 0).is_err());

    let f32 = EarlyExitGateF32::new(2, &[0.5, 0.5], 0.0, 1.0).unwrap();
    let fskipped = f32.evaluate(&[1.0, 0.5], || "ok");
    assert!(!fskipped.was_triggered());

    let ftriggered = f32.evaluate(&[2.0, 2.0], || "full");
    assert!(ftriggered.was_triggered());
    assert_eq!(ftriggered.gate_score(), 2.0);
    assert!(EarlyExitGateF32::new(0, &[], 0.0, 0.0).is_err());
}
