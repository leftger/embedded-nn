//! Ultra-low power TinyML Anomaly Detection & Predictive Maintenance.
//!
//! Provides integer and fixed-point unsupervised anomaly detection for industrial IoT,
//! motor vibration analysis, and condition monitoring on Cortex-M microcontrollers.

use crate::types::{Error, Result};

/// Anomaly scoring result containing anomaly flag, computed score, and threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnomalyResult {
    /// True if the computed error score exceeds the configured anomaly threshold.
    pub is_anomaly: bool,
    /// The computed error score (MSE or Mahalanobis distance).
    pub score: f32,
    /// The anomaly decision threshold.
    pub threshold: f32,
}

/// Fixed-point Euclidean Reconstruction Error Anomaly Detector (Autoencoder output vs input).
#[derive(Debug, Clone, PartialEq)]
pub struct ReconstructionAnomalyDetector {
    /// Mean squared reconstruction error threshold.
    pub threshold: f32,
}

impl ReconstructionAnomalyDetector {
    /// Creates a new reconstruction anomaly detector with the given MSE threshold.
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Evaluates mean squared reconstruction error between sensor input and autoencoder output.
    pub fn evaluate_i8(&self, original: &[i8], reconstructed: &[i8]) -> Result<AnomalyResult> {
        if original.len() != reconstructed.len() || original.is_empty() {
            return Err(Error::ArgumentError);
        }

        let mut sum_sq_diff: u32 = 0;
        for i in 0..original.len() {
            let diff = original[i] as i32 - reconstructed[i] as i32;
            sum_sq_diff += (diff * diff) as u32;
        }

        let mse = sum_sq_diff as f32 / original.len() as f32;
        Ok(AnomalyResult {
            is_anomaly: mse > self.threshold,
            score: mse,
            threshold: self.threshold,
        })
    }

    /// Evaluates float mean squared reconstruction error.
    pub fn evaluate_f32(&self, original: &[f32], reconstructed: &[f32]) -> Result<AnomalyResult> {
        if original.len() != reconstructed.len() || original.is_empty() {
            return Err(Error::ArgumentError);
        }

        let mut sum_sq_diff: f32 = 0.0;
        for i in 0..original.len() {
            let diff = original[i] - reconstructed[i];
            sum_sq_diff += diff * diff;
        }

        let mse = sum_sq_diff / original.len() as f32;
        Ok(AnomalyResult {
            is_anomaly: mse > self.threshold,
            score: mse,
            threshold: self.threshold,
        })
    }
}

/// Gaussian / Mahalanobis Distance Anomaly Detector for multi-channel sensor baselines.
#[derive(Debug, Clone)]
pub struct MahalanobisAnomalyDetector<'a> {
    /// Mean feature baseline vector.
    pub mean: &'a [f32],
    /// Inverse variance (1 / sigma^2) per feature.
    pub inv_variance: &'a [f32],
    /// Distance decision threshold.
    pub threshold: f32,
}

impl<'a> MahalanobisAnomalyDetector<'a> {
    /// Creates a new Mahalanobis anomaly detector with baseline mean and inverse variance vectors.
    pub fn new(mean: &'a [f32], inv_variance: &'a [f32], threshold: f32) -> Self {
        Self {
            mean,
            inv_variance,
            threshold,
        }
    }

    /// Computes diagonal Mahalanobis distance sum((x_i - mu_i)^2 / var_i).
    pub fn score(&self, sample: &[f32]) -> Result<AnomalyResult> {
        if sample.len() != self.mean.len() || sample.len() != self.inv_variance.len() {
            return Err(Error::ArgumentError);
        }

        let mut dist: f32 = 0.0;
        for i in 0..sample.len() {
            let diff = sample[i] - self.mean[i];
            dist += diff * diff * self.inv_variance[i];
        }

        Ok(AnomalyResult {
            is_anomaly: dist > self.threshold,
            score: dist,
            threshold: self.threshold,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reconstruction_anomaly_detector_i8() {
        let detector = ReconstructionAnomalyDetector::new(25.0); // MSE threshold = 25
        let orig = [10i8, 20, 30, 40];
        let good_recon = [11i8, 19, 31, 39]; // diff = 1, 1, 1, 1 -> MSE = 1.0
        let bad_recon = [30i8, 0, 10, 60]; // large error

        let res_good = detector.evaluate_i8(&orig, &good_recon).unwrap();
        assert!(!res_good.is_anomaly);
        assert_eq!(res_good.score, 1.0);

        let res_bad = detector.evaluate_i8(&orig, &bad_recon).unwrap();
        assert!(res_bad.is_anomaly);
    }

    #[test]
    fn test_mahalanobis_detector() {
        let mean = [0.0f32, 0.0];
        let inv_var = [1.0f32, 1.0]; // unit variance
        let detector = MahalanobisAnomalyDetector::new(&mean, &inv_var, 9.0); // 3-sigma distance squared

        let normal = [1.0f32, 1.0]; // dist = 1 + 1 = 2 < 9
        let anomaly = [3.0f32, 3.0]; // dist = 9 + 9 = 18 > 9

        assert!(!detector.score(&normal).unwrap().is_anomaly);
        assert!(detector.score(&anomaly).unwrap().is_anomaly);
    }
}
