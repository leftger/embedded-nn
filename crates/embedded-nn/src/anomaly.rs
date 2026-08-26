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

/// Distance metric for vector embedding comparison and few-shot prototype matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Cosine similarity: higher value = closer match.
    CosineSimilarity,
    /// Squared Euclidean distance: lower value = closer match.
    EuclideanDistance,
    /// Manhattan L1 distance: lower value = closer match.
    ManhattanDistance,
}

/// Computes squared Euclidean distance between two INT8 vectors: `sum((a_i - b_i)^2)`.
pub fn euclidean_distance_s8(a: &[i8], b: &[i8]) -> u32 {
    let len = a.len().min(b.len());
    let mut sum_sq: u32 = 0;
    for i in 0..len {
        let diff = a[i] as i32 - b[i] as i32;
        sum_sq += (diff * diff) as u32;
    }
    sum_sq
}

/// Computes Manhattan L1 distance between two INT8 vectors: `sum(|a_i - b_i|)`.
pub fn manhattan_distance_s8(a: &[i8], b: &[i8]) -> u32 {
    let len = a.len().min(b.len());
    let mut sum: u32 = 0;
    for i in 0..len {
        let diff = (a[i] as i32 - b[i] as i32).abs();
        sum += diff as u32;
    }
    sum
}

/// Fast integer square-root approximation for u64 inputs.
fn integer_sqrt_u64(val: u64) -> u64 {
    if val == 0 {
        return 0;
    }
    let mut x = val;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + val / x) / 2;
    }
    x
}

/// Computes normalized Cosine similarity between two INT8 vectors in Q15 format `[-32768, 32767]`.
pub fn cosine_similarity_s8(a: &[i8], b: &[i8]) -> i32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0;
    }

    let mut dot: i64 = 0;
    let mut norm_a_sq: u64 = 0;
    let mut norm_b_sq: u64 = 0;

    for i in 0..len {
        let ai = a[i] as i64;
        let bi = b[i] as i64;
        dot += ai * bi;
        norm_a_sq += (ai * ai) as u64;
        norm_b_sq += (bi * bi) as u64;
    }

    let norm_prod = integer_sqrt_u64(norm_a_sq * norm_b_sq);
    if norm_prod == 0 {
        return 0;
    }

    let sim = (dot * 32767) / (norm_prod as i64);
    sim.clamp(-32768, 32767) as i32
}

/// Computes squared Euclidean distance between two f32 vectors: `sum((a_i - b_i)^2)`.
pub fn euclidean_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum_sq: f32 = 0.0;
    for i in 0..len {
        let diff = a[i] - b[i];
        sum_sq += diff * diff;
    }
    sum_sq
}

/// Computes Manhattan L1 distance between two f32 vectors: `sum(|a_i - b_i|)`.
pub fn manhattan_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum: f32 = 0.0;
    for i in 0..len {
        sum += (a[i] - b[i]).abs();
    }
    sum
}

#[inline(always)]
fn sqrt_f32_no_std(x: f32) -> f32 {
    #[cfg(feature = "libm")]
    {
        libm::sqrtf(x)
    }
    #[cfg(not(feature = "libm"))]
    {
        if x <= 0.0 {
            return 0.0;
        }
        let mut s = x;
        for _ in 0..10 {
            s = 0.5 * (s + x / s);
        }
        s
    }
}

/// Computes Cosine similarity between two f32 vectors in `[-1.0, 1.0]`.
pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot: f32 = 0.0;
    let mut norm_a_sq: f32 = 0.0;
    let mut norm_b_sq: f32 = 0.0;

    for i in 0..len {
        dot += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }

    let denom = sqrt_f32_no_std(norm_a_sq * norm_b_sq);
    if denom > 0.0 { dot / denom } else { 0.0 }
}

/// Zero-allocation, few-shot prototype matcher for quantized INT8 embeddings.
///
/// Matches neural network backbone feature vectors against enrolled class prototypes in Flash/SRAM.
pub struct FewShotPrototypeMatcherS8<'a> {
    /// Number of enrolled prototype classes.
    pub num_classes: usize,
    /// Dimensionality of the embedding vector.
    pub embedding_dim: usize,
    /// Row-major prototype matrix of shape `[num_classes, embedding_dim]`.
    pub prototypes: &'a [i8],
    /// Distance metric used for comparison.
    pub metric: DistanceMetric,
}

impl<'a> FewShotPrototypeMatcherS8<'a> {
    /// Creates a new few-shot prototype matcher over provided prototype embeddings.
    pub fn new(
        num_classes: usize,
        embedding_dim: usize,
        prototypes: &'a [i8],
        metric: DistanceMetric,
    ) -> Result<Self> {
        if prototypes.len() != num_classes * embedding_dim || num_classes == 0 || embedding_dim == 0
        {
            return Err(Error::ArgumentError);
        }
        Ok(Self {
            num_classes,
            embedding_dim,
            prototypes,
            metric,
        })
    }

    /// Evaluates input embedding against all stored prototypes.
    ///
    /// Returns `(best_class_index, score)`.
    pub fn predict(&self, embedding: &[i8]) -> Result<(usize, i32)> {
        if embedding.len() < self.embedding_dim {
            return Err(Error::ArgumentError);
        }

        let mut best_class = 0;
        let mut best_score = match self.metric {
            DistanceMetric::CosineSimilarity => i32::MIN,
            DistanceMetric::EuclideanDistance | DistanceMetric::ManhattanDistance => i32::MAX,
        };

        for c in 0..self.num_classes {
            let row_offset = c * self.embedding_dim;
            let proto_slice = &self.prototypes[row_offset..row_offset + self.embedding_dim];

            let score = match self.metric {
                DistanceMetric::CosineSimilarity => cosine_similarity_s8(embedding, proto_slice),
                DistanceMetric::EuclideanDistance => {
                    euclidean_distance_s8(embedding, proto_slice) as i32
                }
                DistanceMetric::ManhattanDistance => {
                    manhattan_distance_s8(embedding, proto_slice) as i32
                }
            };

            match self.metric {
                DistanceMetric::CosineSimilarity => {
                    if score > best_score {
                        best_score = score;
                        best_class = c;
                    }
                }
                DistanceMetric::EuclideanDistance | DistanceMetric::ManhattanDistance => {
                    if score < best_score {
                        best_score = score;
                        best_class = c;
                    }
                }
            }
        }

        Ok((best_class, best_score))
    }

    /// Matches embedding and rejects unknown/out-of-distribution inputs exceeding `threshold`.
    pub fn predict_with_threshold(&self, embedding: &[i8], threshold: i32) -> Result<usize> {
        let (best_class, score) = self.predict(embedding)?;
        match self.metric {
            DistanceMetric::CosineSimilarity => {
                if score >= threshold {
                    Ok(best_class)
                } else {
                    Err(Error::Failure)
                }
            }
            DistanceMetric::EuclideanDistance | DistanceMetric::ManhattanDistance => {
                if score <= threshold {
                    Ok(best_class)
                } else {
                    Err(Error::Failure)
                }
            }
        }
    }
}

/// Zero-allocation, few-shot prototype matcher for floating-point embeddings.
pub struct FewShotPrototypeMatcherF32<'a> {
    /// Number of enrolled prototype classes.
    pub num_classes: usize,
    /// Dimensionality of the embedding vector.
    pub embedding_dim: usize,
    /// Row-major prototype matrix of shape `[num_classes, embedding_dim]`.
    pub prototypes: &'a [f32],
    /// Distance metric used for comparison.
    pub metric: DistanceMetric,
}

impl<'a> FewShotPrototypeMatcherF32<'a> {
    /// Creates a new few-shot float prototype matcher.
    pub fn new(
        num_classes: usize,
        embedding_dim: usize,
        prototypes: &'a [f32],
        metric: DistanceMetric,
    ) -> Result<Self> {
        if prototypes.len() != num_classes * embedding_dim || num_classes == 0 || embedding_dim == 0
        {
            return Err(Error::ArgumentError);
        }
        Ok(Self {
            num_classes,
            embedding_dim,
            prototypes,
            metric,
        })
    }

    /// Evaluates input float embedding against all stored prototypes.
    pub fn predict(&self, embedding: &[f32]) -> Result<(usize, f32)> {
        if embedding.len() < self.embedding_dim {
            return Err(Error::ArgumentError);
        }

        let mut best_class = 0;
        let mut best_score = match self.metric {
            DistanceMetric::CosineSimilarity => f32::NEG_INFINITY,
            DistanceMetric::EuclideanDistance | DistanceMetric::ManhattanDistance => f32::INFINITY,
        };

        for c in 0..self.num_classes {
            let row_offset = c * self.embedding_dim;
            let proto_slice = &self.prototypes[row_offset..row_offset + self.embedding_dim];

            let score = match self.metric {
                DistanceMetric::CosineSimilarity => cosine_similarity_f32(embedding, proto_slice),
                DistanceMetric::EuclideanDistance => euclidean_distance_f32(embedding, proto_slice),
                DistanceMetric::ManhattanDistance => manhattan_distance_f32(embedding, proto_slice),
            };

            match self.metric {
                DistanceMetric::CosineSimilarity => {
                    if score > best_score {
                        best_score = score;
                        best_class = c;
                    }
                }
                DistanceMetric::EuclideanDistance | DistanceMetric::ManhattanDistance => {
                    if score < best_score {
                        best_score = score;
                        best_class = c;
                    }
                }
            }
        }

        Ok((best_class, best_score))
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

    #[test]
    fn test_few_shot_matcher_s8_cosine() {
        // 3 classes x 4 embedding dimensions
        let prototypes = [
            100i8, 0, 0, 0, // Class 0: X axis
            0, 100, 0, 0, // Class 1: Y axis
            0, 0, 100, 0, // Class 2: Z axis
        ];

        let matcher =
            FewShotPrototypeMatcherS8::new(3, 4, &prototypes, DistanceMetric::CosineSimilarity)
                .unwrap();

        let sample_y = [5i8, 90, 0, 2];
        let (pred_class, sim) = matcher.predict(&sample_y).unwrap();
        assert_eq!(pred_class, 1);
        assert!(sim > 30000); // High positive cosine similarity in Q15

        let sample_unknown = [0i8, 0, 0, 100]; // W axis (orthogonal)
        assert!(
            matcher
                .predict_with_threshold(&sample_unknown, 20000)
                .is_err()
        );
    }

    #[test]
    fn test_few_shot_matcher_f32_euclidean() {
        let prototypes = [1.0f32, 0.0, 0.0, 1.0];

        let matcher =
            FewShotPrototypeMatcherF32::new(2, 2, &prototypes, DistanceMetric::EuclideanDistance)
                .unwrap();

        let query = [0.9f32, 0.1f32];
        let (pred_class, dist) = matcher.predict(&query).unwrap();
        assert_eq!(pred_class, 0);
        assert!(dist < 0.05);
    }
}
