//! On-device output-layer calibration and fine-tuning engine.
//!
//! Provides zero-allocation, `#![no_std]` algorithms for continuous learning,
//! sensor drift adaptation, and user-specific personalization directly on embedded silicon.

use crate::support::clamp;

/// Floating-point on-device output layer calibrator.
///
/// Updates the classification head (weights and bias) using gradient descent on-device
/// without heap allocations.
pub struct OutputLayerCalibratorF32<'a> {
    /// Number of output classes.
    pub num_classes: usize,
    /// Number of input features.
    pub num_features: usize,
    /// Mutable row-major weight matrix of shape `[num_classes, num_features]`.
    pub weights: &'a mut [f32],
    /// Mutable bias vector of shape `[num_classes]`.
    pub bias: &'a mut [f32],
}

impl<'a> OutputLayerCalibratorF32<'a> {
    /// Creates a new calibrator over provided weight and bias buffers.
    pub fn new(
        num_classes: usize,
        num_features: usize,
        weights: &'a mut [f32],
        bias: &'a mut [f32],
    ) -> Result<Self, &'static str> {
        if num_classes == 0 || num_features == 0 {
            return Err("num_classes and num_features must be greater than zero");
        }
        if weights.len() != num_classes * num_features {
            return Err("Weights buffer length must match num_classes * num_features");
        }
        if bias.len() != num_classes {
            return Err("Bias buffer length must match num_classes");
        }
        Ok(Self {
            num_classes,
            num_features,
            weights,
            bias,
        })
    }

    /// Computes forward inference logits into `logits_out`.
    pub fn predict(&self, features: &[f32], logits_out: &mut [f32]) {
        assert!(features.len() >= self.num_features);
        assert!(logits_out.len() >= self.num_classes);

        for c in 0..self.num_classes {
            let mut acc = self.bias[c];
            let row_offset = c * self.num_features;
            for f in 0..self.num_features {
                acc += self.weights[row_offset + f] * features[f];
            }
            logits_out[c] = acc;
        }
    }

    /// Performs one gradient descent step using Mean-Squared Error (MSE) loss.
    ///
    /// Returns the MSE loss before the parameter update.
    pub fn train_step_mse(
        &mut self,
        features: &[f32],
        targets: &[f32],
        learning_rate: f32,
        logits_scratch: &mut [f32],
    ) -> f32 {
        self.predict(features, logits_scratch);

        let mut loss = 0.0f32;
        for c in 0..self.num_classes {
            let err = logits_scratch[c] - targets[c];
            loss += 0.5 * err * err;

            // Gradient: dL/dW_cf = err * feature_f, dL/dB_c = err
            let row_offset = c * self.num_features;
            let grad_factor = learning_rate * err;
            for f in 0..self.num_features {
                self.weights[row_offset + f] -= grad_factor * features[f];
            }
            self.bias[c] -= grad_factor;
        }

        loss
    }

    /// Performs one gradient descent step using Softmax Cross-Entropy loss.
    ///
    /// Returns the predicted class label before the update.
    pub fn train_step_cross_entropy(
        &mut self,
        features: &[f32],
        target_class: usize,
        learning_rate: f32,
        probs_scratch: &mut [f32],
    ) -> usize {
        assert!(target_class < self.num_classes);
        self.predict(features, probs_scratch);

        // In-place Softmax
        let mut max_val = probs_scratch[0];
        let mut best_class = 0;
        for c in 1..self.num_classes {
            if probs_scratch[c] > max_val {
                max_val = probs_scratch[c];
                best_class = c;
            }
        }

        let mut sum_exp = 0.0f32;
        for c in 0..self.num_classes {
            let diff = probs_scratch[c] - max_val;
            // Pure fast no-std exponential approximation: (1 + diff / 64)^64
            let y = diff / 64.0;
            let mut poly = 1.0 + y * (1.0 + y * (0.5 + y * (1.0 / 6.0)));
            for _ in 0..6 {
                poly *= poly;
            }
            probs_scratch[c] = poly;
            sum_exp += poly;
        }

        let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 1.0 };
        for c in 0..self.num_classes {
            probs_scratch[c] *= inv_sum;
        }

        // Error gradient: e_c = p_c - y_c
        for c in 0..self.num_classes {
            let target_indicator = if c == target_class { 1.0 } else { 0.0 };
            let err = probs_scratch[c] - target_indicator;
            let row_offset = c * self.num_features;
            let grad_factor = learning_rate * err;
            for f in 0..self.num_features {
                self.weights[row_offset + f] -= grad_factor * features[f];
            }
            self.bias[c] -= grad_factor;
        }

        best_class
    }
}

/// Quantized INT8 on-device output layer calibrator.
///
/// Enables continuous integer weight updates and perception fine-tuning without FPU or heap.
pub struct OutputLayerCalibratorS8<'a> {
    /// Number of output classes.
    pub num_classes: usize,
    /// Number of input features.
    pub num_features: usize,
    /// Mutable row-major quantized weight matrix of shape `[num_classes, num_features]`.
    pub weights: &'a mut [i8],
    /// Mutable bias vector of shape `[num_classes]`.
    pub bias: &'a mut [i32],
}

impl<'a> OutputLayerCalibratorS8<'a> {
    /// Creates a new INT8 calibrator over provided weight and bias buffers.
    pub fn new(
        num_classes: usize,
        num_features: usize,
        weights: &'a mut [i8],
        bias: &'a mut [i32],
    ) -> Result<Self, &'static str> {
        if num_classes == 0 || num_features == 0 {
            return Err("num_classes and num_features must be greater than zero");
        }
        if weights.len() != num_classes * num_features {
            return Err("Weights buffer length must match num_classes * num_features");
        }
        if bias.len() != num_classes {
            return Err("Bias buffer length must match num_classes");
        }
        Ok(Self {
            num_classes,
            num_features,
            weights,
            bias,
        })
    }

    /// Computes forward inference integer dot-products into `logits_out`.
    pub fn predict(&self, features: &[i8], logits_out: &mut [i32]) -> usize {
        assert!(features.len() >= self.num_features);
        assert!(logits_out.len() >= self.num_classes);

        let mut max_val = i32::MIN;
        let mut best_class = 0;

        for c in 0..self.num_classes {
            let mut acc = self.bias[c];
            let row_offset = c * self.num_features;
            for f in 0..self.num_features {
                acc += (self.weights[row_offset + f] as i32) * (features[f] as i32);
            }
            logits_out[c] = acc;
            if acc > max_val {
                max_val = acc;
                best_class = c;
            }
        }

        best_class
    }

    /// Performs one integer Perceptron / Margin SGD step.
    ///
    /// If the predicted class differs from `target_class`, reinforces target class weights
    /// and dampens mispredicted class weights with step `lr_step`.
    pub fn train_step_sgd(
        &mut self,
        features: &[i8],
        target_class: usize,
        lr_step: i32,
        logits_scratch: &mut [i32],
    ) -> usize {
        let pred_class = self.predict(features, logits_scratch);

        if pred_class != target_class {
            let target_offset = target_class * self.num_features;
            let pred_offset = pred_class * self.num_features;

            for f in 0..self.num_features {
                let feat = features[f] as i32;
                let sign = if feat > 0 {
                    1
                } else if feat < 0 {
                    -1
                } else {
                    0
                };

                // Reinforce target class
                let cur_target_w = self.weights[target_offset + f] as i32;
                let new_target_w = clamp(cur_target_w + sign * lr_step, -128, 127);
                self.weights[target_offset + f] = new_target_w as i8;

                // Penalize mispredicted class
                let cur_pred_w = self.weights[pred_offset + f] as i32;
                let new_pred_w = clamp(cur_pred_w - sign * lr_step, -128, 127);
                self.weights[pred_offset + f] = new_pred_w as i8;
            }

            self.bias[target_class] += lr_step;
            self.bias[pred_class] -= lr_step;
        }

        pred_class
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibrator_f32_mse_convergence() {
        let mut weights = [0.0f32; 4]; // 2 classes x 2 features
        let mut bias = [0.0f32; 2];
        let mut calibrator = OutputLayerCalibratorF32::new(2, 2, &mut weights, &mut bias).unwrap();

        let sample_a = [1.0f32, 0.0f32];
        let target_a = [1.0f32, 0.0f32];
        let mut scratch = [0.0f32; 2];

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        for step in 0..50 {
            let loss = calibrator.train_step_mse(&sample_a, &target_a, 0.1, &mut scratch);
            if step == 0 {
                initial_loss = loss;
            }
            final_loss = loss;
        }

        assert!(initial_loss > final_loss);
        assert!(final_loss < 0.01);
    }

    #[test]
    fn test_calibrator_s8_sgd_online_learning() {
        let mut weights = [0i8; 6]; // 3 classes x 2 features
        let mut bias = [0i32; 3];
        let mut calibrator = OutputLayerCalibratorS8::new(3, 2, &mut weights, &mut bias).unwrap();

        let sample = [10i8, -10i8];
        let mut scratch = [0i32; 3];

        // Initially class 0 is chosen by tie-break
        let pred0 = calibrator.train_step_sgd(&sample, 2, 2, &mut scratch);
        assert_eq!(pred0, 0);

        // After a few SGD steps, class 2 should be predicted
        for _ in 0..10 {
            calibrator.train_step_sgd(&sample, 2, 2, &mut scratch);
        }

        let pred_final = calibrator.predict(&sample, &mut scratch);
        assert_eq!(pred_final, 2);
    }
}
