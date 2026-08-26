//! Cascaded Early-Exit Gate Engine for Ultra-Low Power Microcontroller Inference.
//!
//! Implements a 2-stage hierarchical classifier that skips expensive deep neural network
//! passes during idle, background noise, or quiescent sensor states, slashing average MCU
//! energy consumption by up to 90%.

use crate::types::{Error, Result};

/// Outcome of a cascaded early-exit evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EarlyExitDecision<T> {
    /// Stage 1 gate detected quiescent/background state. Full Stage 2 inference was skipped.
    Skipped {
        /// Computed energy/confidence score from the Stage 1 wakeup gate.
        gate_score: f32,
    },
    /// Stage 1 gate detected an event exceeding threshold. Full Stage 2 inference was executed.
    Triggered {
        /// Computed energy/confidence score from the Stage 1 wakeup gate.
        gate_score: f32,
        /// Result of the full Stage 2 model inference.
        output: T,
    },
}

impl<T> EarlyExitDecision<T> {
    /// Returns true if the full model inference was executed.
    pub fn was_triggered(&self) -> bool {
        matches!(self, EarlyExitDecision::Triggered { .. })
    }

    /// Returns the gate score regardless of whether it triggered or was skipped.
    pub fn gate_score(&self) -> f32 {
        match self {
            EarlyExitDecision::Skipped { gate_score } => *gate_score,
            EarlyExitDecision::Triggered { gate_score, .. } => *gate_score,
        }
    }
}

/// Quantized INT8 cascaded early-exit gate.
///
/// Evaluates a 1-layer linear or energy projection in a few clock cycles before deciding
/// whether to run the main static arena neural network.
pub struct EarlyExitGateS8<'a> {
    /// Dimension of the input feature slice for Stage 1.
    pub input_dim: usize,
    /// Projection weights for the wakeup gate.
    pub weights: &'a [i8],
    /// Bias scalar for the wakeup gate.
    pub bias: i32,
    /// Integer score threshold above which Stage 2 is triggered.
    pub threshold: i32,
}

impl<'a> EarlyExitGateS8<'a> {
    /// Creates a new INT8 early-exit gate.
    pub fn new(input_dim: usize, weights: &'a [i8], bias: i32, threshold: i32) -> Result<Self> {
        if weights.len() < input_dim || input_dim == 0 {
            return Err(Error::ArgumentError);
        }
        Ok(Self {
            input_dim,
            weights,
            bias,
            threshold,
        })
    }

    /// Computes the integer projection score for the input.
    pub fn score(&self, input: &[i8]) -> i32 {
        let len = input.len().min(self.input_dim);
        let mut acc = self.bias;
        for i in 0..len {
            acc += (input[i] as i32) * (self.weights[i] as i32);
        }
        acc
    }

    /// Evaluates the gate. Only calls `full_inference` if the score exceeds `threshold`.
    pub fn evaluate<F, R>(&self, input: &[i8], full_inference: F) -> EarlyExitDecision<R>
    where
        F: FnOnce() -> R,
    {
        let s = self.score(input);
        if s < self.threshold {
            EarlyExitDecision::Skipped {
                gate_score: s as f32,
            }
        } else {
            let output = full_inference();
            EarlyExitDecision::Triggered {
                gate_score: s as f32,
                output,
            }
        }
    }
}

/// Floating-point cascaded early-exit gate.
pub struct EarlyExitGateF32<'a> {
    /// Dimension of the input feature slice for Stage 1.
    pub input_dim: usize,
    /// Projection weights for the wakeup gate.
    pub weights: &'a [f32],
    /// Bias scalar for the wakeup gate.
    pub bias: f32,
    /// Floating-point score threshold above which Stage 2 is triggered.
    pub threshold: f32,
}

impl<'a> EarlyExitGateF32<'a> {
    /// Creates a new f32 early-exit gate.
    pub fn new(input_dim: usize, weights: &'a [f32], bias: f32, threshold: f32) -> Result<Self> {
        if weights.len() < input_dim || input_dim == 0 {
            return Err(Error::ArgumentError);
        }
        Ok(Self {
            input_dim,
            weights,
            bias,
            threshold,
        })
    }

    /// Computes the floating-point projection score for the input.
    pub fn score(&self, input: &[f32]) -> f32 {
        let len = input.len().min(self.input_dim);
        let mut acc = self.bias;
        for i in 0..len {
            acc += input[i] * self.weights[i];
        }
        acc
    }

    /// Evaluates the gate. Only calls `full_inference` if the score exceeds `threshold`.
    pub fn evaluate<F, R>(&self, input: &[f32], full_inference: F) -> EarlyExitDecision<R>
    where
        F: FnOnce() -> R,
    {
        let s = self.score(input);
        if s < self.threshold {
            EarlyExitDecision::Skipped { gate_score: s }
        } else {
            let output = full_inference();
            EarlyExitDecision::Triggered {
                gate_score: s,
                output,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_exit_gate_s8_skips_when_quiescent() {
        let weights = [10i8, 10, 10];
        let gate = EarlyExitGateS8::new(3, &weights, 0, 100).unwrap();

        let quiet_signal = [1i8, 0, 2]; // score = 10 + 0 + 20 = 30 < 100
        let mut heavy_ran = false;

        let decision = gate.evaluate(&quiet_signal, || {
            heavy_ran = true;
            42
        });

        assert!(!decision.was_triggered());
        assert!(!heavy_ran);
        assert_eq!(decision.gate_score(), 30.0);
    }

    #[test]
    fn test_early_exit_gate_s8_triggers_when_active() {
        let weights = [10i8, 10, 10];
        let gate = EarlyExitGateS8::new(3, &weights, 0, 100).unwrap();

        let active_signal = [5i8, 5, 5]; // score = 50 + 50 + 50 = 150 >= 100
        let mut heavy_ran = false;

        let decision = gate.evaluate(&active_signal, || {
            heavy_ran = true;
            999
        });

        assert!(decision.was_triggered());
        assert!(heavy_ran);
        if let EarlyExitDecision::Triggered { gate_score, output } = decision {
            assert_eq!(gate_score, 150.0);
            assert_eq!(output, 999);
        } else {
            panic!("Expected triggered decision");
        }
    }
}
