//! Sensor Data Augmentation Pipeline for TinyML Training.
//!
//! Provides waveform jitter, amplitude scaling, Gaussian noise injection,
//! and SpecAugment (time/frequency masking) on Mel spectrograms to improve generalization
//! and prevent overfitting on small embedded sensor datasets.

use std::vec::Vec;

/// Configuration for sensor waveform and spectrogram augmentations.
#[derive(Debug, Clone, Copy)]
pub struct AugmentConfig {
    /// Standard deviation of Gaussian noise added to raw sensor values (in g).
    pub noise_std_dev: f32,
    /// Minimum amplitude scaling factor (e.g. 0.85 for 85%).
    pub min_scale: f32,
    /// Maximum amplitude scaling factor (e.g. 1.15 for 115%).
    pub max_scale: f32,
    /// Maximum number of consecutive Mel frequency channels to zero-mask (SpecAugment).
    pub max_freq_mask_channels: usize,
    /// Maximum number of consecutive time frames to zero-mask (SpecAugment).
    pub max_time_mask_frames: usize,
}

impl Default for AugmentConfig {
    fn default() -> Self {
        Self {
            noise_std_dev: 0.02,
            min_scale: 0.85,
            max_scale: 1.15,
            max_freq_mask_channels: 2,
            max_time_mask_frames: 2,
        }
    }
}

/// Apply amplitude scaling to a sensor waveform.
pub fn apply_scaling(waveform: &[f32], scale: f32) -> Vec<f32> {
    waveform.iter().map(|&v| v * scale).collect()
}

/// Apply additive pseudo-random noise to a sensor waveform using a simple LCG generator.
pub fn apply_noise(waveform: &[f32], seed: u64, std_dev: f32) -> Vec<f32> {
    let mut state = seed.max(1);
    waveform
        .iter()
        .map(|&v| {
            // Linear congruential generator for deterministic pseudo-random float [-1.0, 1.0]
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let rand_val = ((state >> 32) as i32 as f32) / (i32::MAX as f32);
            v + rand_val * std_dev
        })
        .collect()
}

/// Apply SpecAugment frequency masking to a sequence of Mel filterbank frames.
///
/// Masks `mask_width` consecutive frequency bins starting at `start_channel`.
pub fn apply_frequency_mask(frames: &mut [Vec<f32>], start_channel: usize, mask_width: usize) {
    for frame in frames.iter_mut() {
        let end = (start_channel + mask_width).min(frame.len());
        if start_channel < end {
            frame[start_channel..end].fill(0.0);
        }
    }
}

/// Apply SpecAugment time masking to a sequence of Mel filterbank frames.
///
/// Zeroes `mask_width` consecutive frames starting at `start_frame`.
pub fn apply_time_mask(frames: &mut [Vec<f32>], start_frame: usize, mask_width: usize) {
    let end = (start_frame + mask_width).min(frames.len());
    if start_frame < end {
        for frame in &mut frames[start_frame..end] {
            frame.fill(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_scaling() {
        let raw = vec![1.0, -2.0, 3.0];
        let scaled = apply_scaling(&raw, 0.5);
        assert_eq!(scaled, vec![0.5, -1.0, 1.5]);
    }

    #[test]
    fn test_apply_noise_is_bounded() {
        let raw = vec![0.0; 100];
        let noisy = apply_noise(&raw, 42, 0.05);
        for &v in &noisy {
            assert!(v.abs() <= 0.051);
        }
    }

    #[test]
    fn test_specaugment_frequency_mask() {
        let mut frames = vec![vec![1.0; 16]; 4];
        apply_frequency_mask(&mut frames, 2, 3);

        for frame in &frames {
            assert_eq!(frame[0], 1.0);
            assert_eq!(frame[1], 1.0);
            assert_eq!(frame[2], 0.0);
            assert_eq!(frame[3], 0.0);
            assert_eq!(frame[4], 0.0);
            assert_eq!(frame[5], 1.0);
        }
    }

    #[test]
    fn test_specaugment_time_mask() {
        let mut frames = vec![vec![1.0; 16]; 5];
        apply_time_mask(&mut frames, 1, 2);

        assert_eq!(frames[0][0], 1.0);
        assert_eq!(frames[1][0], 0.0);
        assert_eq!(frames[2][0], 0.0);
        assert_eq!(frames[3][0], 1.0);
        assert_eq!(frames[4][0], 1.0);
    }

    #[test]
    fn test_apply_noise_zero_std_dev_identity() {
        let raw = vec![0.123, -0.456, 0.789];
        let noisy = apply_noise(&raw, 12345, 0.0);
        assert_eq!(raw, noisy);
    }

    #[test]
    fn test_specaugment_oversized_mask_clamping() {
        let mut frames = vec![vec![1.0; 8]; 3];
        // Request mask width 100 on 8 channels
        apply_frequency_mask(&mut frames, 4, 100);

        for frame in &frames {
            assert_eq!(frame[0], 1.0);
            assert_eq!(frame[3], 1.0);
            assert_eq!(frame[4], 0.0);
            assert_eq!(frame[7], 0.0);
        }
    }
}
