//! Host/device-identical Mel feature extraction for `dsp_contract.json`.
//!
//! Window → optional high-pass → FFT magnitude → Mel filterbank. Callers own every
//! buffer so this stays allocation-free on a microcontroller.

use crate::mel_filterbank_f32;
use embedded_dsp::complex_math::cmplx_mag_f32;
use embedded_dsp::filter_design::biquad_highpass_coeffs;
use embedded_dsp::filtering::{BiquadCascadeInstanceF32, biquad_cascade_df1_f32};
use embedded_dsp::transform::rfft_f32;
use embedded_dsp::window::{apply_window_f32, hamming_f32, hanning_f32};

/// Largest FFT window this helper will accept.
pub const MAX_WINDOW: usize = 128;
/// Largest capture length this helper will accept.
pub const MAX_CAPTURE: usize = 512;
/// Largest Mel filter count this helper will accept.
pub const MAX_MEL_BINS: usize = 32;
/// Per-frame Mel energy floor used when max-normalizing a frame.
///
/// Quiet / still windows whose peak bin is below this value are divided by the
/// floor instead of by a near-zero max, so they stay near zero instead of
/// being stretched to 1.0. Must match [`embedded_nn_compiler::dsp_contract::DspContract`].
pub const DEFAULT_MEL_ENERGY_FLOOR: f32 = 0.05;

/// Window applied before the FFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowKind {
    /// Hann window.
    Hann,
    /// Hamming window.
    Hamming,
    /// No taper.
    Rectangular,
}

/// DSP parameters that must match [`crate`] consumers of `dsp_contract.json`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureDspConfig {
    /// Power-of-two FFT length.
    pub window_size: usize,
    /// Analysis window.
    pub window_kind: WindowKind,
    /// Mel filter count.
    pub num_mel_bins: usize,
    /// High-pass cutoff in Hz; `<= 0` disables the filter.
    pub high_pass_cutoff_hz: f32,
    /// Sample rate in Hz.
    pub sample_rate_hz: f32,
    /// Frame hop in samples.
    pub frame_hop_size: usize,
    /// Raw waveform is truncated or zero-padded to this length.
    pub capture_samples: usize,
    /// Symmetric s8 scale used after Mel extraction (`1/127` matches Studio).
    pub input_scale: f32,
    /// Minimum divisor when max-normalizing each Mel frame. See [`DEFAULT_MEL_ENERGY_FLOOR`].
    pub mel_energy_floor: f32,
}

/// A DSP configuration the helper cannot execute with its stack limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureDspError {
    /// Window, capture, hop, or Mel count is out of range.
    UnsupportedConfig,
    /// Output slice is shorter than `num_frames * num_mel_bins`.
    OutputTooSmall,
}

/// Per-Channel Energy Normalization (PCAN) AGC configuration.
///
/// Implements automatic gain control and adaptive background noise suppression
/// inspired by TensorFlow Lite Micro (`tflite-micro/signal/src/pcan_argc_fixed.h`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcanConfig {
    /// Smoothing factor for running noise floor estimation: `0.0 < alpha < 1.0` (typically 0.05).
    pub smoothing_alpha: f32,
    /// Gain compression exponent: `gamma` (typically 0.5 for square-root or 0.32).
    pub strength: f32,
    /// Offset added to noise estimate to prevent division by zero (typically 1.0).
    pub offset: f32,
    /// Multiplicative output scale factor.
    pub gain: f32,
}

impl Default for PcanConfig {
    fn default() -> Self {
        Self {
            smoothing_alpha: 0.05,
            strength: 0.5,
            offset: 1.0,
            gain: 1.0,
        }
    }
}

impl FeatureDspConfig {
    /// Number of analysis frames produced for `capture_samples`.
    pub fn num_frames(&self) -> usize {
        let capture = self.capture_samples.max(self.window_size);
        let hop = self.frame_hop_size.max(1);
        (capture - self.window_size) / hop + 1
    }
}

/// Fills `out` with `num_frames * num_mel_bins` Mel energies, row-major per frame.
pub fn extract_mel_sequence(
    config: &FeatureDspConfig,
    raw: &[f32],
    out: &mut [f32],
) -> core::result::Result<usize, FeatureDspError> {
    if config.window_size == 0
        || config.window_size > MAX_WINDOW
        || !config.window_size.is_power_of_two()
        || config.capture_samples == 0
        || config.capture_samples > MAX_CAPTURE
        || config.num_mel_bins == 0
        || config.num_mel_bins > MAX_MEL_BINS
        || config.frame_hop_size == 0
        || !config.mel_energy_floor.is_finite()
        || config.mel_energy_floor <= 0.0
    {
        return Err(FeatureDspError::UnsupportedConfig);
    }

    let n_frames = config.num_frames();
    let need = n_frames * config.num_mel_bins;
    if out.len() < need {
        return Err(FeatureDspError::OutputTooSmall);
    }

    let mut captured = [0.0f32; MAX_CAPTURE];
    let capture = config.capture_samples;
    let copy = raw.len().min(capture);
    captured[..copy].copy_from_slice(&raw[..copy]);

    let mut filtered = [0.0f32; MAX_CAPTURE];
    if config.high_pass_cutoff_hz <= 0.0 {
        filtered[..capture].copy_from_slice(&captured[..capture]);
    } else {
        let coeffs =
            biquad_highpass_coeffs(config.high_pass_cutoff_hz, config.sample_rate_hz, 0.7071);
        let mut hp_state = [0.0f32; 4];
        let mut instance = BiquadCascadeInstanceF32::init(1, &coeffs, &mut hp_state);
        biquad_cascade_df1_f32(
            &mut instance,
            &captured[..capture],
            &mut filtered[..capture],
        );
    }

    let n = config.window_size;
    let hop = config.frame_hop_size.max(1);
    for frame_idx in 0..n_frames {
        let start = frame_idx * hop;
        let mut frame = [0.0f32; MAX_WINDOW];
        let end = (start + n).min(capture);
        let take = end.saturating_sub(start);
        frame[..take].copy_from_slice(&filtered[start..start + take]);

        match config.window_kind {
            WindowKind::Hann => {
                let mut win = [0.0f32; MAX_WINDOW];
                hanning_f32(&mut win[..n]);
                apply_window_f32(&mut frame[..n], &win[..n]);
            }
            WindowKind::Hamming => {
                let mut win = [0.0f32; MAX_WINDOW];
                hamming_f32(&mut win[..n]);
                apply_window_f32(&mut frame[..n], &win[..n]);
            }
            WindowKind::Rectangular => {}
        }

        let mut spectrum = [0.0f32; MAX_WINDOW * 2];
        rfft_f32(&frame[..n], &mut spectrum[..n * 2], n, 0);
        let mut mag = [0.0f32; MAX_WINDOW];
        cmplx_mag_f32(&spectrum[..n * 2], &mut mag[..n]);

        let half = n / 2;
        let min_freq = config.high_pass_cutoff_hz.max(1.0);
        let max_freq = (config.sample_rate_hz / 2.0).max(min_freq + 1.0);
        let mut mel = [0.0f32; MAX_MEL_BINS];
        mel_filterbank_f32(
            &mag[..half],
            config.sample_rate_hz,
            min_freq,
            max_freq,
            &mut mel[..config.num_mel_bins],
        );
        let dst = frame_idx * config.num_mel_bins;
        out[dst..dst + config.num_mel_bins].copy_from_slice(&mel[..config.num_mel_bins]);
        let max_e = out[dst..dst + config.num_mel_bins]
            .iter()
            .copied()
            .fold(config.mel_energy_floor, f32::max);
        for value in &mut out[dst..dst + config.num_mel_bins] {
            *value = (*value / max_e).clamp(0.0, 1.0);
        }
    }
    Ok(n_frames)
}

/// Symmetric s8 quantization using `config.input_scale`.
pub fn quantize_mel_s8(values: &[f32], scale: f32, out: &mut [i8]) {
    let n = values.len().min(out.len());
    let scale = if scale.abs() < 1e-12 {
        1.0 / 127.0
    } else {
        scale
    };
    for i in 0..n {
        let q = libm::roundf(values[i] / scale);
        out[i] = q.clamp(-128.0, 127.0) as i8;
    }
}

/// Fills `out` with `num_frames * num_mel_bins` Mel energies from 16-bit PCM input.
pub fn extract_mel_sequence_i16(
    config: &FeatureDspConfig,
    pcm: &[i16],
    out: &mut [f32],
) -> core::result::Result<usize, FeatureDspError> {
    let mut float_raw = [0.0f32; MAX_CAPTURE];
    let copy = pcm.len().min(config.capture_samples.min(MAX_CAPTURE));
    for i in 0..copy {
        float_raw[i] = pcm[i] as f32 / 32768.0;
    }
    extract_mel_sequence(config, &float_raw[..copy], out)
}

/// Micro audio frontend for embedded keyword spotting and audio classification.
///
/// Bridges microphone PCM streams (`i16` / `f32`) directly to quantized `s8` feature matrices
/// compatible with TensorFlow Lite Micro and LiteRT speech models.
#[derive(Debug, Clone)]
pub struct MicroAudioFrontend {
    /// DSP window and filterbank configuration.
    pub config: FeatureDspConfig,
    /// Optional PCAN (Per-Channel Energy Normalization) automatic gain control.
    pub pcan: Option<PcanConfig>,
    /// Whether to apply log-mel compression (`ln(1 + E)`) instead of linear normalization.
    pub log_mel: bool,
    noise_floor: [f32; MAX_MEL_BINS],
}

impl MicroAudioFrontend {
    /// Creates a new frontend with the given DSP configuration.
    pub fn new(config: FeatureDspConfig) -> Self {
        Self {
            config,
            pcan: None,
            log_mel: false,
            noise_floor: [0.0; MAX_MEL_BINS],
        }
    }

    /// Enables PCAN automatic gain control with the specified configuration.
    pub fn with_pcan(mut self, pcan: PcanConfig) -> Self {
        self.pcan = Some(pcan);
        self
    }

    /// Enables or disables logarithmic Mel energy compression (`ln(1 + E)`).
    pub fn with_log_mel(mut self, log_mel: bool) -> Self {
        self.log_mel = log_mel;
        self
    }

    /// Resets the internal running noise floor estimate.
    pub fn reset(&mut self) {
        self.noise_floor = [0.0; MAX_MEL_BINS];
    }

    /// Processes raw `f32` input waveform and extracts Mel features.
    pub fn process_f32(
        &mut self,
        raw: &[f32],
        out: &mut [f32],
    ) -> core::result::Result<usize, FeatureDspError> {
        let n_frames = extract_mel_sequence(&self.config, raw, out)?;
        self.apply_postprocessing(n_frames, out);
        Ok(n_frames)
    }

    /// Processes raw 16-bit signed PCM samples (`&[i16]`) directly into Mel features.
    pub fn process_i16(
        &mut self,
        pcm: &[i16],
        out: &mut [f32],
    ) -> core::result::Result<usize, FeatureDspError> {
        let n_frames = extract_mel_sequence_i16(&self.config, pcm, out)?;
        self.apply_postprocessing(n_frames, out);
        Ok(n_frames)
    }

    /// Processes raw 16-bit signed PCM samples (`&[i16]`) directly into quantized `s8` features.
    pub fn process_i16_quantized(
        &mut self,
        pcm: &[i16],
        out_s8: &mut [i8],
    ) -> core::result::Result<usize, FeatureDspError> {
        let n_frames = self.config.num_frames();
        let need = n_frames * self.config.num_mel_bins;
        if out_s8.len() < need {
            return Err(FeatureDspError::OutputTooSmall);
        }

        let mut float_buf = [0.0f32; MAX_MEL_BINS * 32];
        if need > float_buf.len() {
            return Err(FeatureDspError::UnsupportedConfig);
        }

        self.process_i16(pcm, &mut float_buf[..need])?;
        quantize_mel_s8(
            &float_buf[..need],
            self.config.input_scale,
            &mut out_s8[..need],
        );
        Ok(n_frames)
    }

    fn apply_postprocessing(&mut self, n_frames: usize, out: &mut [f32]) {
        let bins = self.config.num_mel_bins;
        for frame_idx in 0..n_frames {
            let offset = frame_idx * bins;
            let frame = &mut out[offset..offset + bins];

            // 1. Optional PCAN noise estimation and gain control
            if let Some(pcan) = self.pcan {
                for i in 0..bins {
                    let e = frame[i];
                    self.noise_floor[i] = (1.0 - pcan.smoothing_alpha) * self.noise_floor[i]
                        + pcan.smoothing_alpha * e;
                    let denom = self.noise_floor[i] + pcan.offset;
                    let ratio = (e / denom).max(0.0);
                    let compressed = if pcan.strength == 0.5 {
                        libm::sqrtf(ratio)
                    } else if (pcan.strength - 1.0).abs() < 1e-4 {
                        ratio
                    } else {
                        libm::powf(ratio, pcan.strength)
                    };
                    frame[i] = compressed * pcan.gain;
                }
            }

            // 2. Optional Log-Mel compression
            if self.log_mel {
                for value in frame.iter_mut() {
                    *value = libm::logf(1.0 + value.max(0.0));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract_cfg() -> FeatureDspConfig {
        FeatureDspConfig {
            window_size: 64,
            window_kind: WindowKind::Hann,
            num_mel_bins: 16,
            high_pass_cutoff_hz: 10.0,
            sample_rate_hz: 100.0,
            frame_hop_size: 32,
            capture_samples: 256,
            input_scale: 1.0 / 127.0,
            mel_energy_floor: DEFAULT_MEL_ENERGY_FLOOR,
        }
    }

    #[test]
    fn default_contract_shape_is_stable() {
        let cfg = contract_cfg();
        assert_eq!(cfg.num_frames(), 7);
        let mut out = [0.0f32; 7 * 16];
        let frames = extract_mel_sequence(&cfg, &[0.2; 40], &mut out).unwrap();
        assert_eq!(frames, 7);
        assert!(out.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn test_feature_dsp_error_on_unsupported_config() {
        let invalid_cfg = FeatureDspConfig {
            window_size: 0,
            window_kind: WindowKind::Rectangular,
            num_mel_bins: 16,
            high_pass_cutoff_hz: 0.0,
            sample_rate_hz: 100.0,
            frame_hop_size: 32,
            capture_samples: 256,
            input_scale: 1.0 / 127.0,
            mel_energy_floor: DEFAULT_MEL_ENERGY_FLOOR,
        };
        let mut out = [0.0f32; 128];
        assert_eq!(
            extract_mel_sequence(&invalid_cfg, &[0.1; 64], &mut out),
            Err(FeatureDspError::UnsupportedConfig)
        );
    }

    #[test]
    fn test_feature_dsp_error_on_output_too_small() {
        let cfg = FeatureDspConfig {
            window_size: 64,
            window_kind: WindowKind::Hamming,
            num_mel_bins: 16,
            high_pass_cutoff_hz: 0.0,
            sample_rate_hz: 100.0,
            frame_hop_size: 32,
            capture_samples: 256,
            input_scale: 1.0 / 127.0,
            mel_energy_floor: DEFAULT_MEL_ENERGY_FLOOR,
        };
        let mut out = [0.0f32; 10]; // Output buffer too small (requires 7 * 16 = 112)
        assert_eq!(
            extract_mel_sequence(&cfg, &[0.5; 256], &mut out),
            Err(FeatureDspError::OutputTooSmall)
        );
    }

    #[test]
    fn test_quantize_mel_s8_rounding_and_clamping() {
        let values = [0.0, 0.5, 1.0, -1.0, 1.5, -2.0];
        let scale = 1.0 / 127.0;
        let mut out = [0i8; 6];
        quantize_mel_s8(&values, scale, &mut out);

        assert_eq!(out[0], 0);
        assert_eq!(out[1], 64); // 0.5 * 127 = 63.5 -> 64
        assert_eq!(out[2], 127); // 1.0 * 127 = 127
        assert_eq!(out[3], -127);
        assert_eq!(out[4], 127); // Clamped at 127
        assert_eq!(out[5], -128); // Clamped at -128
    }

    #[test]
    fn even_non_power_of_two_window_is_rejected() {
        let mut cfg = contract_cfg();
        cfg.window_size = 6;
        let mut out = [0.0f32; 128];
        assert_eq!(
            extract_mel_sequence(&cfg, &[0.1; 64], &mut out),
            Err(FeatureDspError::UnsupportedConfig)
        );
    }

    #[test]
    fn energy_floor_keeps_quiet_frames_near_zero() {
        let cfg = contract_cfg();
        let mut out = [0.0f32; 7 * 16];
        extract_mel_sequence(&cfg, &[1e-4; 40], &mut out).unwrap();
        assert!(out.iter().all(|v| *v < 0.05));
    }

    #[test]
    fn test_micro_audio_frontend_i16_and_pcan() {
        let cfg = contract_cfg();
        let mut frontend = MicroAudioFrontend::new(cfg)
            .with_pcan(PcanConfig::default())
            .with_log_mel(true);

        // Generate simulated 16-bit PCM audio (440Hz sine equivalent)
        let mut pcm = [0i16; 256];
        for (i, sample) in pcm.iter_mut().enumerate() {
            let phase = (i as f32) * 0.1;
            *sample = (libm::sinf(phase) * 16000.0) as i16;
        }

        let mut mel_out = [0.0f32; 7 * 16];
        let frames = frontend.process_i16(&pcm, &mut mel_out).unwrap();
        assert_eq!(frames, 7);
        assert!(mel_out.iter().any(|&v| v > 0.0));
        assert!(mel_out.iter().all(|&v| v.is_finite()));

        // Test quantized output
        let mut quantized_s8 = [0i8; 7 * 16];
        let q_frames = frontend
            .process_i16_quantized(&pcm, &mut quantized_s8)
            .unwrap();
        assert_eq!(q_frames, 7);
        assert!(quantized_s8.iter().any(|&v| v != 0));
    }
}
