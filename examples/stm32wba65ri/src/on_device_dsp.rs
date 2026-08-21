//! On-device DSP matching Studio's `dsp_contract.json` defaults.

use embedded_nn::feature_dsp::{
    FeatureDspConfig, WindowKind, extract_mel_sequence, quantize_mel_s8,
};

/// Default contract used by Studio (`DspConfig::default`).
pub const DEFAULT_DSP: FeatureDspConfig = FeatureDspConfig {
    window_size: 64,
    window_kind: WindowKind::Hann,
    num_mel_bins: 16,
    high_pass_cutoff_hz: 10.0,
    sample_rate_hz: 100.0,
    frame_hop_size: 32,
    capture_samples: 256,
    input_scale: 1.0 / 127.0,
};

/// Extracts the first Mel frame as s8, for models whose `INPUT_DIM` is `num_mel_bins`.
pub fn first_frame_s8(raw: &[f32], out: &mut [i8]) -> usize {
    let n_frames = DEFAULT_DSP.num_frames();
    let mut mel = [0.0f32; 7 * 16];
    let _ = extract_mel_sequence(&DEFAULT_DSP, raw, &mut mel[..n_frames * 16]);
    let n = DEFAULT_DSP.num_mel_bins.min(out.len());
    quantize_mel_s8(&mel[..n], DEFAULT_DSP.input_scale, &mut out[..n]);
    n
}
