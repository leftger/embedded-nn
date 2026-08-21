use serde::{Deserialize, Serialize};

/// Versioned DSP/feature-extraction contract shipped next to exported models.
///
/// MCU firmware should apply the same window, hop, and Mel parameters before
/// integer `predict`. Use `embedded_nn::feature_dsp` (enabled with the `dsp`
/// feature) as the shared implementation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DspContract {
    pub version: u32,
    pub window_type: String,
    pub window_size: usize,
    pub num_mel_bins: usize,
    pub high_pass_cutoff_hz: f32,
    pub sample_rate_hz: f32,
    pub frame_hop_size: usize,
    pub capture_samples: usize,
    pub input_scale: f32,
    pub input_zero_point: i32,
}

impl DspContract {
    pub const VERSION: u32 = 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_roundtrips_json() {
        let contract = DspContract {
            version: DspContract::VERSION,
            window_type: "hann".into(),
            window_size: 64,
            num_mel_bins: 16,
            high_pass_cutoff_hz: 10.0,
            sample_rate_hz: 100.0,
            frame_hop_size: 32,
            capture_samples: 256,
            input_scale: 1.0 / 127.0,
            input_zero_point: 0,
        };
        let json = serde_json::to_string(&contract).unwrap();
        let parsed: DspContract = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, contract);
    }
}
