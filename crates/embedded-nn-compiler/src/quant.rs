use crate::ir::QuantParams;
use embedded_nn::subbyte::pack_s4_pair;

/// Calculate symmetric s8 quantization parameters from float min/max
pub fn calculate_symmetric_quant_s8(abs_max: f32) -> QuantParams {
    let scale = if abs_max > 1e-7 {
        abs_max / 127.0
    } else {
        1.0 / 127.0
    };
    let (multiplier, shift) = quantize_multiplier(scale);
    QuantParams {
        multiplier,
        shift,
        zero_point: 0,
        scale,
    }
}

/// Convert real scale multiplier (0..1) to fixed-point (multiplier in Q31, shift)
pub fn quantize_multiplier(real_multiplier: f32) -> (i32, i32) {
    if real_multiplier <= 0.0 {
        return (0, 0);
    }
    let mut shift = 0;
    let mut q = real_multiplier;
    while q < 0.5 && shift > -31 {
        q *= 2.0;
        shift -= 1;
    }
    while q >= 1.0 {
        q /= 2.0;
        shift += 1;
    }
    let q_fixed = (q * (1i64 << 31) as f32).round() as i64;
    let multiplier = q_fixed.clamp(0, 0x7FFF_FFFF) as i32;
    (multiplier, shift)
}

/// Quantize f32 weights to signed 8-bit integers [-128..127]
pub fn quantize_weights_s8(weights: &[f32], scale: f32) -> Vec<i8> {
    weights
        .iter()
        .map(|&w| {
            let val = (w / scale).round() as i32;
            val.clamp(-128, 127) as i8
        })
        .collect()
}

/// Quantize f32 weights to signed 4-bit integers [-8..7] and pack into nibble bytes (i8)
pub fn quantize_and_pack_weights_s4(weights: &[f32], scale: f32) -> Vec<i8> {
    let s4_values: Vec<i8> = weights
        .iter()
        .map(|&w| {
            let val = (w / scale).round() as i32;
            val.clamp(-8, 7) as i8
        })
        .collect();

    let mut packed = Vec::with_capacity((s4_values.len() + 1) / 2);
    for chunk in s4_values.chunks(2) {
        let low = chunk[0];
        let high = if chunk.len() > 1 { chunk[1] } else { 0 };
        packed.push(pack_s4_pair(low, high));
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_multiplier_conversion() {
        let (m, s) = quantize_multiplier(0.5);
        assert!(m > 0);
        assert_eq!(s, 0);
    }

    #[test]
    fn test_quantize_and_pack_s4() {
        let weights = [0.2f32, -0.4, 0.6, -0.8];
        let scale = 0.1;
        let packed = quantize_and_pack_weights_s4(&weights, scale);
        assert_eq!(packed.len(), 2);
    }
}
