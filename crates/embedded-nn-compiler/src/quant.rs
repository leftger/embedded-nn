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

/// Calculate asymmetric int8 quantization parameters (scale + zero-point) from a float
/// activation range `[min, max]`, per the standard TFLite/CMSIS-NN convention. Unlike
/// `calculate_symmetric_quant_s8` (used for weights, which are always kept symmetric here),
/// this allows the zero float value to map to a non-zero int8 code, which is necessary for
/// ranges that aren't centered on zero (e.g. post-ReLU activations, which are always >= 0).
///
/// The returned `multiplier`/`shift` are `quantize_multiplier(scale)` as a standalone fallback;
/// callers that are requantizing a specific layer's accumulator (rather than using this tensor
/// value directly) should instead combine scales via `calculate_output_requant_multiplier`.
pub fn calculate_asymmetric_quant_s8(min: f32, max: f32) -> QuantParams {
    const QMIN: f32 = -128.0;
    const QMAX: f32 = 127.0;

    // The float range must include zero for a zero-point to be representable at all.
    let min = min.min(0.0);
    let max = max.max(0.0);

    let scale = if (max - min) > 1e-7 {
        (max - min) / (QMAX - QMIN)
    } else {
        1.0 / (QMAX - QMIN)
    };

    let initial_zero_point = QMIN - min / scale;
    let zero_point = initial_zero_point.round().clamp(QMIN, QMAX) as i32;

    let (multiplier, shift) = quantize_multiplier(scale);
    QuantParams {
        multiplier,
        shift,
        zero_point,
        scale,
    }
}

/// Combines a layer's input/weight float scales with its output tensor's float scale into the
/// fixed-point `(multiplier, shift)` pair needed to requantize the layer's int32 accumulator
/// (standard CMSIS-NN/TFLite convention: `real_multiplier = input_scale * weight_scale /
/// output_scale`). This is the value actually consumed by the runtime's `requantize` step --
/// `calculate_asymmetric_quant_s8`'s own `multiplier`/`shift` are not it, since a tensor's
/// requantization depends on what produced it, not on its own value range alone.
pub fn calculate_output_requant_multiplier(
    input_scale: f32,
    weight_scale: f32,
    output_scale: f32,
) -> (i32, i32) {
    let real_multiplier = if output_scale > 1e-12 {
        (input_scale * weight_scale) / output_scale
    } else {
        0.0
    };
    quantize_multiplier(real_multiplier)
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

/// Calculate per-output-channel symmetric s8 quantization parameters. `weights` is a flat
/// row-major array of `out_channels` equal-sized slices (one per output channel).
pub fn calculate_per_channel_quant_s8(
    weights: &[f32],
    out_channels: usize,
) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let elems_per_channel = weights.len() / out_channels;
    let mut multipliers = Vec::with_capacity(out_channels);
    let mut shifts = Vec::with_capacity(out_channels);
    let mut scales = Vec::with_capacity(out_channels);

    for c in 0..out_channels {
        let channel = &weights[c * elems_per_channel..(c + 1) * elems_per_channel];
        let abs_max = channel.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
        let quant = calculate_symmetric_quant_s8(abs_max);
        multipliers.push(quant.multiplier);
        shifts.push(quant.shift);
        scales.push(quant.scale);
    }

    (multipliers, shifts, scales)
}

/// Quantize f32 weights to s8 using one scale per output channel (from
/// `calculate_per_channel_quant_s8`), rather than a single scale for the whole tensor.
pub fn quantize_weights_s8_per_channel(weights: &[f32], scales: &[f32]) -> Vec<i8> {
    let out_channels = scales.len();
    let elems_per_channel = weights.len() / out_channels;
    let mut out = Vec::with_capacity(weights.len());

    for c in 0..out_channels {
        let channel = &weights[c * elems_per_channel..(c + 1) * elems_per_channel];
        out.extend(quantize_weights_s8(channel, scales[c]));
    }

    out
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

    #[test]
    fn test_per_channel_quant_uses_independent_scales() {
        // Channel 0 has a much larger range than channel 1, so per-channel scales must differ
        // (unlike a single per-tensor scale, which would waste channel 1's precision).
        let weights = [10.0f32, -10.0, 5.0, 0.1, -0.1, 0.05];
        let out_channels = 2;
        let (multipliers, shifts, scales) = calculate_per_channel_quant_s8(&weights, out_channels);

        assert_eq!(multipliers.len(), out_channels);
        assert_eq!(shifts.len(), out_channels);
        assert_eq!(scales.len(), out_channels);
        assert!(scales[0] > scales[1] * 10.0);

        let quantized = quantize_weights_s8_per_channel(&weights, &scales);
        assert_eq!(quantized.len(), weights.len());
        // Channel 0's max magnitude value should saturate near the s8 range.
        assert!(quantized[0].abs() >= 120 || quantized[1].abs() >= 120);
        // Channel 1's much smaller values should not collapse to zero.
        assert!(quantized[3] != 0 || quantized[4] != 0 || quantized[5] != 0);
    }

    #[test]
    fn test_per_channel_quant_matches_per_tensor_for_uniform_channels() {
        // If every channel has the same abs-max, per-channel quantization should degenerate to
        // the same result as per-tensor quantization.
        let weights = [1.0f32, -1.0, 1.0, -1.0];
        let (_, _, scales) = calculate_per_channel_quant_s8(&weights, 2);
        let per_tensor = calculate_symmetric_quant_s8(1.0);
        for scale in scales {
            assert!((scale - per_tensor.scale).abs() < 1e-6);
        }
    }

    #[test]
    fn test_asymmetric_quant_zero_based_range_maps_min_to_qmin() {
        // A post-ReLU range [0, 1] should map 0.0 -> -128 exactly (the whole range sits on the
        // positive side of zero, so zero itself becomes the most negative int8 code).
        let q = calculate_asymmetric_quant_s8(0.0, 1.0);
        assert_eq!(q.zero_point, -128);
        assert!((q.scale - 1.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn test_asymmetric_quant_symmetric_range_is_near_zero_point() {
        // A symmetric range around zero should produce a zero-point near 0 (not -128 or 127),
        // matching what calculate_symmetric_quant_s8 would give for the same range.
        let q = calculate_asymmetric_quant_s8(-1.0, 1.0);
        assert!(q.zero_point.abs() <= 1);
    }

    #[test]
    fn test_asymmetric_quant_clamps_degenerate_range() {
        // min > max or a zero-width range shouldn't panic or divide by zero.
        let q = calculate_asymmetric_quant_s8(0.0, 0.0);
        assert!(q.scale > 0.0);
        assert!(q.zero_point >= -128 && q.zero_point <= 127);
    }

    #[test]
    fn test_output_requant_multiplier_matches_manual_ratio() {
        let input_scale = 1.0 / 127.0;
        let weight_scale = 0.02;
        let output_scale = 1.0 / 255.0;
        let (multiplier, shift) =
            calculate_output_requant_multiplier(input_scale, weight_scale, output_scale);
        let (expected_m, expected_s) =
            quantize_multiplier((input_scale * weight_scale) / output_scale);
        assert_eq!(multiplier, expected_m);
        assert_eq!(shift, expected_s);
    }

    #[test]
    fn test_output_requant_multiplier_handles_zero_output_scale() {
        let (multiplier, shift) = calculate_output_requant_multiplier(0.01, 0.01, 0.0);
        assert_eq!((multiplier, shift), (0, 0));
    }
}
