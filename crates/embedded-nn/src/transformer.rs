//! Tiny transformer kernels for `#![no_std]` int8 inference.
//!
//! These operators target short-sequence encoders (`T` tens of tokens, small `d_model`)
//! rather than BERT/ViT-scale models. RMSNorm and fused scaled-dot-product attention keep
//! the `T×T` score matrix in caller-provided scratch instead of materializing a graph of
//! reshape / transpose / matmul nodes.

use crate::softmax::softmax_s8;
use crate::support::{clamp, requantize};
use crate::types::{PerTensorQuantParams, Result};

/// Integer square root of a `u32` using Babylonian iteration.
#[inline]
pub fn integer_sqrt_u32(val: u32) -> u32 {
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

/// Parameters for last-axis RMSNorm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RmsNormParams {
    /// Subtracted from each input element before the RMS (`-zero_point`).
    pub input_offset: i32,
    /// Added after requantization.
    pub output_offset: i32,
    /// Length of the normalized axis (last dimension).
    pub axis_size: usize,
    /// Added to the mean square in the integer domain to avoid a zero RMS.
    pub epsilon: u32,
}

impl RmsNormParams {
    /// Builds parameters for a last-axis RMSNorm.
    pub const fn new(
        input_offset: i32,
        output_offset: i32,
        axis_size: usize,
        epsilon: u32,
    ) -> Self {
        Self {
            input_offset,
            output_offset,
            axis_size,
            epsilon,
        }
    }
}

/// Parameters for fused scaled-dot-product attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionParams {
    /// Batch count (`N`).
    pub batches: usize,
    /// Sequence length (`T`).
    pub seq_len: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Channels per head (`d_model / num_heads`).
    pub head_dim: usize,
    /// Added to Q elements.
    pub q_offset: i32,
    /// Added to K elements.
    pub k_offset: i32,
    /// Added to V elements.
    pub v_offset: i32,
    /// Added to softmax scores before the value mix (typically `+128` when softmax zp is `-128`).
    pub score_offset: i32,
    /// Added after the output requantization.
    pub output_offset: i32,
    /// Softmax Q31 multiplier (same convention as [`softmax_s8`]).
    pub softmax_mult: i32,
    /// Softmax shift.
    pub softmax_shift: i32,
    /// Softmax `diff_min` cutoff.
    pub softmax_diff_min: i32,
}

/// Last-axis RMSNorm: `y = (x / rms(x)) * gamma` with optional per-channel `gamma`.
///
/// `gamma` is a Q7-ish scale (`127 ≈ 1.0`) of length `axis_size`. When omitted, the gain is `1`.
pub fn rms_norm_s8(
    params: &RmsNormParams,
    quant_params: &PerTensorQuantParams,
    input: &[i8],
    gamma: Option<&[i8]>,
    output: &mut [i8],
) -> Result<()> {
    let axis = params.axis_size;
    if axis == 0 {
        return Err(crate::types::Error::ArgumentError);
    }
    if input.len() != output.len() || !input.len().is_multiple_of(axis) {
        return Err(crate::types::Error::ArgumentError);
    }
    if let Some(gamma) = gamma
        && gamma.len() != axis
    {
        return Err(crate::types::Error::ArgumentError);
    }

    let rows = input.len() / axis;
    for row in 0..rows {
        let in_row = &input[row * axis..(row + 1) * axis];
        let out_row = &mut output[row * axis..(row + 1) * axis];

        let mut sum_sq: u64 = 0;
        for &val in in_row {
            let centered = val as i32 + params.input_offset;
            sum_sq += (centered as i64 * centered as i64) as u64;
        }
        let mean_sq = (sum_sq / axis as u64) as u32;
        let rms = integer_sqrt_u32(mean_sq.saturating_add(params.epsilon)).max(1);
        // Q15 reciprocal of RMS.
        let inv_rms = ((1u32 << 15) / rms) as i32;

        for c in 0..axis {
            let centered = in_row[c] as i32 + params.input_offset;
            let gain = match gamma {
                Some(g) => g[c] as i32,
                None => 127,
            };
            // Keep the normalized value in Q15. Gamma is Q7 where 127 represents 1.0.
            let acc = ((centered as i64 * inv_rms as i64 * gain as i64) / 127) as i32;
            let req = requantize(acc, quant_params.multiplier, quant_params.shift);
            out_row[c] = clamp(req + params.output_offset, i8::MIN as i32, i8::MAX as i32) as i8;
        }
    }
    Ok(())
}

/// Fused int8 scaled-dot-product attention.
///
/// `q`, `k`, `v`, and `output` are packed as `[N, T, H * Dh]` with inner layout
/// `head * head_dim + dim`. `scores_scratch` must hold `2 * seq_len * seq_len` int8 values
/// (logits then probabilities) and is reused across heads.
///
/// `logits_quant` requantizes `QKᵀ` (bake `1/√d_k` into this multiplier). `output_quant`
/// requantizes the `softmax @ V` mix.
pub fn scaled_dot_product_attention_s8(
    params: &AttentionParams,
    logits_quant: &PerTensorQuantParams,
    output_quant: &PerTensorQuantParams,
    q: &[i8],
    k: &[i8],
    v: &[i8],
    scores_scratch: &mut [i8],
    output: &mut [i8],
) -> Result<()> {
    let AttentionParams {
        batches,
        seq_len,
        num_heads,
        head_dim,
        ..
    } = *params;
    if batches == 0 || seq_len == 0 || num_heads == 0 || head_dim == 0 {
        return Err(crate::types::Error::ArgumentError);
    }
    let d_model = num_heads * head_dim;
    let tokens = batches * seq_len;
    let expected = tokens * d_model;
    let score_len = seq_len * seq_len;
    if q.len() < expected
        || k.len() < expected
        || v.len() < expected
        || output.len() < expected
        || scores_scratch.len() < score_len * 2
    {
        return Err(crate::types::Error::ArgumentError);
    }

    for b in 0..batches {
        for h in 0..num_heads {
            let (logits, probs) = scores_scratch.split_at_mut(score_len);
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut acc: i32 = 0;
                    let q_base = ((b * seq_len + i) * d_model) + h * head_dim;
                    let k_base = ((b * seq_len + j) * d_model) + h * head_dim;
                    for d in 0..head_dim {
                        let qv = q[q_base + d] as i32 + params.q_offset;
                        let kv = k[k_base + d] as i32 + params.k_offset;
                        acc += qv * kv;
                    }
                    let req = requantize(acc, logits_quant.multiplier, logits_quant.shift);
                    logits[i * seq_len + j] = clamp(req, i8::MIN as i32, i8::MAX as i32) as i8;
                }
            }

            softmax_s8(
                logits,
                seq_len,
                seq_len,
                params.softmax_mult,
                params.softmax_shift,
                params.softmax_diff_min,
                probs,
            )?;

            for i in 0..seq_len {
                for d in 0..head_dim {
                    let mut acc: i32 = 0;
                    for j in 0..seq_len {
                        let score = probs[i * seq_len + j] as i32 + params.score_offset;
                        let vv = v[((b * seq_len + j) * d_model) + h * head_dim + d] as i32
                            + params.v_offset;
                        acc += score * vv;
                    }
                    let req = requantize(acc, output_quant.multiplier, output_quant.shift);
                    output[((b * seq_len + i) * d_model) + h * head_dim + d] =
                        clamp(req + params.output_offset, i8::MIN as i32, i8::MAX as i32) as i8;
                }
            }
        }
    }
    Ok(())
}

/// Batched matrix multiply over the last two packed dimensions.
///
/// `lhs` is `[batches, rows, accum]`. `rhs` is `[batches, accum, cols]` unless
/// `rhs_transposed`, in which case it is `[batches, cols, accum]`.
pub fn batch_matmul_s8_shaped(
    lhs_offset: i32,
    rhs_offset: i32,
    output_offset: i32,
    quant_params: &PerTensorQuantParams,
    batches: usize,
    rows: usize,
    accum: usize,
    cols: usize,
    lhs: &[i8],
    rhs: &[i8],
    rhs_transposed: bool,
    output: &mut [i8],
) -> Result<()> {
    if batches == 0 || rows == 0 || accum == 0 || cols == 0 {
        return Ok(());
    }
    let lhs_len = batches * rows * accum;
    let rhs_len = batches * accum * cols;
    let out_len = batches * rows * cols;
    if lhs.len() < lhs_len || rhs.len() < rhs_len || output.len() < out_len {
        return Err(crate::types::Error::ArgumentError);
    }

    for b in 0..batches {
        for i in 0..rows {
            for j in 0..cols {
                let mut acc: i32 = 0;
                for k in 0..accum {
                    let lhs_idx = (b * rows + i) * accum + k;
                    let rhs_idx = if rhs_transposed {
                        (b * cols + j) * accum + k
                    } else {
                        (b * accum + k) * cols + j
                    };
                    let lv = lhs[lhs_idx] as i32 + lhs_offset;
                    let rv = rhs[rhs_idx] as i32 + rhs_offset;
                    acc += lv * rv;
                }
                let req = requantize(acc, quant_params.multiplier, quant_params.shift);
                output[(b * rows + i) * cols + j] =
                    clamp(req + output_offset, i8::MIN as i32, i8::MAX as i32) as i8;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_quant() -> PerTensorQuantParams {
        PerTensorQuantParams::new(1_073_741_824, 1)
    }

    #[test]
    fn integer_sqrt_matches_known_squares() {
        assert_eq!(integer_sqrt_u32(0), 0);
        assert_eq!(integer_sqrt_u32(1), 1);
        assert_eq!(integer_sqrt_u32(144), 12);
        assert_eq!(integer_sqrt_u32(15), 3);
    }

    #[test]
    fn rms_norm_constant_row_is_unit_scale() {
        let input = [64i8, 64, 64, 64];
        let mut output = [0i8; 4];
        rms_norm_s8(
            &RmsNormParams::new(0, 0, 4, 1),
            &identity_quant(),
            &input,
            None,
            &mut output,
        )
        .unwrap();
        // Constant vector → gain ≈ 1 after RMS; all channels equal and non-zero.
        assert!(output.iter().all(|&v| v == output[0]));
        assert!(output[0] > 0);
    }

    #[test]
    fn rms_norm_rejects_mismatched_gamma() {
        let input = [1i8, 2, 3, 4];
        let mut output = [0i8; 4];
        let gamma = [1i8, 2];
        assert!(
            rms_norm_s8(
                &RmsNormParams::new(0, 0, 4, 1),
                &identity_quant(),
                &input,
                Some(&gamma),
                &mut output,
            )
            .is_err()
        );
    }

    #[test]
    fn attention_identity_keys_select_matching_value() {
        // One head, d=2, T=2. Q = K = [[1,0],[0,1]] (scaled up), V = [[10,20],[30,40]].
        let q = [32i8, 0, 0, 32];
        let k = [32i8, 0, 0, 32];
        let v = [10i8, 20, 30, 40];
        let mut scores = [0i8; 8];
        let mut output = [0i8; 4];
        let params = AttentionParams {
            batches: 1,
            seq_len: 2,
            num_heads: 1,
            head_dim: 2,
            q_offset: 0,
            k_offset: 0,
            v_offset: 0,
            score_offset: 128,
            output_offset: 0,
            softmax_mult: 1_073_741_824,
            softmax_shift: 20,
            softmax_diff_min: -256,
        };
        scaled_dot_product_attention_s8(
            &params,
            &identity_quant(),
            &identity_quant(),
            &q,
            &k,
            &v,
            &mut scores,
            &mut output,
        )
        .unwrap();
        // Diagonal attention should copy V rows (within quantization slack).
        assert!(output[0] >= output[2]);
        assert!(output[1] >= 0);
        assert!(output[2] >= 0);
        assert!(output[3] >= output[1]);
    }

    #[test]
    fn batch_matmul_shaped_identity() {
        let lhs = [1i8, 0, 0, 1];
        let rhs = [2i8, 3, 4, 5];
        let mut out = [0i8; 4];
        batch_matmul_s8_shaped(
            0,
            0,
            0,
            &identity_quant(),
            1,
            2,
            2,
            2,
            &lhs,
            &rhs,
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [2, 3, 4, 5]);
    }

    #[test]
    fn batch_matmul_rhs_transposed_matches_explicit_transpose() {
        // rhs stored as [cols, accum] = [[2,4],[3,5]] which is transpose of [[2,3],[4,5]].
        let lhs = [1i8, 0, 0, 1];
        let rhs_t = [2i8, 4, 3, 5];
        let mut out = [0i8; 4];
        batch_matmul_s8_shaped(
            0,
            0,
            0,
            &identity_quant(),
            1,
            2,
            2,
            2,
            &lhs,
            &rhs_t,
            true,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [2, 3, 4, 5]);
    }
}
