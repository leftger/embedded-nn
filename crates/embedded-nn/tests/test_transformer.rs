//! Coverage tests for tiny-transformer kernels: RMSNorm, fused attention, and shaped matmul.

use embedded_nn::{
    AttentionParams, PerTensorQuantParams, RmsNormParams, batch_matmul_s8_shaped, integer_sqrt_u32,
    rms_norm_s8, scaled_dot_product_attention_s8, softmax_last_axis_s8,
};

fn identity_quant() -> PerTensorQuantParams {
    PerTensorQuantParams::new(1_073_741_824, 1)
}

fn attn_params(seq_len: usize, num_heads: usize, head_dim: usize) -> AttentionParams {
    AttentionParams {
        batches: 1,
        seq_len,
        num_heads,
        head_dim,
        q_offset: 0,
        k_offset: 0,
        v_offset: 0,
        score_offset: 128,
        output_offset: 0,
        softmax_mult: 1_073_741_824,
        softmax_shift: 20,
        softmax_diff_min: -256,
    }
}

#[test]
fn rms_norm_argument_errors_and_gamma_path() {
    let input = [8i8, -8, 8, -8];
    let mut output = [0i8; 4];
    assert!(
        rms_norm_s8(
            &RmsNormParams::new(0, 0, 0, 1),
            &identity_quant(),
            &input,
            None,
            &mut output
        )
        .is_err()
    );
    assert!(
        rms_norm_s8(
            &RmsNormParams::new(0, 0, 3, 1),
            &identity_quant(),
            &input,
            None,
            &mut output
        )
        .is_err()
    );
    let mut short = [0i8; 2];
    assert!(
        rms_norm_s8(
            &RmsNormParams::new(0, 0, 4, 1),
            &identity_quant(),
            &input,
            None,
            &mut short
        )
        .is_err()
    );

    let gamma = [127i8, 64, 32, 16];
    rms_norm_s8(
        &RmsNormParams::new(0, 0, 4, 1),
        &identity_quant(),
        &input,
        Some(&gamma),
        &mut output,
    )
    .unwrap();
    assert_ne!(output, [0; 4]);
}

#[test]
fn attention_rejects_bad_shapes_and_runs_two_heads() {
    let q = [8i8, 0, 0, 8, 0, 8, 8, 0];
    let k = q;
    let v = [10i8, 20, 30, 40, 50, 60, 70, 80];
    let mut scores = [0i8; 8];
    let mut output = [0i8; 8];
    assert!(
        scaled_dot_product_attention_s8(
            &attn_params(2, 0, 4),
            &identity_quant(),
            &identity_quant(),
            &q,
            &k,
            &v,
            &mut scores,
            &mut output,
        )
        .is_err()
    );
    let mut tiny_scratch = [0i8; 2];
    assert!(
        scaled_dot_product_attention_s8(
            &attn_params(2, 2, 2),
            &identity_quant(),
            &identity_quant(),
            &q,
            &k,
            &v,
            &mut tiny_scratch,
            &mut output,
        )
        .is_err()
    );

    scaled_dot_product_attention_s8(
        &attn_params(2, 2, 2),
        &identity_quant(),
        &identity_quant(),
        &q,
        &k,
        &v,
        &mut scores,
        &mut output,
    )
    .unwrap();
    assert!(output.iter().any(|&x| x != 0));
}

#[test]
fn batch_matmul_empty_and_short_buffers() {
    let mut out = [0i8; 4];
    assert!(
        batch_matmul_s8_shaped(
            0,
            0,
            0,
            &identity_quant(),
            0,
            2,
            2,
            2,
            &[],
            &[],
            false,
            &mut out,
        )
        .is_ok()
    );
    assert!(
        batch_matmul_s8_shaped(
            0,
            0,
            0,
            &identity_quant(),
            1,
            2,
            2,
            2,
            &[1, 0],
            &[1, 0, 0, 1],
            false,
            &mut out,
        )
        .is_err()
    );
}

#[test]
fn softmax_last_axis_empty_channels_and_sqrt_coverage() {
    let input = [1i8, 2];
    let mut output = [0i8; 2];
    softmax_last_axis_s8(&input, 1, 1, 1, 0, 1073741824, 20, -256, &mut output).unwrap();
    assert_eq!(integer_sqrt_u32(2), 1);
    assert_eq!(integer_sqrt_u32(10_000), 100);
}
