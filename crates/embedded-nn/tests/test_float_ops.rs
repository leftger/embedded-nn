#![allow(
    clippy::approx_constant,
    clippy::excessive_precision,
    clippy::identity_op
)]
use embedded_nn::{
    Dims, Padding2D, Tile,
    float_ops::{
        convolve_f32, f16_to_f32, f32_to_f16, fully_connected_f32, relu_f32, relu6_f32, softmax_f32,
    },
};

#[test]
fn test_f16_to_f32_and_back_roundtrip() {
    let test_values = [
        0.0f32,
        -0.0f32,
        1.0f32,
        -1.0f32,
        0.5f32,
        -0.5f32,
        3.14159f32,
        65500.0f32,
        0.00006103515625f32, // f16 min normal
    ];

    for &val in test_values.iter() {
        let f16_bits = f32_to_f16(val);
        let back = f16_to_f32(f16_bits);
        if val == 0.0 {
            assert_eq!(back, val);
        } else {
            let rel_diff = (back - val).abs() / val.abs();
            assert!(
                rel_diff < 0.005,
                "Failed for val {}, got back {}, rel diff {}",
                val,
                back,
                rel_diff
            );
        }
    }
}

#[test]
fn test_f16_special_cases_inf_nan_subnormal() {
    // Pos inf
    let pos_inf = f32::INFINITY;
    let pos_inf_h = f32_to_f16(pos_inf);
    let back_pos_inf = f16_to_f32(pos_inf_h);
    assert!(back_pos_inf.is_infinite() && back_pos_inf > 0.0);

    // Neg inf
    let neg_inf = f32::NEG_INFINITY;
    let neg_inf_h = f32_to_f16(neg_inf);
    let back_neg_inf = f16_to_f32(neg_inf_h);
    assert!(back_neg_inf.is_infinite() && back_neg_inf < 0.0);

    // Subnormal f16 conversion (exp=0, mant!=0)
    let subnormal_h = 0x0001u16; // smallest subnormal f16
    let subnormal_f32 = f16_to_f32(subnormal_h);
    assert!(subnormal_f32 > 0.0 && subnormal_f32 < 0.0001);

    // NaN conversion (ensure low-mantissa NaNs never turn into Infinity)
    let nan_f32 = f32::from_bits(0x7F80_0001); // Smallest mantissa NaN
    let nan_h = f32_to_f16(nan_f32);
    let back_nan = f16_to_f32(nan_h);
    assert!(back_nan.is_nan(), "Expected NaN, got {}", back_nan);
}

#[test]
fn test_convolve_f32_multi_channel_bias_stride() {
    let input_dims = Dims::new(1, 4, 4, 2);
    let input = [1.0f32; 32];
    let filter_dims = Dims::new(2, 2, 2, 2);
    let kernel = [1.0f32; 16];
    let bias = [0.5f32, -0.5f32];
    let output_dims = Dims::new(1, 2, 2, 2);
    let mut output = [0.0f32; 8];

    convolve_f32(
        Tile::new(2, 2),
        Padding2D::default(),
        Tile::new(1, 1),
        &input_dims,
        &input,
        &filter_dims,
        &kernel,
        Some(&bias),
        &output_dims,
        &mut output,
    )
    .unwrap();

    // 2x2 spatial kernel * 2 input channels = 8 multiplications of 1.0 = 8.0
    // Ch 0: 0.5 + 8.0 = 8.5
    // Ch 1: -0.5 + 8.0 = 7.5
    assert_eq!(output[0], 8.5);
    assert_eq!(output[1], 7.5);
}

#[test]
fn test_fully_connected_f32_multi_batch() {
    let input_dims = Dims::new(2, 1, 1, 2);
    let input = [
        1.0f32, 2.0f32, // batch 0
        3.0f32, 4.0f32, // batch 1
    ];
    let filter_dims = Dims::new(2, 1, 1, 2);
    let kernel = [
        2.0f32, 3.0f32, // out_c 0
        4.0f32, 5.0f32, // out_c 1
    ];
    let output_dims = Dims::new(2, 1, 1, 2);
    let mut output = [0.0f32; 4];

    fully_connected_f32(
        &input_dims,
        &input,
        &filter_dims,
        &kernel,
        None,
        &output_dims,
        &mut output,
    )
    .unwrap();

    // Batch 0:
    // out_c 0: 1*2 + 2*3 = 8
    // out_c 1: 1*4 + 2*5 = 14
    assert_eq!(output[0], 8.0);
    assert_eq!(output[1], 14.0);

    // Batch 1:
    // out_c 0: 3*2 + 4*3 = 18
    // out_c 1: 3*4 + 4*5 = 32
    assert_eq!(output[2], 18.0);
    assert_eq!(output[3], 32.0);
}

#[test]
fn test_float_activations() {
    let mut data = [-10.0f32, -0.5, 0.0, 0.5, 10.0];
    relu_f32(&mut data);
    assert_eq!(data, [0.0, 0.0, 0.0, 0.5, 10.0]);

    let mut data6 = [-10.0f32, -0.5, 0.0, 0.5, 5.5, 6.5, 10.0];
    relu6_f32(&mut data6);
    assert_eq!(data6, [0.0, 0.0, 0.0, 0.5, 5.5, 6.0, 6.0]);
}

#[test]
fn test_softmax_f32_extreme_values() {
    let input = [-1000.0f32, -500.0f32, 0.0f32, 10.0f32];
    let mut output = [0.0f32; 4];

    softmax_f32(&input, 1, 4, &mut output).unwrap();

    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
    assert!(output[3] > output[2]);
    assert!(output[2] > output[1]);
}
