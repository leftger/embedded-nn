use embedded_nn::{
    concat::concatenation_s8,
    pad::pad_s8,
    reshape::reshape_s8,
    transpose::{transpose_2d_s8, transpose_spatial_s8},
    Dims, Error, Tile,
};

#[test]
fn test_concatenation_s8_depth_and_multi_tensor() {
    let in1_dims = Dims::new(1, 2, 2, 1);
    let in1 = [1i8, 2i8, 3i8, 4i8];

    let in2_dims = Dims::new(1, 2, 2, 2);
    let in2 = [10i8, 20i8, 30i8, 40i8, 50i8, 60i8, 70i8, 80i8];

    let out_dims = Dims::new(1, 2, 2, 3);
    let mut out = [0i8; 12];

    concatenation_s8(&in1_dims, &in1, &in2_dims, &in2, &out_dims, &mut out).unwrap();

    // Row 0, Col 0: [1, 10, 20]
    // Row 0, Col 1: [2, 30, 40]
    // Row 1, Col 0: [3, 50, 60]
    // Row 1, Col 1: [4, 70, 80]
    assert_eq!(out, [1, 10, 20, 2, 30, 40, 3, 50, 60, 4, 70, 80]);
}

#[test]
fn test_concatenation_s8_mismatched_dimensions_error() {
    let in1_dims = Dims::new(1, 2, 2, 1);
    let in1 = [1i8, 2i8, 3i8, 4i8];
    let in2_dims = Dims::new(1, 3, 2, 1); // Mismatched H!
    let in2 = [1i8; 6];
    let out_dims = Dims::new(1, 2, 2, 2);
    let mut out = [0i8; 8];

    let err = concatenation_s8(&in1_dims, &in1, &in2_dims, &in2, &out_dims, &mut out);
    assert_eq!(err, Err(Error::ArgumentError));
}

#[test]
fn test_pad_s8_asymmetric_and_values() {
    let in_dims = Dims::new(1, 1, 2, 1);
    let input = [5i8, 10i8];

    let pad_before = Tile::new(1, 1);
    let pad_after = Tile::new(1, 1);

    let out_dims = Dims::new(1, 3, 4, 1);
    let mut out = [0i8; 12];

    pad_s8(
        &in_dims,
        &input,
        &pad_before,
        &pad_after,
        -1i8,
        &out_dims,
        &mut out,
    )
    .unwrap();

    // Check padding value -1 outside
    assert_eq!(out[0], -1);
    // Center elements
    assert_eq!(out[5], 5);
    assert_eq!(out[6], 10);
}

#[test]
fn test_transpose_2d_s8_matrix_variations() {
    // 1x5 row matrix to 5x1 column matrix
    let input = [1i8, 2i8, 3i8, 4i8, 5i8];
    let mut out = [0i8; 5];
    transpose_2d_s8(1, 5, &input, &mut out).unwrap();
    assert_eq!(out, [1, 2, 3, 4, 5]);

    // 3x2 matrix
    let input3x2 = [1i8, 2i8, 3i8, 4i8, 5i8, 6i8];
    let mut out2x3 = [0i8; 6];
    transpose_2d_s8(3, 2, &input3x2, &mut out2x3).unwrap();
    // Row 0: [1, 2], Row 1: [3, 4], Row 2: [5, 6] -> transposed 2x3: Col 0: [1, 3, 5], Col 1: [2, 4, 6]
    assert_eq!(out2x3, [1, 3, 5, 2, 4, 6]);
}

#[test]
fn test_transpose_spatial_s8_multi_channel() {
    let in_dims = Dims::new(1, 2, 3, 2); // N=1, H=2, W=3, C=2
    let input = [
        1i8, 10i8, 2i8, 20i8, 3i8, 30i8, // row 0
        4i8, 40i8, 5i8, 50i8, 6i8, 60i8, // row 1
    ];

    let mut out = [0i8; 12]; // N=1, W=3, H=2, C=2

    transpose_spatial_s8(&in_dims, &input, &mut out).unwrap();

    // Transposed height & width with channels preserved per spatial position
    assert_eq!(out[0..2], [1, 10]);
    assert_eq!(out[2..4], [4, 40]);
}

#[test]
fn test_reshape_s8_mismatched_size_error() {
    let in_dims = Dims::new(1, 2, 3, 1);
    let input = [10i8, 20i8, 30i8, 40i8, 50i8, 60i8];

    let out_dims_bad = Dims::new(1, 1, 5, 1); // Size 5 vs 6
    let mut out = [0i8; 5];

    let err = reshape_s8(&in_dims, &input, &out_dims_bad, &mut out);
    assert_eq!(err, Err(Error::ArgumentError));
}

#[test]
fn test_tensor_view_spatial_indexing_and_padding() {
    use embedded_nn::{TensorView, TensorViewPadding};

    let dims = Dims::new(1, 2, 2, 1);
    let input = [10i8, 20i8, 30i8, 40i8];
    let view = TensorView::new(
        &input,
        dims,
        TensorViewPadding::Same,
        Tile::new(1, 1),
        Tile::new(3, 3),
    );

    // Test inside bounds
    assert_eq!(view.get_or_pad(0, 0, 0, 0, 0), 10);
    assert_eq!(view.get_or_pad(0, 1, 1, 0, 0), 40);

    // Test padded out-of-bounds
    assert_eq!(view.get_or_pad(0, -1, 0, 0, -1), -1);
    assert_eq!(view.get_or_pad(0, 2, 0, 0, 0), 0);

    let (oh, ow) = view.output_spatial_dims();
    assert_eq!((oh, ow), (2, 2));
}

#[test]
fn test_fused_activation_mapping() {
    use embedded_nn::FusedActivation;

    let relu = FusedActivation::Relu;
    let act_s8 = relu.to_activation(false);
    assert_eq!(act_s8.min, 0);
    assert_eq!(act_s8.max, 127);

    let relu6 = FusedActivation::Relu6;
    let act_r6 = relu6.to_activation(false);
    assert_eq!(act_r6.min, 0);
    assert_eq!(act_r6.max, 6);
}
