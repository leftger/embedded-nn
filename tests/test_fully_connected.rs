use embedded_nn::{
    float_ops::fully_connected_f32,
    fully_connected::{
        batch_matmul_s16, batch_matmul_s8, fully_connected_per_channel_s8, fully_connected_s16,
        fully_connected_s8,
    },
    subbyte::{fully_connected_s4, pack_s4_pair},
    Activation, Dims, FcParams, PerChannelQuantParams, PerTensorQuantParams,
};

#[test]
fn test_fully_connected_s8_per_tensor_variations() {
    let fc_params = FcParams {
        input_offset: 5,
        filter_offset: 0,
        output_offset: -10,
        activation: Activation::new(-20, 20),
    };
    let quant_params = PerTensorQuantParams::new(1073741824, 0); // 0.5 Q31

    let input_dims = Dims::new(2, 1, 1, 3); // 2 batches, 3 accum depth
    let input = [
        5i8, 15i8, 25i8, // batch 0 -> +5 offset = [10, 20, 30]
        -5i8, 0i8, 5i8, // batch 1 -> +5 offset = [0, 5, 10]
    ];
    let filter_dims = Dims::new(3, 1, 1, 2); // accum_depth=3, output_depth=2
    let kernel = [
        1i8, 2i8, 3i8, // out_c 0
        2i8, 0i8, -1i8, // out_c 1
    ];
    let bias = [10i32, -20i32];

    let output_dims = Dims::new(2, 1, 1, 2);
    let mut output = [0i8; 4];

    fully_connected_s8(
        &fc_params,
        &quant_params,
        &input_dims,
        &input,
        &filter_dims,
        &kernel,
        Some(&bias),
        &output_dims,
        &mut output,
    )
    .unwrap();

    // Batch 0:
    // out_c 0: 10 + (10*1 + 20*2 + 30*3) = 10 + 140 = 150 -> requantized 0.5 = 75 -> offset -10 = 65 -> clamped to 20
    assert_eq!(output[0], 20);
    // out_c 1: -20 + (10*2 + 20*0 + 30*-1) = -20 - 10 = -30 -> requantized 0.5 = -15 -> offset -10 = -25 -> clamped to -20
    assert_eq!(output[1], -20);
}

#[test]
fn test_fully_connected_per_channel_s8_comprehensive() {
    let fc_params = FcParams {
        input_offset: 0,
        filter_offset: 0,
        output_offset: 0,
        activation: Activation::int8_unconstrained(),
    };

    let mults = [1073741824, 2147483647]; // Ch 0: 0.5, Ch 1: 1.0
    let shifts = [0, 0];
    let quant_params = PerChannelQuantParams::new(&mults, &shifts);

    let input_dims = Dims::new(1, 1, 1, 2);
    let input = [10i8, 20i8];
    let filter_dims = Dims::new(2, 1, 1, 2);
    let kernel = [
        1i8, 1i8, // out_c 0
        2i8, 2i8, // out_c 1
    ];
    let output_dims = Dims::new(1, 1, 1, 2);
    let mut output = [0i8; 2];

    fully_connected_per_channel_s8(
        &fc_params,
        &quant_params,
        &input_dims,
        &input,
        &filter_dims,
        &kernel,
        None,
        &output_dims,
        &mut output,
    )
    .unwrap();

    // Ch 0: (10*1 + 20*1) * 0.5 = 15
    // Ch 1: (10*2 + 20*2) * 1.0 = 60
    assert_eq!(output[0], 15);
    assert_eq!(output[1], 60);
}

#[test]
fn test_fully_connected_s16_comprehensive() {
    let fc_params = FcParams {
        input_offset: 0,
        filter_offset: 0,
        output_offset: 0,
        activation: Activation::int16_unconstrained(),
    };
    let quant_params = PerTensorQuantParams::new(2147483647, 0); // 1.0

    let input_dims = Dims::new(1, 1, 1, 2);
    let input = [1000i16, 2000i16];
    let filter_dims = Dims::new(2, 1, 1, 2);
    let kernel = [10i8, 20i8, 5i8, 15i8];
    let bias = [100i64, -200i64];

    let output_dims = Dims::new(1, 1, 1, 2);
    let mut output = [0i16; 2];

    fully_connected_s16(
        &fc_params,
        &quant_params,
        &input_dims,
        &input,
        &filter_dims,
        &kernel,
        Some(&bias),
        &output_dims,
        &mut output,
    )
    .unwrap();

    // acc = 100 + (1000*10 + 2000*20) = 50100 -> >> 15 = 1 -> requantized 1.0 = 1
    assert_eq!(output[0], 1);
}

#[test]
fn test_batch_matmul_s8_broadcasting_and_execution() {
    let fc_params = FcParams {
        input_offset: 0,
        filter_offset: 0,
        output_offset: 0,
        activation: Activation::int8_unconstrained(),
    };
    let quant_params = PerTensorQuantParams::new(2147483647, 0); // 1.0

    // Batch 2, Matrix A: (2, 3), Matrix B: (3, 2)
    let lhs_dims = Dims::new(2, 2, 3, 1); // N=2, H=2 (rows), W=3 (accum), C=1
    let lhs = [
        1i8, 2i8, 3i8, 4i8, 5i8, 6i8, // Batch 0
        7i8, 8i8, 9i8, 10i8, 11i8, 12i8, // Batch 1
    ];
    let rhs_dims = Dims::new(1, 3, 1, 2); // Broadcasting N=1, H=3 (accum), W=1, C=2 (cols)
    let rhs = [
        1i8, 0i8, // row 0
        0i8, 1i8, // row 1
        1i8, 1i8, // row 2
    ];
    let output_dims = Dims::new(2, 2, 2, 1); // N=2, H=2, W=2
    let mut output = [0i8; 8];

    batch_matmul_s8(
        &fc_params,
        &quant_params,
        &lhs_dims,
        &lhs,
        &rhs_dims,
        &rhs,
        &output_dims,
        &mut output,
    )
    .unwrap();

    // Batch 0 row 0: [1, 2, 3] * B = [1*1 + 2*0 + 3*1, 1*0 + 2*1 + 3*1] = [4, 5]
    assert_eq!(output[0], 4);
    assert_eq!(output[1], 5);
}

#[test]
fn test_batch_matmul_s16_execution() {
    let fc_params = FcParams {
        input_offset: 0,
        filter_offset: 0,
        output_offset: 0,
        activation: Activation::int16_unconstrained(),
    };
    let quant_params = PerTensorQuantParams::new(2147483647, 0);

    let lhs_dims = Dims::new(1, 1, 2, 1); // N=1, H=1, W=2
    let lhs = [100i16, 200i16];
    let rhs_dims = Dims::new(1, 2, 1, 1); // N=1, H=2, W=1, C=1
    let rhs = [3i16, 4i16];
    let output_dims = Dims::new(1, 1, 1, 1);
    let mut output = [0i16; 1];

    batch_matmul_s16(
        &fc_params,
        &quant_params,
        &lhs_dims,
        &lhs,
        &rhs_dims,
        &rhs,
        &output_dims,
        &mut output,
    )
    .unwrap();

    // 100*3 + 200*4 = 1100 -> >> 15 = 0 -> requantized = 0
    assert_eq!(output[0], 0);
}

#[test]
fn test_fully_connected_s4_odd_accum_depth() {
    let fc_params = FcParams {
        input_offset: 0,
        filter_offset: 0,
        output_offset: 0,
        activation: Activation::int8_unconstrained(),
    };
    let quant_params = PerTensorQuantParams::new(2147483647, 0); // 1.0

    let input_dims = Dims::new(1, 1, 1, 3); // K = 3 (odd!)
    let input = [2i8, 3i8, 4i8];
    let filter_dims = Dims::new(3, 1, 1, 1); // 1 output channel
    let packed_kernel = [
        pack_s4_pair(1i8, 2i8), // w0=1, w1=2
        pack_s4_pair(3i8, 0i8), // w2=3, w3=0 (ignored)
    ];
    let output_dims = Dims::new(1, 1, 1, 1);
    let mut output = [0i8; 1];

    fully_connected_s4(
        &fc_params,
        &quant_params,
        &input_dims,
        &input,
        &filter_dims,
        &packed_kernel,
        None,
        &output_dims,
        &mut output,
    )
    .unwrap();

    // 2*1 + 3*2 + 4*3 = 2 + 6 + 12 = 20
    assert_eq!(output[0], 20);
}

#[test]
fn test_fully_connected_f32_execution() {
    let input_dims = Dims::new(1, 1, 1, 3);
    let input = [1.0f32, 2.0f32, 3.0f32];
    let filter_dims = Dims::new(3, 1, 1, 2);
    let kernel = [
        1.0f32, 2.0f32, 3.0f32, // out_c 0
        4.0f32, 5.0f32, 6.0f32, // out_c 1
    ];
    let bias = [0.5f32, -1.0f32];
    let output_dims = Dims::new(1, 1, 1, 2);
    let mut output = [0.0f32; 2];

    fully_connected_f32(
        &input_dims,
        &input,
        &filter_dims,
        &kernel,
        Some(&bias),
        &output_dims,
        &mut output,
    )
    .unwrap();

    // Ch 0: 0.5 + (1*1 + 2*2 + 3*3) = 0.5 + 14 = 14.5
    // Ch 1: -1.0 + (1*4 + 2*5 + 3*6) = -1.0 + 32 = 31.0
    assert_eq!(output[0], 14.5);
    assert_eq!(output[1], 31.0);
}
