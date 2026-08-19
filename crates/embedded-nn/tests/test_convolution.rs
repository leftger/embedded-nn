use embedded_nn::{
    Activation, ConvParams, Dims, DwConvParams, Error, PerChannelQuantParams, PerTensorQuantParams,
    Tile,
    convolution::{
        convolve_1_x_n_s8, convolve_per_channel_s8, convolve_s8, depthwise_conv_per_channel_s8,
        transpose_conv_s8,
    },
    float_ops::convolve_f32,
};

#[test]
fn test_convolve_s8_standard_per_tensor_variations() {
    // Test 3x3 kernel, batch=1, stride=1, padding=0, dilation=1
    let conv_params = ConvParams {
        input_offset: 128,
        output_offset: -128,
        stride: Tile::new(1, 1),
        padding: Tile::new(0, 0),
        dilation: Tile::new(1, 1),
        activation: Activation::int8_unconstrained(),
    };
    let quant_params = PerTensorQuantParams::new(1073741824, 0); // 0.5 Q31

    let input_dims = Dims::new(1, 3, 3, 1);
    let input = [10i8; 9];
    let filter_dims = Dims::new(1, 3, 3, 1); // out_c=1, kernel_h=3, kernel_w=3, in_c=1
    let kernel = [1i8; 9];
    let bias = [100i32];

    let output_dims = Dims::new(1, 1, 1, 1);
    let mut output = [0i8; 1];

    convolve_s8(
        &conv_params,
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

    // sum = 100 (bias) + 9 * (10 + 128) * 1 = 100 + 1242 = 1342
    // requantized 0.5 = 671
    // output offset -128 = 543 -> clamped to 127
    assert_eq!(output[0], 127);
}

#[test]
fn test_convolve_s8_stride_padding_dilation() {
    // 5x5 input, 3x3 kernel, stride=2, padding=1, dilation=1
    let conv_params = ConvParams {
        input_offset: 0,
        output_offset: 0,
        stride: Tile::new(2, 2),
        padding: Tile::new(1, 1),
        dilation: Tile::new(1, 1),
        activation: Activation::new(0, 100),
    };
    let quant_params = PerTensorQuantParams::new(2147483647, 0); // 1.0 Q31

    let input_dims = Dims::new(1, 5, 5, 1);
    let input = [1i8; 25];
    let filter_dims = Dims::new(1, 3, 3, 1);
    let kernel = [1i8; 9];
    let output_dims = Dims::new(1, 3, 3, 1);
    let mut output = [0i8; 9];

    convolve_s8(
        &conv_params,
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

    // Top-left corner with padding 1x1 covers 2x2 real input elements -> sum = 4
    assert_eq!(output[0], 4);
    // Center element covers 3x3 real input elements -> sum = 9
    assert_eq!(output[4], 9);
}

#[test]
fn test_convolve_s8_multi_batch_multi_channel() {
    let conv_params = ConvParams {
        input_offset: 0,
        output_offset: 0,
        stride: Tile::new(1, 1),
        padding: Tile::new(0, 0),
        dilation: Tile::new(1, 1),
        activation: Activation::int8_unconstrained(),
    };
    let quant_params = PerTensorQuantParams::new(2147483647, 0);

    let input_dims = Dims::new(2, 2, 2, 2); // N=2, H=2, W=2, C=2
    let input = [
        1i8, 2i8, 3i8, 4i8, 5i8, 6i8, 7i8, 8i8, // batch 0
        9i8, 10i8, 11i8, 12i8, 13i8, 14i8, 15i8, 16i8, // batch 1
    ];
    let filter_dims = Dims::new(2, 2, 2, 2); // out_c=2, kh=2, kw=2, in_c=2
    let kernel = [1i8; 16];
    let output_dims = Dims::new(2, 1, 1, 2);
    let mut output = [0i8; 4];

    convolve_s8(
        &conv_params,
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

    // Batch 0: sum of 1..8 = 36
    assert_eq!(output[0], 36);
    assert_eq!(output[1], 36);
    // Batch 1: sum of 9..16 = 100
    assert_eq!(output[2], 100);
    assert_eq!(output[3], 100);
}

#[test]
fn test_convolve_per_channel_s8_comprehensive() {
    let conv_params = ConvParams {
        input_offset: 0,
        output_offset: 0,
        stride: Tile::new(1, 1),
        padding: Tile::new(0, 0),
        dilation: Tile::new(1, 1),
        activation: Activation::int8_unconstrained(),
    };

    let mults = [1073741824, 2147483647]; // Ch 0: 0.5, Ch 1: 1.0
    let shifts = [0, 0];
    let quant_params = PerChannelQuantParams::new(&mults, &shifts);

    let input_dims = Dims::new(1, 2, 2, 1);
    let input = [10i8, 10i8, 10i8, 10i8];
    let filter_dims = Dims::new(2, 2, 2, 1);
    let kernel = [1i8; 8];
    let output_dims = Dims::new(1, 1, 1, 2);
    let mut output = [0i8; 2];

    convolve_per_channel_s8(
        &conv_params,
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

    // Ch 0: 40 * 0.5 = 20
    // Ch 1: 40 * 1.0 = 40
    assert_eq!(output[0], 20);
    assert_eq!(output[1], 40);
}

#[test]
fn test_depthwise_conv_per_channel_s8_execution() {
    let dw_params = DwConvParams {
        input_offset: 0,
        output_offset: 0,
        ch_mult: 1,
        stride: Tile::new(1, 1),
        padding: Tile::new(0, 0),
        dilation: Tile::new(1, 1),
        activation: Activation::int8_unconstrained(),
    };

    let mults = [1073741824, 2147483647];
    let shifts = [0, 0];
    let quant_params = PerChannelQuantParams::new(&mults, &shifts);

    let input_dims = Dims::new(1, 2, 2, 2);
    let input = [10i8; 8];
    let filter_dims = Dims::new(2, 2, 2, 1);
    let kernel = [1i8; 8];
    let output_dims = Dims::new(1, 1, 1, 2);
    let mut output = [0i8; 2];

    depthwise_conv_per_channel_s8(
        &dw_params,
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

    // Ch 0: 40 * 0.5 = 20
    // Ch 1: 40 * 1.0 = 40
    assert_eq!(output[0], 20);
    assert_eq!(output[1], 40);
}

#[test]
fn test_convolve_1_x_n_s8_temporal() {
    let conv_params = ConvParams {
        input_offset: 0,
        output_offset: 0,
        stride: Tile::new(1, 1),
        padding: Tile::new(0, 0),
        dilation: Tile::new(1, 1),
        activation: Activation::int8_unconstrained(),
    };
    let quant_params = PerTensorQuantParams::new(2147483647, 0);

    let input_dims = Dims::new(1, 1, 4, 1); // 1D temporal signal length 4
    let input = [1i8, 2i8, 3i8, 4i8];
    let filter_dims = Dims::new(1, 1, 2, 1); // 1D kernel length 2
    let kernel = [1i8, 1i8];
    let output_dims = Dims::new(1, 1, 3, 1);
    let mut output = [0i8; 3];

    convolve_1_x_n_s8(
        &conv_params,
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

    // [1+2, 2+3, 3+4] = [3, 5, 7]
    assert_eq!(output, [3, 5, 7]);
}

#[test]
fn test_transpose_conv_s8_execution() {
    let conv_params = ConvParams {
        input_offset: 0,
        output_offset: 0,
        stride: Tile::new(2, 2),
        padding: Tile::new(0, 0),
        dilation: Tile::new(1, 1),
        activation: Activation::int8_unconstrained(),
    };
    let mults = [2147483647];
    let shifts = [0];
    let quant_params = PerChannelQuantParams::new(&mults, &shifts);

    let input_dims = Dims::new(1, 2, 2, 1);
    let input = [1i8, 2i8, 3i8, 4i8];
    let filter_dims = Dims::new(1, 2, 2, 1);
    let kernel = [1i8, 1i8, 1i8, 1i8];
    let output_dims = Dims::new(1, 4, 4, 1);
    let mut output = [0i8; 16];

    transpose_conv_s8(
        &conv_params,
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

    assert_eq!(output[0], 1);
    assert_eq!(output[5], 1);
    assert_eq!(output[6], 2);
    assert_eq!(output[10], 4);
}

#[test]
fn test_convolve_f32_execution() {
    let input_dims = Dims::new(1, 3, 3, 1);
    let input = [1.0f32; 9];
    let filter_dims = Dims::new(1, 3, 3, 1);
    let kernel = [2.0f32; 9];
    let bias = [5.0f32];
    let output_dims = Dims::new(1, 1, 1, 1);
    let mut output = [0.0f32; 1];

    convolve_f32(
        Tile::new(1, 1),
        Tile::new(0, 0),
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

    // 5.0 + 9 * 2.0 = 23.0
    assert_eq!(output[0], 23.0);
}

#[test]
fn test_convolve_s8_error_paths() {
    let conv_params = ConvParams {
        input_offset: 0,
        output_offset: 0,
        stride: Tile::new(1, 1),
        padding: Tile::new(0, 0),
        dilation: Tile::new(1, 1),
        activation: Activation::int8_unconstrained(),
    };
    let quant_params = PerTensorQuantParams::new(2147483647, 0);

    let input_dims = Dims::new(1, 2, 2, 0); // 0 channels!
    let input = [];
    let filter_dims = Dims::new(1, 1, 1, 0);
    let kernel = [];
    let output_dims = Dims::new(1, 2, 2, 1);
    let mut output = [0i8; 4];

    let err = convolve_s8(
        &conv_params,
        &quant_params,
        &input_dims,
        &input,
        &filter_dims,
        &kernel,
        None,
        &output_dims,
        &mut output,
    );
    assert_eq!(err, Err(Error::ArgumentError));
}
