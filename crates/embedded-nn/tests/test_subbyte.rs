use embedded_nn::{
    Activation, ConvParams, Dims, FcParams, PerTensorQuantParams, Tile,
    subbyte::{convolve_s4, fully_connected_s4, pack_s4_pair, unpack_s4_pair},
};

#[test]
fn test_s4_nibble_packing_full_space() {
    for low in -8i8..=7i8 {
        for high in -8i8..=7i8 {
            let packed = pack_s4_pair(low, high);
            let (unpacked_low, unpacked_high) = unpack_s4_pair(packed);
            assert_eq!(unpacked_low, low, "Low nibble mismatch");
            assert_eq!(unpacked_high, high, "High nibble mismatch");
        }
    }
}

#[test]
fn test_fully_connected_s4_variations() {
    let fc_params = FcParams {
        input_offset: 2,
        filter_offset: 0,
        output_offset: -5,
        activation: Activation::new(-10, 10),
    };
    let quant_params = PerTensorQuantParams::new(1073741824, 0); // 0.5 Q31

    let input_dims = Dims::new(2, 1, 1, 4); // 2 batches, accum depth 4
    let input = [
        1i8, 2i8, 3i8, 4i8, // batch 0 -> +2 offset = [3, 4, 5, 6]
        -1i8, -2i8, 0i8, 1i8, // batch 1 -> +2 offset = [1, 0, 2, 3]
    ];
    let filter_dims = Dims::new(4, 1, 1, 2); // accum depth 4, out_c 2
    let packed_kernel = [
        pack_s4_pair(1i8, 2i8),
        pack_s4_pair(-3i8, 4i8), // out_c 0 weights: [1, 2, -3, 4]
        pack_s4_pair(-1i8, -2i8),
        pack_s4_pair(0i8, 1i8), // out_c 1 weights: [-1, -2, 0, 1]
    ];
    let bias = [10i32, 20i32];

    let output_dims = Dims::new(2, 1, 1, 2);
    let mut output = [0i8; 4];

    fully_connected_s4(
        &fc_params,
        &quant_params,
        &input_dims,
        &input,
        &filter_dims,
        &packed_kernel,
        Some(&bias),
        &output_dims,
        &mut output,
    )
    .unwrap();

    // Batch 0 out_c 0:
    // 10 + (3*1 + 4*2 + 5*-3 + 6*4) = 10 + (3 + 8 - 15 + 24) = 30
    // requantized 0.5 = 15 -> offset -5 = 10 -> clamped to 10
    assert_eq!(output[0], 10);

    // Batch 0 out_c 1:
    // 20 + (3*-1 + 4*-2 + 5*0 + 6*1) = 20 + (-3 - 8 + 0 + 6) = 15
    // requantized 0.5 = 8 -> offset -5 = 3
    assert_eq!(output[1], 3);
}

#[test]
fn test_convolve_s4_variations() {
    let conv_params = ConvParams {
        input_offset: 0,
        output_offset: 0,
        stride: Tile::new(1, 1),
        padding: Tile::new(0, 0),
        dilation: Tile::new(1, 1),
        activation: Activation::int8_unconstrained(),
    };
    let quant_params = PerTensorQuantParams::new(2147483647, 0); // 1.0

    let input_dims = Dims::new(1, 2, 2, 1);
    let input = [10i8, 20i8, 30i8, 40i8];
    let filter_dims = Dims::new(1, 2, 2, 1); // 1 out_c, 2x2 kernel, 1 in_c -> 4 spatial elements
    let packed_kernel = [
        pack_s4_pair(1i8, 1i8), // w0=1, w1=1
        pack_s4_pair(1i8, 1i8), // w2=1, w3=1
    ];
    let output_dims = Dims::new(1, 1, 1, 1);
    let mut output = [0i8; 1];

    convolve_s4(
        &conv_params,
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

    // 10 + 20 + 30 + 40 = 100
    assert_eq!(output[0], 100);
}
