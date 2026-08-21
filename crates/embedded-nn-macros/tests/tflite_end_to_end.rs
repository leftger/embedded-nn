//! End-to-end execution of constructed TFLite fixtures through the proc macro.
//!
//! Expected vectors are calculated from each model's simple affine arithmetic, independently of
//! the importer/code generator. See `embedded-nn-tflite/fixtures/constructed/README.md`.

mod sine {
    use embedded_nn_macros::embedded_nn_model;

    #[embedded_nn_model("../embedded-nn-tflite/fixtures/constructed/sine_fc_int8.tflite")]
    pub struct SineFc;
}

mod tinyconv {
    use embedded_nn_macros::embedded_nn_model;

    #[embedded_nn_model("../embedded-nn-tflite/fixtures/constructed/tinyconv_int8.tflite")]
    pub struct TinyConv;
}

mod uint8 {
    use embedded_nn_macros::embedded_nn_model;

    #[embedded_nn_model("../embedded-nn-tflite/fixtures/constructed/uint8_fc.tflite")]
    pub struct Uint8Fc;
}

mod add_transpose {
    use embedded_nn_macros::embedded_nn_model;

    #[embedded_nn_model("../embedded-nn-tflite/fixtures/constructed/add_transpose_int8.tflite")]
    pub struct AddTranspose;
}

#[test]
fn sine_fc_direct_tflite_macro_executes_int8_and_f32_boundaries() {
    use sine::SineFc;

    for quantized in [-127i8, -64, 0, 64, 127] {
        let mut arena = [0u8; SineFc::ARENA_SIZE];
        let output = SineFc::predict(&[quantized], &mut arena).unwrap();
        assert_eq!(output, [quantized]);

        let input_f32 = [quantized as f32 * SineFc::INPUT_SCALE];
        let mut quantized_input = [0i8; SineFc::INPUT_DIM];
        let mut arena = [0u8; SineFc::ARENA_SIZE];
        let mut output_f32 = [0.0f32; SineFc::OUTPUT_DIM];
        SineFc::predict_f32(
            &input_f32,
            &mut quantized_input,
            &mut arena,
            &mut output_f32,
        )
        .unwrap();
        assert_eq!(quantized_input, [quantized]);
        assert_eq!(output_f32, [quantized as f32 * SineFc::OUTPUT_SCALE]);
    }
}

#[test]
fn tinyconv_executes_conv_pool_reshape_fc_and_softmax_chain() {
    use tinyconv::TinyConv;

    let input = [0i8; TinyConv::INPUT_DIM];
    let mut arena = [0u8; TinyConv::ARENA_SIZE];
    let output = TinyConv::predict(&input, &mut arena).unwrap();

    // Zero real input produces four equal logits. Quantized softmax therefore assigns exactly
    // 64/256 probability to every class: 64 - 128 == -64.
    let mut standalone_softmax = [0i8; 4];
    embedded_nn::softmax_s8(
        &[2, 2, 2, 2],
        1,
        4,
        1073741824,
        20,
        -256,
        &mut standalone_softmax,
    )
    .unwrap();
    assert_eq!(standalone_softmax, [-64, -64, -64, -64]);
    assert_eq!(output, [-64, -64, -64, -64]);
}

#[test]
fn uint8_storage_is_rewritten_and_executes_through_s8_fc() {
    use uint8::Uint8Fc;

    let mut arena = [0u8; Uint8Fc::ARENA_SIZE];

    // Original UINT8 zero point 129 is rewritten to signed code 1.
    assert_eq!(Uint8Fc::INPUT_ZERO_POINT, 1);
    assert_eq!(Uint8Fc::OUTPUT_ZERO_POINT, 3);
    assert_eq!(Uint8Fc::predict(&[1, 1, 1, 1], &mut arena).unwrap(), [3, 3]);

    // Non-zero values exercise rewritten weights, including original UINT8 endpoints 0 and 255.
    assert_eq!(
        Uint8Fc::predict(&[127, 127, 127, 127], &mut arena).unwrap(),
        [-22, 54]
    );
}

#[test]
fn add_then_rank2_transpose_executes_with_two_graph_inputs() {
    use add_transpose::AddTranspose;

    // Inputs are flattened in graph-input order: left[2,3], then right[2,3].
    let input = [-3, -2, -1, 0, 1, 2, 7, 8, 9, 10, 11, 12];
    let mut arena = [0u8; AddTranspose::ARENA_SIZE];
    let output = AddTranspose::predict(&input, &mut arena).unwrap();

    // Affine sums before transpose are [-9,-3,3,9,15,21]. A [2,3] -> [3,2]
    // transpose reorders row-major indices as [0,3,1,4,2,5].
    assert_eq!(output, [-9, 9, -3, 15, 3, 21]);
}

#[test]
fn test_predict_tensor_2d_and_4d_macro_interface() {
    use embedded_nn::{Tensor2D, Tensor4D};
    use sine::SineFc;
    use tinyconv::TinyConv;

    // Test Tensor2D predict on SineFc
    let mut arena_sine = [0u8; SineFc::ARENA_SIZE];
    let input_2d =
        Tensor2D::<i8, 1, 1>::new([[64]], [SineFc::INPUT_SCALE], [SineFc::INPUT_ZERO_POINT]);
    let output_2d = SineFc::predict_tensor(&input_2d, &mut arena_sine).unwrap();
    assert_eq!(output_2d.data, [[64]]);

    // Test Tensor4D predict on TinyConv
    let mut arena_conv = [0u8; TinyConv::ARENA_SIZE];
    let input_4d =
        Tensor4D::<i8, 1, 49, 40, 1>::zero([TinyConv::INPUT_SCALE], [TinyConv::INPUT_ZERO_POINT]);
    let output_4d = TinyConv::predict_tensor(&input_4d, &mut arena_conv).unwrap();
    assert_eq!(output_4d.as_slice(), &[-64, -64, -64, -64]);
}
