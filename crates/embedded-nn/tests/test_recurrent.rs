use embedded_nn::{
    Activation, PerTensorQuantParams,
    recurrent::{LstmGateParams, lstm_step_s8_s16, lstm_step_s16, svdf_s8, svdf_state_s16_s8},
};

#[test]
fn test_svdf_s8_rank1_execution() {
    let input = [10i8, 20i8]; // 2 input channels
    let weights_feature = [1i8, 2i8, 3i8, 4i8]; // 2 filters, rank 1, 2 in_c
    let weights_time = [1i8, 1i8]; // 2 filters, rank 1, memory_size 1
    let bias = [0i32, 0i32];

    let mut state = [0i8; 2]; // 2 filters * rank 1 * memory_size 1 = 2
    let mut output = [0i8; 2];

    let input_quant = PerTensorQuantParams::new(1073741824, 0); // 0.5
    let output_quant = PerTensorQuantParams::new(1073741824, 0); // 0.5
    let act = Activation::int8_unconstrained();

    // Step 1
    svdf_s8(
        0, // input_offset
        0, // output_offset
        1, // rank
        &input,
        &mut state,
        &weights_feature,
        &weights_time,
        Some(&bias),
        &input_quant,
        &output_quant,
        &act,
        &mut output,
    )
    .unwrap();

    // Feature dot:
    // Filter 0: (10*1 + 20*2) * 0.5 = 25
    // Filter 1: (10*3 + 20*4) * 0.5 = 55
    // Time dot with state:
    // Filter 0: 25 * 0.5 = 12.5 -> rounds to 13
    // Filter 1: 55 * 0.5 = 27.5 -> rounds to 28
    assert_eq!(output[0], 13);
    assert_eq!(output[1], 28);
}

#[test]
fn test_svdf_state_s16_s8_execution() {
    let input = [5i8, 10i8];
    let weights_feature = [2i8, 2i8];
    let weights_time = [1i16];
    let bias = [0i32];

    let mut state = [0i16; 1];
    let mut output = [0i8; 1];

    let input_quant = PerTensorQuantParams::new(2147483647, 0); // 1.0
    let output_quant = PerTensorQuantParams::new(2147483647, 0); // 1.0
    let act = Activation::int8_unconstrained();

    svdf_state_s16_s8(
        0, // input_offset
        0, // output_offset
        1, // rank
        &input,
        &mut state,
        &weights_feature,
        &weights_time,
        Some(&bias),
        &input_quant,
        &output_quant,
        &act,
        &mut output,
    )
    .unwrap();

    // Feature dot: 5*2 + 10*2 = 30 -> written to state[0]
    assert_eq!(state[0], 30);
    // Time dot: 30 * 1 = 30 -> >> 15 = 0
    assert_eq!(output[0], 0);
}

#[test]
fn test_lstm_step_s8_s16_execution() {
    let input = [10i8, 20i8]; // input_dim = 2
    let mut hidden_state = [0i8; 1]; // hidden_dim = 1
    let mut cell_state = [0i16; 1];

    // Weights: 4 * hidden_dim x input_dim = 4 * 1 * 2 = 8
    let weight_input = [1i8, 1i8, 1i8, 1i8, 1i8, 1i8, 1i8, 1i8];
    // Weights hidden: 4 * hidden_dim x hidden_dim = 4 * 1 * 1 = 4
    let weight_hidden = [0i8, 0i8, 0i8, 0i8];
    // Bias: 4 * hidden_dim = 4
    let bias = [0i32, 0i32, 0i32, 0i32];

    let gate_params = LstmGateParams {
        input_offset: 0,
        hidden_offset: 0,
        multiplier: 1073741824,
        shift: 0,
    };
    let output_quant = PerTensorQuantParams::new(1073741824, 0);

    lstm_step_s8_s16(
        &input,
        &mut hidden_state,
        &mut cell_state,
        &weight_input,
        &weight_hidden,
        &bias,
        &gate_params,
        32767,
        &output_quant,
        0,
        &Activation::int8_unconstrained(),
    )
    .unwrap();
}

#[test]
fn test_lstm_step_s16_execution() {
    let input = [100i16, 200i16];
    let mut hidden_state = [0i16; 1];
    let mut cell_state = [0i16; 1];

    let weight_input = [1i8, 1i8, 1i8, 1i8, 1i8, 1i8, 1i8, 1i8];
    let weight_hidden = [0i8, 0i8, 0i8, 0i8];
    let bias = [0i64, 0i64, 0i64, 0i64];

    let gate_params = LstmGateParams {
        input_offset: 0,
        hidden_offset: 0,
        multiplier: 1073741824,
        shift: 0,
    };
    let output_quant = PerTensorQuantParams::new(1073741824, 0);

    lstm_step_s16(
        &input,
        &mut hidden_state,
        &mut cell_state,
        &weight_input,
        &weight_hidden,
        &bias,
        &gate_params,
        32767,
        &output_quant,
        &Activation::int16_unconstrained(),
    )
    .unwrap();
}
