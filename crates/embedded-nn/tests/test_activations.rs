use embedded_nn::{
    Activation,
    activations::{
        SIGMOID_TABLE_UINT16, activation_s8, activation_s16, leaky_relu_s8, relu_s8, relu_s16,
        relu6_s8, sigmoid_s8, sigmoid_s16, tanh_s8, tanh_s16,
    },
    float_ops::{relu_f32, relu6_f32},
};

#[test]
fn test_relu_s8_comprehensive() {
    let mut data = [i8::MIN, -128, -50, -1, 0, 1, 50, 127, i8::MAX];
    relu_s8(&mut data);
    assert_eq!(data, [0, 0, 0, 0, 0, 1, 50, 127, 127]);

    let mut empty: [i8; 0] = [];
    relu_s8(&mut empty);
}

#[test]
fn test_relu6_s8_comprehensive() {
    let mut data = [-128i8, -10, 0, 1, 5, 6, 7, 50, 127];
    relu6_s8(&mut data);
    assert_eq!(data, [0, 0, 0, 1, 5, 6, 6, 6, 6]);

    let mut empty: [i8; 0] = [];
    relu6_s8(&mut empty);
}

#[test]
fn test_activation_clamping_s8_comprehensive() {
    let mut data = [-100i8, -50, -10, 0, 5, 10, 50, 100];
    let act = Activation::new(-10, 10);
    activation_s8(&mut data, act);
    assert_eq!(data, [-10, -10, -10, 0, 5, 10, 10, 10]);

    let unconstrained = Activation::int8_unconstrained();
    let mut data2 = [-128i8, 0, 127];
    activation_s8(&mut data2, unconstrained);
    assert_eq!(data2, [-128, 0, 127]);
}

#[test]
fn test_relu_s16_comprehensive() {
    let mut data = [i16::MIN, -32768, -1000, -1, 0, 1, 1000, 32767, i16::MAX];
    relu_s16(&mut data);
    assert_eq!(data, [0, 0, 0, 0, 0, 1, 1000, 32767, 32767]);

    let mut empty: [i16; 0] = [];
    relu_s16(&mut empty);
}

#[test]
fn test_activation_clamping_s16_comprehensive() {
    let mut data = [-20000i16, -1000, 0, 1000, 20000];
    let act = Activation::new(-500, 500);
    activation_s16(&mut data, act);
    assert_eq!(data, [-500, -500, 0, 500, 500]);

    let unconstrained = Activation::int16_unconstrained();
    let mut data2 = [-32768i16, 0, 32767];
    activation_s16(&mut data2, unconstrained);
    assert_eq!(data2, [-32768, 0, 32767]);
}

#[test]
fn test_leaky_relu_s8_offsets_and_alpha() {
    let input = [-20i8, -10i8, 0i8, 10i8, 20i8];
    let mut output = [0i8; 5];

    // Alpha multiplier = 0.5 (Q31 mult = 1073741824, shift = 0)
    // input_offset = 0, output_offset = 0
    leaky_relu_s8(&input, &mut output, 1073741824, 0, 0, 0);
    assert_eq!(output[0], -10);
    assert_eq!(output[1], -5);
    assert_eq!(output[2], 0);
    assert_eq!(output[3], 10);
    assert_eq!(output[4], 20);

    // Test with offsets
    let mut output_off = [0i8; 5];
    // input_offset = 10 -> shifted inputs: [-10, 0, 10, 20, 30]
    // output_offset = -5
    leaky_relu_s8(&input, &mut output_off, 1073741824, 0, 10, -5);
    // -10 * 0.5 - 5 = -10
    // 0 - 5 = -5
    // 10 - 5 = 5
    // 20 - 5 = 15
    // 30 - 5 = 25
    assert_eq!(output_off, [-10, -5, 5, 15, 25]);
}

#[test]
fn test_sigmoid_s8_full_range() {
    let input = [-128i8, -90, -40, -10, 0, 10, 40, 90, 127];
    let mut output = [0i8; 9];
    sigmoid_s8(&input, &mut output);

    // Sigmoid must be strictly monotonic
    for i in 0..output.len() - 1 {
        assert!(
            output[i] <= output[i + 1],
            "Failed monotonicity at index {}",
            i
        );
    }
    // Around 0 input, output should be around 0 in centered s8
    assert!((output[4] as i32).abs() <= 5);
    // Extreme values should saturate near -128 and 127
    assert!(output[0] < -100);
    assert!(output[8] > 100);
}

#[test]
fn test_tanh_s8_full_range() {
    let input = [-128i8, -80, -40, 0, 40, 80, 127];
    let mut output = [0i8; 7];
    tanh_s8(&input, &mut output);

    // Tanh must be symmetric: tanh(-x) approx -tanh(x)
    assert_eq!(output[3], 0); // tanh(0) == 0
    assert!((output[2] as i32 + output[4] as i32).abs() <= 2);
    assert!((output[1] as i32 + output[5] as i32).abs() <= 2);
    assert!(output[0] <= output[1]);
    assert!(output[5] <= output[6]);
}

#[test]
fn test_sigmoid_s16_shifts() {
    let input = [-1000i16, -100, 0, 100, 1000];

    for shift in [-3, -1, 0, 1, 3] {
        let mut output = [0i16; 5];
        sigmoid_s16(&input, &mut output, shift);
        // Sigmoid must be monotonic
        for i in 0..output.len() - 1 {
            assert!(output[i] <= output[i + 1]);
        }
    }
}

#[test]
fn test_tanh_s16_shifts() {
    let input = [-1000i16, -100, 0, 100, 1000];

    for shift in [-3, -1, 0, 1, 3] {
        let mut output = [0i16; 5];
        tanh_s16(&input, &mut output, shift);
        println!("shift={}: {:?}", shift, output);
        // Tanh must be monotonic
        for i in 0..output.len() - 1 {
            assert!(
                output[i] <= output[i + 1],
                "Failed at shift {} index {}: {:?}",
                shift,
                i,
                output
            );
        }
    }
}

#[test]
fn test_sigmoid_table_properties() {
    assert_eq!(SIGMOID_TABLE_UINT16.len(), 256);
    assert_eq!(SIGMOID_TABLE_UINT16[0], 32768); // Sigmoid(0) scaled Q0.16 is 0.5 * 65536 = 32768
    assert!(SIGMOID_TABLE_UINT16[255] >= 65530);
}

#[test]
fn test_f32_activations_edge_cases() {
    let mut data = [-100.0f32, -0.001, 0.0, 0.001, 5.999, 6.0, 6.001, 100.0];
    relu_f32(&mut data);
    assert_eq!(data, [0.0, 0.0, 0.0, 0.001, 5.999, 6.0, 6.001, 100.0]);

    let mut data6 = [-100.0f32, -0.001, 0.0, 0.001, 5.999, 6.0, 6.001, 100.0];
    relu6_f32(&mut data6);
    assert_eq!(data6, [0.0, 0.0, 0.0, 0.001, 5.999, 6.0, 6.0, 6.0]);
}
