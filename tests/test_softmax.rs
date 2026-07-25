use embedded_nn::{
    float_ops::softmax_f32,
    softmax::{exp_on_negative_values, one_over_one_plus_x_for_x_in_0_1, softmax_s16, softmax_s8},
};

#[test]
fn test_softmax_s8_multi_row_and_monotonicity() {
    let input = [
        10i8, 20i8, 30i8, 40i8, // row 0
        40i8, 30i8, 20i8, 10i8, // row 1
    ];
    let mut output = [0i8; 8];

    softmax_s8(&input, 2, 4, 1073741824, 20, -256, &mut output).unwrap();

    // Row 0: monotonic increasing
    assert!(output[3] > output[2]);
    assert!(output[2] > output[1]);
    assert!(output[1] > output[0]);

    // Row 1: monotonic decreasing
    assert!(output[4] > output[5]);
    assert!(output[5] > output[6]);
    assert!(output[6] > output[7]);
}

#[test]
fn test_softmax_s8_identical_values() {
    let input = [20i8, 20i8, 20i8, 20i8];
    let mut output = [0i8; 4];

    softmax_s8(&input, 1, 4, 1073741824, 20, -256, &mut output).unwrap();

    // Equal inputs should produce equal outputs
    assert_eq!(output[0], output[1]);
    assert_eq!(output[1], output[2]);
    assert_eq!(output[2], output[3]);
}

#[test]
fn test_softmax_s16_multi_row_and_monotonicity() {
    let input = [
        100i16, 200i16, 300i16, // row 0
        300i16, 200i16, 100i16, // row 1
    ];
    let mut output = [0i16; 6];

    softmax_s16(&input, 2, 3, 1073741824, 20, -256, &mut output).unwrap();

    assert!(output[2] >= output[1]);
    assert!(output[1] >= output[0]);
    assert!(output[3] >= output[4]);
    assert!(output[4] >= output[5]);
}

#[test]
fn test_softmax_f32_properties() {
    let input = [
        1.0f32, 2.0f32, 3.0f32, 4.0f32, // row 0
        10.0f32, 10.0f32, 10.0f32, 10.0f32, // row 1
    ];
    let mut output = [0.0f32; 8];

    softmax_f32(&input, 2, 4, &mut output).unwrap();

    // Row 0 sum to 1.0
    let sum0: f32 = output[0..4].iter().sum();
    assert!((sum0 - 1.0).abs() < 1e-4);
    assert!(output[3] > output[2]);
    assert!(output[2] > output[1]);
    assert!(output[1] > output[0]);

    // Row 1 uniform probability 0.25 each
    let sum1: f32 = output[4..8].iter().sum();
    assert!((sum1 - 1.0).abs() < 1e-4);
    for val in &output[4..8] {
        assert!((val - 0.25).abs() < 1e-4);
    }
}

#[test]
fn test_softmax_f32_translation_invariance() {
    let input1 = [1.0f32, 2.0f32, 3.0f32];
    let input2 = [101.0f32, 102.0f32, 103.0f32]; // input1 + 100
    let mut out1 = [0.0f32; 3];
    let mut out2 = [0.0f32; 3];

    softmax_f32(&input1, 1, 3, &mut out1).unwrap();
    softmax_f32(&input2, 1, 3, &mut out2).unwrap();

    for i in 0..3 {
        assert!((out1[i] - out2[i]).abs() < 1e-5);
    }
}

#[test]
fn test_fixed_point_exp_and_reciprocal_helpers() {
    let exp_zero = exp_on_negative_values(0);
    assert_eq!(exp_zero, i32::MAX);

    let exp_neg = exp_on_negative_values(-100000);
    assert!(exp_neg > 0);

    let recip_val = one_over_one_plus_x_for_x_in_0_1(1000000);
    assert!(recip_val != 0);
}
