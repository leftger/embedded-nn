use embedded_nn::{
    Activation,
    basic_math::{
        ElementwiseAddParams, ElementwiseMulParams, elementwise_add_s8, elementwise_add_s16,
        elementwise_mul_s8, elementwise_mul_s16, elementwise_sub_s8,
    },
};

#[test]
fn test_elementwise_add_s8_basic_and_clamping() {
    let input1 = [10i8, 20i8, 30i8, -50i8, 100i8];
    let input2 = [5i8, 15i8, 25i8, -60i8, 50i8];
    let mut output = [0i8; 5];

    let params = ElementwiseAddParams {
        input1_offset: 0,
        input1_mult: 1073741824, // 0.5 in Q31
        input1_shift: 0,
        input2_offset: 0,
        input2_mult: 1073741824, // 0.5 in Q31
        input2_shift: 0,
        left_shift: 0,
        output_offset: 0,
        output_mult: 1073741824, // 0.5 in Q31
        output_shift: 0,
        activation: Activation::new(-50, 50),
    };

    elementwise_add_s8(&input1, &input2, &mut output, &params).unwrap();

    // (10*0.5 + 5*0.5)*0.5 = 3.75 -> 4
    assert!((output[0] - 4).abs() <= 1);
    // (20*0.5 + 15*0.5)*0.5 = 8.75 -> 9
    assert!((output[1] - 9).abs() <= 1);
    // (-50*0.5 + -60*0.5)*0.5 = -27.5 -> -28 (clamped within [-50, 50])
    assert!(output[3] >= -50 && output[3] <= 50);
    // (100*0.5 + 50*0.5)*0.5 = 37.5 -> 38
    assert!((output[4] - 38).abs() <= 1);
}

#[test]
fn test_elementwise_add_s8_with_offsets_and_left_shift() {
    let input1 = [-10i8, 0i8, 10i8];
    let input2 = [-5i8, 5i8, 15i8];
    let mut output = [0i8; 3];

    let params = ElementwiseAddParams {
        input1_offset: 10,
        input1_mult: 1073741824,
        input1_shift: 0,
        input2_offset: 5,
        input2_mult: 1073741824,
        input2_shift: 0,
        left_shift: 1,
        output_offset: -10,
        output_mult: 1073741824,
        output_shift: 0,
        activation: Activation::int8_unconstrained(),
    };

    elementwise_add_s8(&input1, &input2, &mut output, &params).unwrap();
    // Input 1 shifted: [0, 10, 20] << 1 = [0, 20, 40]
    // Requantized mult 0.5: [0, 10, 20]
    // Input 2 shifted: [0, 10, 20] << 1 = [0, 20, 40]
    // Requantized mult 0.5: [0, 10, 20]
    // Sum: [0, 20, 40]
    // Requantized output mult 0.5: [0, 10, 20]
    // Output offset -10: [-10, 0, 10]
    assert!((output[0] - (-10)).abs() <= 1);
    assert!((output[1] - 0).abs() <= 1);
    assert!((output[2] - 10).abs() <= 1);
}

#[test]
fn test_elementwise_sub_s8_comprehensive() {
    let input1 = [30i8, 20i8, 10i8];
    let input2 = [10i8, 20i8, 30i8];
    let mut output = [0i8; 3];

    let params = ElementwiseAddParams {
        input1_offset: 0,
        input1_mult: 2147483647, // 1.0
        input1_shift: 0,
        input2_offset: 0,
        input2_mult: 2147483647, // 1.0
        input2_shift: 0,
        left_shift: 0,
        output_offset: 0,
        output_mult: 2147483647, // 1.0
        output_shift: 0,
        activation: Activation::int8_unconstrained(),
    };

    elementwise_sub_s8(&input1, &input2, &mut output, &params).unwrap();
    assert_eq!(output, [20, 0, -20]);
}

#[test]
fn test_elementwise_mul_s8_comprehensive() {
    let input1 = [2i8, -4i8, 8i8];
    let input2 = [3i8, 5i8, -2i8];
    let mut output = [0i8; 3];

    let params = ElementwiseMulParams {
        input1_offset: 0,
        input2_offset: 0,
        output_offset: 0,
        output_mult: 1073741824, // 0.5
        output_shift: 0,
        activation: Activation::int8_unconstrained(),
    };

    elementwise_mul_s8(&input1, &input2, &mut output, &params).unwrap();
    // 2*3 = 6 -> requantized 0.5 = 3
    // -4*5 = -20 -> requantized 0.5 = -10
    // 8*-2 = -16 -> requantized 0.5 = -8
    assert_eq!(output, [3, -10, -8]);
}

#[test]
fn test_elementwise_add_s16_comprehensive() {
    let input1 = [1000i16, 2000i16, -3000i16];
    let input2 = [500i16, -1000i16, 1500i16];
    let mut output = [0i16; 3];

    elementwise_add_s16(
        &input1,
        &input2,
        &mut output,
        1073741824, // mult1 = 0.5
        0,          // shift1
        1073741824, // mult2 = 0.5
        0,          // shift2
        2147483647, // output_mult = 1.0
        0,          // output_shift
        Activation::int16_unconstrained(),
    )
    .unwrap();

    // (1000*0.5 + 500*0.5) * 1.0 = 750
    // (2000*0.5 + -1000*0.5) * 1.0 = 500
    // (-3000*0.5 + 1500*0.5) * 1.0 = -750
    assert_eq!(output, [750, 500, -750]);
}

#[test]
fn test_elementwise_mul_s16_comprehensive() {
    let input1 = [100i16, -200i16, 300i16];
    let input2 = [400i16, 500i16, -600i16];
    let mut output = [0i16; 3];

    elementwise_mul_s16(
        &input1,
        &input2,
        &mut output,
        2147483647, // output_mult = 1.0
        0,
        Activation::int16_unconstrained(),
    )
    .unwrap();

    // 100 * 400 >> 15 = 1
    // -200 * 500 >> 15 = -3
    // 300 * -600 >> 15 = -5
    assert!((output[0] - 1).abs() <= 1);
    assert!((output[1] - (-3)).abs() <= 1);
    assert!((output[2] - (-5)).abs() <= 1);
}

#[test]
fn test_elementwise_mismatched_lengths() {
    let input1 = [10i8, 20i8, 30i8, 40i8];
    let input2 = [5i8, 15i8];
    let mut output = [0i8; 4];

    let params = ElementwiseAddParams {
        input1_offset: 0,
        input1_mult: 2147483647,
        input1_shift: 0,
        input2_offset: 0,
        input2_mult: 2147483647,
        input2_shift: 0,
        left_shift: 0,
        output_offset: 0,
        output_mult: 2147483647,
        output_shift: 0,
        activation: Activation::int8_unconstrained(),
    };

    // Should compute up to min length (2 elements) without panicking
    elementwise_add_s8(&input1, &input2, &mut output, &params).unwrap();
    assert_eq!(output[0], 15);
    assert_eq!(output[1], 35);
}
