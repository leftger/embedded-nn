use embedded_nn::{
    pack_q15x2_32x1, pack_s8x4_32x1, requantize_s64,
    simd::{vec_dot_s16, vec_dot_s8},
    support::{clamp, divide_by_power_of_two, doubling_high_mult_no_sat, requantize},
    Activation, Context, Dims, Error, PerChannelQuantParams, PerTensorQuantParams, QuantParams,
    Tile,
};

#[test]
fn test_clamp_boundaries_and_ordering() {
    assert_eq!(clamp(100, -10, 10), 10);
    assert_eq!(clamp(-100, -10, 10), -10);
    assert_eq!(clamp(0, -10, 10), 0);
    assert_eq!(clamp(-10, -10, 10), -10);
    assert_eq!(clamp(10, -10, 10), 10);
}

#[test]
fn test_doubling_high_mult_no_sat_cases() {
    // Q31 multiplication
    let res1 = doubling_high_mult_no_sat(1073741824, 1073741824); // 0.5 * 0.5 = 0.25 -> 536870912
    assert!((res1 - 536870912).abs() <= 1);

    let res2 = doubling_high_mult_no_sat(2147483647, 1073741824); // ~1.0 * 0.5 = ~0.5 -> 1073741823
    assert!((res2 - 1073741823).abs() <= 2);

    let res_neg = doubling_high_mult_no_sat(-1073741824, 1073741824); // -0.5 * 0.5 = -0.25
    assert!((res_neg - (-536870912)).abs() <= 1);
}

#[test]
fn test_divide_by_power_of_two_rounding_and_edge_cases() {
    assert_eq!(divide_by_power_of_two(100, 0), 100);
    assert_eq!(divide_by_power_of_two(100, -1), 100);
    assert_eq!(divide_by_power_of_two(100, 31), 0);
    assert_eq!(divide_by_power_of_two(100, 35), 0);

    // Positive dividends
    assert_eq!(divide_by_power_of_two(16, 2), 4);
    assert_eq!(divide_by_power_of_two(18, 2), 5); // 4.5 -> rounds to 5

    // Negative dividends rounding
    assert_eq!(divide_by_power_of_two(-16, 2), -4);
    assert_eq!(divide_by_power_of_two(-18, 2), -5); // -4.5 -> rounds away from zero (-5)
}

#[test]
fn test_requantize_positive_and_negative_shifts() {
    let val = 1000;
    let mult = 1073741824; // 0.5

    // Right shift (shift < 0)
    let res_right = requantize(val, mult, -1); // 1000 * 0.5 / 2 = 250
    assert_eq!(res_right, 250);

    // Left shift (shift >= 0)
    let res_left = requantize(val, mult, 1); // (1000 << 1) * 0.5 = 1000
    assert_eq!(res_left, 1000);
}

#[test]
fn test_requantize_s64_math() {
    let val: i64 = 1000000;
    let reduced_mult = 16384;
    let shift = 0;
    let res = requantize_s64(val, reduced_mult, shift);
    assert!(res > 0);
}

#[test]
fn test_packing_helpers() {
    let packed_s8 = pack_s8x4_32x1(-1i8, 0i8, 127i8, -128i8);
    assert_eq!(packed_s8 as u32 & 0xFF, 0xFF);
    assert_eq!((packed_s8 as u32 >> 8) & 0xFF, 0x00);
    assert_eq!((packed_s8 as u32 >> 16) & 0xFF, 0x7F);
    assert_eq!((packed_s8 as u32 >> 24) & 0xFF, 0x80);

    let packed_q15 = pack_q15x2_32x1(-1000i16, 2000i16);
    assert_eq!((packed_q15 & 0xFFFF) as i16, -1000);
    assert_eq!((packed_q15 >> 16) as i16, 2000);
}

#[test]
fn test_vec_dot_s8_length_variations() {
    // Test exact chunk boundaries: lengths 0, 1, 3, 7, 8, 9, 16, 17
    let lhs = [1i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
    let rhs = [1i8; 17];

    assert_eq!(vec_dot_s8(&lhs[..0], &rhs[..0], 0), 0);
    assert_eq!(vec_dot_s8(&lhs[..1], &rhs[..1], 0), 1);
    assert_eq!(vec_dot_s8(&lhs[..3], &rhs[..3], 0), 6);
    assert_eq!(vec_dot_s8(&lhs[..7], &rhs[..7], 0), 28);
    assert_eq!(vec_dot_s8(&lhs[..8], &rhs[..8], 0), 36);
    assert_eq!(vec_dot_s8(&lhs[..9], &rhs[..9], 0), 45);
    assert_eq!(vec_dot_s8(&lhs[..16], &rhs[..16], 0), 136);
    assert_eq!(vec_dot_s8(&lhs[..17], &rhs[..17], 0), 153);

    // Test with lhs_offset = 2
    // Sum for length 8 with offset 2: 36 + 8*2 = 52
    assert_eq!(vec_dot_s8(&lhs[..8], &rhs[..8], 2), 52);
}

#[test]
fn test_vec_dot_s16_length_variations() {
    let lhs = [10i16, 20, 30, 40, 50, 60, 70];
    let rhs = [1i16; 7];

    assert_eq!(vec_dot_s16(&lhs[..0], &rhs[..0]), 0);
    assert_eq!(vec_dot_s16(&lhs[..1], &rhs[..1]), 10);
    assert_eq!(vec_dot_s16(&lhs[..4], &rhs[..4]), 100);
    assert_eq!(vec_dot_s16(&lhs[..7], &rhs[..7]), 280);
}

#[test]
fn test_types_and_error_display() {
    let err1 = Error::ArgumentError;
    let err2 = Error::NoImplementation;
    let err3 = Error::Failure;

    assert_eq!(format!("{}", err1), "Invalid or incompatible arguments");
    assert_eq!(format!("{}", err2), "No implementation available");
    assert_eq!(format!("{}", err3), "Operation failure");

    let dims = Dims::new(2, 3, 4, 5);
    assert_eq!(dims.total_size(), 120);

    let tile = Tile::new(3, 3);
    assert_eq!(tile.w, 3);
    assert_eq!(tile.h, 3);

    let pt = PerTensorQuantParams::new(100, 2);
    let mults = [100, 200];
    let shifts = [1, 2];
    let pc = PerChannelQuantParams::new(&mults, &shifts);

    let q1 = QuantParams::PerTensor(pt);
    let q2 = QuantParams::PerChannel(pc);

    match q1 {
        QuantParams::PerTensor(p) => assert_eq!(p.multiplier, 100),
        _ => panic!(),
    }
    match q2 {
        QuantParams::PerChannel(p) => assert_eq!(p.multiplier.len(), 2),
        _ => panic!(),
    }

    let mut buf = [0u8; 128];
    let ctx_empty = Context::empty();
    assert!(ctx_empty.buf.is_none());

    let ctx_new = Context::new(&mut buf);
    assert!(ctx_new.buf.is_some());

    let act8 = Activation::int8_unconstrained();
    assert_eq!(act8.min, -128);
    assert_eq!(act8.max, 127);

    let act16 = Activation::int16_unconstrained();
    assert_eq!(act16.min, -32768);
    assert_eq!(act16.max, 32767);
}
