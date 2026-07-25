//! Fixed-point math and support helper functions for `embedded-nn`.

/// Clamps a scalar `x` between `min` and `max`.
#[inline]
pub const fn clamp(x: i32, min: i32, max: i32) -> i32 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

/// Doubling high multiply without saturation.
///
/// Computes `(2 * m1 * m2 + 0x4000_0000) >> 31`.
/// Equivalent to Arm CMSIS-NN `arm_nn_doubling_high_mult_no_sat`.
#[inline]
pub fn doubling_high_mult_no_sat(m1: i32, m2: i32) -> i32 {
    let mult = (m1 as i64) * (m2 as i64) + (1i64 << 30);
    (mult >> 31) as i32
}

/// Rounding divide by power of two (midpoint away from zero).
///
/// Equivalent to Arm CMSIS-NN `arm_nn_divide_by_power_of_two`.
#[inline]
pub fn divide_by_power_of_two(dividend: i32, exponent: i32) -> i32 {
    if exponent <= 0 {
        return dividend;
    }
    if exponent >= 31 {
        return 0;
    }

    let remainder_mask = (1i32 << exponent) - 1;
    let remainder = remainder_mask & dividend;

    let mut result = dividend >> exponent;
    let mut threshold = remainder_mask >> 1;
    if dividend < 0 {
        threshold += 1;
    }
    if remainder > threshold {
        result += 1;
    }
    result
}

/// Requantizes a 32-bit integer value given a multiplier and shift.
///
/// Equivalent to Arm CMSIS-NN `arm_nn_requantize`.
#[inline]
pub fn requantize(val: i32, multiplier: i32, shift: i32) -> i32 {
    if shift >= 0 {
        let val_shifted = val.wrapping_shl(shift as u32);
        doubling_high_mult_no_sat(val_shifted, multiplier)
    } else {
        let right_shift = -shift;
        let mult = doubling_high_mult_no_sat(val, multiplier);
        divide_by_power_of_two(mult, right_shift)
    }
}

/// Requantizes a 64-bit accumulator value.
///
/// Equivalent to Arm CMSIS-NN `arm_nn_requantize_s64`.
#[inline]
pub fn requantize_s64(val: i64, reduced_multiplier: i32, shift: i32) -> i32 {
    let new_val = val * (reduced_multiplier as i64);
    let shift_amt = 14 - shift;
    let mut result = (new_val >> (shift_amt - 1)) as i32;
    result = (result + 1) >> 1;
    result
}

/// Packs four 8-bit signed integers into a single 32-bit integer.
#[inline]
pub const fn pack_s8x4_32x1(v0: i8, v1: i8, v2: i8, v3: i8) -> i32 {
    ((v0 as u8 as i32) & 0xFF)
        | (((v1 as u8 as i32) << 8) & 0xFF00)
        | (((v2 as u8 as i32) << 16) & 0xFF0000)
        | (((v3 as u8 as i32) << 24) as i32)
}

/// Packs two 16-bit signed integers into a single 32-bit integer.
#[inline]
pub const fn pack_q15x2_32x1(v0: i16, v1: i16) -> i32 {
    ((v0 as u16 as i32) & 0xFFFF) | ((v1 as u16 as i32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(10, 0, 5), 5);
        assert_eq!(clamp(-5, 0, 5), 0);
        assert_eq!(clamp(3, 0, 5), 3);
    }

    #[test]
    fn test_doubling_high_mult() {
        let res = doubling_high_mult_no_sat(1073741824, 1073741824); // 0.5 * 0.5 in Q31
        assert!((res - 536870912).abs() <= 1);
    }

    #[test]
    fn test_divide_by_power_of_two() {
        assert_eq!(divide_by_power_of_two(16, 2), 4);
        assert_eq!(divide_by_power_of_two(18, 2), 5); // 18 / 4 = 4.5 -> rounds to 5
    }

    #[test]
    fn test_requantize() {
        let val = 1000;
        let mult = 1073741824; // 0.5
        let shift = -1;
        let res = requantize(val, mult, shift);
        assert_eq!(res, 250);
    }
}
