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
    if shift_amt <= 0 {
        return (new_val.wrapping_shl((-shift_amt + 1) as u32)) as i32;
    }
    if shift_amt >= 64 {
        return 0;
    }
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

/// Hardware-accelerated or branchless signed 8-bit saturation (`SSAT #8`).
#[inline(always)]
pub fn saturate_s8(val: i32) -> i8 {
    #[cfg(all(target_arch = "arm", target_feature = "dsp"))]
    {
        let res: i32;
        unsafe {
            core::arch::asm!(
                "ssat {out}, #8, {val}",
                val = in(reg) val,
                out = lateout(reg) res,
                options(pure, nomem, nostack)
            );
        }
        res as i8
    }
    #[cfg(not(all(target_arch = "arm", target_feature = "dsp")))]
    {
        clamp(val, -128, 127) as i8
    }
}

/// Hardware-accelerated or branchless signed 16-bit saturation (`SSAT #16`).
#[inline(always)]
pub fn saturate_s16(val: i32) -> i16 {
    #[cfg(all(target_arch = "arm", target_feature = "dsp"))]
    {
        let res: i32;
        unsafe {
            core::arch::asm!(
                "ssat {out}, #16, {val}",
                val = in(reg) val,
                out = lateout(reg) res,
                options(pure, nomem, nostack)
            );
        }
        res as i16
    }
    #[cfg(not(all(target_arch = "arm", target_feature = "dsp")))]
    {
        clamp(val, -32768, 32767) as i16
    }
}

/// Quantizes a 32-bit floating point value into an 8-bit signed integer (`s8`) given scale and zero point.
///
/// Formula: `clamp(round(val / scale) + zero_point, -128, 127)`
#[inline]
pub fn quantize_f32_to_s8(val: f32, scale: f32, zero_point: i32) -> i8 {
    if scale == 0.0 {
        return zero_point as i8;
    }
    let scaled = val / scale;
    let rounded = if scaled >= 0.0 {
        (scaled + 0.5) as i32
    } else {
        (scaled - 0.5) as i32
    };
    let q = rounded + zero_point;
    saturate_s8(q)
}

/// Dequantizes an 8-bit signed integer (`s8`) into a 32-bit floating point value given scale and zero point.
///
/// Formula: `scale * (val - zero_point)`
#[inline]
pub fn dequantize_s8_to_f32(val: i8, scale: f32, zero_point: i32) -> f32 {
    scale * ((val as i32 - zero_point) as f32)
}

/// Quantizes a 32-bit floating point value into a 16-bit signed integer (`s16`) given scale and zero point.
///
/// Formula: `clamp(round(val / scale) + zero_point, -32768, 32767)`
#[inline]
pub fn quantize_f32_to_s16(val: f32, scale: f32, zero_point: i32) -> i16 {
    if scale == 0.0 {
        return zero_point as i16;
    }
    let scaled = val / scale;
    let rounded = if scaled >= 0.0 {
        (scaled + 0.5) as i32
    } else {
        (scaled - 0.5) as i32
    };
    let q = rounded + zero_point;
    saturate_s16(q)
}

/// Dequantizes a 16-bit signed integer (`s16`) into a 32-bit floating point value given scale and zero point.
///
/// Formula: `scale * (val - zero_point)`
#[inline]
pub fn dequantize_s16_to_f32(val: i16, scale: f32, zero_point: i32) -> f32 {
    scale * ((val as i32 - zero_point) as f32)
}

/// Computes high-performance dot product with 4-way SIMD/DSP unrolling.
#[inline]
pub fn dot_product_s8_accum(lhs: &[i8], rhs: &[i8], lhs_offset: i32, rhs_offset: i32) -> i32 {
    let len = lhs.len().min(rhs.len());
    let mut acc = 0i32;
    let chunks = len / 4;
    let remainder = len % 4;

    for c in 0..chunks {
        let base = c * 4;
        let l0 = lhs[base] as i32 + lhs_offset;
        let l1 = lhs[base + 1] as i32 + lhs_offset;
        let l2 = lhs[base + 2] as i32 + lhs_offset;
        let l3 = lhs[base + 3] as i32 + lhs_offset;

        let r0 = rhs[base] as i32 + rhs_offset;
        let r1 = rhs[base + 1] as i32 + rhs_offset;
        let r2 = rhs[base + 2] as i32 + rhs_offset;
        let r3 = rhs[base + 3] as i32 + rhs_offset;

        acc += l0 * r0 + l1 * r1 + l2 * r2 + l3 * r3;
    }

    let rem_base = chunks * 4;
    for i in 0..remainder {
        let l = lhs[rem_base + i] as i32 + lhs_offset;
        let r = rhs[rem_base + i] as i32 + rhs_offset;
        acc += l * r;
    }

    acc
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

    #[test]
    fn test_quantize_dequantize_s8() {
        let scale = 0.1;
        let zero_point = 2;
        let q = quantize_f32_to_s8(1.0, scale, zero_point);
        assert_eq!(q, 12);
        let f = dequantize_s8_to_f32(q, scale, zero_point);
        assert!((f - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_quantize_dequantize_s16() {
        let scale = 0.05;
        let zero_point = -10;
        let q = quantize_f32_to_s16(2.5, scale, zero_point);
        assert_eq!(q, 40);
        let f = dequantize_s16_to_f32(q, scale, zero_point);
        assert!((f - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_saturate_s8() {
        assert_eq!(saturate_s8(0), 0);
        assert_eq!(saturate_s8(100), 100);
        assert_eq!(saturate_s8(-100), -100);
        assert_eq!(saturate_s8(200), 127);
        assert_eq!(saturate_s8(-200), -128);
    }

    #[test]
    fn test_saturate_s16() {
        assert_eq!(saturate_s16(0), 0);
        assert_eq!(saturate_s16(20000), 20000);
        assert_eq!(saturate_s16(-20000), -20000);
        assert_eq!(saturate_s16(50000), 32767);
        assert_eq!(saturate_s16(-50000), -32768);
    }
}
