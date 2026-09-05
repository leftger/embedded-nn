//! Softmax activation operations for quantized neural networks.

use crate::support::{clamp, divide_by_power_of_two, doubling_high_mult_no_sat};
use crate::types::Result;

const ACCUM_BITS: i32 = 12;

/// Saturation multiply (Q31 high multiplication).
#[inline]
fn mul_sat(a: i32, b: i32) -> i32 {
    doubling_high_mult_no_sat(a, b)
}

/// Evaluates fixed-point exponent on negative values.
pub fn exp_on_negative_values(val: i32) -> i32 {
    if val == 0 {
        return i32::MAX;
    }

    let mut shift = 24i32;
    let val_mod_minus_quarter = (val & ((1 << shift) - 1)) - (1 << shift);
    let remainder = val_mod_minus_quarter - val;
    let x = (val_mod_minus_quarter << 5) + (1 << 28);
    let x2 = mul_sat(x, x);

    let op1 = divide_by_power_of_two(mul_sat(x2, x2), 2) + mul_sat(x2, x);
    let op2 = x + divide_by_power_of_two(mul_sat(op1, 715827883) + x2, 1);
    let mut result = 1895147668 + mul_sat(1895147668, op2);

    let constants = [
        1672461947, 1302514674, 790015084, 290630308, 39332535, 720401, 242,
    ];

    for &c in constants.iter() {
        if (remainder & (1 << shift)) != 0 {
            result = mul_sat(result, c);
        }
        shift += 1;
    }

    result
}

/// Evaluates 1 / (1 + x) for x in [0, 1] in fixed-point.
pub fn one_over_one_plus_x_for_x_in_0_1(val: i32) -> i32 {
    let sum = val as i64 + i32::MAX as i64;
    let half_denominator = ((sum + if sum >= 0 { 1 } else { -1 }) / 2) as i32;
    let mut x = 1515870810 + mul_sat(half_denominator, -1010580540);

    let shift = 1i32 << 29;
    for _ in 0..3 {
        let diff = shift - mul_sat(half_denominator, x);
        x = x.wrapping_add(mul_sat(x, diff).wrapping_shl(2));
    }

    // `x` can converge to exactly 2^30 for an input of zero. Doubling that value with a
    // wrapping shift produces `i32::MIN`, turning a positive reciprocal into a negative scale
    // and collapsing a uniform softmax to all `i8::MIN`. Q31's positive endpoint is represented
    // by `i32::MAX`, so saturate this one unrepresentable value.
    x.saturating_mul(2)
}

/// Performs Softmax for int8 tensors.
pub fn softmax_s8(
    input: &[i8],
    num_rows: usize,
    row_size: usize,
    mult: i32,
    shift: i32,
    diff_min: i32,
    output: &mut [i8],
) -> Result<()> {
    if num_rows == 0 || row_size == 0 {
        return Ok(());
    }
    if input.len() < num_rows * row_size || output.len() < num_rows * row_size {
        return Err(crate::types::Error::ArgumentError);
    }
    let mask = if (0..31).contains(&shift) {
        1i32 << shift
    } else {
        1i32
    };

    for row in 0..num_rows {
        let in_row = &input[row * row_size..(row + 1) * row_size];
        let out_row = &mut output[row * row_size..(row + 1) * row_size];

        // 1. Find max
        let mut max_val = in_row[0];
        for &val in &in_row[1..] {
            if val > max_val {
                max_val = val;
            }
        }

        // 2. Accumulate sum of exp
        let mut sum = 0i32;
        for &val in in_row.iter() {
            let diff = val as i32 - max_val as i32;
            if diff >= diff_min {
                let exp_input = mul_sat(diff * mask, mult);
                let exp_val = exp_on_negative_values(exp_input);
                sum += divide_by_power_of_two(exp_val, ACCUM_BITS);
            }
        }

        // 3. Requantize output
        let headroom = if sum > 0 {
            sum.leading_zeros() as i32
        } else {
            32
        };
        let shifted_sum = if sum > 0 {
            (((sum as u32) << headroom) & 0x7FFF_FFFF) as i32
        } else {
            0
        };
        let shifted_scale = one_over_one_plus_x_for_x_in_0_1(shifted_sum);
        let bits_over_unit = ACCUM_BITS - headroom + 23;

        for (i, &val) in in_row.iter().enumerate() {
            let diff = val as i32 - max_val as i32;
            if diff >= diff_min {
                let exp_input = mul_sat(diff * mask, mult);
                let exp_val = exp_on_negative_values(exp_input);
                let scaled = mul_sat(shifted_scale, exp_val);
                let res = divide_by_power_of_two(scaled, bits_over_unit) + (i8::MIN as i32);
                out_row[i] = clamp(res, i8::MIN as i32, i8::MAX as i32) as i8;
            } else {
                out_row[i] = i8::MIN;
            }
        }
    }
    Ok(())
}

/// Performs Softmax for int16 tensors.
pub fn softmax_s16(
    input: &[i16],
    num_rows: usize,
    row_size: usize,
    mult: i32,
    shift: i32,
    diff_min: i32,
    output: &mut [i16],
) -> Result<()> {
    if num_rows == 0 || row_size == 0 {
        return Ok(());
    }
    if input.len() < num_rows * row_size || output.len() < num_rows * row_size {
        return Err(crate::types::Error::ArgumentError);
    }
    let mask = if (0..31).contains(&shift) {
        1i32 << shift
    } else {
        1i32
    };

    for row in 0..num_rows {
        let in_row = &input[row * row_size..(row + 1) * row_size];
        let out_row = &mut output[row * row_size..(row + 1) * row_size];

        let mut max_val = in_row[0];
        for &val in &in_row[1..] {
            if val > max_val {
                max_val = val;
            }
        }

        let mut sum = 0i32;
        for &val in in_row.iter() {
            let diff = val as i32 - max_val as i32;
            if diff >= diff_min {
                let exp_input = mul_sat(diff * mask, mult);
                let exp_val = exp_on_negative_values(exp_input);
                sum += divide_by_power_of_two(exp_val, ACCUM_BITS);
            }
        }

        let headroom = if sum > 0 {
            sum.leading_zeros() as i32
        } else {
            32
        };
        let shifted_sum = if sum > 0 {
            (((sum as u32) << headroom) & 0x7FFF_FFFF) as i32
        } else {
            0
        };
        let shifted_scale = one_over_one_plus_x_for_x_in_0_1(shifted_sum);
        let bits_over_unit = ACCUM_BITS - headroom + 15;

        for (i, &val) in in_row.iter().enumerate() {
            let diff = val as i32 - max_val as i32;
            if diff >= diff_min {
                let exp_input = mul_sat(diff * mask, mult);
                let exp_val = exp_on_negative_values(exp_input);
                let scaled = mul_sat(shifted_scale, exp_val);
                let res = divide_by_power_of_two(scaled, bits_over_unit) + (i16::MIN as i32);
                out_row[i] = clamp(res, i16::MIN as i32, i16::MAX as i32) as i16;
            } else {
                out_row[i] = i16::MIN;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_s8() {
        let input = [10i8, 20i8, 30i8, 40i8];
        let mut output = [0i8; 4];

        softmax_s8(&input, 1, 4, 1073741824, 20, -256, &mut output).unwrap();

        // Check that highest element input[3] produces maximum softmax probability
        assert!(output[3] > output[2]);
        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);
    }

    #[test]
    fn test_softmax_s8_uniform_distribution() {
        let input = [20i8; 4];
        let mut output = [0i8; 4];

        softmax_s8(&input, 1, 4, 1073741824, 20, -256, &mut output).unwrap();

        assert_eq!(output, [-64; 4]);
    }
}
