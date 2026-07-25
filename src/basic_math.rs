//! Basic elementwise mathematical operations on quantized tensors.

use crate::support::{clamp, requantize};
use crate::types::{Activation, Result};

/// Elementwise addition parameters for quantized int8 tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElementwiseAddParams {
    /// Zero point offset for input 1.
    pub input1_offset: i32,
    /// Multiplier for input 1.
    pub input1_mult: i32,
    /// Shift for input 1.
    pub input1_shift: i32,
    /// Zero point offset for input 2.
    pub input2_offset: i32,
    /// Multiplier for input 2.
    pub input2_mult: i32,
    /// Shift for input 2.
    pub input2_shift: i32,
    /// Common left shift for inputs.
    pub left_shift: i32,
    /// Output zero point offset.
    pub output_offset: i32,
    /// Output multiplier.
    pub output_mult: i32,
    /// Output shift.
    pub output_shift: i32,
    /// Output activation range.
    pub activation: Activation,
}

/// Elementwise multiplication parameters for quantized int8 tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElementwiseMulParams {
    /// Zero point offset for input 1.
    pub input1_offset: i32,
    /// Zero point offset for input 2.
    pub input2_offset: i32,
    /// Output zero point offset.
    pub output_offset: i32,
    /// Output multiplier.
    pub output_mult: i32,
    /// Output shift.
    pub output_shift: i32,
    /// Output activation range.
    pub activation: Activation,
}

/// Performs elementwise addition of two int8 tensors.
pub fn elementwise_add_s8(
    input1: &[i8],
    input2: &[i8],
    output: &mut [i8],
    params: &ElementwiseAddParams,
) -> Result<()> {
    let size = input1.len().min(input2.len()).min(output.len());
    for i in 0..size {
        let val1 = (input1[i] as i32 + params.input1_offset) << params.left_shift;
        let val2 = (input2[i] as i32 + params.input2_offset) << params.left_shift;

        let req1 = requantize(val1, params.input1_mult, params.input1_shift);
        let req2 = requantize(val2, params.input2_mult, params.input2_shift);

        let sum = req1 + req2;
        let req_sum = requantize(sum, params.output_mult, params.output_shift);
        let final_val = req_sum + params.output_offset;

        output[i] = clamp(final_val, params.activation.min, params.activation.max) as i8;
    }
    Ok(())
}

/// Performs elementwise multiplication of two int8 tensors.
pub fn elementwise_mul_s8(
    input1: &[i8],
    input2: &[i8],
    output: &mut [i8],
    params: &ElementwiseMulParams,
) -> Result<()> {
    let size = input1.len().min(input2.len()).min(output.len());
    for i in 0..size {
        let val1 = input1[i] as i32 + params.input1_offset;
        let val2 = input2[i] as i32 + params.input2_offset;
        let prod = val1 * val2;

        let req_prod = requantize(prod, params.output_mult, params.output_shift);
        let final_val = req_prod + params.output_offset;

        output[i] = clamp(final_val, params.activation.min, params.activation.max) as i8;
    }
    Ok(())
}

/// Performs elementwise subtraction of two int8 tensors.
pub fn elementwise_sub_s8(
    input1: &[i8],
    input2: &[i8],
    output: &mut [i8],
    params: &ElementwiseAddParams,
) -> Result<()> {
    let size = input1.len().min(input2.len()).min(output.len());
    for i in 0..size {
        let val1 = (input1[i] as i32 + params.input1_offset) << params.left_shift;
        let val2 = (input2[i] as i32 + params.input2_offset) << params.left_shift;

        let req1 = requantize(val1, params.input1_mult, params.input1_shift);
        let req2 = requantize(val2, params.input2_mult, params.input2_shift);

        let diff = req1 - req2;
        let req_diff = requantize(diff, params.output_mult, params.output_shift);
        let final_val = req_diff + params.output_offset;

        output[i] = clamp(final_val, params.activation.min, params.activation.max) as i8;
    }
    Ok(())
}

/// Performs elementwise addition of two int16 tensors.
pub fn elementwise_add_s16(
    input1: &[i16],
    input2: &[i16],
    output: &mut [i16],
    mult1: i32,
    shift1: i32,
    mult2: i32,
    shift2: i32,
    output_mult: i32,
    output_shift: i32,
    act: Activation,
) -> Result<()> {
    let size = input1.len().min(input2.len()).min(output.len());
    for i in 0..size {
        let req1 = requantize(input1[i] as i32, mult1, shift1);
        let req2 = requantize(input2[i] as i32, mult2, shift2);

        let sum = req1 + req2;
        let req_sum = requantize(sum, output_mult, output_shift);

        output[i] = clamp(req_sum, act.min, act.max) as i16;
    }
    Ok(())
}

/// Performs elementwise multiplication of two int16 tensors.
pub fn elementwise_mul_s16(
    input1: &[i16],
    input2: &[i16],
    output: &mut [i16],
    output_mult: i32,
    output_shift: i32,
    act: Activation,
) -> Result<()> {
    let size = input1.len().min(input2.len()).min(output.len());
    for i in 0..size {
        let prod = (input1[i] as i64) * (input2[i] as i64);
        let req_prod = requantize((prod >> 15) as i32, output_mult, output_shift);
        output[i] = clamp(req_prod, act.min, act.max) as i16;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementwise_add_s8() {
        let input1 = [10i8, 20i8, 30i8];
        let input2 = [5i8, 15i8, 25i8];
        let mut output = [0i8; 3];

        let params = ElementwiseAddParams {
            input1_offset: 0,
            input1_mult: 1073741824, // 0.5
            input1_shift: 0,
            input2_offset: 0,
            input2_mult: 1073741824, // 0.5
            input2_shift: 0,
            left_shift: 0,
            output_offset: 0,
            output_mult: 1073741824, // 0.5
            output_shift: 0,
            activation: Activation::int8_unconstrained(),
        };

        elementwise_add_s8(&input1, &input2, &mut output, &params).unwrap();
        // (10*0.5 + 5*0.5) * 0.5 = 3.75 -> rounded to 4
        assert!((output[0] - 4).abs() <= 1);
    }
}
