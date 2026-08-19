//! Fully Connected (Linear / Dense) and Batch Matrix Multiplication operations.

use crate::support::{clamp, requantize};
use crate::types::{Dims, FcParams, PerChannelQuantParams, PerTensorQuantParams, Result};

/// Performs per-tensor quantized int8 Fully Connected layer.
pub fn fully_connected_s8(
    fc_params: &FcParams,
    quant_params: &PerTensorQuantParams,
    input_dims: &Dims,
    input: &[i8],
    filter_dims: &Dims,
    kernel: &[i8],
    bias: Option<&[i32]>,
    output_dims: &Dims,
    output: &mut [i8],
) -> Result<()> {
    let batches = input_dims.n as usize;
    let accum_depth = filter_dims.n as usize;
    let output_depth = output_dims.c as usize;

    for b in 0..batches {
        let input_batch = &input[b * accum_depth..(b + 1) * accum_depth];
        let output_batch = &mut output[b * output_depth..(b + 1) * output_depth];

        for out_c in 0..output_depth {
            let mut acc: i32 = match bias {
                Some(b_slice) => b_slice[out_c],
                None => 0,
            };

            let kernel_row = &kernel[out_c * accum_depth..(out_c + 1) * accum_depth];

            for i in 0..accum_depth {
                let lhs = input_batch[i] as i32 + fc_params.input_offset;
                let rhs = kernel_row[i] as i32 + fc_params.filter_offset;
                acc += lhs * rhs;
            }

            acc = requantize(acc, quant_params.multiplier, quant_params.shift);
            acc += fc_params.output_offset;
            acc = clamp(acc, fc_params.activation.min, fc_params.activation.max);

            output_batch[out_c] = acc as i8;
        }
    }
    Ok(())
}

/// Performs per-channel quantized int8 Fully Connected layer.
pub fn fully_connected_per_channel_s8(
    fc_params: &FcParams,
    quant_params: &PerChannelQuantParams,
    input_dims: &Dims,
    input: &[i8],
    filter_dims: &Dims,
    kernel: &[i8],
    bias: Option<&[i32]>,
    output_dims: &Dims,
    output: &mut [i8],
) -> Result<()> {
    let batches = input_dims.n as usize;
    let accum_depth = filter_dims.n as usize;
    let output_depth = output_dims.c as usize;

    for b in 0..batches {
        let input_batch = &input[b * accum_depth..(b + 1) * accum_depth];
        let output_batch = &mut output[b * output_depth..(b + 1) * output_depth];

        for out_c in 0..output_depth {
            let mut acc: i32 = match bias {
                Some(b_slice) => b_slice[out_c],
                None => 0,
            };

            let kernel_row = &kernel[out_c * accum_depth..(out_c + 1) * accum_depth];

            for i in 0..accum_depth {
                let lhs = input_batch[i] as i32 + fc_params.input_offset;
                let rhs = kernel_row[i] as i32 + fc_params.filter_offset;
                acc += lhs * rhs;
            }

            let mult = quant_params.multiplier[out_c];
            let shift = quant_params.shift[out_c];

            acc = requantize(acc, mult, shift);
            acc += fc_params.output_offset;
            acc = clamp(acc, fc_params.activation.min, fc_params.activation.max);

            output_batch[out_c] = acc as i8;
        }
    }
    Ok(())
}

/// Performs int16 Fully Connected layer.
pub fn fully_connected_s16(
    fc_params: &FcParams,
    quant_params: &PerTensorQuantParams,
    input_dims: &Dims,
    input: &[i16],
    filter_dims: &Dims,
    kernel: &[i8],
    bias: Option<&[i64]>,
    output_dims: &Dims,
    output: &mut [i16],
) -> Result<()> {
    let batches = input_dims.n as usize;
    let accum_depth = filter_dims.n as usize;
    let output_depth = output_dims.c as usize;

    for b in 0..batches {
        let input_batch = &input[b * accum_depth..(b + 1) * accum_depth];
        let output_batch = &mut output[b * output_depth..(b + 1) * output_depth];

        for out_c in 0..output_depth {
            let mut acc: i64 = match bias {
                Some(b_slice) => b_slice[out_c],
                None => 0,
            };

            let kernel_row = &kernel[out_c * accum_depth..(out_c + 1) * accum_depth];

            for i in 0..accum_depth {
                let lhs = input_batch[i] as i64;
                let rhs = kernel_row[i] as i64;
                acc += lhs * rhs;
            }

            let req = requantize(
                (acc >> 15) as i32,
                quant_params.multiplier,
                quant_params.shift,
            );
            let final_val = clamp(req, fc_params.activation.min, fc_params.activation.max);

            output_batch[out_c] = final_val as i16;
        }
    }
    Ok(())
}

/// Performs Batch Matrix Multiplication (`BatchMatMul`) for int8 tensors.
///
/// Computes `Output[b, i, j] = Requantize(sum_k (LHS[b, i, k] + lhs_offset) * (RHS[b, k, j] + rhs_offset))`
pub fn batch_matmul_s8(
    fc_params: &FcParams,
    quant_params: &PerTensorQuantParams,
    lhs_dims: &Dims,
    input_lhs: &[i8],
    rhs_dims: &Dims,
    input_rhs: &[i8],
    output_dims: &Dims,
    output: &mut [i8],
) -> Result<()> {
    let batches = output_dims.n as usize;
    let rows = lhs_dims.h as usize;
    let cols = rhs_dims.c as usize;
    let accum_dim = lhs_dims.w as usize;

    for b in 0..batches {
        let lhs_b_idx = if lhs_dims.n == 1 { 0 } else { b };
        let rhs_b_idx = if rhs_dims.n == 1 { 0 } else { b };

        for i in 0..rows {
            for j in 0..cols {
                let mut acc: i32 = 0;

                for k in 0..accum_dim {
                    let lhs_idx = (lhs_b_idx * rows + i) * accum_dim + k;
                    let rhs_idx = (rhs_b_idx * accum_dim + k) * cols + j;

                    let lhs_val = input_lhs[lhs_idx] as i32 + fc_params.input_offset;
                    let rhs_val = input_rhs[rhs_idx] as i32 + fc_params.filter_offset;
                    acc += lhs_val * rhs_val;
                }

                let req = requantize(acc, quant_params.multiplier, quant_params.shift);
                let final_val = clamp(
                    req + fc_params.output_offset,
                    fc_params.activation.min,
                    fc_params.activation.max,
                );

                let out_idx = (b * rows + i) * cols + j;
                if out_idx < output.len() {
                    output[out_idx] = final_val as i8;
                }
            }
        }
    }

    Ok(())
}

/// Performs Batch Matrix Multiplication (`BatchMatMul`) for int16 tensors.
pub fn batch_matmul_s16(
    fc_params: &FcParams,
    quant_params: &PerTensorQuantParams,
    lhs_dims: &Dims,
    input_lhs: &[i16],
    rhs_dims: &Dims,
    input_rhs: &[i16],
    output_dims: &Dims,
    output: &mut [i16],
) -> Result<()> {
    let batches = output_dims.n as usize;
    let rows = lhs_dims.h as usize;
    let cols = rhs_dims.c as usize;
    let accum_dim = lhs_dims.w as usize;

    for b in 0..batches {
        let lhs_b_idx = if lhs_dims.n == 1 { 0 } else { b };
        let rhs_b_idx = if rhs_dims.n == 1 { 0 } else { b };

        for i in 0..rows {
            for j in 0..cols {
                let mut acc: i64 = 0;

                for k in 0..accum_dim {
                    let lhs_idx = (lhs_b_idx * rows + i) * accum_dim + k;
                    let rhs_idx = (rhs_b_idx * accum_dim + k) * cols + j;

                    let lhs_val = input_lhs[lhs_idx] as i64;
                    let rhs_val = input_rhs[rhs_idx] as i64;
                    acc += lhs_val * rhs_val;
                }

                let req = requantize(
                    (acc >> 15) as i32,
                    quant_params.multiplier,
                    quant_params.shift,
                );
                let final_val = clamp(req, fc_params.activation.min, fc_params.activation.max);

                let out_idx = (b * rows + i) * cols + j;
                if out_idx < output.len() {
                    output[out_idx] = final_val as i16;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Activation;

    #[test]
    fn test_fully_connected_s8() {
        let fc_params = FcParams {
            input_offset: 0,
            filter_offset: 0,
            output_offset: 0,
            activation: Activation::int8_unconstrained(),
        };
        let quant_params = PerTensorQuantParams::new(1073741824, 0); // 0.5
        let input_dims = Dims::new(1, 1, 1, 3);
        let input = [2i8, 4i8, 6i8];
        let filter_dims = Dims::new(3, 1, 1, 2); // 3 input, 2 output channels
        let kernel = [
            1i8, 2i8, 3i8, // row 0
            4i8, 5i8, 6i8, // row 1
        ];
        let bias = [0i32, 0i32];
        let output_dims = Dims::new(1, 1, 1, 2);
        let mut output = [0i8; 2];

        fully_connected_s8(
            &fc_params,
            &quant_params,
            &input_dims,
            &input,
            &filter_dims,
            &kernel,
            Some(&bias),
            &output_dims,
            &mut output,
        )
        .unwrap();

        assert_eq!(output[0], 14);
        assert_eq!(output[1], 32);
    }

    #[test]
    fn test_batch_matmul_s8() {
        let fc_params = FcParams {
            input_offset: 0,
            filter_offset: 0,
            output_offset: 0,
            activation: Activation::int8_unconstrained(),
        };
        let quant_params = PerTensorQuantParams::new(1073741824, 0); // 0.5

        let lhs_dims = Dims::new(1, 2, 2, 0); // 2 rows x 2 cols
        let input_lhs = [1i8, 2i8, 3i8, 4i8];

        let rhs_dims = Dims::new(1, 2, 0, 2); // 2 rows x 2 cols
        let input_rhs = [5i8, 6i8, 7i8, 8i8];

        let output_dims = Dims::new(1, 2, 0, 2);
        let mut output = [0i8; 4];

        batch_matmul_s8(
            &fc_params,
            &quant_params,
            &lhs_dims,
            &input_lhs,
            &rhs_dims,
            &input_rhs,
            &output_dims,
            &mut output,
        )
        .unwrap();

        // Row 0, Col 0: 1*5 + 2*7 = 19 -> *0.5 = 10 (rounded)
        // Row 0, Col 1: 1*6 + 2*8 = 22 -> *0.5 = 11
        // Row 1, Col 0: 3*5 + 4*7 = 43 -> *0.5 = 22
        // Row 1, Col 1: 3*6 + 4*8 = 50 -> *0.5 = 25
        assert_eq!(output, [10, 11, 22, 25]);
    }
}
