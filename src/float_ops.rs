//! Floating-point (`f32` and `f16`) fallback operations and layers.

use crate::types::{Dims, Result, Tile};

/// IEEE-754 16-bit half-precision floating-point conversion to 32-bit single precision float.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 0x0001;
    let exp = (h >> 10) & 0x001F;
    let mant = h & 0x03FF;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits((sign as u32) << 31)
        } else {
            // Subnormal f16
            let mut m = mant as u32;
            let mut e = 0i32;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            let exp_f32 = (127 - 15 + 1 + e) as u32;
            f32::from_bits(((sign as u32) << 31) | (exp_f32 << 23) | (m << 13))
        }
    } else if exp == 0x1F {
        // Inf / NaN
        f32::from_bits(((sign as u32) << 31) | (0xFF << 23) | ((mant as u32) << 13))
    } else {
        // Normal f16
        let exp_f32 = (exp as u32 + 127 - 15) & 0xFF;
        f32::from_bits(((sign as u32) << 31) | (exp_f32 << 23) | ((mant as u32) << 13))
    }
}

/// 32-bit single precision float conversion to IEEE-754 16-bit half-precision float.
pub fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = (bits >> 31) & 0x0001;
    let exp = (bits >> 23) & 0x00FF;
    let mant = bits & 0x007F_FFFF;

    if exp == 0xFF {
        // Inf / NaN
        ((sign as u16) << 15) | (0x001F << 10) | ((mant >> 13) as u16)
    } else if exp == 0 {
        (sign as u16) << 15
    } else {
        let new_exp = exp as i32 - 127 + 15;
        if new_exp >= 31 {
            // Overflow to Inf
            ((sign as u16) << 15) | (0x001F << 10)
        } else if new_exp <= 0 {
            // Underflow
            (sign as u16) << 15
        } else {
            ((sign as u16) << 15) | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
        }
    }
}

/// Performs 2D Convolution for 32-bit floating point tensors.
pub fn convolve_f32(
    stride: Tile,
    padding: Tile,
    dilation: Tile,
    input_dims: &Dims,
    input: &[f32],
    filter_dims: &Dims,
    kernel: &[f32],
    bias: Option<&[f32]>,
    output_dims: &Dims,
    output: &mut [f32],
) -> Result<()> {
    let input_batches = input_dims.n as usize;
    let input_h = input_dims.h as usize;
    let input_w = input_dims.w as usize;
    let input_c = input_dims.c as usize;

    let kernel_h = filter_dims.h as usize;
    let kernel_w = filter_dims.w as usize;
    let kernel_c = filter_dims.c as usize;

    let output_h = output_dims.h as usize;
    let output_w = output_dims.w as usize;
    let output_c = output_dims.c as usize;

    let groups = input_c / kernel_c;
    let output_c_per_group = output_c / groups;

    for b in 0..input_batches {
        for out_y in 0..output_h {
            let base_y = out_y as i32 * stride.h - padding.h;
            for out_x in 0..output_w {
                let base_x = out_x as i32 * stride.w - padding.w;

                for g in 0..groups {
                    for out_ch_idx in 0..output_c_per_group {
                        let out_c = g * output_c_per_group + out_ch_idx;
                        let mut acc: f32 = match bias {
                            Some(b_slice) => b_slice[out_c],
                            None => 0.0,
                        };

                        for ky in 0..kernel_h {
                            let in_y = base_y + ky as i32 * dilation.h;
                            if in_y >= 0 && in_y < input_dims.h {
                                for kx in 0..kernel_w {
                                    let in_x = base_x + kx as i32 * dilation.w;
                                    if in_x >= 0 && in_x < input_dims.w {
                                        let in_idx_base = ((b * input_h + in_y as usize) * input_w
                                            + in_x as usize)
                                            * input_c
                                            + g * kernel_c;
                                        let ker_idx_base =
                                            ((out_c * kernel_h + ky) * kernel_w + kx) * kernel_c;

                                        for ic in 0..kernel_c {
                                            acc +=
                                                input[in_idx_base + ic] * kernel[ker_idx_base + ic];
                                        }
                                    }
                                }
                            }
                        }

                        let out_idx =
                            ((b * output_h + out_y) * output_w + out_x) * output_c + out_c;
                        output[out_idx] = acc;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Performs Fully Connected layer for 32-bit floating point tensors.
pub fn fully_connected_f32(
    input_dims: &Dims,
    input: &[f32],
    filter_dims: &Dims,
    kernel: &[f32],
    bias: Option<&[f32]>,
    output_dims: &Dims,
    output: &mut [f32],
) -> Result<()> {
    let batches = input_dims.n as usize;
    let accum_depth = filter_dims.n as usize;
    let output_depth = output_dims.c as usize;

    for b in 0..batches {
        let input_batch = &input[b * accum_depth..(b + 1) * accum_depth];
        let output_batch = &mut output[b * output_depth..(b + 1) * output_depth];

        for out_c in 0..output_depth {
            let mut acc: f32 = match bias {
                Some(b_slice) => b_slice[out_c],
                None => 0.0,
            };

            let kernel_row = &kernel[out_c * accum_depth..(out_c + 1) * accum_depth];

            for i in 0..accum_depth {
                acc += input_batch[i] * kernel_row[i];
            }

            output_batch[out_c] = acc;
        }
    }
    Ok(())
}

/// In-place ReLU for 32-bit float buffer (`max(val, 0.0)`).
pub fn relu_f32(data: &mut [f32]) {
    for val in data.iter_mut() {
        if *val < 0.0 {
            *val = 0.0;
        }
    }
}

/// In-place ReLU6 for 32-bit float buffer (`min(max(val, 0.0), 6.0)`).
pub fn relu6_f32(data: &mut [f32]) {
    for val in data.iter_mut() {
        if *val < 0.0 {
            *val = 0.0;
        } else if *val > 6.0 {
            *val = 6.0;
        }
    }
}

/// Fast and accurate pure no-std exp(x) implementation for f32 using range reduction.
fn exp_f32(x: f32) -> f32 {
    if x < -88.0 {
        return 0.0;
    }
    if x > 88.0 {
        return f32::INFINITY;
    }
    // exp(x) = exp(x / 64)^64
    let y = x / 64.0;
    let mut poly = 1.0
        + y * (1.0
            + y * (0.5
                + y * (1.0 / 6.0
                    + y * (1.0 / 24.0
                        + y * (1.0 / 120.0 + y * (1.0 / 720.0 + y * (1.0 / 5040.0)))))));
    for _ in 0..6 {
        poly *= poly;
    }
    poly
}

/// Softmax for 32-bit floating point tensors.
pub fn softmax_f32(
    input: &[f32],
    num_rows: usize,
    row_size: usize,
    output: &mut [f32],
) -> Result<()> {
    for row in 0..num_rows {
        let in_row = &input[row * row_size..(row + 1) * row_size];
        let out_row = &mut output[row * row_size..(row + 1) * row_size];

        let mut max_val = in_row[0];
        for &val in &in_row[1..] {
            if val > max_val {
                max_val = val;
            }
        }

        let mut sum = 0.0f32;
        for (i, &val) in in_row.iter().enumerate() {
            let diff = val - max_val;
            let exp_val = exp_f32(diff);
            out_row[i] = exp_val;
            sum += exp_val;
        }

        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for val in out_row.iter_mut() {
                *val *= inv_sum;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_conversion() {
        let val = 3.14159f32;
        let h = f32_to_f16(val);
        let back = f16_to_f32(h);
        assert!((back - val).abs() < 0.01);
    }

    #[test]
    fn test_fully_connected_f32() {
        let input_dims = Dims::new(1, 1, 1, 2);
        let input = [1.5f32, 2.5f32];
        let filter_dims = Dims::new(2, 1, 1, 1);
        let kernel = [2.0f32, 4.0f32];
        let output_dims = Dims::new(1, 1, 1, 1);
        let mut output = [0.0f32; 1];

        fully_connected_f32(
            &input_dims,
            &input,
            &filter_dims,
            &kernel,
            None,
            &output_dims,
            &mut output,
        )
        .unwrap();

        // 1.5 * 2.0 + 2.5 * 4.0 = 3.0 + 10.0 = 13.0
        assert_eq!(output[0], 13.0);
    }
}
