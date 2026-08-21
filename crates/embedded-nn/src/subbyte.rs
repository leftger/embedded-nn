//! Sub-byte 4-bit (`s4`) quantization operations and layers.

use crate::support::{clamp, requantize};
use crate::types::{ConvParams, Dims, FcParams, PerTensorQuantParams, Result};

/// Unpacks a byte containing two 4-bit signed integers into `(low_nibble, high_nibble)`.
///
/// Both nibbles are sign-extended into `i8` values in the range `[-8, 7]`.
#[inline]
pub const fn unpack_s4_pair(packed_byte: i8) -> (i8, i8) {
    let b = packed_byte as i32;
    let low = (b << 28) >> 28;
    let high = b >> 4;
    (low as i8, high as i8)
}

/// Packs two 4-bit signed integers `(low, high)` in the range `[-8, 7]` into a single byte.
#[inline]
pub const fn pack_s4_pair(low: i8, high: i8) -> i8 {
    ((low as u8 & 0x0F) | ((high as u8 & 0x0F) << 4)) as i8
}

/// Performs per-tensor quantized int4 (`s4`) Fully Connected layer.
///
/// `packed_kernel` contains pairs of 4-bit signed weights packed into bytes.
pub fn fully_connected_s4(
    fc_params: &FcParams,
    quant_params: &PerTensorQuantParams,
    input_dims: &Dims,
    input: &[i8],
    filter_dims: &Dims,
    packed_kernel: &[i8],
    bias: Option<&[i32]>,
    output_dims: &Dims,
    output: &mut [i8],
) -> Result<()> {
    let batches = input_dims.n as usize;
    let accum_depth = filter_dims.n as usize;
    let output_depth = output_dims.c as usize;

    let packed_cols = (accum_depth + 1) / 2;

    for b in 0..batches {
        let input_batch = &input[b * accum_depth..(b + 1) * accum_depth];
        let output_batch = &mut output[b * output_depth..(b + 1) * output_depth];

        for out_c in 0..output_depth {
            let mut acc: i32 = match bias {
                Some(b_slice) => b_slice[out_c],
                None => 0,
            };

            let kernel_row_packed = &packed_kernel[out_c * packed_cols..(out_c + 1) * packed_cols];

            for p in 0..packed_cols {
                let (w0, w1) = unpack_s4_pair(kernel_row_packed[p]);
                let idx0 = p * 2;
                if idx0 < accum_depth {
                    let lhs0 = input_batch[idx0] as i32 + fc_params.input_offset;
                    acc += lhs0 * (w0 as i32);
                }
                let idx1 = idx0 + 1;
                if idx1 < accum_depth {
                    let lhs1 = input_batch[idx1] as i32 + fc_params.input_offset;
                    acc += lhs1 * (w1 as i32);
                }
            }

            acc = requantize(acc, quant_params.multiplier, quant_params.shift);
            acc += fc_params.output_offset;
            acc = clamp(acc, fc_params.activation.min, fc_params.activation.max);

            output_batch[out_c] = acc as i8;
        }
    }
    Ok(())
}

/// Performs per-tensor quantized int4 (`s4`) 2D Convolution layer.
pub fn convolve_s4(
    conv_params: &ConvParams,
    quant_params: &PerTensorQuantParams,
    input_dims: &Dims,
    input: &[i8],
    filter_dims: &Dims,
    packed_kernel: &[i8],
    bias: Option<&[i32]>,
    output_dims: &Dims,
    output: &mut [i8],
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

    let kernel_spatial_c = kernel_h * kernel_w * kernel_c;
    let packed_kernel_cols = (kernel_spatial_c + 1) / 2;

    for b in 0..input_batches {
        for out_y in 0..output_h {
            let base_y = out_y as i32 * conv_params.stride.h - conv_params.padding.top;
            for out_x in 0..output_w {
                let base_x = out_x as i32 * conv_params.stride.w - conv_params.padding.left;

                for out_c in 0..output_c {
                    let mut acc: i32 = match bias {
                        Some(b_slice) => b_slice[out_c],
                        None => 0,
                    };

                    let ker_packed_row = &packed_kernel
                        [out_c * packed_kernel_cols..(out_c + 1) * packed_kernel_cols];
                    let mut k_flat_idx = 0usize;

                    for ky in 0..kernel_h {
                        let in_y = base_y + ky as i32 * conv_params.dilation.h;
                        for kx in 0..kernel_w {
                            let in_x = base_x + kx as i32 * conv_params.dilation.w;
                            for ic in 0..kernel_c {
                                if in_y >= 0
                                    && in_y < input_dims.h
                                    && in_x >= 0
                                    && in_x < input_dims.w
                                {
                                    let in_idx = ((b * input_h + in_y as usize) * input_w
                                        + in_x as usize)
                                        * input_c
                                        + ic;

                                    let packed_byte = ker_packed_row[k_flat_idx / 2];
                                    let weight = if k_flat_idx % 2 == 0 {
                                        unpack_s4_pair(packed_byte).0
                                    } else {
                                        unpack_s4_pair(packed_byte).1
                                    };

                                    let lhs = input[in_idx] as i32 + conv_params.input_offset;
                                    acc += lhs * (weight as i32);
                                }
                                k_flat_idx += 1;
                            }
                        }
                    }

                    acc = requantize(acc, quant_params.multiplier, quant_params.shift);
                    acc += conv_params.output_offset;
                    acc = clamp(acc, conv_params.activation.min, conv_params.activation.max);

                    let out_idx = ((b * output_h + out_y) * output_w + out_x) * output_c + out_c;
                    output[out_idx] = acc as i8;
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
    fn test_s4_packing() {
        let (low, high) = (-5i8, 7i8);
        let packed = pack_s4_pair(low, high);
        let (unpacked_low, unpacked_high) = unpack_s4_pair(packed);
        assert_eq!(unpacked_low, -5);
        assert_eq!(unpacked_high, 7);
    }

    #[test]
    fn test_fully_connected_s4() {
        let fc_params = FcParams {
            input_offset: 0,
            filter_offset: 0,
            output_offset: 0,
            activation: Activation::int8_unconstrained(),
        };
        let quant_params = PerTensorQuantParams::new(1073741824, 0); // 0.5
        let input_dims = Dims::new(1, 1, 1, 2);
        let input = [4i8, 6i8];

        let filter_dims = Dims::new(2, 1, 1, 1);
        let packed_kernel = [pack_s4_pair(2i8, 3i8)]; // w0 = 2, w1 = 3
        let output_dims = Dims::new(1, 1, 1, 1);
        let mut output = [0i8; 1];

        fully_connected_s4(
            &fc_params,
            &quant_params,
            &input_dims,
            &input,
            &filter_dims,
            &packed_kernel,
            None,
            &output_dims,
            &mut output,
        )
        .unwrap();

        // 4*2 + 6*3 = 26. Requantized * 0.5 = 13
        assert_eq!(output[0], 13);
    }
}
