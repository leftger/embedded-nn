//! Convolution layer operations for quantized neural networks.

use crate::support::{clamp, requantize};
use crate::types::{
    ConvParams, Dims, DwConvParams, Error, PerChannelQuantParams, PerTensorQuantParams, Result,
};

/// Performs standard 2D Convolution with per-tensor quantization.
pub fn convolve_s8(
    conv_params: &ConvParams,
    quant_params: &PerTensorQuantParams,
    input_dims: &Dims,
    input: &[i8],
    filter_dims: &Dims,
    kernel: &[i8],
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

    if input_c == 0 || output_c == 0 {
        return Err(Error::ArgumentError);
    }

    let groups = input_c / kernel_c;
    let output_c_per_group = output_c / groups;

    for b in 0..input_batches {
        for out_y in 0..output_h {
            let base_y = out_y as i32 * conv_params.stride.h - conv_params.padding.h;
            for out_x in 0..output_w {
                let base_x = out_x as i32 * conv_params.stride.w - conv_params.padding.w;

                for g in 0..groups {
                    for out_ch_idx in 0..output_c_per_group {
                        let out_c = g * output_c_per_group + out_ch_idx;
                        let mut acc: i32 = match bias {
                            Some(b_slice) => b_slice[out_c],
                            None => 0,
                        };

                        for ky in 0..kernel_h {
                            let in_y = base_y + ky as i32 * conv_params.dilation.h;
                            if in_y >= 0 && in_y < input_dims.h {
                                for kx in 0..kernel_w {
                                    let in_x = base_x + kx as i32 * conv_params.dilation.w;
                                    if in_x >= 0 && in_x < input_dims.w {
                                        let in_idx_base = ((b * input_h + in_y as usize) * input_w
                                            + in_x as usize)
                                            * input_c
                                            + g * kernel_c;
                                        let ker_idx_base =
                                            ((out_c * kernel_h + ky) * kernel_w + kx) * kernel_c;

                                        for ic in 0..kernel_c {
                                            let lhs = input[in_idx_base + ic] as i32
                                                + conv_params.input_offset;
                                            let rhs = kernel[ker_idx_base + ic] as i32;
                                            acc += lhs * rhs;
                                        }
                                    }
                                }
                            }
                        }

                        acc = requantize(acc, quant_params.multiplier, quant_params.shift);
                        acc += conv_params.output_offset;
                        acc = clamp(acc, conv_params.activation.min, conv_params.activation.max);

                        let out_idx =
                            ((b * output_h + out_y) * output_w + out_x) * output_c + out_c;
                        output[out_idx] = acc as i8;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Performs standard 2D Convolution with per-channel quantization.
pub fn convolve_per_channel_s8(
    conv_params: &ConvParams,
    quant_params: &PerChannelQuantParams,
    input_dims: &Dims,
    input: &[i8],
    filter_dims: &Dims,
    kernel: &[i8],
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

    if input_c == 0 || output_c == 0 {
        return Err(Error::ArgumentError);
    }

    let groups = input_c / kernel_c;
    let output_c_per_group = output_c / groups;

    for b in 0..input_batches {
        for out_y in 0..output_h {
            let base_y = out_y as i32 * conv_params.stride.h - conv_params.padding.h;
            for out_x in 0..output_w {
                let base_x = out_x as i32 * conv_params.stride.w - conv_params.padding.w;

                for g in 0..groups {
                    for out_ch_idx in 0..output_c_per_group {
                        let out_c = g * output_c_per_group + out_ch_idx;
                        let mut acc: i32 = match bias {
                            Some(b_slice) => b_slice[out_c],
                            None => 0,
                        };

                        for ky in 0..kernel_h {
                            let in_y = base_y + ky as i32 * conv_params.dilation.h;
                            if in_y >= 0 && in_y < input_dims.h {
                                for kx in 0..kernel_w {
                                    let in_x = base_x + kx as i32 * conv_params.dilation.w;
                                    if in_x >= 0 && in_x < input_dims.w {
                                        let in_idx_base = ((b * input_h + in_y as usize) * input_w
                                            + in_x as usize)
                                            * input_c
                                            + g * kernel_c;
                                        let ker_idx_base =
                                            ((out_c * kernel_h + ky) * kernel_w + kx) * kernel_c;

                                        for ic in 0..kernel_c {
                                            let lhs = input[in_idx_base + ic] as i32
                                                + conv_params.input_offset;
                                            let rhs = kernel[ker_idx_base + ic] as i32;
                                            acc += lhs * rhs;
                                        }
                                    }
                                }
                            }
                        }

                        let mult = quant_params.multiplier[out_c];
                        let shift = quant_params.shift[out_c];

                        acc = requantize(acc, mult, shift);
                        acc += conv_params.output_offset;
                        acc = clamp(acc, conv_params.activation.min, conv_params.activation.max);

                        let out_idx =
                            ((b * output_h + out_y) * output_w + out_x) * output_c + out_c;
                        output[out_idx] = acc as i8;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Performs Depthwise 2D Convolution with per-channel quantization.
pub fn depthwise_conv_per_channel_s8(
    dw_params: &DwConvParams,
    quant_params: &PerChannelQuantParams,
    input_dims: &Dims,
    input: &[i8],
    filter_dims: &Dims,
    kernel: &[i8],
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

    let output_h = output_dims.h as usize;
    let output_w = output_dims.w as usize;
    let output_c = output_dims.c as usize;

    let ch_mult = dw_params.ch_mult as usize;

    for b in 0..input_batches {
        for out_y in 0..output_h {
            let base_y = out_y as i32 * dw_params.stride.h - dw_params.padding.h;
            for out_x in 0..output_w {
                let base_x = out_x as i32 * dw_params.stride.w - dw_params.padding.w;

                for out_c in 0..output_c {
                    let in_c = out_c / ch_mult;

                    let mut acc: i32 = match bias {
                        Some(b_slice) => b_slice[out_c],
                        None => 0,
                    };

                    for ky in 0..kernel_h {
                        let in_y = base_y + ky as i32 * dw_params.dilation.h;
                        if in_y >= 0 && in_y < input_dims.h {
                            for kx in 0..kernel_w {
                                let in_x = base_x + kx as i32 * dw_params.dilation.w;
                                if in_x >= 0 && in_x < input_dims.w {
                                    let in_idx = ((b * input_h + in_y as usize) * input_w
                                        + in_x as usize)
                                        * input_c
                                        + in_c;
                                    let ker_idx = ((ky * kernel_w + kx) * output_c) + out_c;

                                    let lhs = input[in_idx] as i32 + dw_params.input_offset;
                                    let rhs = kernel[ker_idx] as i32;
                                    acc += lhs * rhs;
                                }
                            }
                        }
                    }

                    let mult = quant_params.multiplier[out_c];
                    let shift = quant_params.shift[out_c];

                    acc = requantize(acc, mult, shift);
                    acc += dw_params.output_offset;
                    acc = clamp(acc, dw_params.activation.min, dw_params.activation.max);

                    let out_idx = ((b * output_h + out_y) * output_w + out_x) * output_c + out_c;
                    output[out_idx] = acc as i8;
                }
            }
        }
    }

    Ok(())
}

/// Performs Transposed 2D Convolution (Deconvolution) for int8 tensors with per-channel quantization.
pub fn transpose_conv_s8(
    conv_params: &ConvParams,
    quant_params: &PerChannelQuantParams,
    input_dims: &Dims,
    input: &[i8],
    filter_dims: &Dims,
    kernel: &[i8],
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

    let output_h = output_dims.h as usize;
    let output_w = output_dims.w as usize;
    let output_c = output_dims.c as usize;

    let stride_y = conv_params.stride.h as usize;
    let stride_x = conv_params.stride.w as usize;
    let pad_y = conv_params.padding.h as usize;
    let pad_x = conv_params.padding.w as usize;

    let mut accum_buffer = [0i32; 1024]; // Stack scratch accumulator for 1024 elements chunk or dynamic iteration
    let scratch_size = output_h * output_w * output_c;

    for b in 0..input_batches {
        for out_idx in 0..scratch_size {
            let out_c = out_idx % output_c;
            let b_val = match bias {
                Some(b_slice) => b_slice[out_c],
                None => 0,
            };
            if out_idx < accum_buffer.len() {
                accum_buffer[out_idx] = b_val;
            }
        }

        // Scatter-accumulate input into output positions
        for in_y in 0..input_h {
            for in_x in 0..input_w {
                for ky in 0..kernel_h {
                    let out_y = in_y * stride_y + ky;
                    if out_y >= pad_y && out_y < output_h + pad_y {
                        let actual_out_y = out_y - pad_y;
                        for kx in 0..kernel_w {
                            let out_x = in_x * stride_x + kx;
                            if out_x >= pad_x && out_x < output_w + pad_x {
                                let actual_out_x = out_x - pad_x;

                                for out_c in 0..output_c {
                                    for in_c in 0..input_c {
                                        let in_idx = ((b * input_h + in_y) * input_w + in_x)
                                            * input_c
                                            + in_c;
                                        let ker_idx = ((out_c * kernel_h + ky) * kernel_w + kx)
                                            * input_c
                                            + in_c;

                                        let lhs = input[in_idx] as i32 + conv_params.input_offset;
                                        let rhs = kernel[ker_idx] as i32;

                                        let buf_idx = (actual_out_y * output_w + actual_out_x)
                                            * output_c
                                            + out_c;
                                        if buf_idx < accum_buffer.len() {
                                            accum_buffer[buf_idx] += lhs * rhs;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Requantize and write back output
        for out_y in 0..output_h {
            for out_x in 0..output_w {
                for out_c in 0..output_c {
                    let buf_idx = (out_y * output_w + out_x) * output_c + out_c;
                    let acc = if buf_idx < accum_buffer.len() {
                        accum_buffer[buf_idx]
                    } else {
                        match bias {
                            Some(b_slice) => b_slice[out_c],
                            None => 0,
                        }
                    };

                    let mult = quant_params.multiplier[out_c];
                    let shift = quant_params.shift[out_c];

                    let req = requantize(acc, mult, shift);
                    let final_val = clamp(
                        req + conv_params.output_offset,
                        conv_params.activation.min,
                        conv_params.activation.max,
                    );

                    let out_idx = ((b * output_h + out_y) * output_w + out_x) * output_c + out_c;
                    if out_idx < output.len() {
                        output[out_idx] = final_val as i8;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Performs 1D Temporal Convolution for int8 tensors (`convolve_1_x_n_s8`).
pub fn convolve_1_x_n_s8(
    conv_params: &ConvParams,
    quant_params: &PerTensorQuantParams,
    input_dims: &Dims,
    input: &[i8],
    filter_dims: &Dims,
    kernel: &[i8],
    bias: Option<&[i32]>,
    output_dims: &Dims,
    output: &mut [i8],
) -> Result<()> {
    // 1D Conv is equivalent to 2D Conv with Height = 1
    convolve_s8(
        conv_params,
        quant_params,
        input_dims,
        input,
        filter_dims,
        kernel,
        bias,
        output_dims,
        output,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Activation, Tile};

    #[test]
    fn test_convolve_s8_simple() {
        let conv_params = ConvParams {
            input_offset: 0,
            output_offset: 0,
            stride: Tile::new(1, 1),
            padding: Tile::new(0, 0),
            dilation: Tile::new(1, 1),
            activation: Activation::int8_unconstrained(),
        };

        let quant_params = PerTensorQuantParams::new(1073741824, 0); // 0.5

        let input_dims = Dims::new(1, 3, 3, 1);
        let input = [1i8, 2i8, 3i8, 4i8, 5i8, 6i8, 7i8, 8i8, 9i8];

        let filter_dims = Dims::new(1, 2, 2, 1); // 1 out_channel, 2x2 kernel, 1 in_channel
        let kernel = [1i8, 0i8, 0i8, 1i8];

        let output_dims = Dims::new(1, 2, 2, 1);
        let mut output = [0i8; 4];

        convolve_s8(
            &conv_params,
            &quant_params,
            &input_dims,
            &input,
            &filter_dims,
            &kernel,
            None,
            &output_dims,
            &mut output,
        )
        .unwrap();

        assert_eq!(output, [3, 4, 6, 7]);
    }
}
