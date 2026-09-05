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

    if input_c == 0
        || output_c == 0
        || kernel_c == 0
        || kernel_c > input_c
        || input_c % kernel_c != 0
    {
        return Err(Error::ArgumentError);
    }

    let groups = input_c / kernel_c;
    if groups == 0 || output_c % groups != 0 {
        return Err(Error::ArgumentError);
    }
    let output_c_per_group = output_c / groups;

    for b in 0..input_batches {
        for out_y in 0..output_h {
            let base_y = out_y as i32 * conv_params.stride.h - conv_params.padding.top;
            for out_x in 0..output_w {
                let base_x = out_x as i32 * conv_params.stride.w - conv_params.padding.left;

                for g in 0..groups {
                    for out_ch_idx in 0..output_c_per_group {
                        let out_c = g * output_c_per_group + out_ch_idx;
                        let mut acc: i32 = match bias {
                            Some(b_slice) => b_slice[out_c],
                            None => 0,
                        };

                        for ky in 0..kernel_h {
                            let in_y = base_y + ky as i32 * conv_params.dilation.h;
                            let y_in_bounds = in_y >= 0 && in_y < input_dims.h;
                            for kx in 0..kernel_w {
                                let in_x = base_x + kx as i32 * conv_params.dilation.w;
                                let x_in_bounds = in_x >= 0 && in_x < input_dims.w;
                                let ker_idx_base =
                                    ((out_c * kernel_h + ky) * kernel_w + kx) * kernel_c;

                                if y_in_bounds && x_in_bounds {
                                    let in_idx_base = ((b * input_h + in_y as usize) * input_w
                                        + in_x as usize)
                                        * input_c
                                        + g * kernel_c;

                                    for ic in 0..kernel_c {
                                        let lhs = input[in_idx_base + ic] as i32
                                            + conv_params.input_offset;
                                        let rhs = kernel[ker_idx_base + ic] as i32;
                                        acc += lhs * rhs;
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

    if input_c == 0
        || output_c == 0
        || kernel_c == 0
        || kernel_c > input_c
        || input_c % kernel_c != 0
    {
        return Err(Error::ArgumentError);
    }

    let groups = input_c / kernel_c;
    if groups == 0 || output_c % groups != 0 {
        return Err(Error::ArgumentError);
    }
    let output_c_per_group = output_c / groups;

    for b in 0..input_batches {
        for out_y in 0..output_h {
            let base_y = out_y as i32 * conv_params.stride.h - conv_params.padding.top;
            for out_x in 0..output_w {
                let base_x = out_x as i32 * conv_params.stride.w - conv_params.padding.left;

                for g in 0..groups {
                    for out_ch_idx in 0..output_c_per_group {
                        let out_c = g * output_c_per_group + out_ch_idx;
                        let mut acc: i32 = match bias {
                            Some(b_slice) => b_slice[out_c],
                            None => 0,
                        };

                        for ky in 0..kernel_h {
                            let in_y = base_y + ky as i32 * conv_params.dilation.h;
                            let y_in_bounds = in_y >= 0 && in_y < input_dims.h;
                            for kx in 0..kernel_w {
                                let in_x = base_x + kx as i32 * conv_params.dilation.w;
                                let x_in_bounds = in_x >= 0 && in_x < input_dims.w;
                                let ker_idx_base =
                                    ((out_c * kernel_h + ky) * kernel_w + kx) * kernel_c;

                                if y_in_bounds && x_in_bounds {
                                    let in_idx_base = ((b * input_h + in_y as usize) * input_w
                                        + in_x as usize)
                                        * input_c
                                        + g * kernel_c;

                                    for ic in 0..kernel_c {
                                        let lhs = input[in_idx_base + ic] as i32
                                            + conv_params.input_offset;
                                        let rhs = kernel[ker_idx_base + ic] as i32;
                                        acc += lhs * rhs;
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

    if input_c == 0 || output_c == 0 || dw_params.ch_mult == 0 {
        return Err(Error::ArgumentError);
    }

    let ch_mult = dw_params.ch_mult as usize;

    for b in 0..input_batches {
        for out_y in 0..output_h {
            let base_y = out_y as i32 * dw_params.stride.h - dw_params.padding.top;
            for out_x in 0..output_w {
                let base_x = out_x as i32 * dw_params.stride.w - dw_params.padding.left;

                for out_c in 0..output_c {
                    let in_c = out_c / ch_mult;

                    let mut acc: i32 = match bias {
                        Some(b_slice) => b_slice[out_c],
                        None => 0,
                    };

                    for ky in 0..kernel_h {
                        let in_y = base_y + ky as i32 * dw_params.dilation.h;
                        let y_in_bounds = in_y >= 0 && in_y < input_dims.h;
                        for kx in 0..kernel_w {
                            let in_x = base_x + kx as i32 * dw_params.dilation.w;
                            let x_in_bounds = in_x >= 0 && in_x < input_dims.w;
                            let ker_idx = ((ky * kernel_w + kx) * output_c) + out_c;

                            if y_in_bounds && x_in_bounds {
                                let in_idx = ((b * input_h + in_y as usize) * input_w
                                    + in_x as usize)
                                    * input_c
                                    + in_c;

                                let lhs = input[in_idx] as i32 + dw_params.input_offset;
                                let rhs = kernel[ker_idx] as i32;
                                acc += lhs * rhs;
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

    let stride_y = conv_params.stride.h;
    let stride_x = conv_params.stride.w;
    let pad_y = conv_params.padding.top;
    let pad_x = conv_params.padding.left;

    if stride_y <= 0 || stride_x <= 0 || input_c == 0 || output_c == 0 {
        return Err(Error::ArgumentError);
    }

    for b in 0..input_batches {
        for out_y in 0..output_h {
            for out_x in 0..output_w {
                for out_c in 0..output_c {
                    let mut acc: i32 = match bias {
                        Some(b_slice) => b_slice[out_c],
                        None => 0,
                    };

                    for ky in 0..kernel_h {
                        let y_val = out_y as i32 + pad_y - ky as i32;
                        if y_val >= 0 && y_val % stride_y == 0 {
                            let in_y = (y_val / stride_y) as usize;
                            if in_y < input_h {
                                for kx in 0..kernel_w {
                                    let x_val = out_x as i32 + pad_x - kx as i32;
                                    if x_val >= 0 && x_val % stride_x == 0 {
                                        let in_x = (x_val / stride_x) as usize;
                                        if in_x < input_w {
                                            for in_c in 0..input_c {
                                                let in_idx = ((b * input_h + in_y) * input_w
                                                    + in_x)
                                                    * input_c
                                                    + in_c;
                                                let ker_idx = ((out_c * kernel_h + ky) * kernel_w
                                                    + kx)
                                                    * input_c
                                                    + in_c;

                                                let lhs =
                                                    input[in_idx] as i32 + conv_params.input_offset;
                                                let rhs = kernel[ker_idx] as i32;
                                                acc += lhs * rhs;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

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
    use crate::Padding2D;
    use crate::types::{Activation, Tile};

    #[test]
    fn test_convolve_s8_simple() {
        let conv_params = ConvParams {
            input_offset: 0,
            output_offset: 0,
            stride: Tile::new(1, 1),
            padding: Padding2D::symmetric(0, 0),
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

    #[test]
    fn test_convolve_s8_padding_is_quantized_zero() {
        // 1x3x3x1 input, single channel:
        // 1 2 3
        // 4 5 6
        // 7 8 9
        let input = [1i8, 2, 3, 4, 5, 6, 7, 8, 9];
        let input_dims = Dims::new(1, 3, 3, 1);

        // 1 output channel, 3x3 all-ones kernel.
        let kernel = [1i8; 9];
        let filter_dims = Dims::new(1, 3, 3, 1);

        // SAME padding (pad=1), stride 1: output stays 3x3.
        let output_dims = Dims::new(1, 3, 3, 1);
        let mut output = [0i8; 9];

        let input_offset = 10;
        let conv_params = ConvParams {
            input_offset,
            output_offset: 0,
            stride: Tile::new(1, 1),
            padding: Padding2D::symmetric(1, 1),
            dilation: Tile::new(1, 1),
            activation: Activation::int8_unconstrained(),
        };
        let quant_params = PerTensorQuantParams::new(1073741824, 0); // 0.5

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

        // Top-left output (0,0): only the bottom-right 2x2 taps are in bounds. TFLite padding is
        // real zero, represented by the input zero-point, so centered padded values contribute
        // zero. In particular, padding must not contribute `input_offset * weight`.
        let raw = (1 + 10) + (2 + 10) + (4 + 10) + (5 + 10);
        let expected = requantize(raw, quant_params.multiplier, quant_params.shift) as i8;
        assert_eq!(output[0], expected);

        let incorrect_full_field =
            requantize(raw + 5 * 10, quant_params.multiplier, quant_params.shift) as i8;
        assert_ne!(output[0], incorrect_full_field);
    }

    #[test]
    fn test_convolve_s8_zero_point_masking_is_noop_when_symmetric() {
        // With input_offset == 0 (symmetric quantization), masking must be a pure no-op: padded
        // taps contribute `0 * weight = 0`, identical to the old "skip" behavior.
        let input = [1i8, 2, 3, 4, 5, 6, 7, 8, 9];
        let input_dims = Dims::new(1, 3, 3, 1);
        let kernel = [1i8; 9];
        let filter_dims = Dims::new(1, 3, 3, 1);
        let output_dims = Dims::new(1, 3, 3, 1);
        let mut output = [0i8; 9];

        let conv_params = ConvParams {
            input_offset: 0,
            output_offset: 0,
            stride: Tile::new(1, 1),
            padding: Padding2D::symmetric(1, 1),
            dilation: Tile::new(1, 1),
            activation: Activation::int8_unconstrained(),
        };
        let quant_params = PerTensorQuantParams::new(1073741824, 0);

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

        let raw = 1 + 2 + 4 + 5;
        let expected = requantize(raw, quant_params.multiplier, quant_params.shift) as i8;
        assert_eq!(output[0], expected);
    }
}
