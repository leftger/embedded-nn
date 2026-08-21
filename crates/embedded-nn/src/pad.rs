//! Tensor padding operations for quantized tensors.

use crate::types::{Dims, Result, Tile};

/// Pads an int8 4D tensor with a specified pad value.
pub fn pad_s8(
    input_dims: &Dims,
    input: &[i8],
    padding_before: &Tile,
    _padding_after: &Tile,
    pad_value: i8,
    output_dims: &Dims,
    output: &mut [i8],
) -> Result<()> {
    let batches = input_dims.n as usize;
    let input_h = input_dims.h as usize;
    let input_w = input_dims.w as usize;
    let channels = input_dims.c as usize;

    let pad_h_before = padding_before.h as usize;
    let pad_w_before = padding_before.w as usize;

    let output_h = output_dims.h as usize;
    let output_w = output_dims.w as usize;

    output.fill(pad_value);

    for b in 0..batches {
        for y in 0..input_h {
            let out_y = y + pad_h_before;
            if out_y < output_h {
                for x in 0..input_w {
                    let out_x = x + pad_w_before;
                    if out_x < output_w {
                        let in_idx = ((b * input_h + y) * input_w + x) * channels;
                        let out_idx = ((b * output_h + out_y) * output_w + out_x) * channels;

                        output[out_idx..out_idx + channels]
                            .copy_from_slice(&input[in_idx..in_idx + channels]);
                    }
                }
            }
        }
    }

    Ok(())
}

/// Averages int8 NHWC values over the selected spatial/channel axes.
pub fn reduce_mean_s8(
    batches: usize,
    height: usize,
    width: usize,
    channels: usize,
    reduce_height: bool,
    reduce_width: bool,
    reduce_channels: bool,
    input: &[i8],
    output: &mut [i8],
) -> Result<()> {
    let out_h = if reduce_height { 1 } else { height };
    let out_w = if reduce_width { 1 } else { width };
    let out_c = if reduce_channels { 1 } else { channels };
    let expected = batches * out_h * out_w * out_c;
    if output.len() < expected {
        return Err(crate::types::Error::ArgumentError);
    }

    for b in 0..batches {
        for oh in 0..out_h {
            for ow in 0..out_w {
                for oc in 0..out_c {
                    let h_range = if reduce_height { 0..height } else { oh..oh + 1 };
                    let w_range = if reduce_width { 0..width } else { ow..ow + 1 };
                    let c_range = if reduce_channels {
                        0..channels
                    } else {
                        oc..oc + 1
                    };
                    let mut sum = 0i32;
                    let mut count = 0i32;
                    for h in h_range.clone() {
                        for w in w_range.clone() {
                            for c in c_range.clone() {
                                let idx = ((b * height + h) * width + w) * channels + c;
                                sum += i32::from(input[idx]);
                                count += 1;
                            }
                        }
                    }
                    let mean = if count == 0 { 0 } else { (sum + count / 2) / count };
                    let out_idx = ((b * out_h + oh) * out_w + ow) * out_c + oc;
                    output[out_idx] = mean.clamp(-128, 127) as i8;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_s8() {
        let in_dims = Dims::new(1, 2, 2, 1);
        let input = [1i8, 2i8, 3i8, 4i8];

        let pad_before = Tile::new(1, 1);
        let pad_after = Tile::new(1, 1);

        let out_dims = Dims::new(1, 4, 4, 1);
        let mut out = [0i8; 16];

        pad_s8(
            &in_dims,
            &input,
            &pad_before,
            &pad_after,
            0i8,
            &out_dims,
            &mut out,
        )
        .unwrap();

        // Check center 2x2 elements match input
        assert_eq!(out[5], 1);
        assert_eq!(out[6], 2);
        assert_eq!(out[9], 3);
        assert_eq!(out[10], 4);
        // Border element is 0
        assert_eq!(out[0], 0);
    }
}
