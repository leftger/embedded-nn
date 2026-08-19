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
