//! Tensor transpose operations.

use crate::types::{Dims, Result};

/// Transposes a 2D matrix (rows x cols) for int8 tensors.
pub fn transpose_2d_s8(rows: usize, cols: usize, input: &[i8], output: &mut [i8]) -> Result<()> {
    for r in 0..rows {
        for c in 0..cols {
            output[c * rows + r] = input[r * cols + c];
        }
    }
    Ok(())
}

/// Transposes a 4D tensor (N, H, W, C) -> (N, W, H, C).
pub fn transpose_spatial_s8(input_dims: &Dims, input: &[i8], output: &mut [i8]) -> Result<()> {
    let batches = input_dims.n as usize;
    let input_h = input_dims.h as usize;
    let input_w = input_dims.w as usize;
    let channels = input_dims.c as usize;

    for b in 0..batches {
        for y in 0..input_h {
            for x in 0..input_w {
                let in_idx = ((b * input_h + y) * input_w + x) * channels;
                let out_idx = ((b * input_w + x) * input_h + y) * channels;

                output[out_idx..out_idx + channels]
                    .copy_from_slice(&input[in_idx..in_idx + channels]);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_2d_s8() {
        let input = [1i8, 2i8, 3i8, 4i8, 5i8, 6i8]; // 2 rows x 3 cols
        let mut out = [0i8; 6]; // 3 rows x 2 cols
        transpose_2d_s8(2, 3, &input, &mut out).unwrap();
        assert_eq!(out, [1, 4, 2, 5, 3, 6]);
    }
}
