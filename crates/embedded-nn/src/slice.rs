//! Strided slice for quantized NHWC tensors.

use crate::types::{Dims, Error, Result};

fn clamp_index(index: i32, dim: usize) -> usize {
    if index < 0 {
        (dim as i32 + index).max(0) as usize
    } else {
        (index as usize).min(dim)
    }
}

/// Strided slice over a rank-4 NHWC int8 tensor. Rank-1 tensors use `n=h=w=1`.
pub fn strided_slice_s8(
    input_dims: &Dims,
    begin: &[i32; 4],
    end: &[i32; 4],
    stride: &[i32; 4],
    input: &[i8],
    output: &mut [i8],
) -> Result<()> {
    let dims = [
        input_dims.n as usize,
        input_dims.h as usize,
        input_dims.w as usize,
        input_dims.c as usize,
    ];
    let mut starts = [0usize; 4];
    let mut stops = [0usize; 4];
    let mut steps = [1isize; 4];
    for i in 0..4 {
        if stride[i] == 0 {
            return Err(Error::ArgumentError);
        }
        steps[i] = stride[i] as isize;
        starts[i] = clamp_index(begin[i], dims[i]);
        stops[i] = clamp_index(end[i], dims[i]);
    }

    let mut out_i = 0usize;
    let mut n = starts[0] as isize;
    while (steps[0] > 0 && n < stops[0] as isize) || (steps[0] < 0 && n > stops[0] as isize) {
        let mut h = starts[1] as isize;
        while (steps[1] > 0 && h < stops[1] as isize) || (steps[1] < 0 && h > stops[1] as isize) {
            let mut w = starts[2] as isize;
            while (steps[2] > 0 && w < stops[2] as isize) || (steps[2] < 0 && w > stops[2] as isize)
            {
                let mut c = starts[3] as isize;
                while (steps[3] > 0 && c < stops[3] as isize)
                    || (steps[3] < 0 && c > stops[3] as isize)
                {
                    let idx = (((n as usize) * dims[1] + h as usize) * dims[2] + w as usize)
                        * dims[3]
                        + c as usize;
                    if out_i >= output.len() || idx >= input.len() {
                        return Err(Error::ArgumentError);
                    }
                    output[out_i] = input[idx];
                    out_i += 1;
                    c += steps[3];
                }
                w += steps[2];
            }
            h += steps[1];
        }
        n += steps[0];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slices_channels() {
        let dims = Dims::new(1, 1, 1, 4);
        let input = [1i8, 2, 3, 4];
        let mut out = [0i8; 2];
        strided_slice_s8(
            &dims,
            &[0, 0, 0, 1],
            &[1, 1, 1, 3],
            &[1, 1, 1, 1],
            &input,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [2, 3]);
    }
}
