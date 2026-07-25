//! Concatenation operations along channel/depth dimension for quantized tensors.

use crate::types::{Dims, Error, Result};

/// Concatenates two int8 tensors along the channel (depth) dimension.
pub fn concatenation_s8(
    input1_dims: &Dims,
    input1: &[i8],
    input2_dims: &Dims,
    input2: &[i8],
    output_dims: &Dims,
    output: &mut [i8],
) -> Result<()> {
    if input1_dims.n != input2_dims.n
        || input1_dims.h != input2_dims.h
        || input1_dims.w != input2_dims.w
    {
        return Err(Error::ArgumentError);
    }

    let out_c = input1_dims.c + input2_dims.c;
    if output_dims.c != out_c {
        return Err(Error::ArgumentError);
    }

    let outer_size = (input1_dims.n * input1_dims.h * input1_dims.w) as usize;
    let c1 = input1_dims.c as usize;
    let c2 = input2_dims.c as usize;

    for i in 0..outer_size {
        let in1_slice = &input1[i * c1..(i + 1) * c1];
        let in2_slice = &input2[i * c2..(i + 1) * c2];
        let out_slice = &mut output[i * (c1 + c2)..(i + 1) * (c1 + c2)];

        out_slice[..c1].copy_from_slice(in1_slice);
        out_slice[c1..c1 + c2].copy_from_slice(in2_slice);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concatenation_s8() {
        let in1_dims = Dims::new(1, 1, 1, 2);
        let in1 = [1i8, 2i8];
        let in2_dims = Dims::new(1, 1, 1, 3);
        let in2 = [3i8, 4i8, 5i8];

        let out_dims = Dims::new(1, 1, 1, 5);
        let mut out = [0i8; 5];

        concatenation_s8(&in1_dims, &in1, &in2_dims, &in2, &out_dims, &mut out).unwrap();
        assert_eq!(out, [1, 2, 3, 4, 5]);
    }
}
