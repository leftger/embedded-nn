//! Tensor reshape operations.

use crate::types::{Dims, Error, Result};

/// Reshapes an int8 tensor into target dimensions without moving data (copying contiguous buffer).
pub fn reshape_s8(
    input_dims: &Dims,
    input: &[i8],
    output_dims: &Dims,
    output: &mut [i8],
) -> Result<()> {
    if input_dims.total_size() != output_dims.total_size() {
        return Err(Error::ArgumentError);
    }

    let len = input_dims.total_size();
    if input.len() < len || output.len() < len {
        return Err(Error::ArgumentError);
    }
    output[..len].copy_from_slice(&input[..len]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reshape_s8() {
        let in_dims = Dims::new(1, 2, 2, 1);
        let input = [10i8, 20i8, 30i8, 40i8];

        let out_dims = Dims::new(1, 1, 4, 1);
        let mut output = [0i8; 4];

        reshape_s8(&in_dims, &input, &out_dims, &mut output).unwrap();
        assert_eq!(output, input);
    }
}
