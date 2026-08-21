//! Pooling layer operations (Max Pooling, Average Pooling) for quantized neural networks.

use crate::support::clamp;
use crate::types::{Dims, PoolParams, Result, Tile};

/// Performs Max Pooling 2D for int8 tensors.
pub fn max_pool_s8(
    pool_params: &PoolParams,
    filter_dims: &Tile,
    input_dims: &Dims,
    input: &[i8],
    output_dims: &Dims,
    output: &mut [i8],
) -> Result<()> {
    let input_batches = input_dims.n as usize;
    let input_h = input_dims.h as usize;
    let input_w = input_dims.w as usize;
    let channels = input_dims.c as usize;

    let kernel_h = filter_dims.h as usize;
    let kernel_w = filter_dims.w as usize;

    let output_h = output_dims.h as usize;
    let output_w = output_dims.w as usize;

    for b in 0..input_batches {
        for out_y in 0..output_h {
            let base_y = out_y as i32 * pool_params.stride.h - pool_params.padding.top;
            for out_x in 0..output_w {
                let base_x = out_x as i32 * pool_params.stride.w - pool_params.padding.left;

                for c in 0..channels {
                    let mut max_val = i8::MIN as i32;

                    for ky in 0..kernel_h {
                        let in_y = base_y + ky as i32;
                        if in_y >= 0 && in_y < input_dims.h {
                            for kx in 0..kernel_w {
                                let in_x = base_x + kx as i32;
                                if in_x >= 0 && in_x < input_dims.w {
                                    let in_idx = ((b * input_h + in_y as usize) * input_w
                                        + in_x as usize)
                                        * channels
                                        + c;
                                    let val = input[in_idx] as i32;
                                    if val > max_val {
                                        max_val = val;
                                    }
                                }
                            }
                        }
                    }

                    max_val = clamp(
                        max_val,
                        pool_params.activation.min,
                        pool_params.activation.max,
                    );
                    let out_idx = ((b * output_h + out_y) * output_w + out_x) * channels + c;
                    output[out_idx] = max_val as i8;
                }
            }
        }
    }
    Ok(())
}

/// Performs Average Pooling 2D for int8 tensors.
pub fn avg_pool_s8(
    pool_params: &PoolParams,
    filter_dims: &Tile,
    input_dims: &Dims,
    input: &[i8],
    output_dims: &Dims,
    output: &mut [i8],
) -> Result<()> {
    let input_batches = input_dims.n as usize;
    let input_h = input_dims.h as usize;
    let input_w = input_dims.w as usize;
    let channels = input_dims.c as usize;

    let kernel_h = filter_dims.h as usize;
    let kernel_w = filter_dims.w as usize;

    let output_h = output_dims.h as usize;
    let output_w = output_dims.w as usize;

    for b in 0..input_batches {
        for out_y in 0..output_h {
            let base_y = out_y as i32 * pool_params.stride.h - pool_params.padding.top;
            for out_x in 0..output_w {
                let base_x = out_x as i32 * pool_params.stride.w - pool_params.padding.left;

                for c in 0..channels {
                    let mut sum: i32 = 0;
                    let mut count: i32 = 0;

                    for ky in 0..kernel_h {
                        let in_y = base_y + ky as i32;
                        if in_y >= 0 && in_y < input_dims.h {
                            for kx in 0..kernel_w {
                                let in_x = base_x + kx as i32;
                                if in_x >= 0 && in_x < input_dims.w {
                                    let in_idx = ((b * input_h + in_y as usize) * input_w
                                        + in_x as usize)
                                        * channels
                                        + c;
                                    sum += input[in_idx] as i32;
                                    count += 1;
                                }
                            }
                        }
                    }

                    let avg = if count > 0 {
                        if sum >= 0 {
                            (sum + count / 2) / count
                        } else {
                            (sum - count / 2) / count
                        }
                    } else {
                        0
                    };

                    let final_val =
                        clamp(avg, pool_params.activation.min, pool_params.activation.max);
                    let out_idx = ((b * output_h + out_y) * output_w + out_x) * channels + c;
                    output[out_idx] = final_val as i8;
                }
            }
        }
    }
    Ok(())
}

/// Performs Max Pooling 2D for int16 tensors.
pub fn max_pool_s16(
    pool_params: &PoolParams,
    filter_dims: &Tile,
    input_dims: &Dims,
    input: &[i16],
    output_dims: &Dims,
    output: &mut [i16],
) -> Result<()> {
    let input_batches = input_dims.n as usize;
    let input_h = input_dims.h as usize;
    let input_w = input_dims.w as usize;
    let channels = input_dims.c as usize;

    let kernel_h = filter_dims.h as usize;
    let kernel_w = filter_dims.w as usize;

    let output_h = output_dims.h as usize;
    let output_w = output_dims.w as usize;

    for b in 0..input_batches {
        for out_y in 0..output_h {
            let base_y = out_y as i32 * pool_params.stride.h - pool_params.padding.top;
            for out_x in 0..output_w {
                let base_x = out_x as i32 * pool_params.stride.w - pool_params.padding.left;

                for c in 0..channels {
                    let mut max_val = i16::MIN as i32;

                    for ky in 0..kernel_h {
                        let in_y = base_y + ky as i32;
                        if in_y >= 0 && in_y < input_dims.h {
                            for kx in 0..kernel_w {
                                let in_x = base_x + kx as i32;
                                if in_x >= 0 && in_x < input_dims.w {
                                    let in_idx = ((b * input_h + in_y as usize) * input_w
                                        + in_x as usize)
                                        * channels
                                        + c;
                                    let val = input[in_idx] as i32;
                                    if val > max_val {
                                        max_val = val;
                                    }
                                }
                            }
                        }
                    }

                    max_val = clamp(
                        max_val,
                        pool_params.activation.min,
                        pool_params.activation.max,
                    );
                    let out_idx = ((b * output_h + out_y) * output_w + out_x) * channels + c;
                    output[out_idx] = max_val as i16;
                }
            }
        }
    }
    Ok(())
}

/// Performs Average Pooling 2D for int16 tensors.
pub fn avg_pool_s16(
    pool_params: &PoolParams,
    filter_dims: &Tile,
    input_dims: &Dims,
    input: &[i16],
    output_dims: &Dims,
    output: &mut [i16],
) -> Result<()> {
    let input_batches = input_dims.n as usize;
    let input_h = input_dims.h as usize;
    let input_w = input_dims.w as usize;
    let channels = input_dims.c as usize;

    let kernel_h = filter_dims.h as usize;
    let kernel_w = filter_dims.w as usize;

    let output_h = output_dims.h as usize;
    let output_w = output_dims.w as usize;

    for b in 0..input_batches {
        for out_y in 0..output_h {
            let base_y = out_y as i32 * pool_params.stride.h - pool_params.padding.top;
            for out_x in 0..output_w {
                let base_x = out_x as i32 * pool_params.stride.w - pool_params.padding.left;

                for c in 0..channels {
                    let mut sum: i64 = 0;
                    let mut count: i64 = 0;

                    for ky in 0..kernel_h {
                        let in_y = base_y + ky as i32;
                        if in_y >= 0 && in_y < input_dims.h {
                            for kx in 0..kernel_w {
                                let in_x = base_x + kx as i32;
                                if in_x >= 0 && in_x < input_dims.w {
                                    let in_idx = ((b * input_h + in_y as usize) * input_w
                                        + in_x as usize)
                                        * channels
                                        + c;
                                    sum += input[in_idx] as i64;
                                    count += 1;
                                }
                            }
                        }
                    }

                    let avg = if count > 0 {
                        if sum >= 0 {
                            (sum + count / 2) / count
                        } else {
                            (sum - count / 2) / count
                        }
                    } else {
                        0
                    };

                    let final_val = clamp(
                        avg as i32,
                        pool_params.activation.min,
                        pool_params.activation.max,
                    );
                    let out_idx = ((b * output_h + out_y) * output_w + out_x) * channels + c;
                    output[out_idx] = final_val as i16;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Activation, Padding2D};

    #[test]
    fn test_max_pool_s8() {
        let pool_params = PoolParams {
            stride: Tile::new(2, 2),
            padding: Padding2D::symmetric(0, 0),
            activation: Activation::int8_unconstrained(),
        };

        let filter_dims = Tile::new(2, 2);
        let input_dims = Dims::new(1, 4, 4, 1);
        let input = [
            1i8, 2i8, 5i8, 6i8, 3i8, 4i8, 7i8, 8i8, 9i8, 10i8, 13i8, 14i8, 11i8, 12i8, 15i8, 16i8,
        ];

        let output_dims = Dims::new(1, 2, 2, 1);
        let mut output = [0i8; 4];

        max_pool_s8(
            &pool_params,
            &filter_dims,
            &input_dims,
            &input,
            &output_dims,
            &mut output,
        )
        .unwrap();

        assert_eq!(output, [4, 8, 12, 16]);
    }
}
