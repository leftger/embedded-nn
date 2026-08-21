//! Const-generic, type-safe tensor abstractions and operator wrappers for `#![no_std]`.
//!
//! Inspired by `microflow-rs`, this module provides compile-time dimension validation,
//! fixed-size buffer backing, sliding-window `TensorView` extraction, and zero-allocation
//! quantized layer execution.

use crate::support::{clamp, quantize_f32_to_s8};
use crate::types::{
    Activation, ConvParams, Dims, DwConvParams, FcParams, FusedActivation, PerChannelQuantParams,
    PerTensorQuantParams, PoolParams, Result, TensorViewPadding, Tile,
};

/// Trait implemented by quantized tensor element types (`i8`, `u8`, `i16`).
pub trait Quantized: Copy + PartialEq + 'static {
    /// Zero representation in the quantized domain.
    const ZERO: Self;
    /// Convert element to `i32` for accumulation.
    fn to_i32(self) -> i32;
    /// Convert `i32` accumulator value to quantized element with clamping.
    fn from_i32_clamped(val: i32) -> Self;
}

impl Quantized for i8 {
    const ZERO: Self = 0;
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn from_i32_clamped(val: i32) -> Self {
        clamp(val, i8::MIN as i32, i8::MAX as i32) as i8
    }
}

impl Quantized for u8 {
    const ZERO: Self = 0;
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn from_i32_clamped(val: i32) -> Self {
        clamp(val, 0, u8::MAX as i32) as u8
    }
}

impl Quantized for i16 {
    const ZERO: Self = 0;
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn from_i32_clamped(val: i32) -> Self {
        clamp(val, i16::MIN as i32, i16::MAX as i32) as i16
    }
}

/// Sliding window extraction region from a 4D tensor with const-generic dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticTensorView<T: Quantized, const ROWS: usize, const COLS: usize, const CHANS: usize>
{
    /// Extracted spatial window data.
    pub buffer: [[[T; CHANS]; COLS]; ROWS],
    /// Validity mask for each spatial position in the window.
    pub mask: [[bool; COLS]; ROWS],
    /// Number of valid in-bound spatial elements.
    pub len: usize,
}

/// A 2-dimensional quantized tensor with statically known shape `[ROWS, COLS]`
/// and `QUANTS` quantization scale/zero-point parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tensor2D<T: Quantized, const ROWS: usize, const COLS: usize, const QUANTS: usize = 1> {
    /// Contiguous row-major storage buffer.
    pub data: [[T; COLS]; ROWS],
    /// Quantization scale(s).
    pub scale: [f32; QUANTS],
    /// Quantization zero-point(s).
    pub zero_point: [i32; QUANTS],
}

impl<T: Quantized, const ROWS: usize, const COLS: usize, const QUANTS: usize>
    Tensor2D<T, ROWS, COLS, QUANTS>
{
    /// Total number of elements in the 2D tensor.
    pub const TOTAL_ELEMENTS: usize = ROWS * COLS;

    /// Creates a new `Tensor2D` from raw buffer and quantization parameters.
    pub const fn new(
        data: [[T; COLS]; ROWS],
        scale: [f32; QUANTS],
        zero_point: [i32; QUANTS],
    ) -> Self {
        Self {
            data,
            scale,
            zero_point,
        }
    }

    /// Creates a zero-initialized `Tensor2D`.
    pub const fn zero(scale: [f32; QUANTS], zero_point: [i32; QUANTS]) -> Self {
        Self {
            data: [[T::ZERO; COLS]; ROWS],
            scale,
            zero_point,
        }
    }

    /// Flattens the tensor to a contiguous slice reference.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.data.as_ptr() as *const T, ROWS * COLS) }
    }

    /// Flattens the tensor to a mutable contiguous slice reference.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, ROWS * COLS) }
    }

    /// Returns the element at `(row, col)`.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> T {
        self.data[row][col]
    }

    /// Sets the element at `(row, col)`.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: T) {
        self.data[row][col] = val;
    }

    /// Transposes the 2D tensor dimensions from `[ROWS, COLS]` to `[COLS, ROWS]`.
    pub fn transpose(self) -> Tensor2D<T, COLS, ROWS, QUANTS> {
        let mut out = Tensor2D::zero(self.scale, self.zero_point);
        for r in 0..ROWS {
            for c in 0..COLS {
                out.data[c][r] = self.data[r][c];
            }
        }
        out
    }

    /// Reshapes this 2D tensor into a 4D tensor shape `[OUT_B, OUT_H, OUT_W, OUT_C]`.
    pub fn reshape_4d<
        const OUT_B: usize,
        const OUT_H: usize,
        const OUT_W: usize,
        const OUT_C: usize,
    >(
        self,
    ) -> Result<Tensor4D<T, OUT_B, OUT_H, OUT_W, OUT_C, QUANTS>> {
        if OUT_B * OUT_H * OUT_W * OUT_C != Self::TOTAL_ELEMENTS {
            return Err(crate::types::Error::ArgumentError);
        }
        let mut out = Tensor4D::zero(self.scale, self.zero_point);
        let src = self.as_slice();
        let dst = out.as_mut_slice();
        dst.copy_from_slice(src);
        Ok(out)
    }
}

impl<const ROWS: usize, const COLS: usize> Tensor2D<i8, ROWS, COLS, 1> {
    /// Quantizes an `f32` 2D array into a `Tensor2D<i8, ROWS, COLS, 1>`.
    pub fn quantize_from_f32(
        input: &[[f32; COLS]; ROWS],
        scale: [f32; 1],
        zero_point: [i32; 1],
    ) -> Self {
        let mut data = [[0i8; COLS]; ROWS];
        for r in 0..ROWS {
            for c in 0..COLS {
                data[r][c] = quantize_f32_to_s8(input[r][c], scale[0], zero_point[0]);
            }
        }
        Self {
            data,
            scale,
            zero_point,
        }
    }

    /// Dequantizes this `Tensor2D<i8>` into an `f32` 2D array.
    pub fn dequantize_to_f32(&self) -> [[f32; COLS]; ROWS] {
        let mut out = [[0.0f32; COLS]; ROWS];
        let s = self.scale[0];
        let zp = self.zero_point[0];
        for r in 0..ROWS {
            for c in 0..COLS {
                out[r][c] = (self.data[r][c] as i32 - zp) as f32 * s;
            }
        }
        out
    }
}

/// A 4-dimensional quantized tensor with statically known shape `[BATCHES, ROWS, COLS, CHANS]`
/// (standard NHWC format for TinyML models).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tensor4D<
    T: Quantized,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
    const QUANTS: usize = 1,
> {
    /// Contiguous NHWC storage buffer.
    pub data: [[[[T; CHANS]; COLS]; ROWS]; BATCHES],
    /// Quantization scale(s).
    pub scale: [f32; QUANTS],
    /// Quantization zero-point(s).
    pub zero_point: [i32; QUANTS],
}

impl<
    T: Quantized,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
    const QUANTS: usize,
> Tensor4D<T, BATCHES, ROWS, COLS, CHANS, QUANTS>
{
    /// Total number of elements in the 4D tensor.
    pub const TOTAL_ELEMENTS: usize = BATCHES * ROWS * COLS * CHANS;

    /// Creates a new `Tensor4D` from nested storage buffer and quantization parameters.
    pub const fn new(
        data: [[[[T; CHANS]; COLS]; ROWS]; BATCHES],
        scale: [f32; QUANTS],
        zero_point: [i32; QUANTS],
    ) -> Self {
        Self {
            data,
            scale,
            zero_point,
        }
    }

    /// Creates a zero-initialized `Tensor4D`.
    pub const fn zero(scale: [f32; QUANTS], zero_point: [i32; QUANTS]) -> Self {
        Self {
            data: [[[[T::ZERO; CHANS]; COLS]; ROWS]; BATCHES],
            scale,
            zero_point,
        }
    }

    /// Flattens the 4D tensor to a contiguous slice reference.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            core::slice::from_raw_parts(
                self.data.as_ptr() as *const T,
                BATCHES * ROWS * COLS * CHANS,
            )
        }
    }

    /// Flattens the 4D tensor to a mutable contiguous slice reference.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            core::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut T,
                BATCHES * ROWS * COLS * CHANS,
            )
        }
    }

    /// Converts the 4D tensor into a 2D matrix shape `[OUT_ROWS, OUT_COLS]`.
    #[inline]
    pub fn reshape_2d<const OUT_ROWS: usize, const OUT_COLS: usize>(
        self,
    ) -> Result<Tensor2D<T, OUT_ROWS, OUT_COLS, QUANTS>> {
        if OUT_ROWS * OUT_COLS != Self::TOTAL_ELEMENTS {
            return Err(crate::types::Error::ArgumentError);
        }
        let mut out = Tensor2D::zero(self.scale, self.zero_point);
        let src_slice = self.as_slice();
        let dst_slice = out.as_mut_slice();
        dst_slice.copy_from_slice(src_slice);
        Ok(out)
    }

    /// Extracts a spatial sliding window view around `(focus_row, focus_col)` for batch index `batch`.
    pub fn view<const VIEW_ROWS: usize, const VIEW_COLS: usize>(
        &self,
        focus: (usize, usize),
        batch: usize,
        padding: TensorViewPadding,
        strides: (usize, usize),
    ) -> StaticTensorView<T, VIEW_ROWS, VIEW_COLS, CHANS> {
        let mut len = VIEW_ROWS * VIEW_COLS;
        let mut mask = [[true; VIEW_COLS]; VIEW_ROWS];
        let mut buffer = [[[T::ZERO; CHANS]; VIEW_COLS]; VIEW_ROWS];

        let shift_r = (VIEW_ROWS - 1) / 2;
        let shift_c = (VIEW_COLS - 1) / 2;

        for m in 0..VIEW_ROWS {
            for n in 0..VIEW_COLS {
                match padding {
                    TensorViewPadding::Same => {
                        let r_idx = (strides.0 * focus.0 + m).checked_sub(shift_r);
                        let c_idx = (strides.1 * focus.1 + n).checked_sub(shift_c);

                        if let (Some(r), Some(c)) = (r_idx, c_idx) {
                            if r < ROWS && c < COLS && batch < BATCHES {
                                buffer[m][n] = self.data[batch][r][c];
                            } else {
                                len -= 1;
                                mask[m][n] = false;
                            }
                        } else {
                            len -= 1;
                            mask[m][n] = false;
                        }
                    }
                    TensorViewPadding::Valid => {
                        let r = strides.0 * focus.0 + m;
                        let c = strides.1 * focus.1 + n;
                        if r < ROWS && c < COLS && batch < BATCHES {
                            buffer[m][n] = self.data[batch][r][c];
                        } else {
                            len -= 1;
                            mask[m][n] = false;
                        }
                    }
                }
            }
        }

        StaticTensorView { buffer, mask, len }
    }
}

impl<const BATCHES: usize, const ROWS: usize, const COLS: usize, const CHANS: usize>
    Tensor4D<i8, BATCHES, ROWS, COLS, CHANS, 1>
{
    /// Quantizes an `f32` 4D array into a `Tensor4D<i8, BATCHES, ROWS, COLS, CHANS, 1>`.
    pub fn quantize_from_f32(
        input: &[[[[f32; CHANS]; COLS]; ROWS]; BATCHES],
        scale: [f32; 1],
        zero_point: [i32; 1],
    ) -> Self {
        let mut data = [[[[0i8; CHANS]; COLS]; ROWS]; BATCHES];
        for b in 0..BATCHES {
            for r in 0..ROWS {
                for c in 0..COLS {
                    for ch in 0..CHANS {
                        data[b][r][c][ch] =
                            quantize_f32_to_s8(input[b][r][c][ch], scale[0], zero_point[0]);
                    }
                }
            }
        }
        Self {
            data,
            scale,
            zero_point,
        }
    }

    /// Dequantizes this `Tensor4D<i8>` into an `f32` 4D array.
    pub fn dequantize_to_f32(&self) -> [[[[f32; CHANS]; COLS]; ROWS]; BATCHES] {
        let mut out = [[[[0.0f32; CHANS]; COLS]; ROWS]; BATCHES];
        let s = self.scale[0];
        let zp = self.zero_point[0];
        for b in 0..BATCHES {
            for r in 0..ROWS {
                for c in 0..COLS {
                    for ch in 0..CHANS {
                        out[b][r][c][ch] = (self.data[b][r][c][ch] as i32 - zp) as f32 * s;
                    }
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Type-Checked Operator Implementations
// ---------------------------------------------------------------------------

/// Performs a statically shape-verified Fully Connected forward pass.
pub fn fully_connected_forward<const BATCHES: usize, const IN_DIM: usize, const OUT_DIM: usize>(
    input: &Tensor2D<i8, BATCHES, IN_DIM>,
    weights: &Tensor2D<i8, OUT_DIM, IN_DIM>,
    bias: Option<&[i32; OUT_DIM]>,
    output_scale: f32,
    output_zero_point: i32,
    fused_activation: FusedActivation,
    multiplier: i32,
    shift: i32,
) -> Result<Tensor2D<i8, BATCHES, OUT_DIM>> {
    let mut output = Tensor2D::zero([output_scale], [output_zero_point]);

    let fc_params = FcParams {
        input_offset: -input.zero_point[0],
        filter_offset: -weights.zero_point[0],
        output_offset: output_zero_point,
        activation: match fused_activation {
            FusedActivation::None => Activation::int8_unconstrained(),
            FusedActivation::Relu => Activation::new(output_zero_point, i8::MAX as i32),
            FusedActivation::Relu6 => {
                let six_q = output_zero_point + ((6.0f32 / output_scale) as i32);
                Activation::new(output_zero_point, six_q.min(i8::MAX as i32))
            }
            FusedActivation::LeakyRelu => Activation::int8_unconstrained(),
        },
    };

    let quant_params = PerTensorQuantParams { multiplier, shift };
    let in_dims = Dims::new(BATCHES as i32, 1, 1, IN_DIM as i32);
    let filter_dims = Dims::new(IN_DIM as i32, 1, 1, OUT_DIM as i32);
    let out_dims = Dims::new(BATCHES as i32, 1, 1, OUT_DIM as i32);

    crate::fully_connected::fully_connected_s8(
        &fc_params,
        &quant_params,
        &in_dims,
        input.as_slice(),
        &filter_dims,
        weights.as_slice(),
        bias.map(|b| b.as_slice()),
        &out_dims,
        output.as_mut_slice(),
    )?;

    Ok(output)
}

/// Performs a statically shape-verified 2D Convolution forward pass.
pub fn conv2d_forward<
    const BATCHES: usize,
    const IN_H: usize,
    const IN_W: usize,
    const IN_C: usize,
    const OUT_C: usize,
    const K_H: usize,
    const K_W: usize,
    const OUT_H: usize,
    const OUT_W: usize,
    const QUANTS: usize,
>(
    input: &Tensor4D<i8, BATCHES, IN_H, IN_W, IN_C>,
    kernel: &Tensor4D<i8, OUT_C, K_H, K_W, IN_C, QUANTS>,
    bias: Option<&[i32; OUT_C]>,
    conv_params: &ConvParams,
    per_channel_quant: Option<&PerChannelQuantParams>,
    per_tensor_quant: Option<&PerTensorQuantParams>,
    output_scale: f32,
    output_zero_point: i32,
) -> Result<Tensor4D<i8, BATCHES, OUT_H, OUT_W, OUT_C>> {
    let mut output = Tensor4D::zero([output_scale], [output_zero_point]);
    let input_dims = Dims::new(BATCHES as i32, IN_H as i32, IN_W as i32, IN_C as i32);
    let filter_dims = Dims::new(OUT_C as i32, K_H as i32, K_W as i32, IN_C as i32);
    let output_dims = Dims::new(BATCHES as i32, OUT_H as i32, OUT_W as i32, OUT_C as i32);

    if let Some(pcq) = per_channel_quant {
        crate::convolution::convolve_per_channel_s8(
            conv_params,
            pcq,
            &input_dims,
            input.as_slice(),
            &filter_dims,
            kernel.as_slice(),
            bias.map(|b| b.as_slice()),
            &output_dims,
            output.as_mut_slice(),
        )?;
    } else if let Some(ptq) = per_tensor_quant {
        crate::convolution::convolve_s8(
            conv_params,
            ptq,
            &input_dims,
            input.as_slice(),
            &filter_dims,
            kernel.as_slice(),
            bias.map(|b| b.as_slice()),
            &output_dims,
            output.as_mut_slice(),
        )?;
    } else {
        return Err(crate::types::Error::ArgumentError);
    }

    Ok(output)
}

/// Performs a statically shape-verified Depthwise 2D Convolution forward pass.
pub fn depthwise_conv2d_forward<
    const BATCHES: usize,
    const IN_H: usize,
    const IN_W: usize,
    const IN_C: usize,
    const OUT_C: usize,
    const K_H: usize,
    const K_W: usize,
    const OUT_H: usize,
    const OUT_W: usize,
    const QUANTS: usize,
>(
    input: &Tensor4D<i8, BATCHES, IN_H, IN_W, IN_C>,
    kernel: &Tensor4D<i8, 1, K_H, K_W, OUT_C, QUANTS>,
    bias: Option<&[i32; OUT_C]>,
    dw_params: &DwConvParams,
    quant_params: &PerChannelQuantParams,
    output_scale: f32,
    output_zero_point: i32,
) -> Result<Tensor4D<i8, BATCHES, OUT_H, OUT_W, OUT_C>> {
    let mut output = Tensor4D::zero([output_scale], [output_zero_point]);
    let input_dims = Dims::new(BATCHES as i32, IN_H as i32, IN_W as i32, IN_C as i32);
    let filter_dims = Dims::new(1, K_H as i32, K_W as i32, OUT_C as i32);
    let output_dims = Dims::new(BATCHES as i32, OUT_H as i32, OUT_W as i32, OUT_C as i32);

    crate::convolution::depthwise_conv_per_channel_s8(
        dw_params,
        quant_params,
        &input_dims,
        input.as_slice(),
        &filter_dims,
        kernel.as_slice(),
        bias.map(|b| b.as_slice()),
        &output_dims,
        output.as_mut_slice(),
    )?;

    Ok(output)
}

/// Performs a statically shape-verified 2D Max Pooling forward pass.
pub fn max_pool2d_forward<
    const BATCHES: usize,
    const IN_H: usize,
    const IN_W: usize,
    const CHANS: usize,
    const K_H: usize,
    const K_W: usize,
    const OUT_H: usize,
    const OUT_W: usize,
>(
    input: &Tensor4D<i8, BATCHES, IN_H, IN_W, CHANS>,
    pool_params: &PoolParams,
) -> Result<Tensor4D<i8, BATCHES, OUT_H, OUT_W, CHANS>> {
    let mut output = Tensor4D::zero(input.scale, input.zero_point);
    let filter_dims = Tile::new(K_W as i32, K_H as i32);
    let input_dims = Dims::new(BATCHES as i32, IN_H as i32, IN_W as i32, CHANS as i32);
    let output_dims = Dims::new(BATCHES as i32, OUT_H as i32, OUT_W as i32, CHANS as i32);

    crate::pooling::max_pool_s8(
        pool_params,
        &filter_dims,
        &input_dims,
        input.as_slice(),
        &output_dims,
        output.as_mut_slice(),
    )?;

    Ok(output)
}

/// Performs a statically shape-verified 2D Average Pooling forward pass.
pub fn avg_pool2d_forward<
    const BATCHES: usize,
    const IN_H: usize,
    const IN_W: usize,
    const CHANS: usize,
    const K_H: usize,
    const K_W: usize,
    const OUT_H: usize,
    const OUT_W: usize,
>(
    input: &Tensor4D<i8, BATCHES, IN_H, IN_W, CHANS>,
    pool_params: &PoolParams,
) -> Result<Tensor4D<i8, BATCHES, OUT_H, OUT_W, CHANS>> {
    let mut output = Tensor4D::zero(input.scale, input.zero_point);
    let filter_dims = Tile::new(K_W as i32, K_H as i32);
    let input_dims = Dims::new(BATCHES as i32, IN_H as i32, IN_W as i32, CHANS as i32);
    let output_dims = Dims::new(BATCHES as i32, OUT_H as i32, OUT_W as i32, CHANS as i32);

    crate::pooling::avg_pool_s8(
        pool_params,
        &filter_dims,
        &input_dims,
        input.as_slice(),
        &output_dims,
        output.as_mut_slice(),
    )?;

    Ok(output)
}

/// Performs a statically shape-verified Softmax forward pass on a 2D tensor.
pub fn softmax_forward<const BATCHES: usize, const CHANS: usize>(
    input: &Tensor2D<i8, BATCHES, CHANS>,
    multiplier: i32,
    shift: i32,
    diff_min: i32,
) -> Result<Tensor2D<i8, BATCHES, CHANS>> {
    let mut output = Tensor2D::zero([1.0 / 256.0], [-128]);

    crate::softmax::softmax_s8(
        input.as_slice(),
        BATCHES,
        CHANS,
        multiplier,
        shift,
        diff_min,
        output.as_mut_slice(),
    )?;

    Ok(output)
}

/// Applies ReLU activation to a 2D tensor in place or returning a copy.
pub fn relu_forward<const ROWS: usize, const COLS: usize>(
    input: &Tensor2D<i8, ROWS, COLS>,
) -> Tensor2D<i8, ROWS, COLS> {
    let mut out = *input;
    let zp = input.zero_point[0] as i8;
    for r in 0..ROWS {
        for c in 0..COLS {
            if out.data[r][c] < zp {
                out.data[r][c] = zp;
            }
        }
    }
    out
}

/// Applies ReLU6 activation to a 2D tensor in place or returning a copy.
pub fn relu6_forward<const ROWS: usize, const COLS: usize>(
    input: &Tensor2D<i8, ROWS, COLS>,
) -> Tensor2D<i8, ROWS, COLS> {
    let mut out = *input;
    let zp = input.zero_point[0];
    let six_q = zp + ((6.0f32 / input.scale[0]) as i32);
    let max_q = six_q.min(i8::MAX as i32) as i8;
    let min_q = zp as i8;

    for r in 0..ROWS {
        for c in 0..COLS {
            let v = out.data[r][c];
            if v < min_q {
                out.data[r][c] = min_q;
            } else if v > max_q {
                out.data[r][c] = max_q;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Padding2D;

    #[test]
    fn test_tensor2d_shape_and_slice() {
        let t = Tensor2D::<i8, 2, 3>::new([[1, 2, 3], [4, 5, 6]], [0.1], [0]);
        assert_eq!(t.get(0, 2), 3);
        assert_eq!(t.get(1, 0), 4);
        assert_eq!(t.as_slice(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_tensor2d_transpose() {
        let t = Tensor2D::<i8, 2, 3>::new([[1, 2, 3], [4, 5, 6]], [0.1], [0]);
        let transposed = t.transpose();
        assert_eq!(transposed.data, [[1, 4], [2, 5], [3, 6]]);
    }

    #[test]
    fn test_tensor4d_flatten_and_reshape() {
        let t = Tensor4D::<i8, 1, 2, 2, 1>::new([[[[10], [20]], [[30], [40]]]], [0.2], [0]);
        assert_eq!(t.as_slice(), &[10, 20, 30, 40]);
        let t2 = t.reshape_2d::<2, 2>().unwrap();
        assert_eq!(t2.get(0, 0), 10);
        assert_eq!(t2.get(1, 1), 40);

        let t4 = t2.reshape_4d::<1, 2, 2, 1>().unwrap();
        assert_eq!(t4.as_slice(), &[10, 20, 30, 40]);
    }

    #[test]
    fn test_tensor_view_extraction() {
        let t = Tensor4D::<i8, 1, 3, 3, 1>::new(
            [[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]],
            [0.1],
            [0],
        );

        // Center focus (1, 1) with 3x3 window Same padding
        let view: StaticTensorView<i8, 3, 3, 1> =
            t.view((1, 1), 0, TensorViewPadding::Same, (1, 1));
        assert_eq!(view.len, 9);
        assert_eq!(view.buffer[1][1][0], 5);
        assert!(view.mask[0][0]);

        // Top-left focus (0, 0) with 3x3 window Valid padding
        let view_valid: StaticTensorView<i8, 2, 2, 1> =
            t.view((0, 0), 0, TensorViewPadding::Valid, (1, 1));
        assert_eq!(view_valid.len, 4);
        assert_eq!(view_valid.buffer[0][0][0], 1);
        assert_eq!(view_valid.buffer[1][1][0], 5);
    }

    #[test]
    fn test_quantize_dequantize_f32() {
        let float_in = [[1.0f32, 2.0f32], [3.0f32, 4.0f32]];
        let scale = [0.5f32];
        let zp = [0i32];

        let q_tensor = Tensor2D::quantize_from_f32(&float_in, scale, zp);
        assert_eq!(q_tensor.data, [[2i8, 4i8], [6i8, 8i8]]);

        let deq = q_tensor.dequantize_to_f32();
        assert_eq!(deq, float_in);
    }

    #[test]
    fn test_conv2d_forward() {
        let input = Tensor4D::<i8, 1, 3, 3, 1>::new(
            [[[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]]],
            [1.0],
            [0],
        );

        let kernel = Tensor4D::<i8, 1, 2, 2, 1, 1>::new([[[[1], [1]], [[1], [1]]]], [1.0], [0]);

        let conv_params = ConvParams {
            input_offset: 0,
            output_offset: 0,
            stride: Tile::new(1, 1),
            padding: Padding2D::new(0, 0, 0, 0),
            dilation: Tile::new(1, 1),
            activation: Activation::int8_unconstrained(),
        };

        let quant = PerTensorQuantParams::new(1073741824, 1); // 1.0 multiplier (0.5 in Q31 with +1 shift)
        let out = conv2d_forward::<1, 3, 3, 1, 1, 2, 2, 2, 2, 1>(
            &input,
            &kernel,
            None,
            &conv_params,
            None,
            Some(&quant),
            1.0,
            0,
        )
        .unwrap();

        assert_eq!(out.data, [[[[4], [4]], [[4], [4]]]]);
    }

    #[test]
    fn test_max_pool2d_forward() {
        let input = Tensor4D::<i8, 1, 4, 4, 1>::new(
            [[
                [[1], [3], [2], [4]],
                [[5], [6], [7], [8]],
                [[9], [2], [1], [3]],
                [[4], [0], [5], [6]],
            ]],
            [1.0],
            [0],
        );

        let pool_params = PoolParams {
            stride: Tile::new(2, 2),
            padding: Padding2D::new(0, 0, 0, 0),
            activation: Activation::int8_unconstrained(),
        };

        let out = max_pool2d_forward::<1, 4, 4, 1, 2, 2, 2, 2>(&input, &pool_params).unwrap();
        assert_eq!(out.data, [[[[6], [8]], [[9], [6]]]]);
    }
}
