//! Core types and parameter structures for `embedded-nn`.

/// Error types for `embedded-nn` operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// One or more input arguments/dimensions are invalid or incompatible.
    ArgumentError,
    /// Function or operation is not implemented for the given configuration.
    NoImplementation,
    /// Execution or calculation error.
    Failure,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::ArgumentError => write!(f, "Invalid or incompatible arguments"),
            Error::NoImplementation => write!(f, "No implementation available"),
            Error::Failure => write!(f, "Operation failure"),
        }
    }
}

/// Result type alias for `embedded-nn`.
pub type Result<T> = core::result::Result<T, Error>;

/// Dimensions for 4D Tensors (Batch, Height, Width, Channels / Output Channels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Dims {
    /// Batch size or generic dimension `n`.
    pub n: i32,
    /// Height `h`.
    pub h: i32,
    /// Width `w`.
    pub w: i32,
    /// Channels `c`.
    pub c: i32,
}

impl Dims {
    /// Creates a new 4D Dims structure.
    pub const fn new(n: i32, h: i32, w: i32, c: i32) -> Self {
        Self { n, h, w, c }
    }

    /// Computes total number of elements.
    pub const fn total_size(&self) -> usize {
        (self.n * self.h * self.w * self.c) as usize
    }
}

/// Tile or kernel spatial dimensions (Width, Height).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Tile {
    /// Width `w`.
    pub w: i32,
    /// Height `h`.
    pub h: i32,
}

impl Tile {
    /// Creates a new Tile structure.
    pub const fn new(w: i32, h: i32) -> Self {
        Self { w, h }
    }
}

/// Quantized activation clamping range (min, max).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Activation {
    /// Minimum clamping threshold.
    pub min: i32,
    /// Maximum clamping threshold.
    pub max: i32,
}

impl Activation {
    /// Creates a new Activation clamping range.
    pub const fn new(min: i32, max: i32) -> Self {
        Self { min, max }
    }

    /// Returns an unconstrained range for int8 (-128 to 127).
    pub const fn int8_unconstrained() -> Self {
        Self {
            min: i8::MIN as i32,
            max: i8::MAX as i32,
        }
    }

    /// Returns an unconstrained range for int16 (-32768 to 32767).
    pub const fn int16_unconstrained() -> Self {
        Self {
            min: i16::MIN as i32,
            max: i16::MAX as i32,
        }
    }
}

/// Fused activation function type (inspired by MicroFlow and TFLite).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FusedActivation {
    /// No post-layer activation.
    #[default]
    None,
    /// Rectified Linear Unit (ReLU: `max(x, 0)`).
    Relu,
    /// Rectified Linear Unit 6 (ReLU6: `clamp(x, 0, 6)`).
    Relu6,
    /// Leaky Rectified Linear Unit.
    LeakyRelu,
}

impl FusedActivation {
    /// Converts fused activation into an [`Activation`] clamping range for quantized types.
    pub const fn to_activation(&self, is_int16: bool) -> Activation {
        match self {
            FusedActivation::None => {
                if is_int16 {
                    Activation::int16_unconstrained()
                } else {
                    Activation::int8_unconstrained()
                }
            }
            FusedActivation::Relu => {
                Activation::new(0, if is_int16 { i16::MAX as i32 } else { i8::MAX as i32 })
            }
            FusedActivation::Relu6 => Activation::new(0, 6),
            FusedActivation::LeakyRelu => {
                if is_int16 {
                    Activation::int16_unconstrained()
                } else {
                    Activation::int8_unconstrained()
                }
            }
        }
    }
}

/// Spatial padding strategy for tensor window views (inspired by MicroFlow).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TensorViewPadding {
    /// Missing boundary elements are padded with zero / pad value.
    #[default]
    Same,
    /// Window extraction is strictly constrained within valid tensor bounds.
    Valid,
}

/// A spatial window view over a 4D Tensor `(N, H, W, C)`.
/// Provides zero-allocation spatial patch indexing and bounds-aware element access.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a, T> {
    /// Reference to underlying contiguous flat tensor slice.
    pub data: &'a [T],
    /// Tensor dimensions `(N, H, W, C)`.
    pub dims: Dims,
    /// Spatial padding mode.
    pub padding: TensorViewPadding,
    /// Window stride `(W, H)`.
    pub stride: Tile,
    /// Window kernel size `(W, H)`.
    pub kernel: Tile,
}

impl<'a, T: Copy> TensorView<'a, T> {
    /// Creates a new `TensorView` over data buffer and dimensions.
    pub const fn new(
        data: &'a [T],
        dims: Dims,
        padding: TensorViewPadding,
        stride: Tile,
        kernel: Tile,
    ) -> Self {
        Self {
            data,
            dims,
            padding,
            stride,
            kernel,
        }
    }

    /// Fetches the value at index `(batch, r, c, ch)`. Returns `pad_value` if position is out of bounds.
    #[inline]
    pub fn get_or_pad(&self, batch: usize, r: isize, c: isize, ch: usize, pad_value: T) -> T {
        if r < 0 || r >= self.dims.h as isize || c < 0 || c >= self.dims.w as isize {
            pad_value
        } else {
            let idx = ((batch * self.dims.h as usize + r as usize) * self.dims.w as usize + c as usize)
                * self.dims.c as usize
                + ch;
            if idx < self.data.len() {
                self.data[idx]
            } else {
                pad_value
            }
        }
    }

    /// Computes output height and width for spatial window extraction under current padding & stride.
    pub fn output_spatial_dims(&self) -> (usize, usize) {
        let (h, w) = (self.dims.h as usize, self.dims.w as usize);
        let (kh, kw) = (self.kernel.h as usize, self.kernel.w as usize);
        let (sh, sw) = (self.stride.h as usize, self.stride.w as usize);

        match self.padding {
            TensorViewPadding::Valid => (
                if h >= kh { (h - kh) / sh + 1 } else { 0 },
                if w >= kw { (w - kw) / sw + 1 } else { 0 },
            ),
            TensorViewPadding::Same => (
                (h + sh - 1) / sh,
                (w + sw - 1) / sw,
            ),
        }
    }
}

/// Per-tensor quantization parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerTensorQuantParams {
    /// Requantization multiplier (Q31).
    pub multiplier: i32,
    /// Requantization shift.
    pub shift: i32,
}

impl PerTensorQuantParams {
    /// Creates new per-tensor quantization parameters.
    pub const fn new(multiplier: i32, shift: i32) -> Self {
        Self { multiplier, shift }
    }
}

/// Per-channel quantization parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerChannelQuantParams<'a> {
    /// Per-channel multipliers.
    pub multiplier: &'a [i32],
    /// Per-channel shifts.
    pub shift: &'a [i32],
}

impl<'a> PerChannelQuantParams<'a> {
    /// Creates new per-channel quantization parameters.
    pub const fn new(multiplier: &'a [i32], shift: &'a [i32]) -> Self {
        Self { multiplier, shift }
    }
}

/// Unified quantization parameters (either per-tensor or per-channel).
#[derive(Debug, Clone)]
pub enum QuantParams<'a> {
    /// Per-tensor quantization parameters.
    PerTensor(PerTensorQuantParams),
    /// Per-channel quantization parameters.
    PerChannel(PerChannelQuantParams<'a>),
}

/// Parameters for Convolution layer operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvParams {
    /// Input zero point offset (-zero_point).
    pub input_offset: i32,
    /// Output zero point offset (+zero_point).
    pub output_offset: i32,
    /// Stride (width, height).
    pub stride: Tile,
    /// Padding (width, height).
    pub padding: Tile,
    /// Dilation (width, height).
    pub dilation: Tile,
    /// Output activation range.
    pub activation: Activation,
}

/// Parameters for Depthwise Convolution layer operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DwConvParams {
    /// Input zero point offset (-zero_point).
    pub input_offset: i32,
    /// Output zero point offset (+zero_point).
    pub output_offset: i32,
    /// Channel multiplier.
    pub ch_mult: i32,
    /// Stride (width, height).
    pub stride: Tile,
    /// Padding (width, height).
    pub padding: Tile,
    /// Dilation (width, height).
    pub dilation: Tile,
    /// Output activation range.
    pub activation: Activation,
}

/// Parameters for Fully Connected (Linear) layer operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FcParams {
    /// Input zero point offset (-zero_point).
    pub input_offset: i32,
    /// Filter zero point offset (-zero_point).
    pub filter_offset: i32,
    /// Output zero point offset (+zero_point).
    pub output_offset: i32,
    /// Output activation range.
    pub activation: Activation,
}

/// Parameters for Pooling operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolParams {
    /// Stride (width, height).
    pub stride: Tile,
    /// Padding (width, height).
    pub padding: Tile,
    /// Output activation range.
    pub activation: Activation,
}

/// Parameters for Softmax operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoftmaxParams {
    /// Input scale for fixed-point exponential calculation.
    pub input_scale: i32,
    /// Minimum input difference threshold.
    pub diff_min: i32,
}

/// Context for scratchpad buffers (optional).
#[derive(Debug)]
pub struct Context<'a> {
    /// Scratch buffer slice.
    pub buf: Option<&'a mut [u8]>,
}

impl<'a> Context<'a> {
    /// Creates a context without scratch memory.
    pub const fn empty() -> Self {
        Self { buf: None }
    }

    /// Creates a context with given scratch buffer.
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self { buf: Some(buf) }
    }
}
