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
