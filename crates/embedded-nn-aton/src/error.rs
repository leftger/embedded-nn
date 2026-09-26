use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum AtonError {
    #[error("Unsupported layer operation for ATON NPU hardware: {0}")]
    UnsupportedLayer(String),

    #[error("Tensor not found with ID {0}")]
    TensorNotFound(usize),

    #[error("Invalid tensor shape for streaming engine: {0}")]
    InvalidShape(String),

    #[error("Relocation offset {0} is out of bounds (blob length: {1})")]
    RelocationOutOfBounds(usize, usize),

    #[error("Container serialization error: {0}")]
    ContainerMalformed(String),

    #[error("Memory alignment error: address or size must be aligned to {0} bytes")]
    Misaligned(usize),
}
