//! # embedded-nn-aton
//!
//! Open-source compiler and Epoch Controller binary (`EcBinary`) assembler for the
//! STMicroelectronics Neural-ART (ATON) NPU on the STM32N6 microcontrollers.
//!
//! Provides an open-source toolchain for STM32N6 NPU deployment:
//! - **Epoch Controller ISA & Assembler**: Encodes structured microcode commands for the ATON engine.
//! - **DMA Streaming Engine Configuration**: Programs `STRENG[0..9]` dimensional transfers.
//! - **Stream Switch Crossbar Routing**: Configures dynamic interconnect paths through `STRSWITCH`.
//! - **`EcBinary` Container Packaging**: Directly emits binary blobs conforming to the Epoch
//!   Controller specification consumed by [`embassy_stm32::npu::ecloader`].
//! - **End-to-End Graph Compiler**: Translates `embedded-nn` [`ModelGraph`] instances into
//!   hybrid hardware/software epoch plans.

pub mod compiler;
pub mod container;
pub mod dma;
pub mod error;
pub mod isa;
pub mod switch;

pub use compiler::{AtonCompiledNetwork, AtonCompiler, EpochKind};
pub use container::EcContainerBuilder;
pub use dma::{DmaChannel, ElementSize, StrengConfig};
pub use error::AtonError;
pub use isa::{BINARY_MAGIC, BLOB_MAGIC, EcInstruction, EcOpcode};
pub use switch::{SwitchDst, SwitchRoute, SwitchSrc, emit_switch_routes};
