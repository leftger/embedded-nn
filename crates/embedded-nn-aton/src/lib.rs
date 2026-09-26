//! # embedded-nn-aton
//!
//! Independent hardware acceleration backend and Epoch Controller binary (`EcBinary`) compiler
//! targeting the STM32N6 Neural-ART (ATON) NPU architecture.
//!
//! Provides a pure open-source hardware compilation backend for STM32N6 NPU deployment:
//! - **Epoch Controller ISA & Assembler**: Encodes structured microcode commands for the on-chip Epoch Controller.
//! - **DMA Streaming Engine Configuration**: Programs `STRENG[0..9]` dimensional transfers.
//! - **Stream Switch Crossbar Routing**: Configures dynamic interconnect paths through `STRSWITCH`.
//! - **`EcBinary` Container Packaging**: Directly emits binary blobs conforming to the hardware
//!   Epoch Controller specification consumed by bare-metal loaders (such as [`embassy_stm32::npu::ecloader`]).
//! - **End-to-End Graph Compiler**: Translates `embedded-nn` [`ModelGraph`] instances into
//!   hybrid hardware/software epoch plans.
//!
//! ## Independence Disclaimer
//! This crate is an independent open-source hardware backend developed as part of the
//! `embedded-nn` project. It is not affiliated with, sponsored by, or endorsed by STMicroelectronics.
//! STM32 is a registered trademark of STMicroelectronics.

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
