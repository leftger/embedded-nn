//! ATON DMA Streaming Engine (STRENG) configuration and instruction generator.

use crate::isa::{
    EcInstruction, streng_addr, streng_ctrl, streng_depth, streng_event, streng_fsize, streng_strd,
};

/// Data element width for streaming engine transfers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementSize {
    Int8 = 0,
    Int16 = 1,
    Int32 = 2,
    Float32 = 3,
}

/// Channel assignments for ATON execution pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaChannel {
    /// STRENG 0: Primary input activation tensor stream.
    InputTensor,
    /// STRENG 1: Filter / weight matrix stream.
    WeightMatrix,
    /// STRENG 2: Bias / per-channel scale stream.
    BiasQuant,
    /// STRENG 3: Output activation writeback stream.
    OutputTensor,
    /// Secondary channels 4..9 for parallel branch execution.
    Auxiliary(u8),
}

impl DmaChannel {
    pub fn index(&self) -> usize {
        match self {
            DmaChannel::InputTensor => 0,
            DmaChannel::WeightMatrix => 1,
            DmaChannel::BiasQuant => 2,
            DmaChannel::OutputTensor => 3,
            DmaChannel::Auxiliary(idx) => *idx as usize,
        }
    }
}

/// Configuration descriptor for programming an ATON Streaming Engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrengConfig {
    pub channel: DmaChannel,
    pub base_addr: u32,
    pub width: u16,
    pub height: u16,
    pub depth_size: u16,
    pub depth_offset: u16,
    pub line_offset: u16,
    pub element_size: ElementSize,
    pub sign_extend: bool,
    pub enable_frame_irq: bool,
}

impl StrengConfig {
    pub fn new_tensor_1d(
        channel: DmaChannel,
        base_addr: u32,
        len: usize,
        element_size: ElementSize,
    ) -> Self {
        Self {
            channel,
            base_addr,
            width: len as u16,
            height: 1,
            depth_size: 1,
            depth_offset: 0,
            line_offset: len as u16,
            element_size,
            sign_extend: true,
            enable_frame_irq: channel == DmaChannel::OutputTensor,
        }
    }

    pub fn new_tensor_2d(
        channel: DmaChannel,
        base_addr: u32,
        rows: usize,
        cols: usize,
        element_size: ElementSize,
    ) -> Self {
        Self {
            channel,
            base_addr,
            width: cols as u16,
            height: rows as u16,
            depth_size: 1,
            depth_offset: 0,
            line_offset: cols as u16,
            element_size,
            sign_extend: true,
            enable_frame_irq: channel == DmaChannel::OutputTensor,
        }
    }

    pub fn new_tensor_shape(
        channel: DmaChannel,
        base_addr: u32,
        shape: &embedded_nn_compiler::ir::TensorShape,
        element_size: ElementSize,
    ) -> Self {
        let width = shape.width.max(1) * shape.channels.max(1);
        let height = shape.batches.max(1) * shape.height.max(1);
        Self {
            channel,
            base_addr,
            width: width as u16,
            height: height as u16,
            depth_size: 1,
            depth_offset: 0,
            line_offset: width as u16,
            element_size,
            sign_extend: true,
            enable_frame_irq: channel == DmaChannel::OutputTensor,
        }
    }

    /// Emits the instructions to configure and arm the streaming engine.
    /// Returns the index of the ADDR instruction word, so it can be recorded in the relocation table.
    pub fn emit_instructions(&self, out: &mut Vec<EcInstruction>) {
        let n = self.channel.index();

        // 1. Program FSIZE: WIDTH in [15:0], HEIGHT in [31:16]
        let fsize = (self.width as u32) | ((self.height as u32) << 16);
        out.push(EcInstruction::WriteReg {
            reg_addr: streng_fsize(n),
            value: fsize,
        });

        // 2. Program DEPTH: SIZE in [15:0], OFFSET in [31:16]
        let depth = (self.depth_size as u32) | ((self.depth_offset as u32) << 16);
        out.push(EcInstruction::WriteReg {
            reg_addr: streng_depth(n),
            value: depth,
        });

        // 3. Program STRD: LOFF in [15:0]
        let strd = self.line_offset as u32;
        out.push(EcInstruction::WriteReg {
            reg_addr: streng_strd(n),
            value: strd,
        });

        // 4. Program EVENT: enable frame overflow if needed
        let mut event = 0u32;
        if self.enable_frame_irq {
            event |= 1 << 19; // EN_OFLOW_FRM
        }
        out.push(EcInstruction::WriteReg {
            reg_addr: streng_event(n),
            value: event,
        });

        // 5. Program ADDR: base address (will be relocated dynamically)
        out.push(EcInstruction::WriteReg {
            reg_addr: streng_addr(n),
            value: self.base_addr,
        });

        // 6. Program CTRL: sign extension, element size, and enable
        let mut ctrl = 1u32; // bit 0 = EN
        if self.sign_extend {
            ctrl |= 1 << 15;
        }
        let size_code = match self.element_size {
            ElementSize::Int8 => 0,
            ElementSize::Int16 => 1,
            ElementSize::Int32 => 2,
            ElementSize::Float32 => 3,
        };
        ctrl |= (size_code as u32) << 16;

        out.push(EcInstruction::WriteReg {
            reg_addr: streng_ctrl(n),
            value: ctrl,
        });
    }
}
