//! Stream Switch (STRSWITCH) dynamic interconnect programming.

use crate::isa::{EcInstruction, STRSWITCH_CTRL, strswitch_dst};

/// Target destination port on the ATON crossbar switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchDst {
    /// Streaming Engine 0..9 writeback port.
    Streng(usize),
    /// Convolution / Matrix Accelerator unit primary input (Unit 0).
    ConvInA,
    /// Convolution / Matrix Accelerator unit weight input (Unit 0).
    ConvInB,
    /// Convolution / Matrix Accelerator unit bias/accum input (Unit 0).
    ConvInC,
    /// Convolution / Matrix Accelerator unit primary input for specific unit (0..3).
    ConvInAUnit(usize),
    /// Convolution / Matrix Accelerator unit weight input for specific unit (0..3).
    ConvInBUnit(usize),
    /// Convolution / Matrix Accelerator unit bias/accum input for specific unit (0..3).
    ConvInCUnit(usize),
    /// Activation / Post-processing unit input (Unit 0).
    ActUnitIn,
    /// Activation / Post-processing unit input for specific unit (0..1).
    ActUnitInUnit(usize),
    /// Arithmetic Accelerator input X (Unit 0).
    ArithInX,
    /// Arithmetic Accelerator input Y (Unit 0).
    ArithInY,
    /// Arithmetic Accelerator input X for specific unit (0..3).
    ArithInXUnit(usize),
    /// Arithmetic Accelerator input Y for specific unit (0..3).
    ArithInYUnit(usize),
    /// Pooling Accelerator unit input (Unit 0).
    PoolIn,
    /// Pooling Accelerator unit input for specific unit (0..1).
    PoolInUnit(usize),
}

impl SwitchDst {
    pub fn index(&self) -> usize {
        match self {
            SwitchDst::Streng(n) => *n,
            SwitchDst::ConvInA => 10,
            SwitchDst::ConvInB => 11,
            SwitchDst::ConvInC => 12,
            SwitchDst::ConvInAUnit(u) => 10 + 3 * u,
            SwitchDst::ConvInBUnit(u) => 11 + 3 * u,
            SwitchDst::ConvInCUnit(u) => 12 + 3 * u,
            SwitchDst::ActUnitIn => 26,
            SwitchDst::ActUnitInUnit(u) => 26 + u,
            SwitchDst::ArithInX => 28,
            SwitchDst::ArithInY => 29,
            SwitchDst::ArithInXUnit(u) => 28 + 2 * u,
            SwitchDst::ArithInYUnit(u) => 29 + 2 * u,
            SwitchDst::PoolIn => 36,
            SwitchDst::PoolInUnit(u) => 36 + u,
        }
    }
}

/// Source port on the ATON crossbar switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchSrc {
    /// Streaming Engine 0..9 read stream.
    Streng(usize),
    /// Convolution / Matrix Accelerator unit output (Unit 0).
    ConvOut,
    /// Convolution / Matrix Accelerator unit output for specific unit (0..3).
    ConvOutUnit(usize),
    /// Activation unit output (Unit 0).
    ActUnitOut,
    /// Activation unit output for specific unit (0..1).
    ActUnitOutUnit(usize),
    /// Arithmetic unit output (Unit 0).
    ArithOut,
    /// Arithmetic unit output for specific unit (0..3).
    ArithOutUnit(usize),
    /// Pooling unit output (Unit 0).
    PoolOut,
    /// Pooling unit output for specific unit (0..1).
    PoolOutUnit(usize),
}

impl SwitchSrc {
    pub fn link_id(&self) -> u32 {
        match self {
            SwitchSrc::Streng(n) => *n as u32,
            SwitchSrc::ConvOut => 10,
            SwitchSrc::ConvOutUnit(u) => 10 + *u as u32,
            SwitchSrc::ActUnitOut => 16,
            SwitchSrc::ActUnitOutUnit(u) => 16 + *u as u32,
            SwitchSrc::ArithOut => 18,
            SwitchSrc::ArithOutUnit(u) => 18 + *u as u32,
            SwitchSrc::PoolOut => 22,
            SwitchSrc::PoolOutUnit(u) => 22 + *u as u32,
        }
    }
}

/// A crossbar routing link connecting a source port to a destination port.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchRoute {
    pub src: SwitchSrc,
    pub dst: SwitchDst,
}

/// Generates instructions to program the Stream Switch crossbar routes.
pub fn emit_switch_routes(routes: &[SwitchRoute], out: &mut Vec<EcInstruction>) {
    // 1. Enable stream switch
    out.push(EcInstruction::WriteReg {
        reg_addr: STRSWITCH_CTRL,
        value: 1, // bit 0 = EN
    });

    // 2. Program each route into destination port registers
    for route in routes {
        let dst_idx = route.dst.index();
        let src_link = route.src.link_id();

        // Layout: EN0 = bit 0, LINK0 = bits[6:1]
        let reg_val = 1u32 | (src_link << 1);

        out.push(EcInstruction::WriteReg {
            reg_addr: strswitch_dst(dst_idx),
            value: reg_val,
        });
    }
}
