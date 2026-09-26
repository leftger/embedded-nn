//! Minimal ATON Neural-ART hardware register map and Epoch Controller instruction set.

/// Magic number at the start of an Epoch Controller binary container.
pub const BINARY_MAGIC: u32 = 0xECBF_0050;

/// Magic number at the start of an Epoch Controller instruction blob.
pub const BLOB_MAGIC: u32 = 0xCA05_7A7A;

/// Base address of the ATON peripheral in the STM32N6 non-secure address space.
pub const ATON_BASE: u32 = 0x480E_0000;

pub const CLKCTRL_BASE: u32 = ATON_BASE;
pub const INTCTRL_BASE: u32 = ATON_BASE + 0x1000;
pub const BUSIF_BASE: u32 = ATON_BASE + 0x2000;
pub const STRSWITCH_BASE: u32 = ATON_BASE + 0x4000;
pub const STRENG_BASE: u32 = ATON_BASE + 0x5000;
pub const CONVACC_BASE: u32 = ATON_BASE + 0xF000;
pub const DECUN_BASE: u32 = ATON_BASE + 0x13000;
pub const ACTIVACC_BASE: u32 = ATON_BASE + 0x15000;
pub const ARITHACC_BASE: u32 = ATON_BASE + 0x17000;
pub const POOLACC_BASE: u32 = ATON_BASE + 0x1B000;
pub const RECBUF_BASE: u32 = ATON_BASE + 0x1D000;
pub const EPOCHCTRL_BASE: u32 = ATON_BASE + 0x1E000;

// ── CLKCTRL Registers ──
pub const CLKCTRL_CTRL: u32 = CLKCTRL_BASE;
pub const CLKCTRL_AGATES0: u32 = CLKCTRL_BASE + 0x08;
pub const CLKCTRL_AGATES1: u32 = CLKCTRL_BASE + 0x0C;
pub const CLKCTRL_BGATES: u32 = CLKCTRL_BASE + 0x10;

// ── INTCTRL Registers ──
pub const INTCTRL_CTRL: u32 = INTCTRL_BASE;
pub const INTCTRL_INTREG: u32 = INTCTRL_BASE + 0x08;
pub const INTCTRL_INTSET: u32 = INTCTRL_BASE + 0x0C;
pub const INTCTRL_INTCLR: u32 = INTCTRL_BASE + 0x10;
pub const INT_ECTRL_EVT: u32 = 1 << 0;

// ── EPOCHCTRL Registers ──
pub const EPOCHCTRL_CTRL: u32 = EPOCHCTRL_BASE;
pub const EPOCHCTRL_ADDR: u32 = EPOCHCTRL_BASE + 0x08;
pub const EPOCHCTRL_IRQ: u32 = EPOCHCTRL_BASE + 0x0C;
pub const EPOCHCTRL_LABEL: u32 = EPOCHCTRL_BASE + 0x1C;
pub const EPOCHCTRL_BC: u32 = EPOCHCTRL_BASE + 0x20;

pub const EPOCHCTRL_CTRL_EN: u32 = 1 << 0;
pub const EPOCHCTRL_CTRL_CLR: u32 = 1 << 1;
pub const EPOCHCTRL_CTRL_SM: u32 = 1 << 3;
pub const EPOCHCTRL_CTRL_CONFCLR: u32 = 1 << 30;
pub const EPOCHCTRL_CTRL_RUNNING: u32 = 1 << 31;

// ── STRENG Registers (per engine n < 10) ──
#[inline(always)]
pub const fn streng_base(n: usize) -> u32 {
    STRENG_BASE + (0x1000 * n as u32)
}

#[inline(always)]
pub const fn streng_ctrl(n: usize) -> u32 {
    streng_base(n)
}

#[inline(always)]
pub const fn streng_addr(n: usize) -> u32 {
    streng_base(n) + 0x08
}

#[inline(always)]
pub const fn streng_fsize(n: usize) -> u32 {
    streng_base(n) + 0x0C
}

#[inline(always)]
pub const fn streng_depth(n: usize) -> u32 {
    streng_base(n) + 0x10
}

#[inline(always)]
pub const fn streng_strd(n: usize) -> u32 {
    streng_base(n) + 0x14
}

#[inline(always)]
pub const fn streng_event(n: usize) -> u32 {
    streng_base(n) + 0x24
}

// ── CONVACC Registers (per unit n < 4) ──
#[inline(always)]
pub const fn convacc_base(n: usize) -> u32 {
    CONVACC_BASE + (0x1000 * n as u32)
}

#[inline(always)]
pub const fn convacc_ctrl(n: usize) -> u32 {
    convacc_base(n)
}

#[inline(always)]
pub const fn convacc_kformat(n: usize) -> u32 {
    convacc_base(n) + 0x08
}

#[inline(always)]
pub const fn convacc_sample(n: usize) -> u32 {
    convacc_base(n) + 0x0C
}

#[inline(always)]
pub const fn convacc_dformat(n: usize) -> u32 {
    convacc_base(n) + 0x10
}

#[inline(always)]
pub const fn convacc_fformat(n: usize) -> u32 {
    convacc_base(n) + 0x14
}

#[inline(always)]
pub const fn convacc_fhcrop(n: usize) -> u32 {
    convacc_base(n) + 0x18
}

#[inline(always)]
pub const fn convacc_fvcrop(n: usize) -> u32 {
    convacc_base(n) + 0x1C
}

#[inline(always)]
pub const fn convacc_kfilt(n: usize) -> u32 {
    convacc_base(n) + 0x20
}

#[inline(always)]
pub const fn convacc_afilt(n: usize) -> u32 {
    convacc_base(n) + 0x24
}

#[inline(always)]
pub const fn convacc_zframe(n: usize) -> u32 {
    convacc_base(n) + 0x28
}

#[inline(always)]
pub const fn convacc_fsub(n: usize) -> u32 {
    convacc_base(n) + 0x30
}

// ── POOLACC Registers (per unit n < 2) ──
#[inline(always)]
pub const fn poolacc_base(n: usize) -> u32 {
    POOLACC_BASE + (0x1000 * n as u32)
}

#[inline(always)]
pub const fn poolacc_ctrl(n: usize) -> u32 {
    poolacc_base(n)
}

#[inline(always)]
pub const fn poolacc_pdims(n: usize) -> u32 {
    poolacc_base(n) + 0x08
}

#[inline(always)]
pub const fn poolacc_fdims(n: usize) -> u32 {
    poolacc_base(n) + 0x0C
}

#[inline(always)]
pub const fn poolacc_outdims(n: usize) -> u32 {
    poolacc_base(n) + 0x10
}

#[inline(always)]
pub const fn poolacc_mulval(n: usize) -> u32 {
    poolacc_base(n) + 0x14
}

#[inline(always)]
pub const fn poolacc_xcrop(n: usize) -> u32 {
    poolacc_base(n) + 0x18
}

#[inline(always)]
pub const fn poolacc_ycrop(n: usize) -> u32 {
    poolacc_base(n) + 0x1C
}

#[inline(always)]
pub const fn poolacc_rndctrl(n: usize) -> u32 {
    poolacc_base(n) + 0x20
}

pub const POOL_OP_MAX: u32 = 1;
pub const POOL_OP_MIN: u32 = 2;
pub const POOL_OP_AVG: u32 = 3;
pub const POOL_OP_GMAX: u32 = 4;
pub const POOL_OP_GMIN: u32 = 5;
pub const POOL_OP_GAVG: u32 = 6;

// ── ARITHACC Registers (per unit n < 4) ──
#[inline(always)]
pub const fn arithacc_base(n: usize) -> u32 {
    ARITHACC_BASE + (0x1000 * n as u32)
}

#[inline(always)]
pub const fn arithacc_ctrl(n: usize) -> u32 {
    arithacc_base(n)
}

#[inline(always)]
pub const fn arithacc_shift(n: usize) -> u32 {
    arithacc_base(n) + 0x08
}

#[inline(always)]
pub const fn arithacc_inccnt(n: usize) -> u32 {
    arithacc_base(n) + 0x0C
}

#[inline(always)]
pub const fn arithacc_rstcnt1(n: usize) -> u32 {
    arithacc_base(n) + 0x10
}

#[inline(always)]
pub const fn arithacc_rstcnt2(n: usize) -> u32 {
    arithacc_base(n) + 0x14
}

#[inline(always)]
pub const fn arithacc_rstcnt3(n: usize) -> u32 {
    arithacc_base(n) + 0x18
}

#[inline(always)]
pub const fn arithacc_coeffac(n: usize) -> u32 {
    arithacc_base(n) + 0x1C
}

#[inline(always)]
pub const fn arithacc_coeffb(n: usize) -> u32 {
    arithacc_base(n) + 0x20
}

#[inline(always)]
pub const fn arithacc_addroffset(n: usize) -> u32 {
    arithacc_base(n) + 0x24
}

#[inline(always)]
pub const fn arithacc_incoffset(n: usize) -> u32 {
    arithacc_base(n) + 0x28
}

#[inline(always)]
pub const fn arithacc_translateaddr(n: usize) -> u32 {
    arithacc_base(n) + 0x2C
}

#[inline(always)]
pub const fn arithacc_coeffaddr(n: usize) -> u32 {
    arithacc_base(n) + 0x30
}

#[inline(always)]
pub const fn arithacc_inshifter(n: usize) -> u32 {
    arithacc_base(n) + 0x34
}

#[inline(always)]
pub const fn arithacc_cliprange(n: usize) -> u32 {
    arithacc_base(n) + 0x38
}

pub const ARITH_OP_AFFINE: u32 = 1;
pub const ARITH_OP_MIN: u32 = 2;
pub const ARITH_OP_MAX: u32 = 3;
pub const ARITH_OP_MUL: u32 = 4;
pub const ARITH_OP_CLIP: u32 = 16;

// ── ACTIVACC Registers (per unit n < 2) ──
#[inline(always)]
pub const fn activacc_base(n: usize) -> u32 {
    ACTIVACC_BASE + (0x1000 * n as u32)
}

#[inline(always)]
pub const fn activacc_ctrl(n: usize) -> u32 {
    activacc_base(n)
}

#[inline(always)]
pub const fn activacc_activparam(n: usize) -> u32 {
    activacc_base(n) + 0x08
}

#[inline(always)]
pub const fn activacc_func(n: usize) -> u32 {
    activacc_base(n) + 0x0C
}

#[inline(always)]
pub const fn activacc_activparam2(n: usize) -> u32 {
    activacc_base(n) + 0x10
}

#[inline(always)]
pub const fn activacc_fsub(n: usize) -> u32 {
    activacc_base(n) + 0x14
}

pub const ACTIV_OP_RELU: u32 = 1;
pub const ACTIV_OP_PRELU: u32 = 2;
pub const ACTIV_OP_TRELU: u32 = 3;
pub const ACTIV_OP_FUNC: u32 = 4;
pub const ACTIV_OP_LUT: u32 = 5;

// ── STRSWITCH Registers ──
pub const STRSWITCH_CTRL: u32 = STRSWITCH_BASE;

#[inline(always)]
pub const fn strswitch_dst(idx: usize) -> u32 {
    STRSWITCH_BASE + 0x08 + 4 * idx as u32
}

/// Epoch Controller Microcode Opcode definitions.
///
/// The Epoch Controller executes 32-bit command words.
/// Format:
/// [31:28] Opcode
/// [27:16] Sub-command / Register offset
/// [15:0]  Immediate / Parameter / Operand
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EcOpcode {
    /// NOP instruction
    Nop = 0x0,
    /// Direct Register Write: writes the following word to target peripheral register.
    WriteReg = 0x1,
    /// Bitwise Set Mask: sets bits in target register.
    SetMask = 0x2,
    /// Bitwise Clear Mask: clears bits in target register.
    ClearMask = 0x3,
    /// Synchronous barrier: stalls epoch controller until specified event mask is asserted.
    WaitEvents = 0x4,
    /// Raises end-of-epoch or intermediate interrupt to INTCTRL.
    TriggerIrq = 0x5,
    /// Label marker for debugging and jump targets.
    Label = 0x6,
    /// Terminates epoch execution and signals EVT_EC_DONE to the CPU.
    End = 0xF,
}

/// A structured Epoch Controller instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EcInstruction {
    Nop,
    /// Writes `value` to the ATON internal register at `reg_addr`.
    WriteReg {
        reg_addr: u32,
        value: u32,
    },
    /// Sets `mask` bits in `reg_addr`.
    SetMask {
        reg_addr: u32,
        mask: u32,
    },
    /// Clears `mask` bits in `reg_addr`.
    ClearMask {
        reg_addr: u32,
        mask: u32,
    },
    /// Stalls until events in `event_mask` are signaled.
    WaitEvents {
        event_mask: u32,
    },
    /// Raises interrupt.
    TriggerIrq,
    /// Debug section label.
    Label(u16),
    /// Terminates the epoch.
    End,
}

impl EcInstruction {
    /// Encodes the instruction into 32-bit microcode words.
    pub fn encode(&self, out: &mut Vec<u32>) {
        match self {
            EcInstruction::Nop => {
                out.push(0x0000_0000);
            }
            EcInstruction::WriteReg { reg_addr, value } => {
                let offset = (reg_addr.wrapping_sub(ATON_BASE) / 4) & 0x0FFF;
                let header = ((EcOpcode::WriteReg as u32) << 28) | (offset << 16);
                out.push(header);
                out.push(*value);
            }
            EcInstruction::SetMask { reg_addr, mask } => {
                let offset = (reg_addr.wrapping_sub(ATON_BASE) / 4) & 0x0FFF;
                let header = ((EcOpcode::SetMask as u32) << 28) | (offset << 16);
                out.push(header);
                out.push(*mask);
            }
            EcInstruction::ClearMask { reg_addr, mask } => {
                let offset = (reg_addr.wrapping_sub(ATON_BASE) / 4) & 0x0FFF;
                let header = ((EcOpcode::ClearMask as u32) << 28) | (offset << 16);
                out.push(header);
                out.push(*mask);
            }
            EcInstruction::WaitEvents { event_mask } => {
                let header = (EcOpcode::WaitEvents as u32) << 28;
                out.push(header);
                out.push(*event_mask);
            }
            EcInstruction::TriggerIrq => {
                out.push((EcOpcode::TriggerIrq as u32) << 28);
            }
            EcInstruction::Label(id) => {
                let header = ((EcOpcode::Label as u32) << 28) | (*id as u32);
                out.push(header);
            }
            EcInstruction::End => {
                out.push((EcOpcode::End as u32) << 28);
            }
        }
    }
}
