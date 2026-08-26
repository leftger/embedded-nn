//! Target SIMD vectorization abstractions and acceleration hooks.

/// Vectorized dot product for int8 slices with an input zero-point offset (`lhs_offset`).
///
/// Efficiently processes 4-element or 8-element chunks to trigger compiler auto-vectorization
/// or ARM SIMD instructions (`SMLAD` on Cortex-M4/M7, `vmladavaq` on Cortex-M55/M85).
#[inline]
pub fn vec_dot_s8(lhs: &[i8], rhs: &[i8], lhs_offset: i32) -> i32 {
    let len = lhs.len().min(rhs.len());
    let mut acc: i32 = 0;

    #[cfg(all(target_arch = "arm", target_feature = "dsp"))]
    {
        // Target ARM DSP hardware acceleration (Cortex-M4/M7 SMLAD)
        let chunks = len / 2;
        let remainder = len % 2;

        let mut i = 0;
        for _ in 0..chunks {
            let l0 = lhs[i] as i32 + lhs_offset;
            let l1 = lhs[i + 1] as i32 + lhs_offset;
            let r0 = rhs[i] as i32;
            let r1 = rhs[i + 1] as i32;

            let val_l = (l0 & 0xFFFF) | ((l1 & 0xFFFF) << 16);
            let val_r = (r0 & 0xFFFF) | ((r1 & 0xFFFF) << 16);

            let res: i32;
            unsafe {
                core::arch::asm!(
                    "smlad {0}, {1}, {2}, {3}",
                    out(reg) res,
                    in(reg) val_l,
                    in(reg) val_r,
                    in(reg) acc,
                );
            }
            acc = res;
            i += 2;
        }

        for j in 0..remainder {
            let l = lhs[i + j] as i32 + lhs_offset;
            let r = rhs[i + j] as i32;
            acc += l * r;
        }

        acc
    }

    #[cfg(target_arch = "riscv32")]
    {
        // Target RISC-V 32-bit (ESP32-C3/C6, RP2350 Hazard3, BL602)
        let chunks = len / 4;
        let remainder = len % 4;

        let mut i = 0;
        for _ in 0..chunks {
            let l0 = lhs[i] as i32 + lhs_offset;
            let l1 = lhs[i + 1] as i32 + lhs_offset;
            let l2 = lhs[i + 2] as i32 + lhs_offset;
            let l3 = lhs[i + 3] as i32 + lhs_offset;
            let r0 = rhs[i] as i32;
            let r1 = rhs[i + 1] as i32;
            let r2 = rhs[i + 2] as i32;
            let r3 = rhs[i + 3] as i32;

            acc += l0 * r0 + l1 * r1 + l2 * r2 + l3 * r3;
            i += 4;
        }

        for j in 0..remainder {
            let l = lhs[i + j] as i32 + lhs_offset;
            let r = rhs[i + j] as i32;
            acc += l * r;
        }

        acc
    }

    #[cfg(not(any(
        all(target_arch = "arm", target_feature = "dsp"),
        target_arch = "riscv32"
    )))]
    {
        let chunks = len / 8;
        let remainder = len % 8;

        let mut i = 0;
        for _ in 0..chunks {
            let l0 = lhs[i] as i32 + lhs_offset;
            let r0 = rhs[i] as i32;
            let l1 = lhs[i + 1] as i32 + lhs_offset;
            let r1 = rhs[i + 1] as i32;
            let l2 = lhs[i + 2] as i32 + lhs_offset;
            let r2 = rhs[i + 2] as i32;
            let l3 = lhs[i + 3] as i32 + lhs_offset;
            let r3 = rhs[i + 3] as i32;
            let l4 = lhs[i + 4] as i32 + lhs_offset;
            let r4 = rhs[i + 4] as i32;
            let l5 = lhs[i + 5] as i32 + lhs_offset;
            let r5 = rhs[i + 5] as i32;
            let l6 = lhs[i + 6] as i32 + lhs_offset;
            let r6 = rhs[i + 6] as i32;
            let l7 = lhs[i + 7] as i32 + lhs_offset;
            let r7 = rhs[i + 7] as i32;

            acc += l0 * r0 + l1 * r1 + l2 * r2 + l3 * r3 + l4 * r4 + l5 * r5 + l6 * r6 + l7 * r7;
            i += 8;
        }

        for j in 0..remainder {
            let l = lhs[i + j] as i32 + lhs_offset;
            let r = rhs[i + j] as i32;
            acc += l * r;
        }

        acc
    }
}

/// Vectorized dot product for int16 slices.
///
/// Exploits ARM Cortex-M DSP assembly `SMLALD` (dual 16-bit signed multiply-accumulate
/// into a 64-bit accumulator) on ARM DSP targets, RISC-V 32-bit SIMD, or 4-way loop unrolling on non-ARM architectures.
#[inline]
pub fn vec_dot_s16(lhs: &[i16], rhs: &[i16]) -> i64 {
    let len = lhs.len().min(rhs.len());
    let mut acc: i64 = 0;

    #[cfg(all(target_arch = "arm", target_feature = "dsp"))]
    {
        let pairs = len / 2;
        let remainder = len % 2;

        let mut i = 0;
        for _ in 0..pairs {
            let l_packed = (lhs[i] as u16 as u32) | ((lhs[i + 1] as u16 as u32) << 16);
            let r_packed = (rhs[i] as u16 as u32) | ((rhs[i + 1] as u16 as u32) << 16);

            let acc_lo = acc as u32;
            let acc_hi = (acc >> 32) as u32;
            let out_lo: u32;
            let out_hi: u32;
            unsafe {
                core::arch::asm!(
                    "smlald {out_lo}, {out_hi}, {a}, {b}",
                    a = in(reg) l_packed,
                    b = in(reg) r_packed,
                    out_lo = inout(reg) acc_lo => out_lo,
                    out_hi = inout(reg) acc_hi => out_hi,
                    options(pure, nomem, nostack)
                );
            }
            acc = ((out_hi as i64) << 32) | (out_lo as i64);
            i += 2;
        }

        if remainder != 0 {
            acc += (lhs[i] as i64) * (rhs[i] as i64);
        }

        acc
    }

    #[cfg(target_arch = "riscv32")]
    {
        let pairs = len / 2;
        let remainder = len % 2;

        let mut i = 0;
        for _ in 0..pairs {
            let l0 = lhs[i] as i64;
            let r0 = rhs[i] as i64;
            let l1 = lhs[i + 1] as i64;
            let r1 = rhs[i + 1] as i64;
            acc += l0 * r0 + l1 * r1;
            i += 2;
        }

        if remainder != 0 {
            acc += (lhs[i] as i64) * (rhs[i] as i64);
        }

        acc
    }

    #[cfg(not(any(
        all(target_arch = "arm", target_feature = "dsp"),
        target_arch = "riscv32"
    )))]
    {
        let chunks = len / 4;
        let remainder = len % 4;

        let mut i = 0;
        for _ in 0..chunks {
            let l0 = lhs[i] as i64;
            let r0 = rhs[i] as i64;
            let l1 = lhs[i + 1] as i64;
            let r1 = rhs[i + 1] as i64;
            let l2 = lhs[i + 2] as i64;
            let r2 = rhs[i + 2] as i64;
            let l3 = lhs[i + 3] as i64;
            let r3 = rhs[i + 3] as i64;

            acc += l0 * r0 + l1 * r1 + l2 * r2 + l3 * r3;
            i += 4;
        }

        for j in 0..remainder {
            acc += (lhs[i + j] as i64) * (rhs[i + j] as i64);
        }

        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_dot_s8() {
        let lhs = [1i8, 2i8, 3i8, 4i8, 5i8, 6i8, 7i8, 8i8, 9i8];
        let rhs = [1i8, 1i8, 1i8, 1i8, 1i8, 1i8, 1i8, 1i8, 1i8];
        assert_eq!(vec_dot_s8(&lhs, &rhs, 0), 45);
        assert_eq!(vec_dot_s8(&lhs, &rhs, 1), 45 + 9);
    }

    #[test]
    fn test_vec_dot_s16() {
        let lhs = [10i16, 20i16, 30i16, 40i16, 50i16];
        let rhs = [1i16, 2i16, 3i16, 4i16, 5i16];
        assert_eq!(vec_dot_s16(&lhs, &rhs), 550);
    }
}
