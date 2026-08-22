//! Industrial Functional Safety (ISO 26262 / IEC 61508) & Memory Integrity Verification.
//!
//! Provides zero-allocation compile-time and boot-time Flash weight verification,
//! CRC32 bitflip detection, and activation arena memory boundary canary protection.

use core::fmt;

/// Safety and memory integrity error kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyError {
    /// Flash memory bitflip detected: CRC32 checksum mismatch.
    WeightCorruption {
        /// The expected CRC32 checksum recorded at compilation.
        expected_crc: u32,
        /// The actual CRC32 computed across the Flash weight array.
        computed_crc: u32,
    },
    /// Activation arena memory capacity is smaller than required by the model.
    ArenaOverflow {
        /// Byte capacity required by the model's static memory plan.
        required_bytes: usize,
        /// Actual byte capacity of the provided arena buffer.
        available_bytes: usize,
    },
    /// Memory boundary canary was overwritten, indicating a buffer overflow or stack collision.
    CanaryViolation {
        /// The expected canary constant (`0xDEAD_CAFE`).
        expected_canary: u32,
        /// The corrupted canary value found at the guard boundary.
        found_canary: u32,
    },
}

impl fmt::Display for SafetyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WeightCorruption {
                expected_crc,
                computed_crc,
            } => {
                write!(
                    f,
                    "Flash weight corruption detected: expected CRC32 0x{:08X}, got 0x{:08X}",
                    expected_crc, computed_crc
                )
            }
            Self::ArenaOverflow {
                required_bytes,
                available_bytes,
            } => {
                write!(
                    f,
                    "Arena memory overflow: model requires {} bytes, but arena has {}",
                    required_bytes, available_bytes
                )
            }
            Self::CanaryViolation {
                expected_canary,
                found_canary,
            } => {
                write!(
                    f,
                    "Arena guard canary corrupted: expected 0x{:08X}, found 0x{:08X} (buffer overflow)",
                    expected_canary, found_canary
                )
            }
        }
    }
}

/// Standard IEEE 802.3 CRC32 implementation optimized for `#![no_std]` embedded execution.
pub fn crc32_fast(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Verifies that Flash weight arrays have not suffered from bitflips or memory corruption.
pub fn verify_weights_integrity(weights: &[i8], expected_crc: u32) -> Result<(), SafetyError> {
    // Cast slice to &[u8] for CRC calculation
    let bytes =
        unsafe { core::slice::from_raw_parts(weights.as_ptr() as *const u8, weights.len()) };
    let computed_crc = crc32_fast(bytes);
    if computed_crc != expected_crc {
        Err(SafetyError::WeightCorruption {
            expected_crc,
            computed_crc,
        })
    } else {
        Ok(())
    }
}

/// Magic canary constant placed at the end of the activation arena for overflow detection.
pub const ARENA_GUARD_CANARY: u32 = 0xDEAD_CAFE;

/// Verifies that an activation arena buffer meets size requirements and guard canaries are intact.
pub fn verify_arena_integrity(
    arena: &[u8],
    required_bytes: usize,
    guard_canary: u32,
) -> Result<(), SafetyError> {
    if arena.len() < required_bytes {
        return Err(SafetyError::ArenaOverflow {
            required_bytes,
            available_bytes: arena.len(),
        });
    }

    if guard_canary != ARENA_GUARD_CANARY {
        return Err(SafetyError::CanaryViolation {
            expected_canary: ARENA_GUARD_CANARY,
            found_canary: guard_canary,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32_fast_known_vector() {
        let text = b"123456789";
        assert_eq!(crc32_fast(text), 0xCBF43926);
    }

    #[test]
    fn test_verify_weights_integrity_detects_bitflip() {
        let mut weights = [1i8, 2, 3, 4, 5, 6, 7, 8, -10, -20];
        let bytes =
            unsafe { core::slice::from_raw_parts(weights.as_ptr() as *const u8, weights.len()) };
        let expected_crc = crc32_fast(bytes);

        assert!(verify_weights_integrity(&weights, expected_crc).is_ok());

        // Simulate single bitflip
        weights[4] ^= 1;
        let err = verify_weights_integrity(&weights, expected_crc).unwrap_err();
        match err {
            SafetyError::WeightCorruption {
                expected_crc: exp,
                computed_crc: comp,
            } => {
                assert_eq!(exp, expected_crc);
                assert_ne!(comp, expected_crc);
            }
            _ => panic!("Expected WeightCorruption error"),
        }
    }

    #[test]
    fn test_verify_arena_integrity_detects_overflow_and_canary() {
        let arena = [0u8; 128];
        assert!(verify_arena_integrity(&arena, 100, ARENA_GUARD_CANARY).is_ok());
        assert!(verify_arena_integrity(&arena, 200, ARENA_GUARD_CANARY).is_err());
        assert!(verify_arena_integrity(&arena, 100, 0x00000000).is_err());
    }
}
