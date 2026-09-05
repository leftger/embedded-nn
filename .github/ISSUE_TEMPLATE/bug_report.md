---
name: 🐛 Bug Report
about: Create a report to help us improve embedded-nn
title: '[BUG] '
labels: ['bug']
assignees: ''
---

## Describe the Bug
A clear and concise description of what the bug is.

## Target Hardware & Environment
- **Architecture**: (e.g. ARM Cortex-M4, Cortex-M33, Cortex-M0+, RISC-V RV32IMAC, x86_64, WASM)
- **Target Chip/Board**: (e.g. STM32WBA65RI, RP2040, nRF52840, ESP32-S3, Native Host)
- **Target Triple**: (e.g. `thumbv7em-none-eabihf`, `thumbv8m.main-none-eabihf`, `thumbv6m-none-eabi`)
- **Rust Version**: `rustc --version`
- **Crate & Version**: (e.g. `embedded-nn 0.2.0`, `embedded-nn-compiler 0.2.0`)

## Steps to Reproduce
Steps to reproduce the behavior:
1. Define model / kernel call '...'
2. Pass input slice '...'
3. Observe unexpected output or error

## Expected vs Actual Behavior
- **Expected**: (e.g. Bit-exact match to reference output `[12, -4, 30]`)
- **Actual**: (e.g. Overflow, incorrect requantization, panic, or compilation error)

## Minimal Code Example
```rust
// Paste minimal reproducing code here
```

## Additional Context
Add any other context, compiler logs, or stack traces here.
