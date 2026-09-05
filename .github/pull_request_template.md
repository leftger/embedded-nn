## Description
Briefly describe the purpose of this PR and what problem it solves.

## Type of Change
- [ ] 🐛 Bug fix (non-breaking change fixing an issue)
- [ ] ✨ New feature (non-breaking change adding functionality or operator)
- [ ] ⚡ Performance optimization (latency, cycle count, or SRAM reduction)
- [ ] 📝 Documentation update
- [ ] 🧹 Refactoring / code quality improvements

## Checklist
- [ ] My code follows the [Zero-Allocation / `#![no_std]` guidelines](CONTRIBUTING.md).
- [ ] All new arithmetic and memory access have bounds checks or safety proofs.
- [ ] New unit and integration tests have been added for modified kernels / components.
- [ ] `cargo fmt --all` has been run.
- [ ] `cargo check --workspace --all-targets` passes without error.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes without warnings.
- [ ] `#![no_std]` targets check passes (`thumbv6m-none-eabi`, `thumbv7em-none-eabihf`, `thumbv8m.main-none-eabihf`, `riscv32imac-unknown-none-elf`).
- [ ] `cargo test --workspace` passes all tests.
