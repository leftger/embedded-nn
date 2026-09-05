# Contributing to embedded-nn

Thank you for your interest in contributing to **`embedded-nn`**!

`embedded-nn` is a pure Rust, `#![no_std]` neural network inference runtime, ahead-of-time compiler, and TinyML platform designed for resource-constrained microcontrollers and edge silicon.

---

## Code of Conduct

We are committed to providing a welcoming, inclusive, and harassment-free environment for everyone. Please be respectful and constructive in all discussions, issues, and pull requests.

---

## Workspace Architecture

The repository is organized as a Cargo workspace:

- **`crates/embedded-nn`**: Core `#![no_std]` runtime containing integer (`s8`, `s16`, `s4`) & float kernel math, sub-byte LUT quantization, safety integrity checks (CRC32, canaries), and anomaly detectors.
- **`crates/embedded-nn-compiler`**: Model graph IR, ahead-of-time interval-colored SRAM arena scheduler, and host integer interpreter.
- **`crates/embedded-nn-codegen`**: Standalone `#![no_std]` Rust code emitter and standalone C99 header generator (`model.h`).
- **`crates/embedded-nn-macros`**: Procedural macro `#[embedded_nn_model("...")]` for compile-time model embedding.
- **`crates/embedded-nn-tflite`**: TensorFlow Lite / LiteRT FlatBuffer parser and importer.
- **`crates/embedded-nn-litert-plugin`**: C-ABI compiler plugin for Google LiteRT runtime.
- **`crates/embedded-nn-train`**: Burn-based QAT/PTQ training, SpecAugment, and Pareto optimization.
- **`crates/embedded-nn-live`**: Binary USB-HS / UART streaming protocol and HIL runner.
- **`crates/embedded-nn-cli`**: `enn` command-line interface.
- **`crates/embedded-nn-studio`**: Interactive Desktop & WebAssembly TinyML studio.

---

## Development & Testing Workflow

### Prerequisites
- **Rust Toolchain**: Stable Rust (Edition 2024, MSRV 1.98+).
- **Target Architectures**: Install bare-metal targets for testing `#![no_std]` builds:
  ```bash
  rustup target add thumbv6m-none-eabi thumbv7em-none-eabihf thumbv8m.main-none-eabihf riscv32imac-unknown-none-elf wasm32-unknown-unknown
  ```

### Local CI Checks
Before submitting a pull request, run the local verification suite:
```bash
# 1. Format code
cargo fmt --all

# 2. Check workspace
cargo check --workspace --all-targets

# 3. Clippy lints (warnings treated as errors)
cargo clippy --workspace --all-targets -- -D warnings

# 4. Check core crate against bare-metal no_std targets
cargo check -p embedded-nn --lib --no-default-features --features libm --target thumbv6m-none-eabi
cargo check -p embedded-nn --lib --no-default-features --features float,libm --target thumbv7em-none-eabihf
cargo check -p embedded-nn --lib --no-default-features --features libm --target riscv32imac-unknown-none-elf

# 5. Run full unit and integration test suite
cargo test --workspace
```

---

## Core Principles & Design Rules

When contributing code to `crates/embedded-nn` (the runtime crate), adhere to these strict rules:

1. **Zero Dynamic Allocation (`#![no_std]`)**:
   - The runtime must never allocate heap memory (`alloc`, `Box`, `Vec`, `String` are forbidden in kernel execution paths).
   - All buffers (inputs, filters, biases, outputs, scratch arenas) must be passed as slices with pre-determined dimensions.

2. **Arithmetic & Boundary Safety**:
   - Never panic or divide by zero. Validate all slice lengths against expected dimensions (`Dims::flat_size()`) and return `Err(Error::ArgumentError)` or `Err(Error::DimensionMismatch)`.
   - Prevent integer overflow by widening accumulators to `i32` or `i64` during dot-products and convolutions before applying fixed-point requantization (`requantize_s32` / `requantize_s64`).

3. **Determinism & Portability**:
   - Fixed-point arithmetic must produce bit-exact identical outputs across all architectures (x86_64, ARM Cortex-M, RISC-V, Xtensa, WASM).

---

## How to Add a New Operator

To contribute a new neural network operator:

1. **Kernel Implementation (`crates/embedded-nn/src/`)**:
   - Add the quantized operator function (e.g., `my_op_s8`) and floating-point variant (if applicable).
   - Export it in `crates/embedded-nn/src/lib.rs`.
2. **Kernel Unit Tests (`crates/embedded-nn/tests/`)**:
   - Add unit tests verifying numerical accuracy against known reference vectors (e.g. from TFLite reference or PyTorch).
   - Include edge case tests: single-element inputs, non-power-of-two dimensions, maximum/minimum integer saturation.
3. **Graph IR Lowering (`crates/embedded-nn-compiler/src/`)**:
   - Add the node variant to `OpType` in `crates/embedded-nn-compiler/src/ir.rs`.
   - Implement tensor shape inference and arena lifetime tracking.
4. **Code Emitters (`crates/embedded-nn-codegen/src/`)**:
   - Implement Rust `#![no_std]` code generation in `rust_emitter.rs`.
   - Implement C99 standalone code generation in `c_emitter.rs`.
5. **TFLite Importer (`crates/embedded-nn-tflite/src/`)**:
   - Map the TFLite FlatBuffer operator code (`tflite::BuiltinOperator`) to the newly added IR node.

---

## Pull Request Guidelines

1. **Focused PRs**: Keep PRs focused on a single bug fix, feature, or operator addition.
2. **Test Coverage**: Every new feature or bug fix must be accompanied by relevant unit or integration tests.
3. **Clear Commit Messages**: Use conventional commits (e.g., `feat(embedded-nn): add s8 gelu activation`, `fix(compiler): arena reuse with multi-branch graphs`).
4. **All CI Checks Green**: Ensure GitHub Actions pass formatting, clippy, `no_std` matrix checks, and test suite.

---

## Questions and Discussions

If you'd like to propose a significant architectural change or discuss a new feature, open a GitHub Discussion or an issue first before writing large amounts of code. We are happy to collaborate and guide design decisions!
