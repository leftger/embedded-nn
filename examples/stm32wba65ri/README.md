# STM32WBA65RI inference stub

This standalone, workspace-excluded firmware initializes `embassy-stm32` for the
STM32WBA65RI, runs the generated model with caller-owned storage, measures `predict` with the
Cortex-M33 DWT cycle counter, and writes the result over RTT with `defmt`. It intentionally does
not initialize a BLE stack.

```console
rustup target add thumbv8m.main-none-eabihf
cargo check
```

No hand-written `memory.x` is included. The `embassy-stm32` `memory-x` feature generates the
linker memory description from `stm32-metapac` for the selected `stm32wba65ri` chip. `link.x`
comes from `cortex-m-rt`, and `defmt.x` comes from the defmt tooling. A board-specific flashing
runner is deliberately not assumed; configure probe-rs for the actual board/debug probe.

The target rustflags enable `+dsp`, which is valid for the STM32WBA65RI's Cortex-M33 with the
Armv8-M DSP extension. The generated model currently uses portable kernels, but this flag allows
DSP-accelerated implementations to be selected when available.
