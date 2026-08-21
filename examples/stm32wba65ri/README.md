# STM32WBA65RI inference stub

This standalone, workspace-excluded firmware initializes `embassy-stm32` for the
STM32WBA65RI, runs the generated model with caller-owned storage, measures `predict` with the
Cortex-M33 DWT cycle counter, and writes the result over RTT with `defmt`. It intentionally does
not initialize a BLE stack.

```console
rustup target add thumbv8m.main-none-eabihf
cargo check
```

## USB-HS HIL agent

`hil_agent` enumerates as a WinUSB vendor bulk device (`VID 0x1209`, `PID 0xE612`) on the
WBA65 USB-HS port (**PD6 = DP**, **PD7 = DM**). Clock tree matches the proven
`stm32wba-tftdisplay` studio agent (96 MHz sysclk, USB PHY from PLL1P). Frames use
[`embedded-nn-live`](../../docs/LIVE_PROTOCOL.md) (`0xE6 0x4E` + CRC-16), not JSON or CDC.

```console
cargo check --features hil-usb --bin hil_agent
# probe-rs run --chip STM32WBA65RI --features hil-usb --bin hil_agent
enn hil ping
enn hil infer --input 64
```

No hand-written `memory.x` is included. The `embassy-stm32` `memory-x` feature generates the
linker memory description from `stm32-metapac` for the selected `stm32wba65ri` chip. `link.x`
comes from `cortex-m-rt`, and `defmt.x` comes from the defmt tooling. A board-specific flashing
runner is deliberately not assumed; configure probe-rs for the actual board/debug probe.

The target rustflags enable `+dsp`, which is valid for the STM32WBA65RI's Cortex-M33 with the
Armv8-M DSP extension. The generated model currently uses portable kernels, but this flag allows
DSP-accelerated implementations to be selected when available.

Stable Rust accepts `-C target-feature=+dsp` but warns that it is not stably supported:

```text
warning: unstable feature specified for `-Ctarget-feature`: `dsp`
         this feature is not stably supported; its behavior can change in the future
```

The warning is expected and the build is unaffected; drop the flag from
[`.cargo/config.toml`](.cargo/config.toml) if you would rather not depend on it.

Studio's Arena tab carries a matching STM32WBA65RI profile (Cortex-M33 FPU + DSP, 100 MHz,
2048 KB Flash, 512 KB SRAM) with an editable radio/protocol-stack SRAM reserve, defaulting to
192 KB, so the planned activation arena is checked against *available* rather than total SRAM.
The USB HIL agent currently runs the 96 MHz PLL required by the HS PHY.
