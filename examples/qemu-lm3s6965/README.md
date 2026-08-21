# LM3S6965 QEMU inference

This standalone, workspace-excluded `no_std` binary expands a TFLite fixture with
`embedded-nn-macros`, runs inference with a caller-owned arena, checks the expected output, and
reports success or failure through semihosting.

```console
rustup target add thumbv7m-none-eabi
cargo run --release
```

The configured runner requires `qemu-system-arm`. The checked-in `memory.x` matches QEMU's
LM3S6965EVB model (256 KiB flash at `0x00000000`, 64 KiB SRAM at `0x20000000`).

Do not add `-C target-feature=+dsp` here: the emulated Cortex-M3 implements the Armv7-M
architecture without the DSP extension.
