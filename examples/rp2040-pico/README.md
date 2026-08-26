# Raspberry Pi Pico (RP2040) Neural Network Inference

Bare-metal `#![no_std]` TinyML neural network inference on the Raspberry Pi Pico (RP2040 Dual ARM Cortex-M0+) with **zero dynamic heap allocations (`alloc`)** using `embedded-nn`.

---

## Hardware Specifications

- **Microcontroller**: Raspberry Pi Pico / RP2040
- **Core**: Dual-core ARM Cortex-M0+ @ 125 MHz
- **SRAM**: 264 KB on-chip SRAM
- **Flash**: 2 MB Quad-SPI external Flash (W25Q16JV)
- **Visual Feedback**: On-board green LED (`GPIO 25`) indicates inference classification results in real time.

---

## Building and Flashing

### 1. Prerequisites

Install the ARM Cortex-M0+ cross-compilation target:

```bash
rustup target add thumbv6m-none-eabi
```

Install your preferred flashing runner:
- **`probe-rs`** (for CMSIS-DAP / Picoprobe / Raspberry Pi Debug Probe):
  ```bash
  cargo install probe-rs-tools
  ```
- Or **`elf2uf2-rs`** (for drag-and-drop USB BOOTSEL mode):
  ```bash
  cargo install elf2uf2-rs --locked
  ```

### 2. Building Firmware

Check compilation:

```bash
cd examples/rp2040-pico
cargo check --target thumbv6m-none-eabi
```

Build release binary:

```bash
cargo build --release --target thumbv6m-none-eabi
```

### 3. Flashing to Pico

Using `probe-rs`:

```bash
probe-rs run --chip RP2040 target/thumbv6m-none-eabi/release/rp2040-pico-inference
```

Or convert to `.uf2` for USB drag-and-drop flashing:

```bash
elf2uf2-rs target/thumbv6m-none-eabi/release/rp2040-pico-inference pico_nn.uf2
```

---

## Memory Footprint

| Resource | Footprint | Utilization on RP2040 |
| :--- | :--- | :--- |
| **Flash Weights (ROM)** | 144 bytes | 0.007% of 2 MB |
| **SRAM Arena (Stack)** | 26 bytes | 0.009% of 264 KB |
| **Dynamic Heap (`alloc`)** | **0 bytes** | Completely zero-allocation |
