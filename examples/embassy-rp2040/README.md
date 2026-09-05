# Embassy Async TinyML Inference on Raspberry Pi Pico (RP2040)

This example demonstrates non-blocking, multi-tasking TinyML inference on a **Raspberry Pi Pico (RP2040 Cortex-M0+)** using the **[Embassy](https://embassy.dev/)** async framework and **[`embedded-nn`](https://github.com/leftger/embedded-nn)**.

---

## Architecture

The firmware utilizes cooperative async tasks connected via zero-copy synchronization channels:

```
[sensor_sampler task] 
       │
       ▼ (50 Hz async Ticker -> SENSOR_CHANNEL)
[inference_worker task] ── embedded-nn INT8 forward pass (zero heap allocations)
       │
       ▼ (INFERENCE_CHANNEL)
[actuator_task] ──────── GPIO 25 LED indication
```

- **Zero Dynamic Allocation**: Inference runs in a fixed SRAM scratch arena.
- **Power Efficiency**: The microcontroller enters low-power WFI sleep between async timer events and sensor batches.
- **Composability**: Adding DMA peripherals (I2S, SPI, ADC) integrates directly into the Embassy async executor.

---

## Hardware Requirements

- **Raspberry Pi Pico** (or any RP2040 board with onboard LED on GPIO 25).
- Debug probe (Pico Probe, J-Link, or ST-Link) OR `elf2uf2-rs` over USB bootloader.

---

## Building and Flashing

### 1. Install Target and Tools
```bash
rustup target add thumbv6m-none-eabi
cargo install probe-rs-tools flip-link
```

### 2. Run with probe-rs
```bash
cargo run --release
```

### 3. Or Convert to UF2
```bash
cargo install elf2uf2-rs
cargo build --release
elf2uf2-rs target/thumbv6m-none-eabi/release/embassy-rp2040-inference -d
```
