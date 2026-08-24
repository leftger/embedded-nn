# STM32WBA65RI inference stub

This firmware DSP-extracts the first Mel frame (`feature_dsp`), classifies it with a 16→2
integer MLP (`models/gesture_mlp.json`), then still runs the generated sine identity model for
the original hello-world check. DWT cycles and SRAM arena size are printed over RTT.

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

## Data Collection & Storage System (`data_collector`)

The `data_collector` binary provides high-speed sensor acquisition and dataset persistence across the stacked hardware rig:

- **LIS2DE12 3-axis Accelerometer**: Polled over `I2C1` (`PB1` SDA / `PB2` SCL, Address `0x18`) on the LR1110 evaluation shield.
- **W25Q32BV 32Mbit SPI NOR Flash**: Low-latency burst buffer on `SPI2` (`PB10` SCK, `PC3` MOSI, `PA9` MISO, `PA3` Flash CS).
- **MicroSD Cradle (DM-TFT28-116)**: Formats dataset records into JSON Lines (`.jsonl`) according to [`DATASET_IMPORT_FORMAT.md`](../../docs/DATASET_IMPORT_FORMAT.md) on `SPI2` (`PA10` SD CS).

### Pin Mapping & Switch Setup

| Signal / Function | Nucleo Pin | Arduino Header | Shield / Target |
| :--- | :--- | :--- | :--- |
| **I2C1 SCL** | `PB2` | `D15` | LIS2DE12 SCL (Pin 1) |
| **I2C1 SDA** | `PB1` | `D14` | LIS2DE12 SDA (Pin 4) |
| **SPI2 SCK** | `PB10` | `D13` | MicroSD CLK / Flash CLK (via 74HC541) |
| **SPI2 MISO** | `PA9` | `D12` | MicroSD MISO / Flash DO |
| **SPI2 MOSI** | `PC3` | `D11` | MicroSD MOSI / Flash DI (via 74HC541) |
| **SD Card CS** | `PA10` | `D8` | MicroSD `DAT3/CD` (via 74HC541) |
| **Flash CS** | `PA3` | `D4` | W25Q32 / W25Q16 `/CS` |
| **User Trigger Button** | `PC13` | B1 | Capture trigger |
| **Status LEDs** | `PD8` (LD1 Blue), `PC4` (LD2 Green), `PB8` (LD3 Red) | LEDs | System / Capture status |

> **Hardware Switches:**
> - On the **DM-TFT28-116 Display Module**, ensure **DIP Switch SW2** positions 1, 2, and 3 are set to **ON** to bridge SPI lines `D11`/`D12`/`D13`.

### Building and Running

```console
# Check compilation
cargo check --bin data_collector

# Flash and run via cargo (using configured probe-rs runner)
cargo run --bin data_collector
```

Press **User Button B1** (`PC13`) to record a 128-sample burst at 100 Hz. The firmware serializes the burst into the canonical `.jsonl` schema:

```json
{"sample_id":"sample_0001","label":null,"sample_rate_hz":100.0,"channel_names":["x","y","z"],"waveform":[[0.012,0.004,0.988],[0.016,0.008,0.992]]}
```

Validate and import directly into Studio:

```console
enn dataset validate dataset.jsonl
cargo run -p embedded-nn-studio
```

## Model Zoo & Silicon Power / Battery Runtime Profiling

The **embedded-nn Studio** provides pre-architected baseline neural networks and silicon specifications specifically tailored for the STM32WBA65RI:

- **Hardware Specs**: 100 MHz Cortex-M33 (DSP + FPU), 512 KB SRAM, 2048 KB Flash, 8.5 mA active current.
- **Model Zoo Presets**:
  - `MicroSpeechDsCnn`: 2D Spectrogram DS-CNN for 4-class keyword spotting (~58 KB Flash, ~12 KB SRAM).
  - `GestureResNet8`: 1D Temporal ResNet with skip connections for 6-axis IMU tracking (~76 KB Flash, ~14 KB SRAM).
  - `VisualWakeWords`: MobileNetV1 0.25x grayscale classifier for vision wake words (~218 KB Flash, ~38 KB SRAM).
  - `AnomalyAutoencoder`: Conv1D-Dense bottleneck autoencoder for vibration maintenance (~28 KB Flash, ~6 KB SRAM).
  - `StreamingSvdf`: Dual-stage streaming delay-line filter (~32 KB Flash, ~8 KB SRAM).
  - `SensorTransformer`: 1D Patch Tokenizer with Multi-Head Self-Attention (~92 KB Flash, ~18 KB SRAM).
  - `SeMobileNetV3`: Squeeze-and-Excitation channel attention CNN (~180 KB Flash, ~32 KB SRAM).
  - `DilatedSoundNet`: Multi-rate dilated temporal convolutions (~64 KB Flash, ~12 KB SRAM).
- **Silicon Power & Battery Life Estimator**: Live MAC cycle counting with battery runtime prediction (CR2032, LiPo, AAA, 18650) based on inference duty cycle and low-power standby current.


