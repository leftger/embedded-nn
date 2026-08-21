# Live HIL protocol (`embedded-nn-live`)

Host tools (`embedded-nn-studio`, `enn hil`) talk to a flashed MCU agent over **vendor USB bulk**, not CDC/JSON.

This is the same framing class as `embedded-gui-live`, with a different magic so the streams cannot be confused.

## USB identity

| Field | Value |
| --- | --- |
| VID | `0x1209` |
| PID | `0xE612` (GUI display agent is `0xE611`) |
| Class | vendor `0xFF` |
| Interface | 0 |
| Endpoints | bulk IN + bulk OUT, 512-byte MPS |
| Windows | WinUSB via MS OS 2.0 descriptors |

STM32WBA65RI agent pins and clock: USB-HS PHY on **PD6 (DP) / PD7 (DM)**. The `hil_agent` firmware uses the same PLL as the proven WBA studio display agent (`sysclk` 96 MHz, `otghssel = Pll1P`). Studio's Arena tab still lists a 100 MHz Cortex-M33 profile; treat 96 MHz as the USB-bring-up clock.

## Frame layout

```
magic0=0xE6  magic1=0x4E  type:u8  len:u32 LE  payload[len]  crc16 LE
```

CRC-16/CCITT-FALSE covers `type`, the four length bytes, and the payload. The device decoder is constant-memory (`Decoder<CAP>`) and resynchronizes on magic after CRC/overflow faults.

## Messages (protocol version 1)

Host → device: `Hello`, `RunInference`, `Ping`  
Device → host: `Ready`, `InferenceResult`, `SensorFrame`, `Nack`, `Pong`

Tensors are raw packed `i8`. Sensor samples are little-endian `f32`.

`Hello` with `model_id` / `input_len` / `output_len` of `0` means “advertise your model”; the agent still replies `Ready` with the flashed shape.

## Host commands

```bash
enn devices
enn hil ping
enn hil infer --input 64
```

Flash the WBA example:

```bash
cd examples/stm32wba65ri
cargo check
cargo check --features hil-usb --bin hil_agent
# then probe-rs run --chip STM32WBA65RI --bin hil_agent --features hil-usb
```

## Studio

Ingest tab: **Refresh USB** / **Connect** uses VID/PID `1209:e612`. Codegen tab: **Run on device** sends the playground i8 vector; **Compare TFLite golden** re-imports the opened `.tflite` and checks the host interpreter against the playground.

Export writes `model.rs` and a versioned `dsp_contract.json` sidecar (window, hop, Mel bins, input scale). The MCU must apply that DSP before `predict`; the sidecar is documentation, not an on-device DSP runtime.
