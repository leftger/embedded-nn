# On-Device Audio Keyword Spotting (KWS)

An end-to-end TinyML audio wake-word / keyword spotting pipeline designed for microcontrollers with zero heap memory allocation.

---

## Processing Pipeline

```
Raw Audio Stream (16 kHz)
   │
   ▼
[High-Pass Filter (80 Hz Cutoff)]
   │
   ▼
[FFT Windowing (Hann, 64-pt, 50% overlap)]
   │
   ▼
[Mel Filterbank (16 frequency channels)]
   │
   ▼
[Quantization to INT8 (112 features)]
   │
   ▼
[Quantized Neural Network (112 -> 32 -> 4)]
   │
   ▼
[Softmax Activation -> Top Class Selection]
```

---

## Running the Example

```bash
cargo run --package embedded-nn --example keyword_spotting --features="dsp,libm"
```

Or from within this directory:

```bash
cd examples/keyword-spotting
cargo run
```

---

## Memory & Latency Profile

- **Peak SRAM Consumption**: < 256 bytes (allocated on stack or static BSS)
- **Flash Weight Storage**: ~3.7 KB (INT8 quantized weights)
- **Target Latency**: < 5 ms per frame on ARM Cortex-M4 / Cortex-M33 @ 64 MHz
