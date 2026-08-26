# Sub-Byte 4-Bit (s4) & Codebook LUT Quantization

Demonstrates ultra-low Flash quantization techniques for microcontrollers with extreme storage constraints using `embedded-nn`.

---

## Quantization Capabilities

1. **Signed 4-Bit (`s4`) Linear Packing**:
   - Packs two 4-bit signed weights `[-8..=7]` into a single byte (`pack_s4_pair` / `unpack_s4_pair`).
   - Cuts Flash weight storage by **50%** relative to standard INT8 CMSIS-NN formats.
   - Zero-allocation direct execution via `fully_connected_s4` and `convolve_s4`.
2. **Nonlinear 16-Entry Codebook LUT (`s4_lut`)**:
   - 4-bit nibbles index into a 16-entry codebook centroids table.
   - Preserves non-uniform and bell-curve weight distributions with negligible accuracy loss.

---

## Running the Example

```bash
cargo run --package embedded-nn --example subbyte_quantization
```

Or from within this directory:

```bash
cd examples/subbyte-quantization
cargo run
```
