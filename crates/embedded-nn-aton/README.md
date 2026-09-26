# embedded-nn-aton: Independent STM32N6 Neural-ART NPU Hardware Backend

`embedded-nn-aton` is an independent, open-source hardware acceleration backend and microcode compiler targeting the **STM32N6 Neural-ART (ATON) NPU** architecture.

It enables pure Rust, `#![no_std]`, and bare-metal environments (such as [Embassy](https://github.com/embassy-rs/embassy), FreeRTOS, or custom HALs) to compile and execute neural networks directly on the on-chip NPU hardware with zero dynamic allocations and without external proprietary host compiler dependencies.

---

## Architecture & Positioning

The STM32N6 family features a dedicated on-chip Neural Processing Unit (Neural-ART / ATON) alongside an **Epoch Controller** execution processor. 

Rather than relying on closed-source host utilities, `embedded-nn-aton` serves as a first-class hardware compilation backend for the `embedded-nn` ecosystem. It translates high-level quantized neural network graphs directly into native hardware register configurations, multi-dimensional DMA descriptors, crossbar switch routing matrices, and binary-compatible `EcBinary` microcode streams.

```
                      +---------------------------------------+
                      |   Model Graph (.tflite / ModelGraph)  |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |        AtonCompiler Partitioning      |
                      +---------------------------------------+
                                   /             \
                                  /               \
            [Supported Layers]   /                 \   [Fallback Layers]
                                v                   v
        +-----------------------------+       +-----------------------------+
        |  Hardware Microcode Engine  |       |   Cortex-M55 Vectorized     |
        | - Convacc0..3 (Conv/DW)     |       |   Software Execution        |
        | - Poolacc0..1 (Max/Avg/Mean)|       |   (Armv8.1-M Helium SIMD)   |
        | - Arithacc0..3 (Add/Mul)    |       +-----------------------------+
        | - Activacc0..1 (ReLU/ReLU6) |                      |
        | - Streng0..9 DMA Descriptors|                      |
        | - Strswitch Crossbar Matrix |                      |
        +-----------------------------+                      |
                       |                                     |
                       v                                     v
        +-----------------------------+       +-----------------------------+
        |   EcBinary Microcode Blob   |       |  Static SRAM Intermediate   |
        |   (0xECBF0050 Container)    |       |  Buffer Lifetimes           |
        +-----------------------------+       +-----------------------------+
                       \                                     /
                        \                                   /
                         v                                 v
               +----------------------------------------------------+
               |           STM32N6 Target Hardware Execution        |
               |        (Embassy / RTOS / Bare-Metal Driver)        |
               +----------------------------------------------------+
```

---

## Key Hardware Modules

1. **Epoch Controller ISA & Assembler ([`isa.rs`](src/isa.rs))**:
   - Hardware register definitions mapped to the non-secure peripheral base (`0x480E_0000`).
   - Opcode generator for Epoch Controller instructions: `WriteReg`, `WaitEvents`, `TriggerIrq`, `End`, and `Branch`.
2. **Streaming Engine DMA Descriptors ([`dma.rs`](src/dma.rs))**:
   - Multi-dimensional tensor DMA programming across 10 independent channels (`STRENG0..9`).
   - Spatial stride calculations, circular buffering, burst alignments, and sub-byte packing.
3. **Crossbar Interconnect Router ([`switch.rs`](src/switch.rs))**:
   - Configures the 41-port `STRSWITCH` crossbar matrix routing DMA streaming channels to compute accelerators (`ConvIn`, `PoolIn`, `ArithIn`, `ActUnitIn`).
4. **Hardware Compute Operators ([`compiler.rs`](src/compiler.rs))**:
   - **`Conv2D` & `DepthwiseConv2D`**: Multi-unit acceleration with hardware SIMD 8-bit mode, padding, and stride configuration.
   - **`MaxPool2D` & `AvgPool2D`**: Arbitrary pooling kernels with fixed-point Q16 reciprocal scaling.
   - **`Mean` (Global Average Pooling)**: Spatial reduction mapped to hardware `POOL_OP_GAVG`.
   - **`ElementwiseAdd` & `ElementwiseMul`**: Dual-stream affine combination ($A \cdot X + B \cdot Y + C$) and product evaluation via `Arithacc`.
   - **Fused Activations**: Hardware-fused `ReLU` and clamped `ReLU6`.
5. **Container & Relocation Packager ([`container.rs`](src/container.rs))**:
   - Packages exact `EcBinary` container images:
     - Binary Magic: `0xECBF0050`
     - Blob Magic: `0xCA057A7A`
     - Dynamic relocation tables for relocatable input, output, and weight buffer base addresses.

---

## File Format Specifications

### `EcBinary` Container Layout
```
+-------------------------------------------------------------+
| Header (20 bytes):                                          |
|   magic = 0xECBF0050                                        |
|   reloc_table_offset (u32)                                  |
|   patch_offset (u32)                                        |
|   debug_offset (u32)                                        |
|   blob_offset (u32)                                         |
+-------------------------------------------------------------+
| Relocation Table:                                           |
|   num_relocs (u32)                                          |
|   Entries: [str_offset, num_locs, loc_array_offset] * N     |
+-------------------------------------------------------------+
| Patch Table:                                                |
|   num_patches (u32)                                         |
|   Entries: [str_offset, shr, mask, num_locs, loc_arr_off] *M|
+-------------------------------------------------------------+
| Blob Section:                                               |
|   blob_magic = 0xCA057A7A                                   |
|   size_words (u32)                                          |
|   instructions: uint32_t[size_words]                        |
+-------------------------------------------------------------+
```

### Dynamic Buffer Relocation
When loading an `EcBinary` container, base buffer addresses (input activations, output tensors, static weights) are relocated at load time:
```c
blob[offset + 2] += (base_address - prev_base_address);
```

---

## CLI Usage

Compile a model directly targeting the STM32N6 Neural-ART NPU hardware backend:
```bash
# Compile a quantized TFLite model to an EcBinary container
enn aton --model model.tflite --out model_ec.bin

# Inspect/disassemble an existing container
enn aton --model model.json
```

---

## Independence & Trademark Notice

This project is an independent open-source hardware backend developed under the `embedded-nn` project. It is not affiliated with, sponsored by, or endorsed by STMicroelectronics. STM32 is a registered trademark of STMicroelectronics.

---

## License

Dual-licensed under either of:
- Apache License, Version 2.0 ([`LICENSE-APACHE`](../../LICENSE-APACHE))
- MIT License ([`LICENSE-MIT`](../../LICENSE-MIT))
