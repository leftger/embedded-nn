# embedded-nn-aton

Open-source compiler, microcode generator, and binary container packager for the **STM32N6 Neural-ART (ATON) NPU** and **Epoch Controller (`EcBinary`)** runtime.

## Overview

The STM32N6 microcontrollers feature a dedicated Neural-ART (ATON) Neural Processing Unit (NPU) and an on-chip **Epoch Controller** that executes compiled microcode streams.

`embedded-nn-aton` is an open-source, `#![no_std]`-compatible Rust implementation that parses quantized neural network graphs, partitions them into hardware execution epochs, synthesizes multidimensional DMA (`STRENG[0..9]`) and crossbar router (`STRSWITCH`) configurations, and packages binary-compatible **`EcBinary`** microcode containers.

---

## Architecture & Subsystems

1. **Epoch Controller ISA ([`isa.rs`](src/isa.rs))**:
   - Instruction opcodes (`WriteReg`, `WaitEvents`, `TriggerIrq`, `End`, `Branch`).
   - Hardware register address definitions for ATON streaming, convolution (`Convacc[0..3]`), and arithmetic (`Arithacc[0..1]`) units.
2. **Streaming DMA Engine ([`dma.rs`](src/dma.rs))**:
   - Multidimensional tensor descriptors for 10 independent hardware streaming channels (`STRENG0..9`).
   - Configures spatial strides, sub-byte packing, circular buffers, and burst alignments.
3. **Crossbar Router ([`switch.rs`](src/switch.rs))**:
   - 41-port `STRSWITCH` interconnect configuration matrix connecting memory streaming channels to compute units.
4. **Binary Container Packager ([`container.rs`](src/container.rs))**:
   - Generates exact byte-compatible Epoch Controller binary images:
     - Binary Magic: `0xECBF0050` (`ECASM_BINARY_MAGIC`)
     - Blob Magic: `0xCA057A7A` (`ECASM_BLOB_MAGIC`)
     - Section offsets for relocations, patches, debug symbols, and instructions.
     - 3-word relocation entries for dynamic buffer base-address updates.
     - 5-word patch entries for dynamic bitfield masking and shifting.
5. **Graph Compiler & Epoch Scheduler ([`compiler.rs`](src/compiler.rs))**:
   - Partitions neural network graphs into sequential hardware epochs and software fallback layers.

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

### Relocation Patch Algorithm
When loading an `EcBinary`, base buffer addresses (inputs, outputs, weights) are patched via:
```c
blob[offset + 2] += (base_address - prev_base_address);
```

---

## Integration with CLI

You can invoke `embedded-nn-aton` directly via the `enn` CLI tool:
```bash
# Compile a quantized TFLite model to an ATON EcBinary container
enn aton compile -i model.tflite -o model_ec.bin

# Inspect/disassemble an existing EcBinary container
enn aton inspect -i model_ec.bin
```

---

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
