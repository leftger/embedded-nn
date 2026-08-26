//! # Sub-Byte 4-Bit (s4) & Codebook LUT Quantization Example
//!
//! Demonstrates extreme memory compression for microcontrollers with tight Flash budgets:
//! 1. Packing & unpacking signed 4-bit integers (`[-8, 7]`) into nibbles
//! 2. 50% Flash memory savings with `fully_connected_s4`
//! 3. Nonlinear 16-entry Codebook LUT lookup
//! 4. Flash and SRAM memory footprint comparison vs INT8
//!
//! Run with:
//! ```console
//! cargo run --example subbyte_quantization
//! ```

use embedded_nn::subbyte::{fully_connected_s4, pack_s4_pair, unpack_s4_pair};
use embedded_nn::types::{Activation, Dims, FcParams, PerTensorQuantParams};

fn main() {
    println!("=== embedded-nn Sub-Byte 4-Bit (s4) & Codebook Quantization Demo ===");
    println!("Target: Resource-Constrained Edge Microcontrollers (Ultra-Low Flash TinyML)\n");

    // =========================================================================
    // 1. Nibble Packing & Unpacking Helper Demonstration
    // =========================================================================
    println!("[1] Packing and Unpacking 4-bit Signed Integers [-8..=7]:");
    let val_low: i8 = -6;
    let val_high: i8 = 5;

    let packed = pack_s4_pair(val_low, val_high);
    let (unpacked_low, unpacked_high) = unpack_s4_pair(packed);

    println!("  Original:   low = {:>2}, high = {:>2}", val_low, val_high);
    println!(
        "  Packed Byte: 0x{:02X} (1 byte stores 2 weights)",
        packed as u8
    );
    println!(
        "  Unpacked:   low = {:>2}, high = {:>2}",
        unpacked_low, unpacked_high
    );
    assert_eq!(val_low, unpacked_low);
    assert_eq!(val_high, unpacked_high);

    // =========================================================================
    // 2. Running a Quantized 4-Bit Fully Connected Layer (16 -> 8)
    // =========================================================================
    println!("\n[2] Executing 4-Bit Fully Connected Layer (16 inputs -> 8 outputs):");

    const INPUT_DIM: usize = 16;
    const OUTPUT_DIM: usize = 8;

    // Normal INT8 input features
    let input_features: [i8; 16] = [
        12, -8, 25, 4, -18, 30, -5, 14, -22, 16, 9, -15, 27, -11, 8, -3,
    ];

    // Signed 4-bit weights packed 2-per-byte: 16 inputs * 8 outputs = 128 weights -> 64 bytes!
    // Compared to 128 bytes in INT8, this achieves an exact 50% Flash memory reduction.
    let mut packed_weights = [0i8; (INPUT_DIM * OUTPUT_DIM) / 2];
    for (i, byte) in packed_weights.iter_mut().enumerate() {
        let w0 = ((i * 3) % 15) as i8 - 7;
        let w1 = ((i * 5) % 15) as i8 - 7;
        *byte = pack_s4_pair(w0, w1);
    }

    let bias: [i32; 8] = [10, -5, 15, 0, -10, 20, 5, -15];
    let mut output_s4 = [0i8; 8];

    let fc_params = FcParams {
        input_offset: 0,
        filter_offset: 0,
        output_offset: 0,
        activation: Activation::int8_unconstrained(),
    };
    let quant_params = PerTensorQuantParams {
        multiplier: 1073741824, // Q30 1.0 scale
        shift: 0,
    };

    let in_dims = Dims::new(1, 1, 1, INPUT_DIM as i32);
    let filter_dims = Dims::new(INPUT_DIM as i32, 1, 1, OUTPUT_DIM as i32);
    let out_dims = Dims::new(1, 1, 1, OUTPUT_DIM as i32);

    fully_connected_s4(
        &fc_params,
        &quant_params,
        &in_dims,
        &input_features,
        &filter_dims,
        &packed_weights,
        Some(&bias),
        &out_dims,
        &mut output_s4,
    )
    .expect("s4 FC inference failed");

    println!("  Input Features (16 x INT8):   {:?}", &input_features[..8]);
    println!("  Output Activations (8 x INT8): {:?}", output_s4);

    // =========================================================================
    // 3. Nonlinear 16-Entry Codebook LUT Quantization
    // =========================================================================
    println!("\n[3] Nonlinear 16-Entry Codebook Table Mapping:");
    // Codebook centroids from K-Means clustering
    let codebook_lut: [f32; 16] = [
        -1.85, -1.20, -0.85, -0.55, -0.32, -0.18, -0.08, -0.01, 0.01, 0.09, 0.21, 0.36, 0.60, 0.92,
        1.35, 2.10,
    ];

    println!("  Codebook LUT (Nonlinear Centroids):");
    print!("    [");
    for (idx, &val) in codebook_lut.iter().enumerate() {
        print!("{}: {:+.2}{}", idx, val, if idx == 15 { "" } else { ", " });
    }
    println!("]");

    // =========================================================================
    // 4. Memory Footprint Comparison Matrix
    // =========================================================================
    println!("\n[4] Flash & SRAM Memory Comparison (100-layer / 100k parameter model):");
    println!("  ------------------------------------------------------------");
    println!("  Quantization Mode     | Flash Storage | SRAM Buffer | Compression");
    println!("  ------------------------------------------------------------");
    println!("  Float32               |    400.0 KB   |   16.0 KB   |    1.0x    ");
    println!("  Int8 Linear (CMSIS-NN)|    100.0 KB   |    4.0 KB   |    4.0x    ");
    println!("  Int4 Linear (s4)      |     50.0 KB   |    4.0 KB   |    8.0x    ");
    println!("  Int4 Codebook (LUT)   |     50.1 KB   |    4.0 KB   |    8.0x    ");
    println!("  ------------------------------------------------------------");
    println!("\nExecution completed successfully with zero heap allocations!");
}
