//! STM32N6 Epoch Controller (`EcBinary`) container packager.

use crate::error::AtonError;
use crate::isa::{BINARY_MAGIC, BLOB_MAGIC};
use std::collections::BTreeMap;

/// Builder for creating STM32N6 Epoch Controller binary containers.
#[derive(Debug, Clone, Default)]
pub struct EcContainerBuilder {
    relocations: BTreeMap<String, Vec<u32>>,
    blob_instructions: Vec<u32>,
}

impl EcContainerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a relocation entry for `symbol` targeting an instruction word offset in the blob.
    ///
    /// The `word_offset` is relative to the first instruction word (instruction 0).
    pub fn add_relocation(&mut self, symbol: impl Into<String>, word_offset: u32) {
        self.relocations
            .entry(symbol.into())
            .or_default()
            .push(word_offset);
    }

    /// Appends a single instruction word to the blob.
    pub fn push_instruction(&mut self, word: u32) {
        self.blob_instructions.push(word);
    }

    /// Extends the blob with an iterator of instruction words.
    pub fn extend_instructions(&mut self, words: impl IntoIterator<Item = u32>) {
        self.blob_instructions.extend(words);
    }

    /// Current instruction word count in the blob.
    pub fn num_instructions(&self) -> usize {
        self.blob_instructions.len()
    }

    /// Builds the complete `EcBinary` container as a vector of 32-bit words.
    pub fn build_words_u32(&self) -> Result<Vec<u32>, AtonError> {
        // Validate relocations
        for list in self.relocations.values() {
            for &off in list {
                if off as usize >= self.blob_instructions.len() {
                    return Err(AtonError::RelocationOutOfBounds(
                        off as usize,
                        self.blob_instructions.len(),
                    ));
                }
            }
        }

        // Section 1: Container header (5 words / 20 bytes)
        // [magic, reloc_off, patch_off, debug_off, blob_off]
        let mut words: Vec<u32> = vec![BINARY_MAGIC, 0, 0, 0, 0];

        // Section 2: Relocations table (starts at word 5, byte 20)
        let reloc_word_offset = words.len();
        let reloc_count = self.relocations.len();

        let mut table_words: Vec<u32> = Vec::new();
        table_words.push(reloc_count as u32);

        // Preallocate space for entries: [id_off, num, list_off]
        for _ in 0..reloc_count {
            table_words.push(0);
            table_words.push(0);
            table_words.push(0);
        }

        // Append payload: strings and lists
        for (i, (symbol, list)) in self.relocations.iter().enumerate() {
            // String offset in bytes relative to table start
            let id_byte_offset = (table_words.len() * 4) as u32;

            // Pack null-terminated string into 32-bit words
            let mut str_bytes = symbol.as_bytes().to_vec();
            str_bytes.push(0); // null terminator
            while !str_bytes.len().is_multiple_of(4) {
                str_bytes.push(0);
            }
            for chunk in str_bytes.as_chunks::<4>().0 {
                let w = u32::from_le_bytes(*chunk);
                table_words.push(w);
            }

            // List offset in bytes relative to table start
            let list_byte_offset = (table_words.len() * 4) as u32;
            for &off in list {
                table_words.push(off);
            }

            // Fill in the entry header
            let entry_idx = 1 + 3 * i;
            table_words[entry_idx] = id_byte_offset;
            table_words[entry_idx + 1] = list.len() as u32;
            table_words[entry_idx + 2] = list_byte_offset;
        }

        // Append table to words
        words.extend(&table_words);

        // Align blob offset to 8-byte boundary (even number of u32 words)
        while !words.len().is_multiple_of(2) {
            words.push(0); // padding
        }

        let blob_word_offset = words.len();

        // Update container header with byte offsets
        words[1] = (reloc_word_offset * 4) as u32;
        words[2] = 0; // patch_off (none)
        words[3] = 0; // debug_off (none)
        words[4] = (blob_word_offset * 4) as u32;

        // Section 3: Blob section
        // blob[0] = BLOB_MAGIC
        // blob[1] = instr_words
        // blob[2..] = instructions
        words.push(BLOB_MAGIC);
        words.push(self.blob_instructions.len() as u32);
        words.extend(&self.blob_instructions);

        // Pad entire container to 8-byte (even u32 count)
        while !words.len().is_multiple_of(2) {
            words.push(0);
        }

        Ok(words)
    }

    /// Builds the complete `EcBinary` container as 64-bit words aligned for STM32N6 NPU DMA.
    pub fn build_words_u64(&self) -> Result<Vec<u64>, AtonError> {
        let u32_words = self.build_words_u32()?;
        let mut u64_words = Vec::with_capacity(u32_words.len().div_ceil(2));

        for chunk in u32_words.chunks(2) {
            let low = chunk[0] as u64;
            let high = if chunk.len() > 1 {
                (chunk[1] as u64) << 32
            } else {
                0
            };
            u64_words.push(low | high);
        }

        Ok(u64_words)
    }

    /// Emits a C header file format (`<name>_ecblobs.h`).
    pub fn emit_c_header(&self, network_name: &str) -> Result<String, AtonError> {
        let u64_words = self.build_words_u64()?;
        let mut out = String::new();
        out.push_str("/* Auto-generated by embedded-nn-aton. DO NOT EDIT. */\n");
        out.push_str("#pragma once\n\n");
        out.push_str("#include <stdint.h>\n\n");
        out.push_str(&format!(
            "const uint64_t _ec_blob_{}_0[{}] = {{\n",
            network_name,
            u64_words.len()
        ));

        for (i, &word) in u64_words.iter().enumerate() {
            if i % 4 == 0 {
                out.push_str("    ");
            }
            out.push_str(&format!("0x{:016x}ULL, ", word));
            if i % 4 == 3 || i == u64_words.len() - 1 {
                out.push('\n');
            }
        }
        out.push_str("};\n");
        Ok(out)
    }

    /// Emits Rust source code format defining an aligned static `u64` array.
    pub fn emit_rust_code(&self, network_name: &str) -> Result<String, AtonError> {
        let u64_words = self.build_words_u64()?;
        let mut out = String::new();
        out.push_str("// Auto-generated by embedded-nn-aton for STM32N6 Neural-ART NPU.\n");
        out.push_str("// Zero-dependency Epoch Controller binary container.\n\n");
        out.push_str(&format!(
            "pub static EC_CONTAINER_{}: [u64; {}] = [\n",
            network_name.to_uppercase(),
            u64_words.len()
        ));

        for (i, &word) in u64_words.iter().enumerate() {
            if i % 4 == 0 {
                out.push_str("    ");
            }
            out.push_str(&format!("0x{:016x}, ", word));
            if i % 4 == 3 || i == u64_words.len() - 1 {
                out.push('\n');
            }
        }
        out.push_str("];\n");
        Ok(out)
    }
}
