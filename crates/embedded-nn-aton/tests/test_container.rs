use embedded_nn_aton::container::EcContainerBuilder;
use embedded_nn_aton::isa::{BINARY_MAGIC, BLOB_MAGIC, EcInstruction};

/// Reference `no_std` parser matching `embassy_stm32::npu::ecloader::EcBinary`.
struct TestEcBinary<'a> {
    words: &'a [u32],
    reloc_off: usize,
    blob_off: usize,
}

impl<'a> TestEcBinary<'a> {
    fn new(words: &'a [u32]) -> Self {
        assert!(words.len() >= 5);
        assert_eq!(words[0], BINARY_MAGIC);
        let reloc_off = (words[1] / 4) as usize;
        let blob_off = (words[4] / 4) as usize;
        Self {
            words,
            reloc_off,
            blob_off,
        }
    }

    fn num_relocs(&self) -> usize {
        self.words[self.reloc_off] as usize
    }

    fn reloc_id(&self, idx: usize) -> &str {
        let t = &self.words[self.reloc_off..];
        let id_byte_off = t[3 * idx + 1] as usize;
        let bytes: &[u8] =
            unsafe { core::slice::from_raw_parts(t.as_ptr() as *const u8, t.len() * 4) };
        let s = &bytes[id_byte_off..];
        let end = s.iter().position(|&b| b == 0).unwrap();
        core::str::from_utf8(&s[..end]).unwrap()
    }

    fn blob_words(&self) -> &[u32] {
        let s = &self.words[self.blob_off..];
        assert_eq!(s[0], BLOB_MAGIC);
        let count = s[1] as usize;
        &s[2..2 + count]
    }
}

#[test]
fn test_ec_container_packaging() {
    let mut builder = EcContainerBuilder::new();

    // Add some instructions
    let mut instr_words = Vec::new();
    EcInstruction::Nop.encode(&mut instr_words);
    EcInstruction::WriteReg {
        reg_addr: 0x480E_5008, // STRENG 0 ADDR
        value: 0x0000_0000,
    }
    .encode(&mut instr_words);
    EcInstruction::WriteReg {
        reg_addr: 0x480E_8008, // STRENG 3 ADDR
        value: 0x0000_0000,
    }
    .encode(&mut instr_words);
    EcInstruction::TriggerIrq.encode(&mut instr_words);
    EcInstruction::End.encode(&mut instr_words);

    builder.extend_instructions(instr_words);
    builder.add_relocation("_user_io_input_0", 2); // word 2 is the value of first WriteReg
    builder.add_relocation("_user_io_output_0", 4); // word 4 is the value of second WriteReg

    let u32_words = builder.build_words_u32().expect("build failed");
    assert_eq!(u32_words[0], BINARY_MAGIC);

    let parsed = TestEcBinary::new(&u32_words);
    assert_eq!(parsed.num_relocs(), 2);
    assert_eq!(parsed.reloc_id(0), "_user_io_input_0");
    assert_eq!(parsed.reloc_id(1), "_user_io_output_0");

    let blob = parsed.blob_words();
    assert_eq!(blob.len(), builder.num_instructions());

    let u64_words = builder.build_words_u64().expect("build u64 failed");
    assert!(!u64_words.is_empty());
    assert_eq!(u64_words[0] as u32, BINARY_MAGIC);
}
