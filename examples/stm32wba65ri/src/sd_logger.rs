//! Dataset SD Card logger, SPI block driver, and JSON Lines (`.jsonl`) serializer for `#![no_std]`.
//!
//! Formats captured accelerometer frames into the exact dataset schema specified in
//! `docs/DATASET_IMPORT_FORMAT.md` for seamless ingestion into `embedded-nn-studio`
//! and `embedded-nn-train`.

#![allow(dead_code)]

use core::fmt::Write;
use embedded_hal::digital::OutputPin;
use embedded_hal::spi::SpiBus;

/// Maximum number of 3-axis samples in a single recorded burst.
pub const MAX_BURST_SAMPLES: usize = 256;

/// SD Block size in bytes (standard 512 bytes).
pub const SD_BLOCK_SIZE: usize = 512;

/// A single 3-axis acceleration sample in g units.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct AccelSample {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// A recorded dataset burst ready for SD card persistence or streaming.
#[derive(Clone, Debug)]
pub struct DatasetBurst {
    pub sample_id: u32,
    pub sample_rate_hz: f32,
    pub samples: [AccelSample; MAX_BURST_SAMPLES],
    pub count: usize,
}

impl DatasetBurst {
    /// Create a new empty burst container.
    pub const fn new(sample_id: u32, sample_rate_hz: f32) -> Self {
        Self {
            sample_id,
            sample_rate_hz,
            samples: [AccelSample {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            }; MAX_BURST_SAMPLES],
            count: 0,
        }
    }

    /// Reset burst container with new sample ID.
    pub fn reset(&mut self, sample_id: u32) {
        self.sample_id = sample_id;
        self.count = 0;
    }

    /// Push a 3-axis sample into the burst.
    pub fn push(&mut self, sample: AccelSample) -> bool {
        if self.count < MAX_BURST_SAMPLES {
            self.samples[self.count] = sample;
            self.count += 1;
            true
        } else {
            false
        }
    }

    /// Check if burst is full.
    pub const fn is_full(&self) -> bool {
        self.count >= MAX_BURST_SAMPLES
    }

    /// Serialize this burst to a JSON Lines (`.jsonl`) record into `out_buf`.
    /// Returns the number of bytes written on success.
    pub fn format_jsonl(&self, out_buf: &mut [u8]) -> Result<usize, ()> {
        let mut writer = BufferWriter::new(out_buf);

        // Header: {"sample_id":"sample_XXXX","label":null,"sample_rate_hz":100.0,"channel_names":["x","y","z"],"waveform":[
        write!(
            writer,
            "{{\"sample_id\":\"sample_{:04}\",\"label\":null,\"sample_rate_hz\":{:.1},\"channel_names\":[\"x\",\"y\",\"z\"],\"waveform\":[",
            self.sample_id, self.sample_rate_hz
        )
        .map_err(|_| ())?;

        // Waveform array: [[x0,y0,z0],[x1,y1,z1],...]
        for i in 0..self.count {
            let s = &self.samples[i];
            if i > 0 {
                writer.write_byte(b',')?;
            }
            write!(writer, "[{:.4},{:.4},{:.4}]", s.x, s.y, s.z).map_err(|_| ())?;
        }

        // Closing: ]}\n
        write!(writer, "]}}\n").map_err(|_| ())?;

        Ok(writer.len())
    }
}

/// Minimal `#![no_std]` byte buffer writer implementing `core::fmt::Write`.
pub struct BufferWriter<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> BufferWriter<'a> {
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn write_byte(&mut self, b: u8) -> Result<(), ()> {
        if self.pos < self.buf.len() {
            self.buf[self.pos] = b;
            self.pos += 1;
            Ok(())
        } else {
            Err(())
        }
    }

    pub const fn len(&self) -> usize {
        self.pos
    }

    pub const fn is_empty(&self) -> bool {
        self.pos == 0
    }
}

impl<'a> Write for BufferWriter<'a> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let bytes = s.as_bytes();
        let remaining = self.buf.len() - self.pos;
        if bytes.len() > remaining {
            return Err(core::fmt::Error);
        }
        self.buf[self.pos..self.pos + bytes.len()].copy_from_slice(bytes);
        self.pos += bytes.len();
        Ok(())
    }
}

/// SPI-mode MicroSD block device commands.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SdCommand {
    Cmd0GoIdle = 0x40 | 0,
    Cmd8SendIfCond = 0x40 | 8,
    Cmd16SetBlockLen = 0x40 | 16,
    Cmd17ReadSingleBlock = 0x40 | 17,
    Cmd24WriteSingleBlock = 0x40 | 24,
    Cmd55AppCmd = 0x40 | 55,
    Cmd58ReadOcr = 0x40 | 58,
    Acmd41SdSendOpCond = 0x40 | 41,
}

impl SdCommand {
    pub const fn code(self) -> u8 {
        self as u8
    }
}

/// Standalone SPI-mode MicroSD card driver.
pub struct SpiSdCard<SPI, CS> {
    spi: SPI,
    cs: CS,
    is_sdhc: bool,
}

impl<SPI: SpiBus, CS: OutputPin> SpiSdCard<SPI, CS> {
    /// Create a new SPI SD Card instance.
    pub fn new(spi: SPI, mut cs: CS) -> Result<Self, CS::Error> {
        cs.set_high()?;
        Ok(Self {
            spi,
            cs,
            is_sdhc: false,
        })
    }

    fn select(&mut self) -> Result<(), CS::Error> {
        self.cs.set_low()
    }

    fn deselect(&mut self) -> Result<(), CS::Error> {
        self.cs.set_high()
    }

    /// Send raw SPI command (6 bytes) and return R1 response byte.
    pub fn send_cmd(&mut self, cmd: u8, arg: u32, crc: u8) -> Result<u8, SPI::Error> {
        let frame = [
            cmd,
            ((arg >> 24) & 0xFF) as u8,
            ((arg >> 16) & 0xFF) as u8,
            ((arg >> 8) & 0xFF) as u8,
            (arg & 0xFF) as u8,
            crc,
        ];

        self.spi.write(&frame)?;

        // Read until non-0xFF R1 response or timeout
        let mut r1 = 0xFF;
        for _ in 0..10 {
            let mut b = [0u8; 1];
            self.spi.read(&mut b)?;
            if b[0] != 0xFF {
                r1 = b[0];
                break;
            }
        }
        Ok(r1)
    }

    /// Initialize card into SPI mode (CMD0 -> CMD8 -> ACMD41 -> CMD58).
    pub fn init(&mut self) -> Result<bool, SPI::Error> {
        // 1. Send >= 74 dummy clock pulses with CS high
        let _ = self.deselect();
        let dummy = [0xFFu8; 10];
        self.spi.write(&dummy)?;

        // 2. Select card and issue CMD0 (GO_IDLE_STATE) with CRC 0x95
        let _ = self.select();
        let r1 = self.send_cmd(SdCommand::Cmd0GoIdle.code(), 0, 0x95)?;
        let _ = self.deselect();

        if r1 != 0x01 {
            return Ok(false); // Failed to enter idle state
        }

        // 3. Issue CMD8 (SEND_IF_COND) with 3.3V pattern (0x1AA) and CRC 0x87
        let _ = self.select();
        let r1_cmd8 = self.send_cmd(SdCommand::Cmd8SendIfCond.code(), 0x1AA, 0x87)?;
        let mut r7 = [0u8; 4];
        if r1_cmd8 == 0x01 {
            self.spi.read(&mut r7)?;
        }
        let _ = self.deselect();

        // 4. Repeatedly send ACMD41 until ready (R1 == 0x00)
        let mut ready = false;
        for _ in 0..200 {
            let _ = self.select();
            let _ = self.send_cmd(SdCommand::Cmd55AppCmd.code(), 0, 0x65);
            let r1 = self.send_cmd(SdCommand::Acmd41SdSendOpCond.code(), 0x4000_0000, 0x77)?;
            let _ = self.deselect();

            if r1 == 0x00 {
                ready = true;
                break;
            }
        }

        if !ready {
            return Ok(false);
        }

        // 5. Query OCR (CMD58) to check CCS (Card Capacity Status bit 30)
        let _ = self.select();
        let _ = self.send_cmd(SdCommand::Cmd58ReadOcr.code(), 0, 0xFD);
        let mut ocr = [0u8; 4];
        self.spi.read(&mut ocr)?;
        let _ = self.deselect();

        self.is_sdhc = (ocr[0] & 0x40) != 0;
        Ok(true)
    }

    /// Write a single 512-byte block to the SD Card (CMD24).
    pub fn write_block(
        &mut self,
        block_address: u32,
        data: &[u8; SD_BLOCK_SIZE],
    ) -> Result<bool, SPI::Error> {
        let addr = if self.is_sdhc {
            block_address
        } else {
            block_address * (SD_BLOCK_SIZE as u32)
        };

        let _ = self.select();
        let r1 = self.send_cmd(SdCommand::Cmd24WriteSingleBlock.code(), addr, 0xFF)?;
        if r1 != 0x00 {
            let _ = self.deselect();
            return Ok(false);
        }

        // Data token 0xFE
        self.spi.write(&[0xFE])?;
        // 512 bytes payload
        self.spi.write(data)?;
        // Dummy 16-bit CRC
        self.spi.write(&[0xFF, 0xFF])?;

        // Read Data Response token (0bxxx00101 = accepted)
        let mut resp = [0xFFu8; 1];
        for _ in 0..10 {
            self.spi.read(&mut resp)?;
            if resp[0] != 0xFF {
                break;
            }
        }

        // Busy wait while card writes block (MISO low)
        let mut busy = [0u8; 1];
        for _ in 0..5000 {
            self.spi.read(&mut busy)?;
            if busy[0] != 0x00 {
                break;
            }
        }

        let _ = self.deselect();
        Ok((resp[0] & 0x1F) == 0x05)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_writer_overflow() {
        let mut buf = [0u8; 4];
        let mut writer = BufferWriter::new(&mut buf);
        assert!(writer.write_byte(b'a').is_ok());
        assert!(writer.write_byte(b'b').is_ok());
        assert!(writer.write_byte(b'c').is_ok());
        assert!(writer.write_byte(b'd').is_ok());
        assert!(writer.write_byte(b'e').is_err());
        assert_eq!(writer.len(), 4);
    }

    #[test]
    fn test_jsonl_formatter_output() {
        let mut burst = JsonlRecordFormatter::new(1, 100.0);
        assert!(burst.push(AccelSample {
            x: 0.12,
            y: -0.34,
            z: 0.98
        }));
        assert!(burst.push(AccelSample {
            x: 0.15,
            y: -0.30,
            z: 0.95
        }));

        let mut buf = [0u8; 512];
        let written = burst.format_jsonl(&mut buf).unwrap();
        assert!(written > 0);

        let str_out = core::str::from_utf8(&buf[..written]).unwrap();
        assert!(str_out.starts_with("{\"sample_id\":\"sample_0001\""));
        assert!(str_out.contains("\"sample_rate_hz\":100.0"));
        assert!(str_out.contains("\"channel_names\":[\"x\",\"y\",\"z\"]"));
        assert!(str_out.ends_with("]}\n"));
    }
}
