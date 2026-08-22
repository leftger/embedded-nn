//! Winbond W25Q32BV / W25Q16BV SPI NOR Flash driver for `#![no_std]`.
//!
//! Handles JEDEC ID verification, sector/block erase, page program, and continuous reads
//! for high-speed sensor burst logging and model weight staging.

#![allow(dead_code)]

use embedded_hal::digital::OutputPin;
use embedded_hal::spi::SpiBus;

/// Winbond Manufacturer ID (`0xEF`).
pub const WINBOND_MANUFACTURER_ID: u8 = 0xEF;

/// Memory type for standard SPI (`0x40`).
pub const WINBOND_MEMORY_TYPE_SPI: u8 = 0x40;

/// Capacity code for 32 Mbit (4 MByte) W25Q32 (`0x16`).
pub const WINBOND_CAPACITY_32MBIT: u8 = 0x16;

/// Capacity code for 16 Mbit (2 MByte) W25Q16 (`0x15`).
pub const WINBOND_CAPACITY_16MBIT: u8 = 0x15;

/// Page size in bytes.
pub const PAGE_SIZE: usize = 256;

/// Sector size in bytes (4 KB).
pub const SECTOR_SIZE: usize = 4096;

/// Commands.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Command {
    WriteEnable = 0x06,
    WriteDisable = 0x04,
    ReadStatus1 = 0x05,
    ReadStatus2 = 0x35,
    WriteStatus = 0x01,
    ReadData = 0x03,
    FastRead = 0x0B,
    PageProgram = 0x02,
    SectorErase4k = 0x20,
    BlockErase32k = 0x52,
    BlockErase64k = 0xD8,
    ChipErase = 0xC7,
    PowerDown = 0xB9,
    ReleasePowerDown = 0xAB,
    JedecId = 0x9F,
}

impl Command {
    pub const fn code(self) -> u8 {
        self as u8
    }
}

/// JEDEC ID identification tuple.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct JedecId {
    pub manufacturer: u8,
    pub memory_type: u8,
    pub capacity: u8,
}

/// W25Q32 SPI NOR Flash driver.
pub struct W25q32<SPI, CS> {
    spi: SPI,
    cs: CS,
}

impl<SPI: SpiBus, CS: OutputPin> W25q32<SPI, CS> {
    /// Create a new W25Q32 driver instance.
    pub fn new(spi: SPI, mut cs: CS) -> Result<Self, CS::Error> {
        cs.set_high()?;
        Ok(Self { spi, cs })
    }

    /// Helper to select chip (drive CS low).
    fn select(&mut self) -> Result<(), CS::Error> {
        self.cs.set_low()
    }

    /// Helper to deselect chip (drive CS high).
    fn deselect(&mut self) -> Result<(), CS::Error> {
        self.cs.set_high()
    }

    /// Read 3-byte JEDEC ID.
    pub fn read_jedec_id(&mut self) -> Result<JedecId, SPI::Error> {
        let _ = self.select();
        let cmd = [Command::JedecId.code()];
        let mut resp = [0u8; 3];

        let res = (|| {
            self.spi.write(&cmd)?;
            self.spi.read(&mut resp)?;
            Ok(())
        })();

        let _ = self.deselect();
        res?;

        Ok(JedecId {
            manufacturer: resp[0],
            memory_type: resp[1],
            capacity: resp[2],
        })
    }

    /// Check if JEDEC ID matches Winbond W25Q32 or W25Q16.
    pub fn is_valid_winbond(&mut self) -> Result<bool, SPI::Error> {
        let id = self.read_jedec_id()?;
        Ok(id.manufacturer == WINBOND_MANUFACTURER_ID
            && id.memory_type == WINBOND_MEMORY_TYPE_SPI
            && (id.capacity == WINBOND_CAPACITY_32MBIT || id.capacity == WINBOND_CAPACITY_16MBIT))
    }

    /// Issue Write Enable command (0x06).
    pub fn write_enable(&mut self) -> Result<(), SPI::Error> {
        let _ = self.select();
        let res = self.spi.write(&[Command::WriteEnable.code()]);
        let _ = self.deselect();
        res
    }

    /// Read Status Register 1 (0x05).
    pub fn read_status1(&mut self) -> Result<u8, SPI::Error> {
        let _ = self.select();
        let cmd = [Command::ReadStatus1.code()];
        let mut status = [0u8; 1];

        let res = (|| {
            self.spi.write(&cmd)?;
            self.spi.read(&mut status)?;
            Ok(())
        })();

        let _ = self.deselect();
        res?;
        Ok(status[0])
    }

    /// Wait until write/erase busy flag (bit 0) clears.
    pub fn wait_busy(&mut self) -> Result<(), SPI::Error> {
        while (self.read_status1()? & 0x01) != 0 {
            // Busy polling loop
            cortex_m::asm::nop();
        }
        Ok(())
    }

    /// Erase 4 KB sector at `address`.
    pub fn sector_erase(&mut self, address: u32) -> Result<(), SPI::Error> {
        self.write_enable()?;

        let _ = self.select();
        let cmd = [
            Command::SectorErase4k.code(),
            ((address >> 16) & 0xFF) as u8,
            ((address >> 8) & 0xFF) as u8,
            (address & 0xFF) as u8,
        ];
        let res = self.spi.write(&cmd);
        let _ = self.deselect();
        res?;

        self.wait_busy()
    }

    /// Erase entire flash chip.
    pub fn chip_erase(&mut self) -> Result<(), SPI::Error> {
        self.write_enable()?;

        let _ = self.select();
        let res = self.spi.write(&[Command::ChipErase.code()]);
        let _ = self.deselect();
        res?;

        self.wait_busy()
    }

    /// Program a single page (up to 256 bytes) at `address`.
    pub fn page_program(&mut self, address: u32, data: &[u8]) -> Result<(), SPI::Error> {
        if data.is_empty() {
            return Ok(());
        }

        self.write_enable()?;

        let _ = self.select();
        let cmd = [
            Command::PageProgram.code(),
            ((address >> 16) & 0xFF) as u8,
            ((address >> 8) & 0xFF) as u8,
            (address & 0xFF) as u8,
        ];

        let res = (|| {
            self.spi.write(&cmd)?;
            self.spi.write(data)?;
            Ok(())
        })();

        let _ = self.deselect();
        res?;

        self.wait_busy()
    }

    /// Read `data.len()` bytes starting at `address`.
    pub fn read(&mut self, address: u32, data: &mut [u8]) -> Result<(), SPI::Error> {
        if data.is_empty() {
            return Ok(());
        }

        let _ = self.select();
        let cmd = [
            Command::ReadData.code(),
            ((address >> 16) & 0xFF) as u8,
            ((address >> 8) & 0xFF) as u8,
            (address & 0xFF) as u8,
        ];

        let res = (|| {
            self.spi.write(&cmd)?;
            self.spi.read(data)?;
            Ok(())
        })();

        let _ = self.deselect();
        res
    }
}
