//! LIS2DE12 3-axis MEMS accelerometer driver for `#![no_std]` embedded environments.
//!
//! Provides I2C register configuration, FIFO modes, full-scale ranges (+/-2g, +/-4g, +/-8g, +/-16g),
//! and conversion to SI units (g / m/s^2).

#![allow(dead_code)]

/// Default 7-bit I2C address of LIS2DE12 when SA0/SDO is tied to GND.
pub const LIS2DE12_I2C_ADDR: u8 = 0x18;

/// Alternate 7-bit I2C address of LIS2DE12 when SA0/SDO is tied to VDD.
pub const LIS2DE12_I2C_ADDR_ALT: u8 = 0x19;

/// Expected WHO_AM_I value for LIS2DE12.
pub const WHO_AM_I_VAL: u8 = 0x33;

/// Register addresses.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Register {
    StatusAux = 0x07,
    OutTempL = 0x0C,
    OutTempH = 0x0D,
    WhoAmI = 0x0F,
    CtrlReg1 = 0x20,
    CtrlReg2 = 0x21,
    CtrlReg3 = 0x22,
    CtrlReg4 = 0x23,
    CtrlReg5 = 0x24,
    CtrlReg6 = 0x25,
    Reference = 0x26,
    StatusReg = 0x27,
    OutXL = 0x28,
    OutXH = 0x29,
    OutYL = 0x2A,
    OutYH = 0x2B,
    OutZL = 0x2C,
    OutZH = 0x2D,
    FifoCtrlReg = 0x2E,
    FifoSrcReg = 0x2F,
    Int1Cfg = 0x30,
    Int1Src = 0x31,
    Int1Ths = 0x32,
    Int1Duration = 0x33,
    Int2Cfg = 0x34,
    Int2Src = 0x35,
    Int2Ths = 0x36,
    Int2Duration = 0x37,
    ClickCfg = 0x38,
    ClickSrc = 0x39,
    ClickThs = 0x3A,
    TimeLimit = 0x3B,
    TimeLatency = 0x3C,
    TimeWindow = 0x3D,
    ActThs = 0x3E,
    ActDur = 0x3F,
}

impl Register {
    pub const fn addr(self) -> u8 {
        self as u8
    }

    /// Auto-increment bit for multi-byte sequential register reads/writes.
    pub const fn auto_inc(self) -> u8 {
        (self as u8) | 0x80
    }
}

/// Output data rate (ODR) options.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Odr {
    PowerDown = 0b0000,
    Hz1 = 0b0001,
    Hz10 = 0b0010,
    Hz25 = 0b0011,
    Hz50 = 0b0100,
    Hz100 = 0b0101,
    Hz200 = 0b0110,
    Hz400 = 0b0111,
    LowPowerHz1620 = 0b1000,
    Hz1344 = 0b1001,
}

/// Full-scale range selection.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FullScale {
    G2 = 0b00,
    G4 = 0b01,
    G8 = 0b10,
    G16 = 0b11,
}

impl FullScale {
    /// Sensitivity in mg/digit (Normal 10-bit mode: 1 LSB = scale mg).
    pub const fn sensitivity_mg(self) -> f32 {
        match self {
            Self::G2 => 4.0,
            Self::G4 => 8.0,
            Self::G8 => 16.0,
            Self::G16 => 48.0,
        }
    }

    /// Conversion factor from 10-bit raw counts (signed i16) to g units.
    pub const fn scale_g(self) -> f32 {
        match self {
            Self::G2 => 0.004,
            Self::G4 => 0.008,
            Self::G8 => 0.016,
            Self::G16 => 0.048,
        }
    }
}

/// Operating mode (resolution).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Mode {
    /// 8-bit resolution, lowest power.
    LowPower,
    /// 10-bit resolution.
    Normal,
    /// 12-bit resolution.
    HighResolution,
}

/// Raw 3-axis accelerometer reading.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct RawAccel {
    pub x: i16,
    pub y: i16,
    pub z: i16,
}

/// 3-axis accelerometer reading in standard gravity units (g).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct AccelG {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl AccelG {
    /// Magnitude of acceleration vector: sqrt(x^2 + y^2 + z^2).
    pub fn magnitude(&self) -> f32 {
        libm::sqrtf(self.x * self.x + self.y * self.y + self.z * self.z)
    }

    /// As an array `[x, y, z]`.
    pub const fn to_array(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

/// LIS2DE12 sensor driver over `embedded-hal` I2C.
pub struct Lis2de12<I2C> {
    i2c: I2C,
    addr: u8,
    full_scale: FullScale,
}

impl<I2C: embedded_hal::i2c::I2c> Lis2de12<I2C> {
    /// Create a new driver instance with default I2C address (0x18).
    pub const fn new(i2c: I2C) -> Self {
        Self::new_with_addr(i2c, LIS2DE12_I2C_ADDR)
    }

    /// Create a new driver instance with explicit I2C address.
    pub const fn new_with_addr(i2c: I2C, addr: u8) -> Self {
        Self {
            i2c,
            addr,
            full_scale: FullScale::G2,
        }
    }

    /// Verify device ID (reads WHO_AM_I register).
    pub fn check_id(&mut self) -> Result<bool, I2C::Error> {
        let mut id = [0u8; 1];
        self.i2c.write_read(self.addr, &[Register::WhoAmI.addr()], &mut id)?;
        Ok(id[0] == WHO_AM_I_VAL)
    }

    /// Auto-detect LIS2DE12 by probing standard (0x18) and alternate (0x19) addresses.
    /// Updates `self.addr` to the responding address if found.
    pub fn auto_detect(&mut self) -> bool {
        for &test_addr in &[LIS2DE12_I2C_ADDR, LIS2DE12_I2C_ADDR_ALT] {
            defmt::info!("[LIS2DE12] Probing I2C address 0x{:02x} for WHO_AM_I (0x33)...", test_addr);
            let mut id = [0u8; 1];
            match self.i2c.write_read(test_addr, &[Register::WhoAmI.addr()], &mut id) {
                Ok(()) => {
                    defmt::info!("[LIS2DE12] Response from 0x{:02x}: WHO_AM_I = 0x{:02x}", test_addr, id[0]);
                    if id[0] == WHO_AM_I_VAL {
                        self.addr = test_addr;
                        return true;
                    } else {
                        defmt::warn!("[LIS2DE12] Device at 0x{:02x} returned unexpected ID 0x{:02x} (expected 0x33)", test_addr, id[0]);
                    }
                }
                Err(_) => {
                    defmt::debug!("[LIS2DE12] No response / NACK from address 0x{:02x}", test_addr);
                }
            }
        }
        false
    }

    /// Return the active I2C address.
    pub const fn address(&self) -> u8 {
        self.addr
    }

    /// Initialize sensor with specified ODR and Full-Scale range.
    pub fn init(&mut self, odr: Odr, fs: FullScale) -> Result<(), I2C::Error> {
        self.full_scale = fs;

        defmt::info!("[LIS2DE12] Configuring CTRL_REG1 (ODR=100Hz, all axes enabled)...");
        // CTRL_REG1: ODR[3:0] | LPen (0 for normal/HR) | Zen (1) | Yen (1) | Xen (1)
        let ctrl1 = ((odr as u8) << 4) | 0x07;
        self.i2c.write(self.addr, &[Register::CtrlReg1.addr(), ctrl1])?;

        defmt::info!("[LIS2DE12] Configuring CTRL_REG4 (BDU=1, FS=+/-2g, HR=1)...");
        // CTRL_REG4: BDU (bit 7 = 1) | BLE (0) | FS[1:0] (bits 5:4) | HR (bit 3 = 1)
        let fs_bits = (fs as u8) << 4;
        let ctrl4 = 0x80 | fs_bits | 0x08;
        self.i2c.write(self.addr, &[Register::CtrlReg4.addr(), ctrl4])?;

        defmt::info!("[LIS2DE12] Configuration registers written successfully");
        Ok(())
    }

    /// Read raw 10-bit/12-bit acceleration values (left-justified in 16-bit registers).
    pub fn read_raw(&mut self) -> Result<RawAccel, I2C::Error> {
        let mut buf = [0u8; 6];
        self.i2c.write_read(self.addr, &[Register::OutXL.auto_inc()], &mut buf)?;

        // 16-bit little endian, shift right by 6 for 10-bit normal representation
        let x = (i16::from_le_bytes([buf[0], buf[1]])) >> 6;
        let y = (i16::from_le_bytes([buf[2], buf[3]])) >> 6;
        let z = (i16::from_le_bytes([buf[4], buf[5]])) >> 6;

        Ok(RawAccel { x, y, z })
    }

    /// Read acceleration in units of standard gravity (g).
    pub fn read_accel_g(&mut self) -> Result<AccelG, I2C::Error> {
        let raw = self.read_raw()?;
        let scale = self.full_scale.scale_g();

        Ok(AccelG {
            x: raw.x as f32 * scale,
            y: raw.y as f32 * scale,
            z: raw.z as f32 * scale,
        })
    }

    /// Check if new acceleration data is ready (reads STATUS_REG).
    pub fn is_data_ready(&mut self) -> Result<bool, I2C::Error> {
        let mut status = [0u8; 1];
        self.i2c.write_read(self.addr, &[Register::StatusReg.addr()], &mut status)?;
        // ZYXDA bit 3 indicates new X, Y, Z data available
        Ok((status[0] & 0x08) != 0)
    }

    /// Release underlying I2C peripheral.
    pub fn release(self) -> I2C {
        self.i2c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accel_g_magnitude() {
        let accel = AccelG { x: 3.0, y: 4.0, z: 0.0 };
        assert_eq!(accel.magnitude(), 5.0);
        assert_eq!(accel.to_array(), [3.0, 4.0, 0.0]);
    }

    #[test]
    fn test_full_scale_factors() {
        assert_eq!(FullScale::G2.scale_g(), 0.0039);
        assert_eq!(FullScale::G4.scale_g(), 0.0078);
        assert_eq!(FullScale::G8.scale_g(), 0.0156);
        assert_eq!(FullScale::G16.scale_g(), 0.0312);
    }
}
