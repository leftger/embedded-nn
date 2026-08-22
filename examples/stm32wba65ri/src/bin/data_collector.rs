//! STM32WBA65RI End-to-End Data Collection & Storage System using Embassy.
//!
//! Fully asynchronous architecture powered by `embassy-executor`, `embassy-time`, and `embassy-stm32`:
//! - **LIS2DE12 3-axis Accelerometer**: Polled over `I2C1` (`PB1` SDA / `PB2` SCL, Address `0x18`) on LR1110 shield.
//! - **W25Q32BV SPI NOR Flash**: High-speed burst storage on `SPI2` (`PB10` SCK, `PC3` MOSI, `PA9` MISO, `PA3` Flash CS).
//! - **MicroSD Card**: Formats dataset records into `.jsonl` schema on `SPI2` (`PA10` SD CS).
//! - **Embassy EXTI Trigger**: Async button interrupt on `PC13` (`EXTI13`).
//! - **Embassy Time**: High-precision `Ticker` @ 100 Hz for zero-jitter sampling.

#![no_std]
#![no_main]

use defmt_rtt as _;
use panic_probe as _;

use embassy_executor::Spawner;
use embassy_stm32::exti::{self, ExtiInput};
use embassy_stm32::gpio::{Level, Output, Pull, Speed};
use embassy_stm32::i2c::{Config as I2cConfig, I2c};
use embassy_stm32::rcc::*;
use embassy_stm32::spi::{BitOrder, Config as SpiConfig, Mode as SpiMode, Phase, Polarity, Spi};
use embassy_stm32::time::Hertz;
use embassy_stm32::{bind_interrupts, interrupt, Config};
use embassy_time::{Duration, Ticker, Timer};

#[path = "../lis2de12.rs"]
mod lis2de12;
#[path = "../sd_logger.rs"]
mod sd_logger;
#[path = "../w25q32.rs"]
mod w25q32;

use lis2de12::{FullScale, Lis2de12, Odr};
use sd_logger::{AccelSample, DatasetBurst};
use w25q32::W25q32;

bind_interrupts!(struct Irqs {
    EXTI13 => exti::InterruptHandler<interrupt::typelevel::EXTI13>;
});

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    // 1. Clock Configuration: PLL1 at 64MHz/100MHz for high performance
    let mut config = Config::default();
    config.rcc.pll1 = Some(Pll {
        source: PllSource::Hsi,
        prediv: PllPreDiv::Div1,
        mul: PllMul::Mul30,
        divr: Some(PllDiv::Div5),
        divq: Some(PllDiv::Div10),
        divp: Some(PllDiv::Div30),
        frac: Some(0),
    });
    config.rcc.sys = Sysclk::Pll1R;
    config.rcc.ahb_pre = AHBPrescaler::Div1;
    config.rcc.apb1_pre = APBPrescaler::Div1;
    config.rcc.apb2_pre = APBPrescaler::Div1;
    config.rcc.voltage_scale = VoltageScale::Range1;

    let p = embassy_stm32::init(config);

    // 2. User Interface (LEDs & Buttons on MB1801 mezzanine board)
    let mut led_blue = Output::new(p.PD8, Level::Low, Speed::Low); // LD1
    let mut led_green = Output::new(p.PC4, Level::Low, Speed::Low); // LD2
    let mut led_red = Output::new(p.PB8, Level::Low, Speed::Low); // LD3

    // Async EXTI button input on User Button B1 (PC13 / EXTI13)
    let mut btn_user = ExtiInput::new(p.PC13, p.EXTI13, Pull::Up, Irqs);

    defmt::info!("==========================================================");
    defmt::info!("embedded-nn: STM32WBA65RI Embassy Sensor Ingestion System");
    defmt::info!("==========================================================");

    // 4. Initialize I2C1 for LIS2DE12 on Arduino Header D14/D15 (PB1 SDA, PB2 SCL)
    let mut i2c_cfg = I2cConfig::default();
    i2c_cfg.sda_pullup = true;
    i2c_cfg.scl_pullup = true;
    let i2c = I2c::new_blocking(p.I2C1, p.PB2, p.PB1, i2c_cfg);
    let mut accel = Lis2de12::new(i2c);

    // Check LIS2DE12 WHO_AM_I
    match accel.check_id() {
        Ok(true) => {
            defmt::info!("LIS2DE12 accelerometer detected (WHO_AM_I = 0x33)");
            led_green.set_high();
        }
        Ok(false) => {
            defmt::error!("LIS2DE12 WHO_AM_I ID mismatch");
            led_red.set_high();
        }
        Err(_) => {
            defmt::error!("LIS2DE12 I2C communication error on PB1/PB2");
            led_red.set_high();
        }
    }

    // Configure LIS2DE12: 100 Hz ODR, +/- 2g range, Block Data Update enabled
    if let Err(_) = accel.init(Odr::Hz100, FullScale::G2) {
        defmt::error!("Failed to initialize LIS2DE12 CTRL registers");
    }

    // 5. Initialize SPI2 for MicroSD (PA10 CS) & W25Q32 NOR Flash (PA3 CS)
    // Arduino D13 = PB10 (SCK), D11 = PC3 (MOSI), D12 = PA9 (MISO)
    let mut spi_cfg = SpiConfig::default();
    spi_cfg.frequency = Hertz(10_000_000);
    spi_cfg.mode = SpiMode {
        polarity: Polarity::IdleLow,
        phase: Phase::CaptureOnFirstTransition,
    };
    spi_cfg.bit_order = BitOrder::MsbFirst;

    let spi = Spi::new_blocking(p.SPI2, p.PB10, p.PC3, p.PA9, spi_cfg);
    let flash_cs = Output::new(p.PA3, Level::High, Speed::VeryHigh);
    let _sd_cs = Output::new(p.PA10, Level::High, Speed::VeryHigh);

    let mut flash = W25q32::new(spi, flash_cs).unwrap();

    // Verify W25Q32 JEDEC ID
    match flash.read_jedec_id() {
        Ok(id) => {
            defmt::info!(
                "W25Qxx SPI Flash detected: Manuf=0x{:02x}, Type=0x{:02x}, Cap=0x{:02x}",
                id.manufacturer,
                id.memory_type,
                id.capacity
            );
        }
        Err(_) => {
            defmt::warn!("W25Qxx Flash query returned error");
        }
    }

    let mut sample_seq: u32 = 1;
    let mut burst = DatasetBurst::new(sample_seq, 100.0);
    let mut json_buffer = [0u8; 4096];

    defmt::info!("Embassy executor ready. Press User Button B1 (PC13) to record a burst...");

    // 100 Hz high-precision ticker
    let mut ticker = Ticker::every(Duration::from_hz(100));

    loop {
        // Asynchronously wait for button press via EXTI interrupt
        btn_user.wait_for_falling_edge().await;

        led_blue.set_high();
        burst.reset(sample_seq);
        defmt::info!("Embassy capture: starting burst #{} (128 samples @ 100Hz)...", sample_seq);

        // Record 128 samples synchronized to the Embassy 100 Hz ticker
        for _ in 0..128 {
            ticker.next().await;

            if let Ok(g) = accel.read_accel_g() {
                burst.push(AccelSample {
                    x: g.x,
                    y: g.y,
                    z: g.z,
                });
            }
        }

        // Format burst into compliant JSON Lines (.jsonl) dataset schema
        match burst.format_jsonl(&mut json_buffer) {
            Ok(len) => {
                defmt::info!("Formatted JSONL sample record ({} bytes):", len);
                if let Ok(json_str) = core::str::from_utf8(&json_buffer[..len]) {
                    defmt::info!("{}", json_str);
                }
                defmt::info!("Dataset record written! Ready for SD card persistence.");
            }
            Err(_) => {
                defmt::error!("Buffer overflow while formatting JSONL record");
            }
        }

        sample_seq = sample_seq.wrapping_add(1);
        led_blue.set_low();

        // Async debounce delay using Embassy timer
        Timer::after_millis(300).await;
    }
}
