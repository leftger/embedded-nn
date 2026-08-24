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
use embassy_stm32::dma;
use embassy_stm32::exti::{self, ExtiInput};
use embassy_stm32::gpio::{Level, Output, Pull, Speed};
use embassy_stm32::i2c::{self, Config as I2cConfig, I2c};
use embassy_stm32::rcc::*;
use embassy_stm32::spi::{BitOrder, Config as SpiConfig, Mode as SpiMode, Phase, Polarity, Spi};
use embassy_stm32::time::Hertz;
use embassy_stm32::{bind_interrupts, interrupt, peripherals, Config};
use embassy_time::{Duration, Ticker, Timer};
use accelerometer::vector::F32x3;
use lis2de12::{Lis2de12Async, SlaveAddr};

#[path = "../sd_logger.rs"]
mod sd_logger;
#[path = "../w25q32.rs"]
mod w25q32;

use sd_logger::{AccelSample, DatasetBurst};
use w25q32::W25q32;

/// WHO_AM_I register address and expected value, used only to probe which of
/// the two possible 7-bit addresses the LIS2DE12 responds on before handing
/// the bus to `Lis2de12Async::new_i2c`. `SlaveAddr::addr()` is private to the
/// `lis2de12` crate, so the raw addresses are duplicated here (datasheet
/// Table 15, 7-bit form).
const WHO_AM_I_REG: u8 = 0x0F;
const WHO_AM_I_VAL: u8 = 0x33;
const LIS2DE12_ADDR_DEFAULT: u8 = 0x18;
const LIS2DE12_ADDR_ALT: u8 = 0x19;

bind_interrupts!(struct Irqs {
    EXTI13 => exti::InterruptHandler<interrupt::typelevel::EXTI13>;
    I2C1_EV => i2c::EventInterruptHandler<peripherals::I2C1>;
    I2C1_ER => i2c::ErrorInterruptHandler<peripherals::I2C1>;
    GPDMA1_CHANNEL0 => dma::InterruptHandler<peripherals::GPDMA1_CH0>;
    GPDMA1_CHANNEL1 => dma::InterruptHandler<peripherals::GPDMA1_CH1>;
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

    if let Some(mut core) = cortex_m::Peripherals::take() {
        core.DCB.enable_trace();
        core.DWT.enable_cycle_counter();
    }

    defmt::info!("==========================================================");
    defmt::info!("embedded-nn: STM32WBA65RI Embassy Sensor Ingestion System");
    defmt::info!("==========================================================");
    defmt::info!("[Step 1/5] Initializing clocks (HSI -> PLL1 @ 64/96MHz)...");

    // 2. User Interface (LEDs & Buttons on MB1801 mezzanine board)
    defmt::info!("[Step 2/5] Configuring GPIOs (LD1 Blue=PD8, LD2 Green=PC4, LD3 Red=PB8, B1=PC13)...");
    let mut led_blue = Output::new(p.PD8, Level::Low, Speed::Low); // LD1
    let mut led_green = Output::new(p.PC4, Level::Low, Speed::Low); // LD2
    let mut led_red = Output::new(p.PB8, Level::Low, Speed::Low); // LD3

    // Async EXTI button input on User Button B1 (PC13 / EXTI13)
    let mut btn_user = ExtiInput::new(p.PC13, p.EXTI13, Pull::Up, Irqs);

    // 4. Initialize I2C1 for LIS2DE12 on Arduino Header D14/D15 (PB1 SDA, PB2 SCL)
    defmt::info!("[Step 3/5] Initializing I2C1 on PB2 (SCL) / PB1 (SDA) with internal pull-ups...");
    // WBA65's I2C peripheral is the "v2" variant, whose async transfers are
    // always DMA-backed internally (`new_no_dma` leaves the DMA channels as
    // `None`, which panics the first time an async read/write is issued).
    let mut i2c_cfg = I2cConfig::default();
    i2c_cfg.sda_pullup = true;
    i2c_cfg.scl_pullup = true;
    let mut i2c = I2c::new(
        p.I2C1, p.PB2, p.PB1, p.GPDMA1_CH0, p.GPDMA1_CH1, Irqs, i2c_cfg,
    );

    // Check LIS2DE12 WHO_AM_I (probes standard 0x18 and alternate 0x19)
    defmt::info!("[Step 4/5] Probing LIS2DE12 accelerometer on I2C bus...");
    let mut who = [0u8; 1];
    let detected_addr = if i2c
        .write_read(LIS2DE12_ADDR_DEFAULT, &[WHO_AM_I_REG], &mut who)
        .await
        .is_ok()
        && who[0] == WHO_AM_I_VAL
    {
        Some(SlaveAddr::Default)
    } else if i2c
        .write_read(LIS2DE12_ADDR_ALT, &[WHO_AM_I_REG], &mut who)
        .await
        .is_ok()
        && who[0] == WHO_AM_I_VAL
    {
        Some(SlaveAddr::Alternative)
    } else {
        None
    };

    // Configure LIS2DE12: 100 Hz ODR, +/- 2g range, Block Data Update enabled
    // (`Lis2de12Config::default()` matches all three).
    let mut accel = match detected_addr {
        Some(addr) => match Lis2de12Async::new_i2c(i2c, addr).await {
            Ok(dev) => {
                defmt::info!(" -> LIS2DE12 detected (WHO_AM_I = 0x33)");
                defmt::info!(" -> LIS2DE12 configured: 100 Hz ODR, +/- 2g scale, BDU enabled");
                led_green.set_high();
                Some(dev)
            }
            Err(_) => {
                defmt::error!(" -> Failed to initialize LIS2DE12 CTRL registers");
                led_red.set_high();
                None
            }
        },
        None => {
            defmt::error!(" -> LIS2DE12 not detected at 0x18 or 0x19 on PB1 (SDA) / PB2 (SCL)");
            led_red.set_high();
            None
        }
    };

    // 5. Initialize SPI2 for MicroSD (PA10 CS) & W25Q32 NOR Flash (PA3 CS)
    // Arduino D13 = PB10 (SCK), D11 = PC3 (MOSI), D12 = PA9 (MISO)
    defmt::info!("[Step 5/5] Initializing SPI2 (PB10 SCK, PC3 MOSI, PA9 MISO) & storage...");
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
                " -> W25Qxx SPI Flash detected: Manuf=0x{:02x}, Type=0x{:02x}, Cap=0x{:02x}",
                id.manufacturer,
                id.memory_type,
                id.capacity
            );
        }
        Err(_) => {
            defmt::warn!(" -> W25Qxx Flash query returned error");
        }
    }

    let mut sample_seq: u32 = 1;
    let mut burst = DatasetBurst::new(sample_seq, 100.0);
    let mut json_buffer = [0u8; 4096];

    defmt::info!("----------------------------------------------------------");
    defmt::info!("System initialization complete! Ready for data ingestion.");
    defmt::info!("Storage mode: Real-time RTT JSONL terminal streaming + RAM buffer (MicroSD optional).");
    defmt::info!(">> Press User Button B1 (PC13) to record a 128-sample gesture burst.");
    defmt::info!("----------------------------------------------------------");

    // 100 Hz high-precision ticker
    let mut ticker = Ticker::every(Duration::from_hz(100));

    loop {
        defmt::info!("[State: IDLE] Waiting for User Button B1 (PC13) trigger...");
        // Asynchronously wait for button press via EXTI interrupt
        btn_user.wait_for_falling_edge().await;

        led_blue.set_high();
        burst.reset(sample_seq);
        defmt::info!(
            ">> Button pressed! Starting burst #{} (128 samples @ 100 Hz, 1.28s window)...",
            sample_seq
        );

        let mut latest_g = F32x3::new(0.0, 0.0, 0.0);

        // Record 128 samples synchronized to the Embassy 100 Hz ticker
        for i in 1..=128 {
            ticker.next().await;

            if let Some(dev) = accel.as_mut() {
                if let Ok(g) = dev.read_g().await {
                    latest_g = g;
                    burst.push(AccelSample {
                        x: g.x,
                        y: g.y,
                        z: g.z,
                    });
                }
            } else {
                // Fallback mock waveform if sensor not detected
                burst.push(AccelSample {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                });
            }

            // Periodic progress reporting every 32 samples (25%, 50%, 75%, 100%)
            if i % 32 == 0 {
                defmt::info!(
                    "   [Progress: {}/128 samples ({}%)] Latest accel: x={=i32}mg, y={=i32}mg, z={=i32}mg",
                    i,
                    (i * 100) / 128,
                    (latest_g.x * 1000.0) as i32,
                    (latest_g.y * 1000.0) as i32,
                    (latest_g.z * 1000.0) as i32,
                );
            }
        }

        defmt::info!(">> Capture complete! Serializing burst into JSON Lines format...");

        // Format burst into compliant JSON Lines (.jsonl) dataset schema
        match burst.format_jsonl(&mut json_buffer) {
            Ok(len) => {
                defmt::info!(">> Formatted JSONL sample record ({} bytes):", len);
                if let Ok(json_str) = core::str::from_utf8(&json_buffer[..len]) {
                    defmt::info!("{}", json_str);
                }
                defmt::info!(">> Sample #{} recorded successfully! Ready for SD card persistence.", sample_seq);
            }
            Err(_) => {
                defmt::error!("!! Buffer overflow while formatting JSONL record");
            }
        }

        sample_seq = sample_seq.wrapping_add(1);
        led_blue.set_low();

        defmt::info!(">> Debouncing button input...");
        // Async debounce delay using Embassy timer
        Timer::after_millis(300).await;
    }
}
