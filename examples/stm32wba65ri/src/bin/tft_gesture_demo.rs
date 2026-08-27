//! STM32WBA65RI Free-Running Neural Network Gesture Recognition with TFT Display.
//!
//! Complete Edge AI pipeline:
//! 1. FIFO chunked decompression: 4-frame slices (400 Hz -> 100 Hz) preserve high-frequency oscillations (SHAKE).
//! 2. Enabled Cortex-M33 DCB TRCENA for accurate DWT cycle counter latency.
//! 3. Real-time RTT / defmt telemetry reporting live physical sensor values and model predictions.
//! 4. 3-Axis (X, Y, Z) dynamic auto-scaling oscilloscope with zero clipping.
//! 5. embedded-nn DSP: Hann windowing + Mel filterbank extraction (112 features).
//! 6. ActiveModel (GestureNeuralNet): Conv1D + Dense Head on Cortex-M33.
//! 7. ILI9341 320x240 RGB565 TFT Display on SPI2 (PB10 SCK, PC3 MOSI, PB9 CS, PB11 DC, PA10 RST).

#![no_std]
#![no_main]

#[path = "../model.rs"]
mod model;

use core::cell::{RefCell, UnsafeCell};
use core::fmt::Write;

use defmt_rtt as _;
use panic_probe as _;

use embassy_executor::Spawner;
use embassy_stm32::bind_interrupts;
use embassy_stm32::dma;
use embassy_stm32::gpio::{Level, Output, Speed};
use embassy_stm32::i2c::{self, Config as I2cConfig, I2c};
use embassy_stm32::peripherals;
use embassy_stm32::rcc::*;
use embassy_stm32::spi::{Config as SpiConfig, Spi};
use embassy_stm32::time::Hertz;
use embassy_stm32::Config;
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::blocking_mutex::Mutex;
use embassy_time::{Duration, Ticker, Timer};
use embedded_hal_bus::spi::ExclusiveDevice;

use embedded_graphics::mono_font::{ascii::FONT_6X10, ascii::FONT_7X13_BOLD, ascii::FONT_9X18_BOLD, MonoTextStyle};
use embedded_graphics::pixelcolor::Rgb565;
use embedded_graphics::prelude::*;
use embedded_graphics::primitives::Rectangle;
use embedded_graphics::text::{Baseline, Text};

use mipidsi::interface::SpiInterface;
use mipidsi::models::ILI9341Rgb565;
use mipidsi::options::{Orientation, Rotation};
use mipidsi::Builder;

use lis2de12::{
    FIFO_CAPACITY, FifoConfig, FifoFrame, FifoMode, Lis2de12Async, Lis2de12Config, Odr, SlaveAddr,
};
use libm::sqrtf;

use embedded_nn::feature_dsp::{extract_mel_sequence, quantize_mel_s8, FeatureDspConfig, WindowKind};
use model::ActiveModel;

bind_interrupts!(struct Irqs {
    I2C1_EV => i2c::EventInterruptHandler<peripherals::I2C1>;
    I2C1_ER => i2c::ErrorInterruptHandler<peripherals::I2C1>;
    GPDMA1_CHANNEL0 => dma::InterruptHandler<peripherals::GPDMA1_CH0>;
    GPDMA1_CHANNEL1 => dma::InterruptHandler<peripherals::GPDMA1_CH1>;
});

const VIEW_WIDTH: usize = 320;
const VIEW_HEIGHT: usize = 240;
const VIEW_PIXELS: usize = VIEW_WIDTH * VIEW_HEIGHT;

struct FrameBuffer([Rgb565; VIEW_PIXELS]);
struct SafeFrameBuf(UnsafeCell<FrameBuffer>);
unsafe impl Sync for SafeFrameBuf {}
static RAW_FRAMEBUF: SafeFrameBuf =
    SafeFrameBuf(UnsafeCell::new(FrameBuffer([Rgb565::BLACK; VIEW_PIXELS])));

struct MicroDelay;
impl embedded_hal::delay::DelayNs for MicroDelay {
    #[inline(always)]
    fn delay_ns(&mut self, ns: u32) {
        let cycles = (ns as u64 * 100) / 1000;
        cortex_m::asm::delay(cycles as u32);
    }
}

struct TxBuf([u8; 32768]);
struct SafeTxBuf(UnsafeCell<TxBuf>);
unsafe impl Sync for SafeTxBuf {}
static RAW_TX_BUF: SafeTxBuf = SafeTxBuf(UnsafeCell::new(TxBuf([0u8; 32768])));

const WHO_AM_I_REG: u8 = 0x0F;
const WHO_AM_I_VAL: u8 = 0x33;
const LIS2DE12_ADDR_DEFAULT: u8 = 0x18;
const LIS2DE12_ADDR_ALT: u8 = 0x19;
const G_PER_LSB: f32 = 0.0156;

const CAPTURE_SAMPLES: usize = 256;
const NUM_CLASSES: usize = 4;
const CLASS_NAMES: [&str; NUM_CLASSES] = ["IDLE", "WAVE_LEFT", "WAVE_RIGHT", "SHAKE"];

struct SensorRing {
    samples_x: [f32; CAPTURE_SAMPLES],
    samples_y: [f32; CAPTURE_SAMPLES],
    samples_z: [f32; CAPTURE_SAMPLES],
    samples_mag: [f32; CAPTURE_SAMPLES],
    pos: usize,
    last_xyz: [f32; 3],
    online: bool,
}

static SENSOR_STATE: Mutex<CriticalSectionRawMutex, RefCell<SensorRing>> =
    Mutex::new(RefCell::new(SensorRing {
        samples_x: [0.0; CAPTURE_SAMPLES],
        samples_y: [0.0; CAPTURE_SAMPLES],
        samples_z: [1.0; CAPTURE_SAMPLES],
        samples_mag: [1.0; CAPTURE_SAMPLES],
        pos: 0,
        last_xyz: [0.0, 0.0, 1.0],
        online: false,
    }));

struct StringBuf<const N: usize> {
    buf: [u8; N],
    len: usize,
}

impl<const N: usize> StringBuf<N> {
    fn new() -> Self {
        Self { buf: [0; N], len: 0 }
    }
    fn as_str(&self) -> &str {
        core::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
    }
}

impl<const N: usize> Write for StringBuf<N> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let bytes = s.as_bytes();
        let remaining = N - self.len;
        let to_copy = bytes.len().min(remaining);
        self.buf[self.len..self.len + to_copy].copy_from_slice(&bytes[..to_copy]);
        self.len += to_copy;
        Ok(())
    }
}

/// Matches hil_agent.rs & wba65_imu_dataset_1.jsonl oversampling calculation
fn average_fifo_frames(frames: &[FifoFrame]) -> [f32; 3] {
    let mut sums = [0i32; 3];
    for frame in frames {
        sums[0] += i32::from(frame[1] as i8);
        sums[1] += i32::from(frame[3] as i8);
        sums[2] += i32::from(frame[5] as i8);
    }
    let scale = G_PER_LSB / frames.len().max(1) as f32;
    [
        sums[0] as f32 * scale,
        sums[1] as f32 * scale,
        sums[2] as f32 * scale,
    ]
}

type AccelDevice = Lis2de12Async<lis2de12::DeviceInterfaceAsync<I2c<'static, embassy_stm32::mode::Async, embassy_stm32::i2c::Master>>>;

#[embassy_executor::task]
async fn sensor_worker_task(mut accel: AccelDevice) {
    let mut ticker = Ticker::every(Duration::from_millis(10)); // Exactly 100 Hz
    let mut fifo_frames = [[0u8; 6]; FIFO_CAPACITY as usize];
    let mut last_xyz = [0.0f32, 0.0f32, 1.0f32];
    let mut sample_count = 0u64;

    SENSOR_STATE.lock(|cell| {
        cell.borrow_mut().online = true;
    });

    defmt::info!("sensor_worker_task: 100 Hz chunked FIFO worker online");

    loop {
        ticker.next().await;
        sample_count += 1;

        let mut read_count = 0usize;
        if let Ok(count) = accel.read_fifo_frames(&mut fifo_frames).await {
            read_count = count;
            if count > 0 {
                // Chunk into 4-frame slices (4 frames @ 400Hz = 10ms @ 100Hz)
                // This ensures all shake oscillation cycles are preserved even when multiple frames buffered
                for chunk in fifo_frames[..count].chunks(4) {
                    let xyz = average_fifo_frames(chunk);
                    let mag = sqrtf(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]);
                    last_xyz = xyz;

                    SENSOR_STATE.lock(|cell| {
                        let mut state = cell.borrow_mut();
                        let p = state.pos;
                        state.samples_x[p] = xyz[0];
                        state.samples_y[p] = xyz[1];
                        state.samples_z[p] = xyz[2];
                        state.samples_mag[p] = mag;
                        state.pos = (p + 1) % CAPTURE_SAMPLES;
                        state.last_xyz = xyz;
                        state.online = true;
                    });
                }
            }
        }

        // Periodic telemetry log every 100 samples (1 Hz)
        if sample_count % 100 == 0 {
            let mag = sqrtf(last_xyz[0] * last_xyz[0] + last_xyz[1] * last_xyz[1] + last_xyz[2] * last_xyz[2]);
            defmt::info!(
                "[Sensor 100Hz] X={=f32}g Y={=f32}g Z={=f32}g |M|={=f32}g (FIFO frames={})",
                last_xyz[0],
                last_xyz[1],
                last_xyz[2],
                mag,
                read_count
            );
        }
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    // 1. Clock Configuration: 96 MHz SYSCLK matching hil_agent
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

    // Enable DWT Cycle Counter with TRCENA enabled for microsecond benchmarks
    let mut cp = cortex_m::Peripherals::take().unwrap();
    cp.DCB.enable_trace();
    cp.DWT.enable_cycle_counter();

    defmt::info!("===========================================================");
    defmt::info!("⚡ embedded-nn Edge Neural Network Live Gesture + TFT Demo");
    defmt::info!("96MHz Cortex-M33 + LIS2DE12 (LR1110) + ILI9341 Display");
    defmt::info!("===========================================================");

    // 2. Initialize I2C1 for LIS2DE12 on LR1110 Header (PB1 SDA, PB2 SCL)
    defmt::info!("Initializing I2C1 (PB1 SDA / PB2 SCL) at 400 kHz with pullups...");
    let mut i2c_cfg = I2cConfig::default();
    i2c_cfg.frequency = Hertz(400_000);
    i2c_cfg.sda_pullup = true;
    i2c_cfg.scl_pullup = true;
    let mut i2c = I2c::new(
        p.I2C1,
        p.PB2,
        p.PB1,
        p.GPDMA1_CH0,
        p.GPDMA1_CH1,
        Irqs,
        i2c_cfg,
    );

    let sensor_config = Lis2de12Config {
        odr: Odr::FourHundredHz,
        fifo: FifoConfig::enabled(FifoMode::Stream),
        ..Default::default()
    };

    // Robust hardware detection loop (guarantees sensor is online before starting)
    defmt::info!("Probing LIS2DE12 WHO_AM_I register (0x0F == 0x33)...");
    let detected_addr = loop {
        let mut who = [0u8; 1];
        if i2c.write_read(LIS2DE12_ADDR_DEFAULT, &[WHO_AM_I_REG], &mut who).await.is_ok() && who[0] == WHO_AM_I_VAL {
            defmt::info!("✓ LIS2DE12 detected at address 0x18! (WHO_AM_I={=u8:#x})", who[0]);
            break SlaveAddr::Default;
        } else if i2c.write_read(LIS2DE12_ADDR_ALT, &[WHO_AM_I_REG], &mut who).await.is_ok() && who[0] == WHO_AM_I_VAL {
            defmt::info!("✓ LIS2DE12 detected at address 0x19! (WHO_AM_I={=u8:#x})", who[0]);
            break SlaveAddr::Alternative;
        }

        defmt::warn!("Waiting for LIS2DE12 on I2C1 (PB1/PB2)... response byte={=u8:#x}", who[0]);
        Timer::after(Duration::from_millis(50)).await;
    };

    let mut accel = Lis2de12Async::new_i2c_with_config(i2c, detected_addr, sensor_config).await.expect("LIS2DE12 init");
    accel.reset_fifo().await.expect("LIS2DE12 reset FIFO");
    defmt::info!("✓ LIS2DE12 initialized with 400 Hz stream FIFO (exact hil_agent match)!");

    // Spawn 100 Hz Background Sensor Sampling Worker
    spawner.spawn(sensor_worker_task(accel).expect("spawn sensor_worker_task"));

    // 3. Initialize Display SPI2 & ILI9341
    defmt::info!("Initializing ILI9341 320x240 display on SPI2 (PB10 SCK, PC3 MOSI, PB9 CS)...");
    let mut display_spi_config = SpiConfig::default();
    display_spi_config.frequency = Hertz(25_000_000);
    let spi = Spi::new_blocking_txonly(p.SPI2, p.PB10, p.PC3, display_spi_config);
    let cs_display = Output::new(p.PB9, Level::High, Speed::VeryHigh);
    let dc = Output::new(p.PB11, Level::Low, Speed::VeryHigh);
    let rst = Output::new(p.PA10, Level::High, Speed::VeryHigh);

    let tx_buf = unsafe { &mut (*RAW_TX_BUF.0.get()).0 };
    let spi_device = ExclusiveDevice::new(spi, cs_display, MicroDelay).unwrap();
    let di = SpiInterface::new(spi_device, dc, tx_buf.as_mut_slice());

    let mut display = Builder::new(ILI9341Rgb565, di)
        .reset_pin(rst)
        .color_order(mipidsi::options::ColorOrder::Bgr)
        .orientation(Orientation::new().rotate(Rotation::Deg90).flip_horizontal())
        .init(&mut embassy_time::Delay)
        .unwrap();

    Timer::after(Duration::from_millis(30)).await;
    defmt::info!("✓ ILI9341 display initialized!");

    // 4. DSP Configuration (Matches Studio DSP Contract)
    let dsp_config = FeatureDspConfig {
        window_size: 64,
        window_kind: WindowKind::Hann,
        num_mel_bins: 16,
        high_pass_cutoff_hz: 10.0,
        sample_rate_hz: 100.0,
        frame_hop_size: 32,
        capture_samples: CAPTURE_SAMPLES,
        input_scale: 1.0 / 127.0,
    };

    let mut ordered_x = [0.0f32; CAPTURE_SAMPLES];
    let mut ordered_y = [0.0f32; CAPTURE_SAMPLES];
    let mut ordered_z = [0.0f32; CAPTURE_SAMPLES];
    let mut ordered_mag = [1.0f32; CAPTURE_SAMPLES];

    let mut mel_features = [0.0f32; ActiveModel::INPUT_DIM];
    let mut input_i8 = [0i8; ActiveModel::INPUT_DIM];
    let mut arena = [0u8; ActiveModel::ARENA_SIZE];

    let mut last_prediction_idx = 0usize;
    let mut last_confidence = 0u32;
    let mut last_probabilities = [0u32; NUM_CLASSES];
    let mut last_infer_cycles = 0u32;
    let mut last_xyz = [0.0f32, 0.0f32, 1.0f32];
    let mut is_online = false;
    let mut infer_count = 0u64;

    let framebuf = unsafe { &mut (*RAW_FRAMEBUF.0.get()).0 };

    defmt::info!("Entering main inference & GUI rendering loop (~15 Hz)...");

    loop {
        // Run display and inference at smooth ~15 Hz
        Timer::after(Duration::from_millis(65)).await;
        infer_count += 1;

        // Atomically copy current snapshot from 100 Hz background sensor worker
        SENSOR_STATE.lock(|cell| {
            let state = cell.borrow();
            let p = state.pos;
            for i in 0..CAPTURE_SAMPLES {
                let idx = (p + i) % CAPTURE_SAMPLES;
                ordered_x[i] = state.samples_x[idx];
                ordered_y[i] = state.samples_y[idx];
                ordered_z[i] = state.samples_z[idx];
                ordered_mag[i] = state.samples_mag[idx];
            }
            last_xyz = state.last_xyz;
            is_online = state.online;
        });

        // 1. Extract DSP Features on Magnitude Waveform
        if extract_mel_sequence(&dsp_config, &ordered_mag, &mut mel_features).is_ok() {
            quantize_mel_s8(&mel_features, dsp_config.input_scale, &mut input_i8);

            // 2. Run Active Neural Network Model on Cortex-M33
            let start_cyc = cortex_m::peripheral::DWT::cycle_count();
            if let Ok(logits) = ActiveModel::predict(&input_i8, &mut arena) {
                let end_cyc = cortex_m::peripheral::DWT::cycle_count();
                last_infer_cycles = end_cyc.wrapping_sub(start_cyc);

                let mut max_score = i8::MIN;
                let mut best_class = 0;
                for (i, &l) in logits.iter().enumerate().take(NUM_CLASSES) {
                    if l > max_score {
                        max_score = l;
                        best_class = i;
                    }
                }
                last_prediction_idx = best_class;

                // True softmax probability distribution over 8-bit logits
                let mut exp_sum = 0.0f32;
                let mut exps = [0.0f32; NUM_CLASSES];
                for (i, &l) in logits.iter().enumerate().take(NUM_CLASSES) {
                    let diff = (l as f32 - max_score as f32) * 0.25f32;
                    let e = libm::expf(diff);
                    exps[i] = e;
                    exp_sum += e;
                }
                if exp_sum > 0.0 {
                    for (i, &e) in exps.iter().enumerate() {
                        last_probabilities[i] = ((e / exp_sum) * 100.0) as u32;
                    }
                }
                last_confidence = last_probabilities[best_class];

                // Periodic inference log
                if infer_count % 15 == 0 {
                    let lat_us = (last_infer_cycles as u64 * 10) / 960;
                    defmt::info!(
                        "🎯 [Inference] Class: '{}' ({}%) | Logits: {:?} | Latency: {}.{:02}ms ({} cyc)",
                        CLASS_NAMES[last_prediction_idx],
                        last_confidence,
                        logits,
                        lat_us / 1000,
                        (lat_us % 1000) / 10,
                        last_infer_cycles
                    );
                }
            }
        }

        // 3. Render Dashboard to Framebuffer with Auto-Scaling 3-Axis Oscilloscope
        render_gui(
            framebuf,
            &ordered_x,
            &ordered_y,
            &ordered_z,
            &last_xyz,
            last_prediction_idx,
            last_confidence,
            &last_probabilities,
            last_infer_cycles,
            is_online,
        );

        // 4. Blit to ILI9341 Display via SPI
        let area = Rectangle::new(Point::new(0, 0), Size::new(VIEW_WIDTH as u32, VIEW_HEIGHT as u32));
        let _ = display.fill_contiguous(&area, framebuf.iter().copied());
    }
}

/// Renders the TinyML gesture dashboard to the offscreen RGB565 framebuffer with zero clipping.
fn render_gui(
    buf: &mut [Rgb565; VIEW_PIXELS],
    wave_x: &[f32; CAPTURE_SAMPLES],
    wave_y: &[f32; CAPTURE_SAMPLES],
    wave_z: &[f32; CAPTURE_SAMPLES],
    last_xyz: &[f32; 3],
    pred_idx: usize,
    confidence: u32,
    probabilities: &[u32; NUM_CLASSES],
    cycles: u32,
    imu_ok: bool,
) {
    // Clear background
    buf.fill(Rgb565::new(2, 4, 8)); // Dark Navy

    let text_white = MonoTextStyle::new(&FONT_6X10, Rgb565::WHITE);
    let text_cyan = MonoTextStyle::new(&FONT_6X10, Rgb565::CYAN);
    let text_yellow = MonoTextStyle::new(&FONT_6X10, Rgb565::YELLOW);
    let text_bold_title = MonoTextStyle::new(&FONT_9X18_BOLD, Rgb565::WHITE);
    let text_header = MonoTextStyle::new(&FONT_7X13_BOLD, Rgb565::CYAN);

    // --- Header Section (y: 0..24) ---
    fill_rect(buf, 0, 0, 320, 24, Rgb565::new(3, 10, 22));
    draw_line_h(buf, 0, 320, 24, Rgb565::new(0, 22, 31));

    let _ = Text::with_baseline("⚡ embedded-nn Live Gesture AI", Point::new(8, 5), text_header, Baseline::Top).draw(&mut FbTarget(buf));

    let mut status_str: StringBuf<40> = StringBuf::new();
    let _ = write!(status_str, "{}", if imu_ok { "LIS2DE12: 100Hz" } else { "SENSOR OFFLINE" });
    let _ = Text::with_baseline(status_str.as_str(), Point::new(215, 7), text_cyan, Baseline::Top).draw(&mut FbTarget(buf));

    // --- Section 1: Non-Clipping Dynamic Auto-Scaling 3-Axis Scope (y: 28..106, height: 78) ---
    fill_rect(buf, 10, 28, 300, 78, Rgb565::new(1, 2, 4));
    draw_rect_border(buf, 10, 28, 300, 78, Rgb565::new(8, 16, 24));

    // Header Legend with live values: X (Red), Y (Green), Z (Cyan)
    let text_red = MonoTextStyle::new(&FONT_6X10, Rgb565::new(31, 10, 10));
    let text_green = MonoTextStyle::new(&FONT_6X10, Rgb565::new(0, 62, 16));
    let text_blue = MonoTextStyle::new(&FONT_6X10, Rgb565::new(8, 28, 31));

    let mut sx: StringBuf<16> = StringBuf::new();
    let _ = write!(sx, "X:{:.2}g", last_xyz[0]);
    let _ = Text::with_baseline(sx.as_str(), Point::new(14, 31), text_red, Baseline::Top).draw(&mut FbTarget(buf));

    let mut sy: StringBuf<16> = StringBuf::new();
    let _ = write!(sy, "Y:{:.2}g", last_xyz[1]);
    let _ = Text::with_baseline(sy.as_str(), Point::new(80, 31), text_green, Baseline::Top).draw(&mut FbTarget(buf));

    let mut sz: StringBuf<16> = StringBuf::new();
    let _ = write!(sz, "Z:{:.2}g", last_xyz[2]);
    let _ = Text::with_baseline(sz.as_str(), Point::new(146, 31), text_blue, Baseline::Top).draw(&mut FbTarget(buf));

    let mag = sqrtf(last_xyz[0]*last_xyz[0] + last_xyz[1]*last_xyz[1] + last_xyz[2]*last_xyz[2]);
    let mut sm: StringBuf<16> = StringBuf::new();
    let _ = write!(sm, "|M|:{:.2}g", mag);
    let _ = Text::with_baseline(sm.as_str(), Point::new(212, 31), text_yellow, Baseline::Top).draw(&mut FbTarget(buf));

    // Calculate maximum absolute amplitude across all 3 axes in the 256-sample window
    let mut peak_val = 1.3f32;
    for i in 0..CAPTURE_SAMPLES {
        let ax = wave_x[i].abs();
        let ay = wave_y[i].abs();
        let az = wave_z[i].abs();
        if ax > peak_val { peak_val = ax; }
        if ay > peak_val { peak_val = ay; }
        if az > peak_val { peak_val = az; }
    }
    // Dynamic peak with comfortable headroom so peaks NEVER touch edges
    let dyn_limit = (peak_val * 1.15f32).max(1.4f32);

    let y_center = 74i32;
    let y_half_span = 28.0f32; // Plotting area from y=46 to y=102
    let scale = y_half_span / dyn_limit;

    let y_min = 45usize;
    let y_max = 103usize;
    let wave_x_offset = 12usize;

    // Draw reference grids
    draw_line_h(buf, 12, 308, y_center as usize, Rgb565::new(6, 12, 18)); // 0.0g Center
    if dyn_limit >= 1.0f32 {
        let y_plus_1g = (y_center - (1.0f32 * scale) as i32).clamp(y_min as i32, y_max as i32) as usize;
        let y_minus_1g = (y_center + (1.0f32 * scale) as i32).clamp(y_min as i32, y_max as i32) as usize;
        draw_line_h(buf, 12, 308, y_plus_1g, Rgb565::new(3, 8, 14)); // +1.0g reference
        draw_line_h(buf, 12, 308, y_minus_1g, Rgb565::new(3, 8, 14)); // -1.0g reference
    }

    // Display current dynamic range in top-right corner of scope
    let mut s_scale: StringBuf<16> = StringBuf::new();
    let _ = write!(s_scale, "+-{:.1}g", dyn_limit);
    let _ = Text::with_baseline(s_scale.as_str(), Point::new(270, 31), text_cyan, Baseline::Top).draw(&mut FbTarget(buf));

    // Plot all 3 axes
    for i in 0..295usize {
        let s1 = (i * CAPTURE_SAMPLES) / 296;
        let s2 = ((i + 1) * CAPTURE_SAMPLES) / 296;
        let x1 = wave_x_offset + i;
        let x2 = wave_x_offset + i + 1;

        // X Axis (Red)
        let yx1 = (y_center - (wave_x[s1] * scale) as i32).clamp(y_min as i32, y_max as i32) as usize;
        let yx2 = (y_center - (wave_x[s2] * scale) as i32).clamp(y_min as i32, y_max as i32) as usize;
        draw_line_segment(buf, x1, yx1, x2, yx2, Rgb565::new(31, 10, 10));

        // Y Axis (Green)
        let yy1 = (y_center - (wave_y[s1] * scale) as i32).clamp(y_min as i32, y_max as i32) as usize;
        let yy2 = (y_center - (wave_y[s2] * scale) as i32).clamp(y_min as i32, y_max as i32) as usize;
        draw_line_segment(buf, x1, yy1, x2, yy2, Rgb565::new(0, 62, 16));

        // Z Axis (Cyan)
        let yz1 = (y_center - (wave_z[s1] * scale) as i32).clamp(y_min as i32, y_max as i32) as usize;
        let yz2 = (y_center - (wave_z[s2] * scale) as i32).clamp(y_min as i32, y_max as i32) as usize;
        draw_line_segment(buf, x1, yz1, x2, yz2, Rgb565::new(8, 28, 31));
    }

    // --- Section 2: Active Gesture Prediction Banner (y: 110..152, height: 42) ---
    let (banner_bg, banner_border) = match pred_idx {
        0 => (Rgb565::new(4, 10, 16), Rgb565::new(8, 24, 31)),    // IDLE: Slate
        1 => (Rgb565::new(0, 14, 28), Rgb565::new(0, 31, 31)),    // WAVE LEFT: Blue
        2 => (Rgb565::new(20, 0, 24), Rgb565::new(31, 10, 31)),   // WAVE RIGHT: Purple
        _ => (Rgb565::new(28, 6, 0), Rgb565::new(31, 20, 0)),     // SHAKE: Orange/Red
    };

    fill_rect(buf, 10, 110, 300, 42, banner_bg);
    draw_rect_border(buf, 10, 110, 300, 42, banner_border);

    let mut pred_text: StringBuf<48> = StringBuf::new();
    let _ = write!(pred_text, "▶ {}", CLASS_NAMES[pred_idx]);
    let _ = Text::with_baseline(pred_text.as_str(), Point::new(18, 115), text_bold_title, Baseline::Top).draw(&mut FbTarget(buf));

    let mut conf_text: StringBuf<32> = StringBuf::new();
    let _ = write!(conf_text, "Conf: {}%", confidence);
    let _ = Text::with_baseline(conf_text.as_str(), Point::new(215, 117), text_yellow, Baseline::Top).draw(&mut FbTarget(buf));

    let mut lat_text: StringBuf<48> = StringBuf::new();
    let lat_us = (cycles as u64 * 10) / 960; // @ 96 MHz: 1 cycle = ~10.4 ns
    let _ = write!(lat_text, "Latency: {}.{:02}ms ({} cyc)", lat_us / 1000, (lat_us % 1000) / 10, cycles);
    let _ = Text::with_baseline(lat_text.as_str(), Point::new(18, 136), text_cyan, Baseline::Top).draw(&mut FbTarget(buf));

    // --- Section 3: Class Probability Progress Bars (y: 156..236, height: 80) ---
    fill_rect(buf, 10, 156, 300, 80, Rgb565::new(2, 4, 10));
    draw_rect_border(buf, 10, 156, 300, 80, Rgb565::new(6, 14, 22));

    for i in 0..NUM_CLASSES {
        let row_y = 160 + i * 19;
        let is_winner = i == pred_idx;

        let label_style = if is_winner { text_yellow } else { text_white };
        let mut class_lbl: StringBuf<16> = StringBuf::new();
        let _ = write!(class_lbl, "{:<10}", CLASS_NAMES[i]);
        let _ = Text::with_baseline(class_lbl.as_str(), Point::new(16, row_y as i32 + 2), label_style, Baseline::Top).draw(&mut FbTarget(buf));

        // Draw Bar Container (x: 100..260, width: 160)
        let bar_x = 100usize;
        let bar_w = 160usize;
        let bar_h = 10usize;
        fill_rect(buf, bar_x, row_y + 2, bar_w, bar_h, Rgb565::new(4, 6, 10));

        let prob = probabilities[i].min(100);
        let fill_w = (bar_w * prob as usize) / 100;
        let bar_color = if is_winner {
            Rgb565::new(0, 60, 28) // Bright Emerald Green
        } else {
            Rgb565::new(0, 20, 28) // Dim Cyan
        };
        if fill_w > 0 {
            fill_rect(buf, bar_x, row_y + 2, fill_w, bar_h, bar_color);
        }
        draw_rect_border(buf, bar_x, row_y + 2, bar_w, bar_h, Rgb565::new(10, 20, 28));

        // Draw % Text
        let mut pct_str: StringBuf<12> = StringBuf::new();
        let _ = write!(pct_str, "{:>3}%", prob);
        let _ = Text::with_baseline(pct_str.as_str(), Point::new(268, row_y as i32 + 2), label_style, Baseline::Top).draw(&mut FbTarget(buf));
    }
}

// ----------------------------------------------------------------------------
// Low-Level Direct Framebuffer Drawing Primitives (Zero Heap Allocation)
// ----------------------------------------------------------------------------

#[inline(always)]
fn fill_rect(buf: &mut [Rgb565; VIEW_PIXELS], x: usize, y: usize, w: usize, h: usize, color: Rgb565) {
    let x_end = (x + w).min(VIEW_WIDTH);
    let y_end = (y + h).min(VIEW_HEIGHT);
    for row in y..y_end {
        let offset = row * VIEW_WIDTH;
        for col in x..x_end {
            buf[offset + col] = color;
        }
    }
}

#[inline(always)]
fn draw_rect_border(buf: &mut [Rgb565; VIEW_PIXELS], x: usize, y: usize, w: usize, h: usize, color: Rgb565) {
    draw_line_h(buf, x, x + w, y, color);
    draw_line_h(buf, x, x + w, y + h.saturating_sub(1), color);
    draw_line_v(buf, x, y, y + h, color);
    draw_line_v(buf, x + w.saturating_sub(1), y, y + h, color);
}

#[inline(always)]
fn draw_line_h(buf: &mut [Rgb565; VIEW_PIXELS], x1: usize, x2: usize, y: usize, color: Rgb565) {
    if y >= VIEW_HEIGHT { return; }
    let start = x1.min(VIEW_WIDTH);
    let end = x2.min(VIEW_WIDTH);
    let offset = y * VIEW_WIDTH;
    for x in start..end {
        buf[offset + x] = color;
    }
}

#[inline(always)]
fn draw_line_v(buf: &mut [Rgb565; VIEW_PIXELS], x: usize, y1: usize, y2: usize, color: Rgb565) {
    if x >= VIEW_WIDTH { return; }
    let start = y1.min(VIEW_HEIGHT);
    let end = y2.min(VIEW_HEIGHT);
    for y in start..end {
        buf[y * VIEW_WIDTH + x] = color;
    }
}

#[inline(always)]
fn draw_line_segment(buf: &mut [Rgb565; VIEW_PIXELS], mut x0: usize, mut y0: usize, x1: usize, y1: usize, color: Rgb565) {
    let dx = (x1 as i32 - x0 as i32).abs();
    let sx = if x0 < x1 { 1i32 } else { -1i32 };
    let dy = -(y1 as i32 - y0 as i32).abs();
    let sy = if y0 < y1 { 1i32 } else { -1i32 };
    let mut err = dx + dy;

    loop {
        if x0 < VIEW_WIDTH && y0 < VIEW_HEIGHT {
            buf[y0 * VIEW_WIDTH + x0] = color;
        }
        if x0 == x1 && y0 == y1 { break; }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 = (x0 as i32 + sx) as usize;
        }
        if e2 <= dx {
            err += dx;
            y0 = (y0 as i32 + sy) as usize;
        }
    }
}

// ----------------------------------------------------------------------------
// embedded-graphics DrawTarget Wrapper for Direct Framebuffer Rendering
// ----------------------------------------------------------------------------

struct FbTarget<'a>(&'a mut [Rgb565; VIEW_PIXELS]);

impl<'a> Dimensions for FbTarget<'a> {
    fn bounding_box(&self) -> Rectangle {
        Rectangle::new(Point::new(0, 0), Size::new(VIEW_WIDTH as u32, VIEW_HEIGHT as u32))
    }
}

impl<'a> DrawTarget for FbTarget<'a> {
    type Color = Rgb565;
    type Error = core::convert::Infallible;

    fn draw_iter<I>(&mut self, pixels: I) -> Result<(), Self::Error>
    where
        I: IntoIterator<Item = Pixel<Self::Color>>,
    {
        for Pixel(coord, color) in pixels.into_iter() {
            if coord.x >= 0 && coord.x < VIEW_WIDTH as i32 && coord.y >= 0 && coord.y < VIEW_HEIGHT as i32 {
                self.0[coord.y as usize * VIEW_WIDTH + coord.x as usize] = color;
            }
        }
        Ok(())
    }
}
