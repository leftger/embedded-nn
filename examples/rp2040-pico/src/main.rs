#![no_std]
#![no_main]

use cortex_m_rt::entry;
use embedded_hal::digital::v2::OutputPin;
use embedded_nn::activations::relu_s8;
use embedded_nn::fully_connected::fully_connected_s8;
use embedded_nn::types::{Activation, Dims, FcParams, PerTensorQuantParams};
use panic_halt as _;
use rp2040_hal as hal;
use rp2040_hal::pac;

/// Boot stage 2 loader for RP2040 (W25Q080 / standard external SPI Flash).
#[link_section = ".boot2"]
#[used]
pub static BOOT2: [u8; 256] = rp2040_boot2::BOOT_LOADER_GENERIC_03H;

const INPUT_DIM: usize = 16;
const HIDDEN_DIM: usize = 8;
const OUTPUT_DIM: usize = 2; // Class 0: Idle, Class 1: Motion Triggered

// Quantized network weights stored directly in Flash memory (ROM)
static LAYER1_WEIGHTS: [i8; INPUT_DIM * HIDDEN_DIM] = [
    3, -2, 5, 1, -4, 2, -1, 3,
    -2, 4, -3, 2, 5, -1, 4, -2,
    1, -3, 4, -5, 2, 3, -1, 4,
    -4, 2, 1, 3, -5, 4, -2, 1,
    2, -1, 3, -4, 5, 2, -3, 1,
    -3, 5, -2, 1, 4, -5, 2, 3,
    4, -2, 1, 5, -3, 2, 4, -1,
    -1, 3, -5, 2, 4, -1, 3, 5,
    2, -4, 3, 1, -5, 4, -2, 1,
    -3, 2, 4, -5, 1, 3, 5, -2,
    4, -1, 3, 2, -5, 4, 1, -3,
    1, 5, -2, 4, 3, -1, 5, 2,
    -2, 4, 1, -5, 3, 2, -4, 1,
    3, 1, -4, 2, 5, -3, 1, 4,
    5, -3, 2, 4, -1, 5, 3, -2,
    -1, 4, -2, 3, 5, -1, 4, 2,
];
static LAYER1_BIAS: [i32; HIDDEN_DIM] = [0, 5, -2, 4, 1, -3, 6, 2];

static LAYER2_WEIGHTS: [i8; HIDDEN_DIM * OUTPUT_DIM] = [
    -5, 6,
    4, -3,
    -2, 5,
    6, -4,
    3, -2,
    -4, 5,
    1, -3,
    5, -6,
];
static LAYER2_BIAS: [i32; OUTPUT_DIM] = [10, -10];

#[entry]
fn main() -> ! {
    // Take ownership of RP2040 peripherals
    let mut pac = pac::Peripherals::take().unwrap();
    let core = pac::CorePeripherals::take().unwrap();

    let mut watchdog = hal::Watchdog::new(pac.WATCHDOG);
    let clocks = hal::clocks::init_clocks_and_plls(
        12_000_000, // 12 MHz external crystal on Raspberry Pi Pico
        pac.XOSC,
        pac.CLOCKS,
        pac.PLL_SYS,
        pac.PLL_USB,
        &mut pac.RESETS,
        &mut watchdog,
    )
    .ok()
    .unwrap();

    let mut delay = cortex_m::delay::Delay::new(core.SYST, clocks.system_clock.freq().to_Hz());

    let sio = hal::Sio::new(pac.SIO);
    let pins = hal::gpio::Pins::new(
        pac.IO_BANK0,
        pac.PADS_BANK0,
        sio.gpio_bank0,
        &mut pac.RESETS,
    );

    // On-board LED on Raspberry Pi Pico (GPIO 25)
    let mut led = pins.gpio25.into_push_pull_output();

    // Statically scheduled activation arena in SRAM (Total SRAM: 26 bytes, zero heap allocations)
    let mut arena_hidden = [0i8; HIDDEN_DIM];
    let mut arena_output = [0i8; OUTPUT_DIM];

    let fc_params = FcParams {
        input_offset: 0,
        filter_offset: 0,
        output_offset: 0,
        activation: Activation::int8_unconstrained(),
    };
    let quant_params = PerTensorQuantParams {
        multiplier: 1073741824,
        shift: 0,
    };

    let in_dims = Dims::new(1, 1, 1, INPUT_DIM as i32);
    let filter1_dims = Dims::new(INPUT_DIM as i32, 1, 1, HIDDEN_DIM as i32);
    let hidden_dims = Dims::new(1, 1, 1, HIDDEN_DIM as i32);
    let filter2_dims = Dims::new(HIDDEN_DIM as i32, 1, 1, OUTPUT_DIM as i32);
    let out_dims = Dims::new(1, 1, 1, OUTPUT_DIM as i32);

    let mut step: i8 = 0;

    loop {
        // Synthesize simulated 16-channel sensor vector
        let mut input_sensor = [0i8; INPUT_DIM];
        for (i, val) in input_sensor.iter_mut().enumerate() {
            *val = ((step as i32 * (i as i32 + 1)) % 120) as i8 - 60;
        }

        // Layer 1: Fully Connected (16 -> 8)
        let _ = fully_connected_s8(
            &fc_params,
            &quant_params,
            &in_dims,
            &input_sensor,
            &filter1_dims,
            &LAYER1_WEIGHTS,
            Some(&LAYER1_BIAS),
            &hidden_dims,
            &mut arena_hidden,
        );
        relu_s8(&mut arena_hidden);

        // Layer 2: Fully Connected (8 -> 2)
        let _ = fully_connected_s8(
            &fc_params,
            &quant_params,
            &hidden_dims,
            &arena_hidden,
            &filter2_dims,
            &LAYER2_WEIGHTS,
            Some(&LAYER2_BIAS),
            &out_dims,
            &mut arena_output,
        );

        // Decision logic: if Class 1 (Motion) > Class 0 (Idle), turn on LED
        if arena_output[1] > arena_output[0] {
            let _ = led.set_high();
        } else {
            let _ = led.set_low();
        }

        step = step.wrapping_add(1);
        delay.delay_ms(100);
    }
}
