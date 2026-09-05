#![no_std]
#![no_main]

use embassy_executor::Spawner;
use embassy_rp::gpio::{Level, Output};
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::channel::Channel;
use embassy_time::{Duration, Ticker, Timer};
use panic_halt as _;

use embedded_nn::activations::relu_s8;
use embedded_nn::fully_connected::fully_connected_s8;
use embedded_nn::types::{Activation, Dims, FcParams, PerTensorQuantParams};

const INPUT_DIM: usize = 16;
const HIDDEN_DIM: usize = 8;
const OUTPUT_DIM: usize = 2; // Class 0: Idle / Normal, Class 1: Anomaly / Motion

// Flash-resident quantized weights (ROM)
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

// Static async message channels between tasks
static SENSOR_CHANNEL: Channel<CriticalSectionRawMutex, [i8; INPUT_DIM], 4> = Channel::new();
static INFERENCE_CHANNEL: Channel<CriticalSectionRawMutex, usize, 4> = Channel::new();

/// Async sensor sampler task: simulates periodic 50 Hz acquisition from an accelerometer / vibration sensor.
#[embassy_executor::task]
async fn sensor_sampler() {
    let mut ticker = Ticker::every(Duration::from_hz(50));
    let mut step: i8 = 0;

    loop {
        let mut sample = [0i8; INPUT_DIM];
        for (i, val) in sample.iter_mut().enumerate() {
            *val = ((step as i32 * (i as i32 + 1)) % 120) as i8 - 60;
        }

        // Send sensor frame non-blockingly to inference worker
        SENSOR_CHANNEL.send(sample).await;
        step = step.wrapping_add(1);
        ticker.next().await;
    }
}

/// Async inference worker task: executes zero-allocation quantized neural network inference on incoming batches.
#[embassy_executor::task]
async fn inference_worker() {
    // Static SRAM scratch arena - zero dynamic allocations
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

    loop {
        let input_tensor = SENSOR_CHANNEL.receive().await;

        // Layer 1: Quantized Fully Connected (16 -> 8) + ReLU
        let _ = fully_connected_s8(
            &fc_params,
            &quant_params,
            &in_dims,
            &input_tensor,
            &filter1_dims,
            &LAYER1_WEIGHTS,
            Some(&LAYER1_BIAS),
            &hidden_dims,
            &mut arena_hidden,
        );
        relu_s8(&mut arena_hidden);

        // Layer 2: Quantized Fully Connected (8 -> 2)
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

        // Classify argmax class
        let predicted_class = if arena_output[1] > arena_output[0] { 1 } else { 0 };
        INFERENCE_CHANNEL.send(predicted_class).await;
    }
}

/// Async actuator / UI task: receives predicted classification events and drives hardware status LED.
#[embassy_executor::task]
async fn actuator_task(mut led: Output<'static>) {
    loop {
        let class = INFERENCE_CHANNEL.receive().await;
        if class == 1 {
            led.set_high();
        } else {
            led.set_low();
        }
        // Small yield to allow other tasks to process
        Timer::after(Duration::from_millis(10)).await;
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());

    // Initialize onboard LED (GPIO 25 on Raspberry Pi Pico)
    let led = Output::new(p.PIN_25, Level::Low);

    // Spawn async tasks on the Embassy cooperative executor
    spawner.spawn(sensor_sampler()).unwrap();
    spawner.spawn(inference_worker()).unwrap();
    spawner.spawn(actuator_task(led)).unwrap();
}
