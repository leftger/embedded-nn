#![no_main]
#![no_std]

use cortex_m::{asm, peripheral::DWT};
use cortex_m_rt::entry;
use defmt_rtt as _;
use panic_probe as _;

mod gesture;
mod model;
mod on_device_dsp;

#[entry]
fn main() -> ! {
    let _peripherals = embassy_stm32::init(Default::default());
    let mut core = cortex_m::Peripherals::take().unwrap();
    core.DCB.enable_trace();
    core.DWT.enable_cycle_counter();

    let mut dsp_features = [0i8; 16];
    let n = on_device_dsp::first_frame_s8(&[0.25; 256], &mut dsp_features);
    defmt::info!("dsp first frame ({} bins): {:?}", n, dsp_features);

    let mut gesture_arena = [0u8; gesture::GestureMlp::ARENA_SIZE];
    let gesture_started = DWT::cycle_count();
    let gesture_out = gesture::GestureMlp::predict(&dsp_features, &mut gesture_arena);
    let gesture_cycles = DWT::cycle_count().wrapping_sub(gesture_started);
    match gesture_out {
        Ok(output) => {
            let class = if output[0] >= output[1] { 0u8 } else { 1u8 };
            defmt::info!(
                "gesture class {} in {} cycles, arena {} B, logits {:?}",
                class,
                gesture_cycles,
                gesture::GestureMlp::ARENA_SIZE,
                output
            );
        }
        Err(error) => defmt::error!("gesture predict failed: {}", error),
    }

    let input = [64i8; model::SineFc::INPUT_DIM];
    let mut arena = [0u8; model::SineFc::ARENA_SIZE];
    let started = DWT::cycle_count();
    let result = model::SineFc::predict(&input, &mut arena);
    let cycles = DWT::cycle_count().wrapping_sub(started);

    match result {
        Ok(output) if output == input => {
            defmt::info!("predict passed in {} CPU cycles: {:?}", cycles, output);
        }
        Ok(output) => {
            defmt::error!(
                "predict mismatch in {} CPU cycles: got {:?}, expected {:?}",
                cycles,
                output,
                input
            );
        }
        Err(error) => {
            defmt::error!("predict failed in {} CPU cycles: {}", cycles, error);
        }
    }

    loop {
        asm::wfi();
    }
}
