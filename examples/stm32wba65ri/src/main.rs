#![no_main]
#![no_std]

use cortex_m::{asm, peripheral::DWT};
use cortex_m_rt::entry;
use defmt_rtt as _;
use panic_probe as _;

mod model {
    use embedded_nn_macros::embedded_nn_model;

    #[embedded_nn_model("../../crates/embedded-nn-tflite/fixtures/constructed/sine_fc_int8.tflite")]
    pub struct SineFc;
}

#[entry]
fn main() -> ! {
    let _peripherals = embassy_stm32::init(Default::default());
    let mut core = cortex_m::Peripherals::take().unwrap();
    core.DCB.enable_trace();
    core.DWT.enable_cycle_counter();

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
