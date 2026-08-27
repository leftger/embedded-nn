#![no_main]
#![no_std]

use cortex_m_rt::entry;
use cortex_m_semihosting::{
    debug::{self, EXIT_FAILURE, EXIT_SUCCESS},
    hprintln,
};
use panic_semihosting as _;

mod model {
    use embedded_nn_macros::embedded_nn_model;

    #[embedded_nn_model("../../crates/embedded-nn-tflite/fixtures/constructed/sine_fc_int8.tflite")]
    pub struct SineFc;
}

#[entry]
fn main() -> ! {
    let input = [64i8; model::SineFc::INPUT_DIM];
    let mut arena = [0u8; model::SineFc::ARENA_SIZE];

    let _ = hprintln!(
        "embedded-nn QEMU analysis: model=sine_fc_int8 arena={} weights={} input_shape={:?}",
        model::SineFc::ARENA_SIZE,
        model::SineFc::FLASH_WEIGHTS,
        model::SineFc::INPUT_SHAPE
    );

    let status = match model::SineFc::predict(&input, &mut arena) {
        Ok(output) if output == input => {
            let _ = hprintln!("embedded-nn QEMU inference passed: {:?}", output);
            EXIT_SUCCESS
        }
        Ok(output) => {
            let _ = hprintln!(
                "embedded-nn QEMU inference mismatch: got {:?}, expected {:?}",
                output,
                input
            );
            EXIT_FAILURE
        }
        Err(error) => {
            let _ = hprintln!("embedded-nn QEMU inference failed: {}", error);
            EXIT_FAILURE
        }
    };

    debug::exit(status);
    loop {
        cortex_m::asm::nop();
    }
}
