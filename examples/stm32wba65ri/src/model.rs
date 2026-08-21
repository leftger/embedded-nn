use embedded_nn_macros::embedded_nn_model;

#[embedded_nn_model("../../crates/embedded-nn-tflite/fixtures/constructed/sine_fc_int8.tflite")]
pub struct SineFc;
