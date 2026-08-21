//! Regenerates deterministic, schema-built TFLite fixtures used by end-to-end tests.
//!
//! These models are constructed directly with the generated TFLite FlatBuffer bindings. They
//! are valid `.tflite` files, but they were not emitted by TensorFlow and their expected vectors
//! are derived independently from the deliberately simple integer arithmetic.

use embedded_nn_tflite::constructed_fixtures::{
    build_add_transpose_model, build_conv_pool_reshape_fc_softmax_model, build_sine_fc_model,
    build_uint8_fc_model,
};
use std::fs;
use std::path::PathBuf;

fn main() {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/constructed");
    fs::create_dir_all(&fixtures).expect("create fixture directory");

    for (name, bytes) in [
        ("sine_fc_int8.tflite", build_sine_fc_model()),
        (
            "tinyconv_int8.tflite",
            build_conv_pool_reshape_fc_softmax_model(),
        ),
        ("uint8_fc.tflite", build_uint8_fc_model()),
        ("add_transpose_int8.tflite", build_add_transpose_model()),
    ] {
        fs::write(fixtures.join(name), bytes).expect("write fixture");
    }
}
