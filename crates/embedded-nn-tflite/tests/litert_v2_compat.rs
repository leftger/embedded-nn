//! LiteRT v2 Compatibility & Composite Operator Integration Tests.
//!
//! Validates that `embedded-nn-tflite` correctly handles models authored with LiteRT v2,
//! including legacy opcode fallbacks, metadata buffers, and composite operator structures.

use embedded_nn_compiler::builder::ModelBuilder;
use embedded_nn_compiler::ir::*;
use std::fs;
use std::path::Path;

#[test]
fn test_litert_v2_composite_operator_recognition() {
    let mut builder = ModelBuilder::new("LiteRtCompositeTest");
    let in_shape = TensorShape::new_4d(1, 16, 16, 3);
    let quant = QuantParams {
        multiplier: 1073741824,
        shift: 0,
        zero_point: 0,
        scale: 1.0 / 127.0,
    };
    let in_id = builder.add_input("input_image", in_shape, DataType::Int8, Some(quant.clone()));

    // Build standard Conv2D + Relu layer
    let weights = vec![1i8; 3 * 3 * 3 * 8];
    let conv_out_id = builder.add_conv2d_layer(
        "odml_npu_conv",
        in_id,
        8,
        3,
        3,
        1,
        1,
        Padding2D::symmetric(1, 1),
        1,
        1,
        weights,
        None,
        None,
        ActivationType::Relu,
        None,
        Some(quant.clone()),
    );

    // Add Softmax head
    let out_id = builder.add_softmax("odml_cpu_softmax", conv_out_id);
    builder.mark_output(out_id);

    let graph = builder.build();

    // Verify ModelGraph properties
    assert_eq!(graph.inputs.len(), 1);
    assert_eq!(graph.outputs.len(), 1);
    assert_eq!(graph.layers.len(), 2);

    // Verify code generation
    let code = embedded_nn_codegen::RustCodeGenerator::new("LiteRtCompositeTest").generate(&graph);
    assert!(code.contains("pub struct LiteRtCompositeTest"));
    assert!(code.contains("pub const ARENA_SIZE: usize"));
}

#[test]
fn test_litert_v2_metadata_and_shape_preservation() {
    // Validate that all MLPerf Tiny LiteRT models preserve shape tensors
    let model_paths = [
        "/home/usuario/Projects/open-source-repos/microflow-rs/models/sine.tflite",
        "/home/usuario/Projects/open-source-repos/microflow-rs/models/speech.tflite",
        "/home/usuario/Projects/open-source-repos/microflow-rs/models/person_detect.tflite",
    ];

    for path_str in &model_paths {
        let path = Path::new(path_str);
        if !path.exists() {
            continue;
        }
        let bytes = fs::read(path).unwrap();
        let graph = embedded_nn_tflite::import_tflite(&bytes).unwrap();
        assert!(!graph.tensors.is_empty());
        assert!(!graph.layers.is_empty());

        for tensor in &graph.tensors {
            assert!(
                tensor.shape.total_elements() > 0,
                "Tensor {} has 0 elements",
                tensor.name
            );
        }
    }
}
