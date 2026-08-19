pub mod emit_rust;

pub use emit_rust::RustCodeGenerator;

#[cfg(test)]
mod tests {
    use super::*;
    use embedded_nn_compiler::builder::ModelBuilder;
    use embedded_nn_compiler::ir::*;

    #[test]
    fn test_rust_code_generation() {
        let mut builder = ModelBuilder::new("TinyGestureClassifier");
        let in_id = builder.add_input("imu_accel", TensorShape::new_1d(6), DataType::Int8);
        let fc1_id = builder.add_dense_layer(
            "fc1",
            in_id,
            12,
            vec![1; 72],
            None,
            Some(vec![0; 12]),
            ActivationType::Relu,
        );
        let sm_id = builder.add_softmax("softmax", fc1_id);
        builder.mark_output(sm_id);

        let graph = builder.build();
        let codegen = RustCodeGenerator::new("TinyGestureClassifier");
        let generated_code = codegen.generate(&graph);

        assert!(generated_code.contains("pub struct TinyGestureClassifier;"));
        assert!(generated_code.contains("pub const ARENA_SIZE_BYTES: usize ="));
        assert!(generated_code.contains("pub fn predict"));
    }
}
