use embedded_nn_aton::compiler::AtonCompiler;
use embedded_nn_compiler::ir::*;

#[test]
fn test_compile_mlp_to_aton_blob() {
    let mut graph = ModelGraph::new("TestMlp");

    // Input: [1, 16]
    graph.tensors.push(TensorDesc {
        id: 0,
        name: "input".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_1d(16),
        quant: QuantParams::default(),
    });

    // Hidden: [1, 16]
    graph.tensors.push(TensorDesc {
        id: 1,
        name: "hidden".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_1d(16),
        quant: QuantParams::default(),
    });

    // Output: [1, 4]
    graph.tensors.push(TensorDesc {
        id: 2,
        name: "output".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_1d(4),
        quant: QuantParams::default(),
    });

    graph.inputs.push(0);
    graph.outputs.push(2);

    // Layer 0: FC 16->16
    graph.layers.push(LayerNode {
        id: 0,
        name: "fc0".into(),
        inputs: vec![0],
        outputs: vec![1],
        op: OpPayload::FullyConnected {
            weights: vec![1i8; 256],
            packed_s4: None,
            bias: Some(vec![0i32; 16]),
            filter_offset: 0,
            activation: ActivationType::Relu,
            per_channel_quant: None,
        },
    });

    // Layer 1: FC 16->4
    graph.layers.push(LayerNode {
        id: 1,
        name: "fc1".into(),
        inputs: vec![1],
        outputs: vec![2],
        op: OpPayload::FullyConnected {
            weights: vec![1i8; 64],
            packed_s4: None,
            bias: Some(vec![0i32; 4]),
            filter_offset: 0,
            activation: ActivationType::None,
            per_channel_quant: None,
        },
    });

    let compiler = AtonCompiler::new();
    let compiled = compiler.compile(&graph).expect("compilation failed");

    assert_eq!(compiled.name, "TestMlp");
    assert_eq!(compiled.input_size_bytes, 16);
    assert_eq!(compiled.output_size_bytes, 4);

    let u64_blob = compiled.build_blob_u64().expect("build blob failed");
    assert!(!u64_blob.is_empty());

    let c_header = compiled.emit_c_header().expect("emit C failed");
    assert!(c_header.contains("const uint64_t _ec_blob_TestMlp_0["));

    let rust_code = compiled.emit_rust_code().expect("emit Rust failed");
    assert!(rust_code.contains("pub static EC_CONTAINER_TESTMLP: [u64;"));
}
