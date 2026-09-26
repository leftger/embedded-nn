use embedded_nn_aton::compiler::{AtonCompiler, EpochKind};
use embedded_nn_compiler::ir::*;

#[test]
fn test_compile_conv2d_fused_relu() {
    let mut graph = ModelGraph::new("Conv2DNet");

    // Input: [1, 28, 28, 1]
    graph.tensors.push(TensorDesc {
        id: 0,
        name: "input".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_4d(1, 28, 28, 1),
        quant: QuantParams::default(),
    });

    // Output: [1, 26, 26, 8]
    graph.tensors.push(TensorDesc {
        id: 1,
        name: "conv_out".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_4d(1, 26, 26, 8),
        quant: QuantParams::default(),
    });

    graph.inputs.push(0);
    graph.outputs.push(1);

    // 3x3 Conv2D with 8 filters
    graph.layers.push(LayerNode {
        id: 0,
        name: "conv0".into(),
        inputs: vec![0],
        outputs: vec![1],
        op: OpPayload::Conv2D {
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 1,
            stride_w: 1,
            padding: Padding2D::default(),
            dilation_h: 1,
            dilation_w: 1,
            weights: vec![1i8; 3 * 3 * 1 * 8],
            packed_s4: None,
            bias: Some(vec![0i32; 8]),
            activation: ActivationType::Relu,
            per_channel_quant: None,
        },
    });

    let compiler = AtonCompiler::new();
    let compiled = compiler.compile(&graph).expect("compilation failed");

    assert_eq!(compiled.name, "Conv2DNet");
    assert_eq!(compiled.input_size_bytes, 28 * 28 * 1);
    assert_eq!(compiled.output_size_bytes, 26 * 26 * 8);
    assert_eq!(compiled.epochs.len(), 1);
    assert!(matches!(compiled.epochs[0], EpochKind::Hardware { .. }));

    // Static weights: 72 bytes weights + 32 bytes bias = 104 bytes
    assert_eq!(compiled.static_weights.len(), 72 + 32);

    let blob = compiled.build_blob_u64().expect("build blob");
    assert!(!blob.is_empty());
}

#[test]
fn test_compile_conv2d_fused_relu6() {
    let mut graph = ModelGraph::new("ConvRelu6Net");

    graph.tensors.push(TensorDesc {
        id: 0,
        name: "input".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_4d(1, 16, 16, 4),
        quant: QuantParams::default(),
    });

    graph.tensors.push(TensorDesc {
        id: 1,
        name: "output".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_4d(1, 16, 16, 8),
        quant: QuantParams::default(),
    });

    graph.inputs.push(0);
    graph.outputs.push(1);

    graph.layers.push(LayerNode {
        id: 0,
        name: "conv_relu6".into(),
        inputs: vec![0],
        outputs: vec![1],
        op: OpPayload::Conv2D {
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            padding: Padding2D::default(),
            dilation_h: 1,
            dilation_w: 1,
            weights: vec![2i8; 4 * 8],
            packed_s4: None,
            bias: Some(vec![1i32; 8]),
            activation: ActivationType::Relu6,
            per_channel_quant: None,
        },
    });

    let compiler = AtonCompiler::new();
    let compiled = compiler.compile(&graph).expect("compilation failed");

    assert_eq!(compiled.input_size_bytes, 16 * 16 * 4);
    assert_eq!(compiled.output_size_bytes, 16 * 16 * 8);

    let blob = compiled.build_blob_u64().expect("build blob");
    assert!(!blob.is_empty());
}

#[test]
fn test_compile_depthwise_conv2d() {
    let mut graph = ModelGraph::new("DwConvNet");

    graph.tensors.push(TensorDesc {
        id: 0,
        name: "input".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_4d(1, 32, 32, 16),
        quant: QuantParams::default(),
    });

    graph.tensors.push(TensorDesc {
        id: 1,
        name: "output".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_4d(1, 16, 16, 16),
        quant: QuantParams::default(),
    });

    graph.inputs.push(0);
    graph.outputs.push(1);

    graph.layers.push(LayerNode {
        id: 0,
        name: "dwconv0".into(),
        inputs: vec![0],
        outputs: vec![1],
        op: OpPayload::DepthwiseConv2D {
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 2,
            stride_w: 2,
            padding: Padding2D::symmetric(1, 1),
            ch_mult: 1,
            weights: vec![1i8; 3 * 3 * 16],
            bias: Some(vec![0i32; 16]),
            activation: ActivationType::Relu,
            per_channel_quant: None,
        },
    });

    let compiler = AtonCompiler::new();
    let compiled = compiler.compile(&graph).expect("compilation failed");

    assert_eq!(compiled.input_size_bytes, 32 * 32 * 16);
    assert_eq!(compiled.output_size_bytes, 16 * 16 * 16);
    assert_eq!(compiled.static_weights.len(), 3 * 3 * 16 + 16 * 4);

    let blob = compiled.build_blob_u64().expect("build blob");
    assert!(!blob.is_empty());
}

#[test]
fn test_compile_max_pool2d() {
    let mut graph = ModelGraph::new("MaxPoolNet");

    graph.tensors.push(TensorDesc {
        id: 0,
        name: "input".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_4d(1, 24, 24, 32),
        quant: QuantParams::default(),
    });

    graph.tensors.push(TensorDesc {
        id: 1,
        name: "output".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_4d(1, 12, 12, 32),
        quant: QuantParams::default(),
    });

    graph.inputs.push(0);
    graph.outputs.push(1);

    graph.layers.push(LayerNode {
        id: 0,
        name: "pool0".into(),
        inputs: vec![0],
        outputs: vec![1],
        op: OpPayload::MaxPool2D {
            pool_h: 2,
            pool_w: 2,
            stride_h: 2,
            stride_w: 2,
            padding: Padding2D::default(),
        },
    });

    let compiler = AtonCompiler::new();
    let compiled = compiler.compile(&graph).expect("compilation failed");

    assert_eq!(compiled.input_size_bytes, 24 * 24 * 32);
    assert_eq!(compiled.output_size_bytes, 12 * 12 * 32);

    let blob = compiled.build_blob_u64().expect("build blob");
    assert!(!blob.is_empty());
}

#[test]
fn test_compile_avg_pool2d() {
    let mut graph = ModelGraph::new("AvgPoolNet");

    graph.tensors.push(TensorDesc {
        id: 0,
        name: "input".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_4d(1, 14, 14, 16),
        quant: QuantParams::default(),
    });

    graph.tensors.push(TensorDesc {
        id: 1,
        name: "output".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_4d(1, 7, 7, 16),
        quant: QuantParams::default(),
    });

    graph.inputs.push(0);
    graph.outputs.push(1);

    graph.layers.push(LayerNode {
        id: 0,
        name: "avgpool0".into(),
        inputs: vec![0],
        outputs: vec![1],
        op: OpPayload::AvgPool2D {
            pool_h: 2,
            pool_w: 2,
            stride_h: 2,
            stride_w: 2,
            padding: Padding2D::default(),
        },
    });

    let compiler = AtonCompiler::new();
    let compiled = compiler.compile(&graph).expect("compilation failed");

    assert_eq!(compiled.input_size_bytes, 14 * 14 * 16);
    assert_eq!(compiled.output_size_bytes, 7 * 7 * 16);

    let blob = compiled.build_blob_u64().expect("build blob");
    assert!(!blob.is_empty());
}

#[test]
fn test_compile_global_mean_pool() {
    let mut graph = ModelGraph::new("GlobalMeanNet");

    graph.tensors.push(TensorDesc {
        id: 0,
        name: "input".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_4d(1, 7, 7, 64),
        quant: QuantParams::default(),
    });

    graph.tensors.push(TensorDesc {
        id: 1,
        name: "output".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_4d(1, 1, 1, 64),
        quant: QuantParams::default(),
    });

    graph.inputs.push(0);
    graph.outputs.push(1);

    graph.layers.push(LayerNode {
        id: 0,
        name: "mean_pool".into(),
        inputs: vec![0],
        outputs: vec![1],
        op: OpPayload::Mean {
            reduce_height: true,
            reduce_width: true,
            reduce_channels: false,
            keep_dims: true,
        },
    });

    let compiler = AtonCompiler::new();
    let compiled = compiler.compile(&graph).expect("compilation failed");

    assert_eq!(compiled.input_size_bytes, 7 * 7 * 64);
    assert_eq!(compiled.output_size_bytes, 64);

    let blob = compiled.build_blob_u64().expect("build blob");
    assert!(!blob.is_empty());
}

#[test]
fn test_compile_elementwise_add() {
    let mut graph = ModelGraph::new("ElemAddNet");

    graph.tensors.push(TensorDesc {
        id: 0,
        name: "input_a".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_1d(128),
        quant: QuantParams::default(),
    });

    graph.tensors.push(TensorDesc {
        id: 1,
        name: "input_b".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_1d(128),
        quant: QuantParams::default(),
    });

    graph.tensors.push(TensorDesc {
        id: 2,
        name: "output".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_1d(128),
        quant: QuantParams::default(),
    });

    graph.inputs.push(0);
    graph.inputs.push(1);
    graph.outputs.push(2);

    graph.layers.push(LayerNode {
        id: 0,
        name: "add0".into(),
        inputs: vec![0, 1],
        outputs: vec![2],
        op: OpPayload::ElementwiseAdd {
            quant: ElementwiseAddQuant {
                input1_offset: 0,
                input1_multiplier: 1073741824,
                input1_shift: 0,
                input2_offset: 0,
                input2_multiplier: 1073741824,
                input2_shift: 0,
                left_shift: 0,
                output_offset: 0,
                output_multiplier: 1073741824,
                output_shift: 0,
            },
            activation: ActivationType::Relu,
        },
    });

    let compiler = AtonCompiler::new();
    let compiled = compiler.compile(&graph).expect("compilation failed");

    assert_eq!(compiled.input_size_bytes, 128);
    assert_eq!(compiled.output_size_bytes, 128);

    let blob = compiled.build_blob_u64().expect("build blob");
    assert!(!blob.is_empty());
}

#[test]
fn test_compile_elementwise_mul() {
    let mut graph = ModelGraph::new("ElemMulNet");

    graph.tensors.push(TensorDesc {
        id: 0,
        name: "in_a".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_1d(64),
        quant: QuantParams::default(),
    });

    graph.tensors.push(TensorDesc {
        id: 1,
        name: "in_b".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_1d(64),
        quant: QuantParams::default(),
    });

    graph.tensors.push(TensorDesc {
        id: 2,
        name: "out".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_1d(64),
        quant: QuantParams::default(),
    });

    graph.inputs.push(0);
    graph.inputs.push(1);
    graph.outputs.push(2);

    graph.layers.push(LayerNode {
        id: 0,
        name: "mul0".into(),
        inputs: vec![0, 1],
        outputs: vec![2],
        op: OpPayload::ElementwiseMul {
            quant: ElementwiseMulQuant {
                input1_offset: 0,
                input2_offset: 0,
                output_offset: 0,
                output_multiplier: 1073741824,
                output_shift: 7,
            },
            activation: ActivationType::None,
        },
    });

    let compiler = AtonCompiler::new();
    let compiled = compiler.compile(&graph).expect("compilation failed");

    assert_eq!(compiled.input_size_bytes, 64);
    assert_eq!(compiled.output_size_bytes, 64);

    let blob = compiled.build_blob_u64().expect("build blob");
    assert!(!blob.is_empty());
}

#[test]
fn test_compile_hybrid_npu_cpu_model() {
    let mut graph = ModelGraph::new("HybridCnnSoftmax");

    // Input: [1, 28, 28, 1]
    graph.tensors.push(TensorDesc {
        id: 0,
        name: "input".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_4d(1, 28, 28, 1),
        quant: QuantParams::default(),
    });

    // Conv Out: [1, 14, 14, 8]
    graph.tensors.push(TensorDesc {
        id: 1,
        name: "conv_out".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_4d(1, 14, 14, 8),
        quant: QuantParams::default(),
    });

    // Pool Out: [1, 7, 7, 8]
    graph.tensors.push(TensorDesc {
        id: 2,
        name: "pool_out".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_4d(1, 7, 7, 8),
        quant: QuantParams::default(),
    });

    // FC Out: [1, 10]
    graph.tensors.push(TensorDesc {
        id: 3,
        name: "fc_out".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_1d(10),
        quant: QuantParams::default(),
    });

    // Softmax Out: [1, 10]
    graph.tensors.push(TensorDesc {
        id: 4,
        name: "prob_out".into(),
        dtype: DataType::Int8,
        shape: TensorShape::new_1d(10),
        quant: QuantParams::default(),
    });

    graph.inputs.push(0);
    graph.outputs.push(4);

    // Layer 0: Conv2D (NPU)
    graph.layers.push(LayerNode {
        id: 0,
        name: "conv0".into(),
        inputs: vec![0],
        outputs: vec![1],
        op: OpPayload::Conv2D {
            kernel_h: 3,
            kernel_w: 3,
            stride_h: 2,
            stride_w: 2,
            padding: Padding2D::symmetric(1, 1),
            dilation_h: 1,
            dilation_w: 1,
            weights: vec![1i8; 3 * 3 * 1 * 8],
            packed_s4: None,
            bias: Some(vec![0i32; 8]),
            activation: ActivationType::Relu,
            per_channel_quant: None,
        },
    });

    // Layer 1: MaxPool2D (NPU)
    graph.layers.push(LayerNode {
        id: 1,
        name: "pool0".into(),
        inputs: vec![1],
        outputs: vec![2],
        op: OpPayload::MaxPool2D {
            pool_h: 2,
            pool_w: 2,
            stride_h: 2,
            stride_w: 2,
            padding: Padding2D::default(),
        },
    });

    // Layer 2: FC (NPU)
    graph.layers.push(LayerNode {
        id: 2,
        name: "fc0".into(),
        inputs: vec![2],
        outputs: vec![3],
        op: OpPayload::FullyConnected {
            weights: vec![1i8; 7 * 7 * 8 * 10],
            packed_s4: None,
            bias: Some(vec![0i32; 10]),
            filter_offset: 0,
            activation: ActivationType::None,
            per_channel_quant: None,
        },
    });

    // Layer 3: Softmax (Cortex-M55 CPU fallback)
    graph.layers.push(LayerNode {
        id: 3,
        name: "softmax0".into(),
        inputs: vec![3],
        outputs: vec![4],
        op: OpPayload::Softmax,
    });

    let compiler = AtonCompiler::new();
    let compiled = compiler.compile(&graph).expect("compilation failed");

    // Hybrid epochs: Hardware epoch (layers 0, 1, 2) + Software epoch (layer 3)
    assert_eq!(compiled.epochs.len(), 2);
    assert_eq!(
        compiled.epochs[0],
        EpochKind::Hardware {
            layer_ids: vec![0, 1, 2],
            input_symbol: "_user_io_input_0".into(),
            output_symbol: "_intermediate_3".into(),
        }
    );
    assert_eq!(
        compiled.epochs[1],
        EpochKind::Software {
            layer_id: 3,
            op_name: "softmax0".into(),
        }
    );

    let blob = compiled.build_blob_u64().expect("build blob");
    assert!(!blob.is_empty());
}
