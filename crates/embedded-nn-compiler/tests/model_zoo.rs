use embedded_nn_compiler::HostInterpreter;
use embedded_nn_compiler::builder::ModelBuilder;
use embedded_nn_compiler::ir::{ActivationType, DataType, QuantParams, TensorShape};

fn identity_quant() -> QuantParams {
    QuantParams {
        multiplier: 1_073_741_824,
        shift: 1,
        zero_point: 0,
        scale: 1.0,
    }
}

fn fc_zoo() -> embedded_nn_compiler::ir::ModelGraph {
    let mut builder = ModelBuilder::new("zoo_fc");
    let input = builder.add_input(
        "input",
        TensorShape::new_1d(4),
        DataType::Int8,
        Some(identity_quant()),
    );
    let hidden = builder.add_dense_layer(
        "h",
        input,
        4,
        vec![1; 16],
        None,
        Some(vec![0; 4]),
        ActivationType::Relu,
        None,
        Some(identity_quant()),
    );
    let output = builder.add_dense_layer(
        "out",
        hidden,
        2,
        vec![1, 0, 0, 0, 0, 1, 0, 0],
        None,
        Some(vec![0, 0]),
        ActivationType::None,
        None,
        Some(identity_quant()),
    );
    builder.mark_output(output);
    builder.build()
}

fn conv1d_zoo() -> embedded_nn_compiler::ir::ModelGraph {
    let mut builder = ModelBuilder::new("zoo_conv1d");
    let input = builder.add_input(
        "input",
        TensorShape::new_4d(1, 1, 4, 2),
        DataType::Int8,
        Some(identity_quant()),
    );
    let conv = builder.add_conv1d_layer(
        "conv",
        input,
        2,
        2,
        1,
        0,
        1,
        vec![1; 8],
        Some(vec![0, 0]),
        ActivationType::Relu,
        Some(identity_quant()),
    );
    let flat = builder.add_reshape_layer("flat", conv, TensorShape::new_1d(6));
    let output = builder.add_dense_layer(
        "out",
        flat,
        2,
        vec![1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        None,
        Some(vec![0, 0]),
        ActivationType::None,
        None,
        Some(identity_quant()),
    );
    builder.mark_output(output);
    builder.build()
}

#[test]
fn zoo_fc_golden_vector() {
    let graph = fc_zoo();
    let mut host = HostInterpreter::new(&graph).unwrap();
    let out = host.run(&[&[1i8, 0, 0, 0]]).unwrap();
    assert_eq!(out[0].len(), 2);
}

#[test]
fn zoo_conv1d_golden_vector() {
    let graph = conv1d_zoo();
    let mut host = HostInterpreter::new(&graph).unwrap();
    let input = [1i8, 0, 0, 0, 0, 0, 0, 0];
    let out = host.run(&[&input]).unwrap();
    assert_eq!(out[0].len(), 2);
}

#[test]
fn zoo_lstm_step_runs() {
    let mut builder = ModelBuilder::new("zoo_lstm");
    let input = builder.add_input(
        "x",
        TensorShape::new_1d(2),
        DataType::Int8,
        Some(identity_quant()),
    );
    let hidden = 2;
    let out = builder.add_lstm_step_layer(
        "cell",
        input,
        hidden,
        vec![1; 4 * hidden * 2],
        vec![1; 4 * hidden * hidden],
        vec![0; 4 * hidden],
        Some(identity_quant()),
    );
    builder.mark_output(out);
    let graph = builder.build();
    let mut host = HostInterpreter::new(&graph).unwrap();
    let y = host.run(&[&[1i8, 0]]).unwrap();
    assert_eq!(y[0].len(), 2);
}
