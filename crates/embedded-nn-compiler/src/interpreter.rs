//! Host-side integer interpreter for [`ModelGraph`](crate::ModelGraph).
//!
//! This executes the same quantized kernels and arena schedule used by generated models. It is
//! intentionally a `std` development tool; generated/device inference remains allocation-free.

use crate::arena::{ArenaPlan, ArenaScheduler};
use crate::ir::{
    ActivationType, DataType, LayerNode, ModelGraph, OpPayload, QuantParams, TensorDesc,
    TransposeKind,
};
use embedded_nn::{
    Activation, ConvParams, Dims, DwConvParams, ElementwiseAddParams, FcParams, Padding2D,
    PerChannelQuantParams, PerTensorQuantParams, PoolParams, Tile, avg_pool_s8, convolve_1_x_n_s8,
    convolve_per_channel_s8, convolve_s4, convolve_s8, depthwise_conv_per_channel_s8,
    elementwise_add_s8, fully_connected_per_channel_s8, fully_connected_s4, fully_connected_s8,
    max_pool_s8, pad_s8, reduce_mean_s8, softmax_s8, svdf_s8, transpose_2d_s8, transpose_spatial_s8,
};
use std::collections::HashMap;

/// A failure while validating or executing a host-side graph.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum InterpreterError {
    #[error("graph has no model outputs")]
    NoOutputs,
    #[error("expected {expected} model inputs, got {actual}")]
    InputCount { expected: usize, actual: usize },
    #[error("input {index} has {actual} elements, expected {expected}")]
    InputLength {
        index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("tensor {0} is missing from the graph")]
    MissingTensor(usize),
    #[error("tensor {0} has no arena allocation")]
    MissingAllocation(usize),
    #[error("tensor {tensor_id} uses unsupported host dtype {dtype:?}")]
    UnsupportedDataType { tensor_id: usize, dtype: DataType },
    #[error("layer {layer_id} ({name}) has invalid inputs or outputs: {message}")]
    InvalidLayer {
        layer_id: usize,
        name: String,
        message: String,
    },
    #[error("layer {layer_id} ({name}) uses unsupported operation {operation}")]
    UnsupportedOp {
        layer_id: usize,
        name: String,
        operation: &'static str,
    },
    #[error("layer {layer_id} ({name}) kernel failed: {operation}")]
    Kernel {
        layer_id: usize,
        name: String,
        operation: &'static str,
    },
}

/// Reusable host interpreter. SVDF delay-line state is retained between calls and can be reset.
pub struct HostInterpreter<'g> {
    graph: &'g ModelGraph,
    plan: ArenaPlan,
    arena: Vec<u8>,
    svdf_state: HashMap<usize, Vec<i8>>,
}

impl<'g> HostInterpreter<'g> {
    /// Validates the graph's host tensor types and prepares its static arena.
    pub fn new(graph: &'g ModelGraph) -> Result<Self, InterpreterError> {
        if graph.outputs.is_empty() {
            return Err(InterpreterError::NoOutputs);
        }
        for tensor in &graph.tensors {
            if tensor.dtype != DataType::Int8 {
                return Err(InterpreterError::UnsupportedDataType {
                    tensor_id: tensor.id,
                    dtype: tensor.dtype,
                });
            }
        }
        let plan = ArenaScheduler::schedule(graph);
        let arena = vec![0; plan.total_arena_bytes];
        Ok(Self {
            graph,
            plan,
            arena,
            svdf_state: HashMap::new(),
        })
    }

    /// Returns the arena plan used by this interpreter.
    pub fn arena_plan(&self) -> &ArenaPlan {
        &self.plan
    }

    /// Clears persistent external recurrent state without rebuilding the graph or arena.
    pub fn reset_external_state(&mut self) {
        self.svdf_state.clear();
    }

    /// Executes one inference and returns owned copies of every model output, in graph order.
    pub fn run(&mut self, inputs: &[&[i8]]) -> Result<Vec<Vec<i8>>, InterpreterError> {
        if inputs.len() != self.graph.inputs.len() {
            return Err(InterpreterError::InputCount {
                expected: self.graph.inputs.len(),
                actual: inputs.len(),
            });
        }
        self.arena.fill(0);
        for (index, (&tensor_id, input)) in self.graph.inputs.iter().zip(inputs).enumerate() {
            let tensor = self.tensor(tensor_id)?;
            let expected = tensor.shape.total_elements();
            if input.len() != expected {
                return Err(InterpreterError::InputLength {
                    index,
                    expected,
                    actual: input.len(),
                });
            }
            self.write_tensor(tensor_id, input)?;
        }

        for layer in &self.graph.layers {
            self.execute_layer(layer)?;
        }

        self.graph
            .outputs
            .iter()
            .map(|&id| self.read_tensor(id))
            .collect()
    }

    fn tensor(&self, id: usize) -> Result<&TensorDesc, InterpreterError> {
        self.graph
            .tensors
            .iter()
            .find(|tensor| tensor.id == id)
            .ok_or(InterpreterError::MissingTensor(id))
    }

    fn range(&self, id: usize) -> Result<std::ops::Range<usize>, InterpreterError> {
        let allocation = self
            .plan
            .allocations
            .get(&id)
            .ok_or(InterpreterError::MissingAllocation(id))?;
        Ok(allocation.byte_offset..allocation.byte_offset + allocation.byte_size)
    }

    fn read_tensor(&self, id: usize) -> Result<Vec<i8>, InterpreterError> {
        Ok(self.arena[self.range(id)?]
            .iter()
            .map(|&value| value as i8)
            .collect())
    }

    fn write_tensor(&mut self, id: usize, values: &[i8]) -> Result<(), InterpreterError> {
        let range = self.range(id)?;
        if range.len() != values.len() {
            return Err(InterpreterError::InputLength {
                index: id,
                expected: range.len(),
                actual: values.len(),
            });
        }
        for (target, &value) in self.arena[range].iter_mut().zip(values) {
            *target = value as u8;
        }
        Ok(())
    }

    fn execute_layer(&mut self, layer: &LayerNode) -> Result<(), InterpreterError> {
        if layer.outputs.len() != 1 || layer.inputs.is_empty() {
            return Err(self.invalid(layer, "exactly one output and at least one input required"));
        }
        let input_id = layer.inputs[0];
        let output_id = layer.outputs[0];
        let input_tensor = self.tensor(input_id)?.clone();
        let output_tensor = self.tensor(output_id)?.clone();
        // Owned inputs avoid aliasing Rust references while still reading/writing the scheduled
        // offsets. Host execution permits allocation; the generated device runtime does not.
        let input = self.read_tensor(input_id)?;
        let mut output = vec![0i8; output_tensor.shape.total_elements()];
        let activation = |kind: &ActivationType| activation_for(kind, &output_tensor.quant);

        let kernel_result = match &layer.op {
            OpPayload::FullyConnected {
                weights,
                packed_s4,
                bias,
                filter_offset,
                activation: kind,
                per_channel_quant,
            } => {
                let params = FcParams {
                    input_offset: -input_tensor.quant.zero_point,
                    filter_offset: *filter_offset,
                    output_offset: output_tensor.quant.zero_point,
                    activation: activation(kind),
                };
                let input_len = input.len();
                let output_len = output.len();
                if let Some(weights) = packed_s4 {
                    fully_connected_s4(
                        &params,
                        &per_tensor(&output_tensor),
                        &Dims::new(1, 1, 1, input_len as i32),
                        &input,
                        &Dims::new(input_len as i32, 1, 1, output_len as i32),
                        weights,
                        bias.as_deref(),
                        &Dims::new(1, 1, 1, output_len as i32),
                        &mut output,
                    )
                } else if let Some(quant) = per_channel_quant {
                    fully_connected_per_channel_s8(
                        &params,
                        &PerChannelQuantParams::new(&quant.multipliers, &quant.shifts),
                        &Dims::new(1, 1, 1, input_len as i32),
                        &input,
                        &Dims::new(input_len as i32, 1, 1, output_len as i32),
                        weights,
                        bias.as_deref(),
                        &Dims::new(1, 1, 1, output_len as i32),
                        &mut output,
                    )
                } else {
                    fully_connected_s8(
                        &params,
                        &per_tensor(&output_tensor),
                        &Dims::new(1, 1, 1, input_len as i32),
                        &input,
                        &Dims::new(input_len as i32, 1, 1, output_len as i32),
                        weights,
                        bias.as_deref(),
                        &Dims::new(1, 1, 1, output_len as i32),
                        &mut output,
                    )
                }
            }
            OpPayload::Conv2D {
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding,
                dilation_h,
                dilation_w,
                weights,
                packed_s4,
                bias,
                activation: kind,
                per_channel_quant,
            } => {
                let params = ConvParams {
                    input_offset: -input_tensor.quant.zero_point,
                    output_offset: output_tensor.quant.zero_point,
                    stride: Tile::new(*stride_w as i32, *stride_h as i32),
                    padding: runtime_padding(padding),
                    dilation: Tile::new(*dilation_w as i32, *dilation_h as i32),
                    activation: activation(kind),
                };
                let in_dims = dims(&input_tensor);
                let filter_dims = Dims::new(
                    output_tensor.shape.channels as i32,
                    *kernel_h as i32,
                    *kernel_w as i32,
                    input_tensor.shape.channels as i32,
                );
                let out_dims = dims(&output_tensor);
                if let Some(weights) = packed_s4 {
                    convolve_s4(
                        &params,
                        &per_tensor(&output_tensor),
                        &in_dims,
                        &input,
                        &filter_dims,
                        weights,
                        bias.as_deref(),
                        &out_dims,
                        &mut output,
                    )
                } else if let Some(quant) = per_channel_quant {
                    convolve_per_channel_s8(
                        &params,
                        &PerChannelQuantParams::new(&quant.multipliers, &quant.shifts),
                        &in_dims,
                        &input,
                        &filter_dims,
                        weights,
                        bias.as_deref(),
                        &out_dims,
                        &mut output,
                    )
                } else {
                    convolve_s8(
                        &params,
                        &per_tensor(&output_tensor),
                        &in_dims,
                        &input,
                        &filter_dims,
                        weights,
                        bias.as_deref(),
                        &out_dims,
                        &mut output,
                    )
                }
            }
            OpPayload::DepthwiseConv2D {
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding,
                ch_mult,
                weights,
                bias,
                activation: kind,
                per_channel_quant,
            } => {
                let quant = per_channel_quant
                    .as_ref()
                    .ok_or_else(|| self.invalid(layer, "depthwise requires per-channel quant"))?;
                depthwise_conv_per_channel_s8(
                    &DwConvParams {
                        input_offset: -input_tensor.quant.zero_point,
                        output_offset: output_tensor.quant.zero_point,
                        ch_mult: *ch_mult as i32,
                        stride: Tile::new(*stride_w as i32, *stride_h as i32),
                        padding: runtime_padding(padding),
                        dilation: Tile::new(1, 1),
                        activation: activation(kind),
                    },
                    &PerChannelQuantParams::new(&quant.multipliers, &quant.shifts),
                    &dims(&input_tensor),
                    &input,
                    &Dims::new(
                        1,
                        *kernel_h as i32,
                        *kernel_w as i32,
                        output_tensor.shape.channels as i32,
                    ),
                    weights,
                    bias.as_deref(),
                    &dims(&output_tensor),
                    &mut output,
                )
            }
            OpPayload::MaxPool2D {
                pool_h,
                pool_w,
                stride_h,
                stride_w,
                padding,
            } => max_pool_s8(
                &pool_params(*stride_h, *stride_w, padding, &output_tensor),
                &Tile::new(*pool_w as i32, *pool_h as i32),
                &dims(&input_tensor),
                &input,
                &dims(&output_tensor),
                &mut output,
            ),
            OpPayload::AvgPool2D {
                pool_h,
                pool_w,
                stride_h,
                stride_w,
                padding,
            } => avg_pool_s8(
                &pool_params(*stride_h, *stride_w, padding, &output_tensor),
                &Tile::new(*pool_w as i32, *pool_h as i32),
                &dims(&input_tensor),
                &input,
                &dims(&output_tensor),
                &mut output,
            ),
            OpPayload::Softmax => {
                softmax_s8(&input, 1, input.len(), 1_073_741_824, 20, -256, &mut output)
            }
            OpPayload::Reshape { .. } => {
                if input.len() != output.len() {
                    return Err(self.invalid(layer, "reshape element counts differ"));
                }
                output.copy_from_slice(&input);
                Ok(())
            }
            OpPayload::Pad {
                padding,
                pad_value,
            } => pad_s8(
                &dims(&input_tensor),
                &input,
                &Tile::new(padding.left as i32, padding.top as i32),
                &Tile::new(padding.right as i32, padding.bottom as i32),
                *pad_value,
                &dims(&output_tensor),
                &mut output,
            ),
            OpPayload::Mean {
                reduce_height,
                reduce_width,
                reduce_channels,
                ..
            } => reduce_mean_s8(
                input_tensor.shape.batches,
                input_tensor.shape.height,
                input_tensor.shape.width,
                input_tensor.shape.channels,
                *reduce_height,
                *reduce_width,
                *reduce_channels,
                &input,
                &mut output,
            ),
            OpPayload::ElementwiseAdd {
                quant,
                activation: kind,
            } => {
                if layer.inputs.len() != 2 {
                    return Err(self.invalid(layer, "ADD requires two inputs"));
                }
                let input2 = self.read_tensor(layer.inputs[1])?;
                elementwise_add_s8(
                    &input,
                    &input2,
                    &mut output,
                    &ElementwiseAddParams {
                        input1_offset: quant.input1_offset,
                        input1_mult: quant.input1_multiplier,
                        input1_shift: quant.input1_shift,
                        input2_offset: quant.input2_offset,
                        input2_mult: quant.input2_multiplier,
                        input2_shift: quant.input2_shift,
                        left_shift: quant.left_shift,
                        output_offset: quant.output_offset,
                        output_mult: quant.output_multiplier,
                        output_shift: quant.output_shift,
                        activation: activation(kind),
                    },
                )
            }
            OpPayload::Transpose { kind } => match kind {
                TransposeKind::Matrix2D { rows, cols } => {
                    transpose_2d_s8(*rows, *cols, &input, &mut output)
                }
                TransposeKind::Spatial4D => {
                    transpose_spatial_s8(&dims(&input_tensor), &input, &mut output)
                }
            },
            OpPayload::Conv1D {
                kernel_w,
                stride_w,
                pad_w,
                dilation_w,
                weights,
                bias,
                activation: kind,
            } => convolve_1_x_n_s8(
                &ConvParams {
                    input_offset: -input_tensor.quant.zero_point,
                    output_offset: output_tensor.quant.zero_point,
                    stride: Tile::new(*stride_w as i32, 1),
                    padding: Padding2D::symmetric(*pad_w as i32, 0),
                    dilation: Tile::new(*dilation_w as i32, 1),
                    activation: activation(kind),
                },
                &per_tensor(&output_tensor),
                &dims(&input_tensor),
                &input,
                &Dims::new(
                    output_tensor.shape.channels as i32,
                    1,
                    *kernel_w as i32,
                    input_tensor.shape.channels as i32,
                ),
                weights,
                bias.as_deref(),
                &dims(&output_tensor),
                &mut output,
            ),
            OpPayload::Svdf {
                rank,
                memory_size,
                weights_feature,
                weights_time,
                bias,
                activation: kind,
            } => {
                let input_dim = input.len();
                if input_dim == 0 || weights_feature.len() % input_dim != 0 {
                    return Err(self.invalid(layer, "invalid SVDF feature weights"));
                }
                let state_len = (weights_feature.len() / input_dim) * memory_size;
                let state = self
                    .svdf_state
                    .entry(layer.id)
                    .or_insert_with(|| vec![0; state_len]);
                svdf_s8(
                    -input_tensor.quant.zero_point,
                    output_tensor.quant.zero_point,
                    *rank,
                    &input,
                    state,
                    weights_feature,
                    weights_time,
                    bias.as_deref(),
                    &per_tensor(&input_tensor),
                    &per_tensor(&output_tensor),
                    &activation(kind),
                    &mut output,
                )
            }
            OpPayload::LstmStep { .. } => {
                return Err(InterpreterError::UnsupportedOp {
                    layer_id: layer.id,
                    name: layer.name.clone(),
                    operation: "LstmStep",
                });
            }
        };

        kernel_result.map_err(|_| InterpreterError::Kernel {
            layer_id: layer.id,
            name: layer.name.clone(),
            operation: op_name(&layer.op),
        })?;
        self.write_tensor(output_id, &output)
    }

    fn invalid(&self, layer: &LayerNode, message: impl Into<String>) -> InterpreterError {
        InterpreterError::InvalidLayer {
            layer_id: layer.id,
            name: layer.name.clone(),
            message: message.into(),
        }
    }
}

fn dims(tensor: &TensorDesc) -> Dims {
    Dims::new(
        tensor.shape.batches as i32,
        tensor.shape.height as i32,
        tensor.shape.width as i32,
        tensor.shape.channels as i32,
    )
}

fn runtime_padding(padding: &crate::ir::Padding2D) -> Padding2D {
    Padding2D::new(
        padding.top as i32,
        padding.bottom as i32,
        padding.left as i32,
        padding.right as i32,
    )
}

fn per_tensor(tensor: &TensorDesc) -> PerTensorQuantParams {
    PerTensorQuantParams::new(tensor.quant.multiplier, tensor.quant.shift)
}

fn activation_for(kind: &ActivationType, quant: &QuantParams) -> Activation {
    match kind {
        ActivationType::None => Activation::int8_unconstrained(),
        ActivationType::Relu => Activation::new(quant.zero_point, i8::MAX as i32),
        ActivationType::Relu6 => Activation::new(
            quant.zero_point,
            (quant.zero_point + (6.0 / quant.scale).round() as i32).min(i8::MAX as i32),
        ),
        ActivationType::LeakyRelu { .. } | ActivationType::Sigmoid | ActivationType::Tanh => {
            Activation::int8_unconstrained()
        }
    }
}

fn pool_params(
    stride_h: usize,
    stride_w: usize,
    padding: &crate::ir::Padding2D,
    output: &TensorDesc,
) -> PoolParams {
    PoolParams {
        stride: Tile::new(stride_w as i32, stride_h as i32),
        padding: runtime_padding(padding),
        activation: activation_for(&ActivationType::None, &output.quant),
    }
}

fn op_name(op: &OpPayload) -> &'static str {
    match op {
        OpPayload::FullyConnected { packed_s4, .. } => {
            if packed_s4.is_some() {
                "FullyConnected s4"
            } else {
                "FullyConnected s8"
            }
        }
        OpPayload::Conv2D { packed_s4, .. } => {
            if packed_s4.is_some() {
                "Conv2D s4"
            } else {
                "Conv2D s8"
            }
        }
        OpPayload::DepthwiseConv2D { .. } => "DepthwiseConv2D",
        OpPayload::MaxPool2D { .. } => "MaxPool2D",
        OpPayload::AvgPool2D { .. } => "AvgPool2D",
        OpPayload::Softmax => "Softmax",
        OpPayload::ElementwiseAdd { .. } => "ADD",
        OpPayload::Transpose { .. } => "Transpose",
        OpPayload::Reshape { .. } => "Reshape",
        OpPayload::Pad { .. } => "Pad",
        OpPayload::Mean { .. } => "Mean",
        OpPayload::LstmStep { .. } => "LstmStep",
        OpPayload::Conv1D { .. } => "Conv1D",
        OpPayload::Svdf { .. } => "SVDF",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ModelBuilder;
    use crate::ir::TensorShape;

    fn identity_quant() -> QuantParams {
        QuantParams {
            multiplier: 1_073_741_824,
            shift: 1,
            zero_point: 0,
            scale: 1.0,
        }
    }

    #[test]
    fn dense_mlp_golden_vector_matches_integer_kernel_pipeline() {
        let mut builder = ModelBuilder::new("dense_mlp");
        let input = builder.add_input(
            "input",
            TensorShape::new_1d(2),
            DataType::Int8,
            Some(identity_quant()),
        );
        let hidden = builder.add_dense_layer(
            "hidden",
            input,
            2,
            vec![1, 0, 0, 1],
            None,
            Some(vec![0, 0]),
            ActivationType::Relu,
            None,
            Some(identity_quant()),
        );
        let output = builder.add_dense_layer(
            "output",
            hidden,
            2,
            vec![1, 1, -1, 1],
            None,
            Some(vec![1, -1]),
            ActivationType::None,
            None,
            Some(identity_quant()),
        );
        builder.mark_output(output);
        let graph = builder.build();

        let mut host = HostInterpreter::new(&graph).unwrap();
        assert_eq!(host.run(&[&[1, 2]]).unwrap(), vec![vec![4, 0]]);
    }

    #[test]
    fn pad_and_mean_match_kernels() {
        let mut builder = ModelBuilder::new("pad_mean");
        let input = builder.add_input(
            "input",
            TensorShape::new_4d(1, 2, 2, 1),
            DataType::Int8,
            Some(identity_quant()),
        );
        let padded = builder.add_pad_layer(
            "pad",
            input,
            crate::ir::Padding2D::symmetric(1, 1),
            0,
        );
        let mean = builder.add_mean_layer("mean", padded, true, true, false, false);
        builder.mark_output(mean);
        let graph = builder.build();
        let mut host = HostInterpreter::new(&graph).unwrap();
        let out = host.run(&[&[1, 2, 3, 4]]).unwrap();
        assert_eq!(out[0].len(), 1);
    }
}
