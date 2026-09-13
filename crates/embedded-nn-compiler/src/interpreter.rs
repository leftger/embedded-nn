//! Host-side integer interpreter for [`ModelGraph`].
//!
//! This executes the same quantized kernels and arena schedule used by generated models. It is
//! intentionally a `std` development tool; generated/device inference remains allocation-free.

use crate::arena::{ArenaPlan, ArenaScheduler};
use crate::ir::{
    ActivationType, DataType, LayerNode, ModelGraph, OpPayload, QuantParams, TensorDesc,
    TransposeKind,
};
use embedded_nn::{
    Activation, AttentionParams, ConvParams, Dims, DwConvParams, ElementwiseAddParams,
    ElementwiseMulParams, FcParams, LstmGateParams, Padding2D, PerChannelQuantParams,
    PerTensorQuantParams, PoolParams, RmsNormParams, Tile, avg_pool_s8, batch_matmul_s8_shaped,
    concatenation_s8, convolve_1_x_n_s8, convolve_per_channel_s8, convolve_s4, convolve_s8,
    depthwise_conv_per_channel_s8, elementwise_add_s8, elementwise_mul_s8,
    fully_connected_per_channel_s8, fully_connected_s4, fully_connected_s8, gelu_s8,
    lstm_step_s8_s16, max_pool_s8, pad_s8, reduce_mean_s8, rms_norm_s8,
    scaled_dot_product_attention_s8, softmax_last_axis_s8, strided_slice_s8, svdf_s8,
    transpose_2d_s8, transpose_nd_s8, transpose_spatial_s8,
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
    lstm_hidden: HashMap<usize, Vec<i8>>,
    lstm_cell: HashMap<usize, Vec<i16>>,
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
            lstm_hidden: HashMap::new(),
            lstm_cell: HashMap::new(),
        })
    }

    /// Returns the arena plan used by this interpreter.
    pub fn arena_plan(&self) -> &ArenaPlan {
        &self.plan
    }

    /// Clears persistent external recurrent state without rebuilding the graph or arena.
    pub fn reset_external_state(&mut self) {
        self.svdf_state.clear();
        self.lstm_hidden.clear();
        self.lstm_cell.clear();
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
        if layer.outputs.is_empty() || layer.inputs.is_empty() {
            return Err(self.invalid(layer, "at least one output and one input required"));
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
                let in_channels = input_tensor.shape.channels.max(1);
                let out_channels = output_tensor.shape.channels.max(1);
                let token_batches = input_len / in_channels;
                let in_features_from_weights = if weights.is_empty() {
                    in_channels
                } else {
                    weights.len() / out_channels.max(1)
                };
                let token_wise = packed_s4.is_none()
                    && in_features_from_weights == in_channels
                    && token_batches > 1
                    && output_len == token_batches * out_channels;
                let (batches, in_features, out_features) = if token_wise {
                    (
                        token_batches as i32,
                        in_channels as i32,
                        out_channels as i32,
                    )
                } else {
                    (1, input_len as i32, output_len as i32)
                };
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
                        &Dims::new(batches, 1, 1, in_features),
                        &input,
                        &Dims::new(in_features, 1, 1, out_features),
                        weights,
                        bias.as_deref(),
                        &Dims::new(batches, 1, 1, out_features),
                        &mut output,
                    )
                } else {
                    fully_connected_s8(
                        &params,
                        &per_tensor(&output_tensor),
                        &Dims::new(batches, 1, 1, in_features),
                        &input,
                        &Dims::new(in_features, 1, 1, out_features),
                        weights,
                        bias.as_deref(),
                        &Dims::new(batches, 1, 1, out_features),
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
            OpPayload::Softmax => softmax_last_axis_s8(
                &input,
                input_tensor.shape.batches,
                input_tensor.shape.height,
                input_tensor.shape.width,
                input_tensor.shape.channels.max(1),
                1_073_741_824,
                20,
                -256,
                &mut output,
            ),
            OpPayload::Gelu { lut, .. } => gelu_s8(&input, &mut output, lut),
            OpPayload::Reshape { .. } => {
                if input.len() != output.len() {
                    return Err(self.invalid(layer, "reshape element counts differ"));
                }
                output.copy_from_slice(&input);
                Ok(())
            }
            OpPayload::Pad { padding, pad_value } => pad_s8(
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
            OpPayload::ElementwiseMul {
                quant,
                activation: kind,
            } => {
                if layer.inputs.len() != 2 {
                    return Err(self.invalid(layer, "MUL requires two inputs"));
                }
                let input2 = self.read_tensor(layer.inputs[1])?;
                elementwise_mul_s8(
                    &input,
                    &input2,
                    &mut output,
                    &ElementwiseMulParams {
                        input1_offset: quant.input1_offset,
                        input2_offset: quant.input2_offset,
                        output_offset: quant.output_offset,
                        output_mult: quant.output_multiplier,
                        output_shift: quant.output_shift,
                        activation: activation(kind),
                    },
                )
            }
            OpPayload::Concat => {
                if layer.inputs.len() != 2 {
                    return Err(self.invalid(layer, "Concat requires two inputs"));
                }
                let input2_tensor = self.tensor(layer.inputs[1])?.clone();
                let input2 = self.read_tensor(layer.inputs[1])?;
                concatenation_s8(
                    &dims(&input_tensor),
                    &input,
                    &dims(&input2_tensor),
                    &input2,
                    &dims(&output_tensor),
                    &mut output,
                )
            }
            OpPayload::StridedSlice { begin, end, stride } => strided_slice_s8(
                &dims(&input_tensor),
                begin,
                end,
                stride,
                &input,
                &mut output,
            ),
            OpPayload::Transpose { kind } => match kind {
                TransposeKind::Matrix2D { rows, cols } => {
                    transpose_2d_s8(*rows, *cols, &input, &mut output)
                }
                TransposeKind::Spatial4D => {
                    transpose_spatial_s8(&dims(&input_tensor), &input, &mut output)
                }
                TransposeKind::Nd { dims, perm } => {
                    transpose_nd_s8(dims, perm, &input, &mut output)
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
            OpPayload::LstmStep {
                hidden_dim,
                input_weights,
                recurrent_weights,
                bias,
            } => {
                let layer_id = layer.id;
                let mut hidden = self
                    .lstm_hidden
                    .remove(&layer_id)
                    .unwrap_or_else(|| vec![0; *hidden_dim]);
                let mut cell = self
                    .lstm_cell
                    .remove(&layer_id)
                    .unwrap_or_else(|| vec![0; *hidden_dim]);
                let gate = LstmGateParams {
                    input_offset: -input_tensor.quant.zero_point,
                    hidden_offset: -output_tensor.quant.zero_point,
                    multiplier: output_tensor.quant.multiplier,
                    shift: output_tensor.quant.shift,
                };
                let result = lstm_step_s8_s16(
                    &input,
                    &mut hidden,
                    &mut cell,
                    input_weights,
                    recurrent_weights,
                    bias,
                    &gate,
                    32767,
                    &per_tensor(&output_tensor),
                    output_tensor.quant.zero_point,
                    &activation(&ActivationType::None),
                );
                if hidden.len() == output.len() {
                    output.copy_from_slice(&hidden);
                }
                self.lstm_hidden.insert(layer_id, hidden);
                self.lstm_cell.insert(layer_id, cell);
                result
            }
            OpPayload::BatchMatMul { rhs_transposed } => {
                if layer.inputs.len() != 2 {
                    return Err(self.invalid(layer, "BatchMatMul requires two inputs"));
                }
                let rhs_tensor = self.tensor(layer.inputs[1])?.clone();
                let rhs = self.read_tensor(layer.inputs[1])?;
                let (lhs_b, rows, accum) = input_tensor.shape.as_batched_matrix();
                let (rhs_b, rhs_d0, rhs_d1) = rhs_tensor.shape.as_batched_matrix();
                let (rhs_accum, cols) = if *rhs_transposed {
                    (rhs_d1, rhs_d0)
                } else {
                    (rhs_d0, rhs_d1)
                };
                if accum != rhs_accum {
                    return Err(self.invalid(layer, "BatchMatMul inner dimensions do not match"));
                }
                let batches = lhs_b.max(rhs_b);
                batch_matmul_s8_shaped(
                    -input_tensor.quant.zero_point,
                    -rhs_tensor.quant.zero_point,
                    output_tensor.quant.zero_point,
                    &per_tensor(&output_tensor),
                    batches,
                    rows,
                    accum,
                    cols,
                    &input,
                    &rhs,
                    *rhs_transposed,
                    &mut output,
                )
            }
            OpPayload::RmsNorm { gamma, epsilon } => rms_norm_s8(
                &RmsNormParams::new(
                    -input_tensor.quant.zero_point,
                    output_tensor.quant.zero_point,
                    input_tensor.shape.channels.max(1),
                    *epsilon,
                ),
                &per_tensor(&output_tensor),
                &input,
                gamma.as_deref(),
                &mut output,
            ),
            OpPayload::ScaledDotProductAttention {
                num_heads,
                logits_multiplier,
                logits_shift,
                softmax_mult,
                softmax_shift,
                softmax_diff_min,
            } => {
                if layer.inputs.len() != 3 {
                    return Err(self.invalid(layer, "Attention requires Q, K, and V"));
                }
                let k = self.read_tensor(layer.inputs[1])?;
                let v = self.read_tensor(layer.inputs[2])?;
                let k_tensor = self.tensor(layer.inputs[1])?.clone();
                let v_tensor = self.tensor(layer.inputs[2])?.clone();
                let seq_len = input_tensor.shape.height.max(input_tensor.shape.width);
                let d_model = input_tensor.shape.channels;
                if *num_heads == 0 || d_model % *num_heads != 0 {
                    return Err(self.invalid(layer, "invalid attention head configuration"));
                }
                let head_dim = d_model / *num_heads;
                let mut scores = vec![0i8; seq_len * seq_len * 2];
                let params = AttentionParams {
                    batches: input_tensor.shape.batches.max(1),
                    seq_len,
                    num_heads: *num_heads,
                    head_dim,
                    q_offset: -input_tensor.quant.zero_point,
                    k_offset: -k_tensor.quant.zero_point,
                    v_offset: -v_tensor.quant.zero_point,
                    score_offset: 128,
                    output_offset: output_tensor.quant.zero_point,
                    softmax_mult: *softmax_mult,
                    softmax_shift: *softmax_shift,
                    softmax_diff_min: *softmax_diff_min,
                };
                scaled_dot_product_attention_s8(
                    &params,
                    &PerTensorQuantParams::new(*logits_multiplier, *logits_shift),
                    &per_tensor(&output_tensor),
                    &input,
                    &k,
                    &v,
                    &mut scores,
                    &mut output,
                )
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
        OpPayload::Gelu { .. } => "GELU",
        OpPayload::ElementwiseAdd { .. } => "ADD",
        OpPayload::ElementwiseMul { .. } => "MUL",
        OpPayload::Concat => "Concat",
        OpPayload::StridedSlice { .. } => "StridedSlice",
        OpPayload::Transpose { .. } => "Transpose",
        OpPayload::Reshape { .. } => "Reshape",
        OpPayload::Pad { .. } => "Pad",
        OpPayload::Mean { .. } => "Mean",
        OpPayload::LstmStep { .. } => "LstmStep",
        OpPayload::Conv1D { .. } => "Conv1D",
        OpPayload::Svdf { .. } => "SVDF",
        OpPayload::BatchMatMul { .. } => "BatchMatMul",
        OpPayload::RmsNorm { .. } => "RmsNorm",
        OpPayload::ScaledDotProductAttention { .. } => "ScaledDotProductAttention",
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
        let padded = builder.add_pad_layer("pad", input, crate::ir::Padding2D::symmetric(1, 1), 0);
        let mean = builder.add_mean_layer("mean", padded, true, true, false, false);
        builder.mark_output(mean);
        let graph = builder.build();
        let mut host = HostInterpreter::new(&graph).unwrap();
        let out = host.run(&[&[1, 2, 3, 4]]).unwrap();
        assert_eq!(out[0].len(), 1);
    }

    #[test]
    fn interpreter_reports_missing_outputs_and_validation_errors() {
        let builder = ModelBuilder::new("no_output");
        let graph = builder.build();
        assert!(matches!(
            HostInterpreter::new(&graph),
            Err(InterpreterError::NoOutputs)
        ));

        let mut builder = ModelBuilder::new("bad_dtype");
        let input = builder.add_input(
            "input",
            TensorShape::new_1d(2),
            DataType::Float32,
            Some(identity_quant()),
        );
        builder.mark_output(input);
        let graph = builder.build();
        assert!(matches!(
            HostInterpreter::new(&graph),
            Err(InterpreterError::UnsupportedDataType { .. })
        ));

        let mut builder = ModelBuilder::new("dense");
        let input = builder.add_input(
            "input",
            TensorShape::new_1d(2),
            DataType::Int8,
            Some(identity_quant()),
        );
        let output = builder.add_dense_layer(
            "out",
            input,
            1,
            vec![1, 1],
            None,
            Some(vec![0]),
            ActivationType::None,
            None,
            Some(identity_quant()),
        );
        builder.mark_output(output);
        let graph = builder.build();
        let mut host = HostInterpreter::new(&graph).unwrap();
        assert!(host.run(&[]).is_err());
        assert!(host.run(&[&[1, 2, 3]]).is_err());
        assert!(host.arena_plan().total_arena_bytes > 0);
        host.reset_external_state();
    }

    #[test]
    fn interpreter_runs_conv_pool_and_concat_graphs() {
        let mut builder = ModelBuilder::new("conv_pool");
        let input = builder.add_input(
            "input",
            TensorShape::new_4d(1, 2, 2, 1),
            DataType::Int8,
            Some(identity_quant()),
        );
        let conv = builder.add_conv2d_layer(
            "conv",
            input,
            1,
            2,
            2,
            1,
            1,
            crate::ir::Padding2D::default(),
            1,
            1,
            vec![1, 1, 1, 1],
            None,
            None,
            ActivationType::None,
            None,
            Some(identity_quant()),
        );
        let pool =
            builder.add_maxpool2d_layer("pool", conv, 1, 1, 1, 1, crate::ir::Padding2D::default());
        builder.mark_output(pool);
        let graph = builder.build();
        let mut host = HostInterpreter::new(&graph).unwrap();
        let out = host.run(&[&[1, 2, 3, 4]]).unwrap();
        assert_eq!(out[0], vec![10]);

        let mut builder = ModelBuilder::new("concat");
        let a = builder.add_input(
            "a",
            TensorShape::new_1d(1),
            DataType::Int8,
            Some(identity_quant()),
        );
        let b = builder.add_input(
            "b",
            TensorShape::new_1d(1),
            DataType::Int8,
            Some(identity_quant()),
        );
        let cat = builder.add_concat_layer("cat", a, b).unwrap();
        builder.mark_output(cat);
        let graph = builder.build();
        let mut host = HostInterpreter::new(&graph).unwrap();
        assert_eq!(host.run(&[&[1], &[2]]).unwrap(), vec![vec![1, 2]]);
    }

    #[test]
    fn batch_matmul_and_last_axis_softmax_run() {
        let mut builder = ModelBuilder::new("bmm_soft");
        let lhs = builder.add_input(
            "lhs",
            TensorShape::new_2d(2, 2),
            DataType::Int8,
            Some(identity_quant()),
        );
        let rhs = builder.add_input(
            "rhs",
            TensorShape::new_2d(2, 2),
            DataType::Int8,
            Some(identity_quant()),
        );
        let mm = builder
            .add_batch_matmul_layer("mm", lhs, rhs, false, Some(identity_quant()))
            .unwrap();
        let sm = builder.add_softmax("sm", mm);
        builder.mark_output(sm);
        let graph = builder.build();
        let mut host = HostInterpreter::new(&graph).unwrap();
        let out = host.run(&[&[1, 0, 0, 1], &[10, 20, 30, 40]]).unwrap();
        assert_eq!(out[0].len(), 4);
        assert!(out[0][1] > out[0][0]);
        assert!(out[0][3] > out[0][2]);
    }

    #[test]
    fn tiny_attention_encoder_runs() {
        let tokens = TensorShape::new_4d(1, 2, 1, 4);
        let mut builder = ModelBuilder::new("tiny_enc");
        let q = builder.add_input("q", tokens, DataType::Int8, Some(identity_quant()));
        let k = builder.add_input("k", tokens, DataType::Int8, Some(identity_quant()));
        let v = builder.add_input("v", tokens, DataType::Int8, Some(identity_quant()));
        let n = builder
            .add_rms_norm_layer("n", q, None, 1, Some(identity_quant()))
            .unwrap();
        let attn = builder
            .add_attention_layer("attn", n, k, v, 2, 1_073_741_824, 1, Some(identity_quant()))
            .unwrap();
        let ffn = builder.add_channel_dense_layer(
            "ffn",
            attn,
            4,
            vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            Some(vec![0, 0, 0, 0]),
            ActivationType::Relu,
            None,
            Some(identity_quant()),
        );
        builder.mark_output(ffn);
        let graph = builder.build();
        let mut host = HostInterpreter::new(&graph).unwrap();
        let x = [8i8, 0, 0, 8, 0, 8, 8, 0];
        let out = host.run(&[&x, &x, &x]).unwrap();
        assert_eq!(out[0].len(), 8);
    }

    #[test]
    fn rms_norm_with_gamma_and_transposed_matmul() {
        let mut builder = ModelBuilder::new("gamma");
        let x = builder.add_input(
            "x",
            TensorShape::new_2d(2, 2),
            DataType::Int8,
            Some(identity_quant()),
        );
        let n = builder
            .add_rms_norm_layer("n", x, Some(vec![127, 64]), 1, Some(identity_quant()))
            .unwrap();
        builder.mark_output(n);
        let graph = builder.build();
        assert_eq!(graph.total_weights_size_bytes(), 2);
        let mut host = HostInterpreter::new(&graph).unwrap();
        let out = host.run(&[&[16i8, -16, 16, -16]]).unwrap();
        assert_eq!(out[0].len(), 4);

        let mut bmm = ModelBuilder::new("bt");
        let lhs = bmm.add_input(
            "lhs",
            TensorShape::new_2d(2, 2),
            DataType::Int8,
            Some(identity_quant()),
        );
        let rhs = bmm.add_input(
            "rhs",
            TensorShape::new_2d(2, 2),
            DataType::Int8,
            Some(identity_quant()),
        );
        let y = bmm
            .add_batch_matmul_layer("mm", lhs, rhs, true, Some(identity_quant()))
            .unwrap();
        bmm.mark_output(y);
        let graph = bmm.build();
        let mut host = HostInterpreter::new(&graph).unwrap();
        let out = host.run(&[&[1, 0, 0, 1], &[2, 4, 3, 5]]).unwrap();
        assert_eq!(out[0], vec![2, 3, 4, 5]);
    }

    #[test]
    fn attention_and_matmul_report_invalid_layer_inputs() {
        let mut graph = ModelGraph::new("bad_attn");
        graph.tensors.push(TensorDesc {
            id: 0,
            name: "q".into(),
            shape: TensorShape::new_4d(1, 2, 1, 4),
            dtype: DataType::Int8,
            quant: identity_quant(),
        });
        graph.tensors.push(TensorDesc {
            id: 1,
            name: "out".into(),
            shape: TensorShape::new_4d(1, 2, 1, 4),
            dtype: DataType::Int8,
            quant: identity_quant(),
        });
        graph.inputs.push(0);
        graph.outputs.push(1);
        graph.layers.push(LayerNode {
            id: 0,
            name: "attn".into(),
            inputs: vec![0],
            outputs: vec![1],
            op: OpPayload::ScaledDotProductAttention {
                num_heads: 2,
                logits_multiplier: 1_073_741_824,
                logits_shift: 1,
                softmax_mult: 1_073_741_824,
                softmax_shift: 20,
                softmax_diff_min: -256,
            },
        });
        let mut host = HostInterpreter::new(&graph).unwrap();
        let err = host.run(&[&[1i8; 8]]).unwrap_err();
        assert!(matches!(err, InterpreterError::InvalidLayer { .. }));

        graph.layers[0].op = OpPayload::BatchMatMul {
            rhs_transposed: false,
        };
        let mut host = HostInterpreter::new(&graph).unwrap();
        let err = host.run(&[&[1i8; 8]]).unwrap_err();
        assert!(matches!(err, InterpreterError::InvalidLayer { .. }));
    }
}
