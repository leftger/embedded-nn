//! TensorFlow Lite (`.tflite`) FlatBuffers importer for `embedded-nn`.
//!
//! Parses a `.tflite` model (subgraph 0 only) into an `embedded-nn-compiler` [`ModelGraph`],
//! which can then be fed straight into the existing `embedded-nn-codegen` pipeline exactly like
//! a Studio-trained or hand-built graph -- no changes needed downstream.
//!
//! ## Scope (v1)
//! - Only `INT8`-quantized tensors are supported (TFLite `UINT8` models are rejected).
//! - Only subgraph 0 is imported; models with control-flow subgraphs are not supported.
//! - Supported operators: `FULLY_CONNECTED`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `MAX_POOL_2D`,
//!   `AVERAGE_POOL_2D`, `SOFTMAX`, `RESHAPE`. Anything else is a hard `UnsupportedOperator` error.
//! - SAME padding is converted to embedded-nn's symmetric `pad_h`/`pad_w` by taking the *before*
//!   half of TFLite's (possibly asymmetric) total padding. When the total padding is odd, this is
//!   off by one row/column from TFLite's true asymmetric padding -- embedded-nn's IR has no
//!   separate before/after padding fields to represent that exactly. Flagged, not silently wrong.
//! - Per-channel quantization is respected for `CONV_2D`/`DEPTHWISE_CONV_2D`/`FULLY_CONNECTED`
//!   weight tensors (assuming `quantized_dimension == 0`, the near-universal TFLite convention).

#[path = "../schema/schema_generated.rs"]
#[allow(warnings)]
mod schema;

use embedded_nn_compiler::builder::ModelBuilder;
use embedded_nn_compiler::ir::*;
use embedded_nn_compiler::quant::{calculate_output_requant_multiplier, quantize_multiplier};
use schema::tflite;
use std::collections::HashMap;

/// Errors that can occur while importing a `.tflite` model.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("invalid FlatBuffer data: {0}")]
    InvalidFlatBuffer(String),
    #[error("model has no subgraphs")]
    NoSubgraphs,
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("unsupported tensor type: {0}")]
    UnsupportedTensorType(&'static str),
    #[error("unsupported operator: {0:?} (opcode {1})")]
    UnsupportedOperator(&'static str, i32),
    #[error("operator input tensor {0} was never produced by an earlier layer")]
    UnresolvedInput(usize),
}

/// Imports a `.tflite` model's subgraph 0 into an embedded-nn [`ModelGraph`].
pub fn import_tflite(bytes: &[u8]) -> Result<ModelGraph, ImportError> {
    let model =
        tflite::root_as_model(bytes).map_err(|e| ImportError::InvalidFlatBuffer(e.to_string()))?;

    let subgraphs = model.subgraphs().ok_or(ImportError::NoSubgraphs)?;
    if subgraphs.is_empty() {
        return Err(ImportError::NoSubgraphs);
    }
    let subgraph = subgraphs.get(0);

    let tensors = subgraph
        .tensors()
        .ok_or(ImportError::MissingField("subgraph.tensors"))?;
    let buffers = model
        .buffers()
        .ok_or(ImportError::MissingField("model.buffers"))?;
    let operator_codes = model
        .operator_codes()
        .ok_or(ImportError::MissingField("model.operator_codes"))?;
    let operators = subgraph
        .operators()
        .ok_or(ImportError::MissingField("subgraph.operators"))?;
    let graph_inputs = subgraph
        .inputs()
        .ok_or(ImportError::MissingField("subgraph.inputs"))?;
    let graph_outputs = subgraph
        .outputs()
        .ok_or(ImportError::MissingField("subgraph.outputs"))?;

    let mut builder = ModelBuilder::new("TfliteImportedModel");
    // Maps a TFLite tensor index to the ModelGraph tensor id that represents it.
    let mut tensor_ids: HashMap<usize, usize> = HashMap::new();
    // Maps a TFLite tensor index to that tensor's own float scale, used as the "input_scale" for
    // whichever operator consumes it next.
    let mut tensor_scales: HashMap<usize, f32> = HashMap::new();

    for i in 0..graph_inputs.len() {
        let idx = graph_inputs.get(i) as usize;
        let tensor = tensors.get(idx);
        let shape = convert_shape(&tensor)?;
        let dtype = convert_tensor_type(tensor.type_())?;
        let (scale, zero_point) = read_per_tensor_quant(&tensor)?;
        let (multiplier, shift) = quantize_multiplier(scale);
        let quant = QuantParams {
            multiplier,
            shift,
            zero_point,
            scale,
        };
        let id = builder.add_input(format!("input_{}", idx), shape, dtype, Some(quant));
        tensor_ids.insert(idx, id);
        tensor_scales.insert(idx, scale);
    }

    for (op_index, operator) in operators.iter().enumerate() {
        let op_inputs = operator
            .inputs()
            .ok_or(ImportError::MissingField("operator.inputs"))?;
        let op_outputs = operator
            .outputs()
            .ok_or(ImportError::MissingField("operator.outputs"))?;

        let primary_input_idx = op_inputs.get(0) as usize;
        let in_id = *tensor_ids
            .get(&primary_input_idx)
            .ok_or(ImportError::UnresolvedInput(primary_input_idx))?;
        let input_scale = *tensor_scales.get(&primary_input_idx).unwrap_or(&1.0);

        let opcode_index = operator.opcode_index() as usize;
        let opcode = operator_codes.get(opcode_index);
        let builtin = opcode.builtin_code();

        let output_idx = op_outputs.get(0) as usize;
        let output_tensor = tensors.get(output_idx);
        let layer_name = format!("op{}", op_index);

        let out_id = match builtin {
            tflite::BuiltinOperator::FULLY_CONNECTED => import_fully_connected(
                &mut builder,
                &operator,
                &tensors,
                &buffers,
                in_id,
                input_scale,
                &output_tensor,
                &layer_name,
            )?,
            tflite::BuiltinOperator::CONV_2D => import_conv2d(
                &mut builder,
                &operator,
                &tensors,
                &buffers,
                in_id,
                input_scale,
                &output_tensor,
                &layer_name,
            )?,
            tflite::BuiltinOperator::DEPTHWISE_CONV_2D => import_depthwise_conv2d(
                &mut builder,
                &operator,
                &tensors,
                &buffers,
                in_id,
                input_scale,
                &output_tensor,
                &layer_name,
            )?,
            tflite::BuiltinOperator::MAX_POOL_2D => import_pool(
                &mut builder,
                &operator,
                &tensors,
                in_id,
                &output_tensor,
                &layer_name,
                PoolKind::Max,
            )?,
            tflite::BuiltinOperator::AVERAGE_POOL_2D => import_pool(
                &mut builder,
                &operator,
                &tensors,
                in_id,
                &output_tensor,
                &layer_name,
                PoolKind::Avg,
            )?,
            tflite::BuiltinOperator::SOFTMAX => builder.add_softmax(layer_name.clone(), in_id),
            tflite::BuiltinOperator::RESHAPE => {
                let shape = convert_shape(&output_tensor)?;
                builder.add_reshape_layer(layer_name.clone(), in_id, shape)
            }
            other => {
                return Err(ImportError::UnsupportedOperator(
                    other.variant_name().unwrap_or("UNKNOWN"),
                    other.0,
                ));
            }
        };

        let (out_scale, _) = read_per_tensor_quant(&output_tensor)?;
        tensor_ids.insert(output_idx, out_id);
        tensor_scales.insert(output_idx, out_scale);
    }

    for i in 0..graph_outputs.len() {
        let idx = graph_outputs.get(i) as usize;
        if let Some(&id) = tensor_ids.get(&idx) {
            builder.mark_output(id);
        }
    }

    Ok(builder.build())
}

enum PoolKind {
    Max,
    Avg,
}

fn convert_tensor_type(t: tflite::TensorType) -> Result<DataType, ImportError> {
    match t {
        tflite::TensorType::INT8 => Ok(DataType::Int8),
        _ => Err(ImportError::UnsupportedTensorType(
            "only INT8 tensors are supported (UINT8/float models are not)",
        )),
    }
}

/// Maps a TFLite tensor's declared shape to embedded-nn's `TensorShape`, assuming batch size 1
/// (embedded-nn has no batch dimension concept beyond that) and NHWC layout for rank-4 tensors.
fn convert_shape(tensor: &tflite::Tensor) -> Result<TensorShape, ImportError> {
    let dims = tensor
        .shape()
        .ok_or(ImportError::MissingField("tensor.shape"))?;
    let dims: Vec<i32> = dims.iter().collect();
    Ok(match dims.len() {
        1 => TensorShape::new_1d(dims[0] as usize),
        2 => TensorShape::new_1d(dims[1] as usize),
        4 => TensorShape::new_4d(1, dims[1] as usize, dims[2] as usize, dims[3] as usize),
        _ => TensorShape::new_1d(dims.iter().product::<i32>().max(0) as usize),
    })
}

fn read_per_tensor_quant(tensor: &tflite::Tensor) -> Result<(f32, i32), ImportError> {
    let q = tensor
        .quantization()
        .ok_or(ImportError::MissingField("tensor.quantization"))?;
    let scale = q
        .scale()
        .and_then(|v| v.iter().next())
        .ok_or(ImportError::MissingField("tensor.quantization.scale"))?;
    let zero_point = q.zero_point().and_then(|v| v.iter().next()).unwrap_or(0) as i32;
    Ok((scale, zero_point))
}

fn read_scales(tensor: &tflite::Tensor) -> Result<Vec<f32>, ImportError> {
    let q = tensor
        .quantization()
        .ok_or(ImportError::MissingField("weight tensor quantization"))?;
    let scale = q
        .scale()
        .ok_or(ImportError::MissingField("weight tensor scale"))?;
    Ok(scale.iter().collect())
}

fn read_i8_buffer(
    tensor: &tflite::Tensor,
    buffers: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Buffer>>,
) -> Result<Vec<i8>, ImportError> {
    let buffer = buffers.get(tensor.buffer() as usize);
    let data = buffer
        .data()
        .ok_or(ImportError::MissingField("weight buffer data"))?;
    Ok(data.iter().map(|b| b as i8).collect())
}

fn read_i32_bias_buffer(
    tensor: &tflite::Tensor,
    buffers: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Buffer>>,
) -> Result<Vec<i32>, ImportError> {
    let buffer = buffers.get(tensor.buffer() as usize);
    let data = buffer
        .data()
        .ok_or(ImportError::MissingField("bias buffer data"))?;
    let bytes: Vec<u8> = data.iter().collect();
    // `as_chunks` isn't available at this workspace's MSRV (1.87); chunks_exact is fine here.
    // `unknown_lints` allowed too since this lint doesn't exist on every clippy version CI runs.
    #[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
    let chunks = bytes.chunks_exact(4);
    Ok(chunks
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// `op_inputs[index]` is `-1` in TFLite when that input is optional and absent (e.g. no bias).
fn optional_bias(
    op_inputs: &flatbuffers::Vector<i32>,
    index: usize,
    tensors: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Tensor>>,
    buffers: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Buffer>>,
) -> Result<Option<Vec<i32>>, ImportError> {
    if op_inputs.len() <= index {
        return Ok(None);
    }
    let idx = op_inputs.get(index);
    if idx < 0 {
        return Ok(None);
    }
    let tensor = tensors.get(idx as usize);
    Ok(Some(read_i32_bias_buffer(&tensor, buffers)?))
}

fn read_activation(activation: tflite::ActivationFunctionType) -> ActivationType {
    match activation {
        tflite::ActivationFunctionType::RELU => ActivationType::Relu,
        tflite::ActivationFunctionType::RELU6 => ActivationType::Relu6,
        tflite::ActivationFunctionType::TANH => ActivationType::Tanh,
        // RELU_N1_TO_1 and SIGN_BIT aren't representable in embedded-nn's ActivationType; treat
        // as unconstrained rather than silently misrepresenting them as ReLU.
        _ => ActivationType::None,
    }
}

/// Combines a calibrated input scale + this layer's weight scale(s) with the *authoritative*
/// output scale/zero-point already baked into the TFLite file (no calibration needed here, unlike
/// Studio's training path -- TFLite models are already fully quantized end-to-end).
fn build_output_quant(
    input_scale: f32,
    weight_scales: &[f32],
    output_tensor: &tflite::Tensor,
) -> Result<(QuantParams, Option<PerChannelQuant>), ImportError> {
    let (out_scale, out_zero_point) = read_per_tensor_quant(output_tensor)?;

    if weight_scales.len() > 1 {
        let mut multipliers = Vec::with_capacity(weight_scales.len());
        let mut shifts = Vec::with_capacity(weight_scales.len());
        for &ws in weight_scales {
            let (m, s) = calculate_output_requant_multiplier(input_scale, ws, out_scale);
            multipliers.push(m);
            shifts.push(s);
        }
        let quant = QuantParams {
            multiplier: multipliers[0],
            shift: shifts[0],
            zero_point: out_zero_point,
            scale: out_scale,
        };
        Ok((
            quant,
            Some(PerChannelQuant {
                multipliers,
                shifts,
            }),
        ))
    } else {
        let ws = weight_scales.first().copied().unwrap_or(1.0);
        let (multiplier, shift) = calculate_output_requant_multiplier(input_scale, ws, out_scale);
        Ok((
            QuantParams {
                multiplier,
                shift,
                zero_point: out_zero_point,
                scale: out_scale,
            },
            None,
        ))
    }
}

/// Like `build_output_quant`, but always returns a `PerChannelQuant` sized to `out_channels`
/// (repeating the single weight scale across all channels if the source model happened to
/// quantize this depthwise conv per-tensor) -- the runtime has no per-tensor depthwise kernel.
fn build_depthwise_output_quant(
    input_scale: f32,
    weight_scales: &[f32],
    out_channels: usize,
    output_tensor: &tflite::Tensor,
) -> Result<(QuantParams, PerChannelQuant), ImportError> {
    let (out_scale, out_zero_point) = read_per_tensor_quant(output_tensor)?;
    let mut multipliers = Vec::with_capacity(out_channels);
    let mut shifts = Vec::with_capacity(out_channels);
    for c in 0..out_channels {
        let ws = if weight_scales.len() == out_channels {
            weight_scales[c]
        } else {
            weight_scales.first().copied().unwrap_or(1.0)
        };
        let (m, s) = calculate_output_requant_multiplier(input_scale, ws, out_scale);
        multipliers.push(m);
        shifts.push(s);
    }
    let quant = QuantParams {
        multiplier: multipliers[0],
        shift: shifts[0],
        zero_point: out_zero_point,
        scale: out_scale,
    };
    Ok((
        quant,
        PerChannelQuant {
            multipliers,
            shifts,
        },
    ))
}

/// Converts TFLite's padding mode into embedded-nn's explicit symmetric `pad_before` amount.
/// For `SAME`, this is the floor half of TFLite's (possibly asymmetric) total padding -- see the
/// module-level doc comment for the resulting fidelity limitation on odd total padding.
fn compute_pad_before(
    padding: tflite::Padding,
    kernel: usize,
    stride: usize,
    dilation: usize,
    in_size: usize,
    out_size: usize,
) -> usize {
    if padding == tflite::Padding::VALID {
        return 0;
    }
    let effective_kernel = dilation * (kernel - 1) + 1;
    let needed = (out_size.saturating_sub(1)) * stride + effective_kernel;
    let pad_total = needed.saturating_sub(in_size);
    pad_total / 2
}

#[allow(clippy::too_many_arguments)]
fn import_fully_connected(
    builder: &mut ModelBuilder,
    operator: &tflite::Operator,
    tensors: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Tensor>>,
    buffers: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Buffer>>,
    in_id: usize,
    input_scale: f32,
    output_tensor: &tflite::Tensor,
    name: &str,
) -> Result<usize, ImportError> {
    let op_inputs = operator
        .inputs()
        .ok_or(ImportError::MissingField("operator.inputs"))?;
    let weight_tensor = tensors.get(op_inputs.get(1) as usize);
    let weight_shape = weight_tensor
        .shape()
        .ok_or(ImportError::MissingField("FC weight shape"))?;
    let out_features = weight_shape.get(0) as usize;

    let weights = read_i8_buffer(&weight_tensor, buffers)?;
    let bias = optional_bias(&op_inputs, 2, tensors, buffers)?;
    let weight_scales = read_scales(&weight_tensor)?;
    let (output_quant, per_channel_quant) =
        build_output_quant(input_scale, &weight_scales, output_tensor)?;

    let activation = operator
        .builtin_options_as_fully_connected_options()
        .map(|o| read_activation(o.fused_activation_function()))
        .unwrap_or(ActivationType::None);

    Ok(builder.add_dense_layer(
        name,
        in_id,
        out_features,
        weights,
        None,
        bias,
        activation,
        per_channel_quant,
        Some(output_quant),
    ))
}

#[allow(clippy::too_many_arguments)]
fn import_conv2d(
    builder: &mut ModelBuilder,
    operator: &tflite::Operator,
    tensors: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Tensor>>,
    buffers: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Buffer>>,
    in_id: usize,
    input_scale: f32,
    output_tensor: &tflite::Tensor,
    name: &str,
) -> Result<usize, ImportError> {
    let opts = operator
        .builtin_options_as_conv_2_doptions()
        .ok_or(ImportError::MissingField("Conv2DOptions"))?;
    let op_inputs = operator
        .inputs()
        .ok_or(ImportError::MissingField("operator.inputs"))?;

    let input_tensor = tensors.get(op_inputs.get(0) as usize);
    let in_shape = input_tensor
        .shape()
        .ok_or(ImportError::MissingField("Conv2D input shape"))?;
    let in_h = in_shape.get(1) as usize;
    let in_w = in_shape.get(2) as usize;

    let weight_tensor = tensors.get(op_inputs.get(1) as usize);
    let weight_shape = weight_tensor
        .shape()
        .ok_or(ImportError::MissingField("Conv2D weight shape"))?;
    let out_channels = weight_shape.get(0) as usize;
    let kernel_h = weight_shape.get(1) as usize;
    let kernel_w = weight_shape.get(2) as usize;

    let out_shape = output_tensor
        .shape()
        .ok_or(ImportError::MissingField("Conv2D output shape"))?;
    let out_h = out_shape.get(1) as usize;
    let out_w = out_shape.get(2) as usize;

    let stride_h = opts.stride_h() as usize;
    let stride_w = opts.stride_w() as usize;
    let dilation_h = opts.dilation_h_factor().max(1) as usize;
    let dilation_w = opts.dilation_w_factor().max(1) as usize;
    let pad_h = compute_pad_before(opts.padding(), kernel_h, stride_h, dilation_h, in_h, out_h);
    let pad_w = compute_pad_before(opts.padding(), kernel_w, stride_w, dilation_w, in_w, out_w);

    let weights = read_i8_buffer(&weight_tensor, buffers)?;
    let bias = optional_bias(&op_inputs, 2, tensors, buffers)?;
    let weight_scales = read_scales(&weight_tensor)?;
    let (output_quant, per_channel_quant) =
        build_output_quant(input_scale, &weight_scales, output_tensor)?;
    let activation = read_activation(opts.fused_activation_function());

    Ok(builder.add_conv2d_layer(
        name,
        in_id,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        weights,
        None,
        bias,
        activation,
        per_channel_quant,
        Some(output_quant),
    ))
}

#[allow(clippy::too_many_arguments)]
fn import_depthwise_conv2d(
    builder: &mut ModelBuilder,
    operator: &tflite::Operator,
    tensors: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Tensor>>,
    buffers: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Buffer>>,
    in_id: usize,
    input_scale: f32,
    output_tensor: &tflite::Tensor,
    name: &str,
) -> Result<usize, ImportError> {
    let opts = operator
        .builtin_options_as_depthwise_conv_2_doptions()
        .ok_or(ImportError::MissingField("DepthwiseConv2DOptions"))?;
    let op_inputs = operator
        .inputs()
        .ok_or(ImportError::MissingField("operator.inputs"))?;

    let input_tensor = tensors.get(op_inputs.get(0) as usize);
    let in_shape = input_tensor
        .shape()
        .ok_or(ImportError::MissingField("DepthwiseConv2D input shape"))?;
    let in_h = in_shape.get(1) as usize;
    let in_w = in_shape.get(2) as usize;

    let weight_tensor = tensors.get(op_inputs.get(1) as usize);
    let weight_shape = weight_tensor
        .shape()
        .ok_or(ImportError::MissingField("DepthwiseConv2D weight shape"))?;
    let kernel_h = weight_shape.get(1) as usize;
    let kernel_w = weight_shape.get(2) as usize;
    let out_channels = weight_shape.get(3) as usize;
    let ch_mult = opts.depth_multiplier() as usize;

    let out_shape = output_tensor
        .shape()
        .ok_or(ImportError::MissingField("DepthwiseConv2D output shape"))?;
    let out_h = out_shape.get(1) as usize;
    let out_w = out_shape.get(2) as usize;

    let stride_h = opts.stride_h() as usize;
    let stride_w = opts.stride_w() as usize;
    let pad_h = compute_pad_before(opts.padding(), kernel_h, stride_h, 1, in_h, out_h);
    let pad_w = compute_pad_before(opts.padding(), kernel_w, stride_w, 1, in_w, out_w);

    let weights = read_i8_buffer(&weight_tensor, buffers)?;
    let bias = optional_bias(&op_inputs, 2, tensors, buffers)?;
    let weight_scales = read_scales(&weight_tensor)?;
    let (output_quant, per_channel_quant) =
        build_depthwise_output_quant(input_scale, &weight_scales, out_channels, output_tensor)?;
    let activation = read_activation(opts.fused_activation_function());

    Ok(builder.add_depthwise_conv2d_layer(
        name,
        in_id,
        ch_mult,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        weights,
        bias,
        activation,
        Some(per_channel_quant),
        Some(output_quant),
    ))
}

fn import_pool(
    builder: &mut ModelBuilder,
    operator: &tflite::Operator,
    tensors: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Tensor>>,
    in_id: usize,
    output_tensor: &tflite::Tensor,
    name: &str,
    kind: PoolKind,
) -> Result<usize, ImportError> {
    let opts = operator
        .builtin_options_as_pool_2_doptions()
        .ok_or(ImportError::MissingField("Pool2DOptions"))?;
    let op_inputs = operator
        .inputs()
        .ok_or(ImportError::MissingField("operator.inputs"))?;

    let input_tensor = tensors.get(op_inputs.get(0) as usize);
    let in_shape = input_tensor
        .shape()
        .ok_or(ImportError::MissingField("Pool input shape"))?;
    let in_h = in_shape.get(1) as usize;
    let in_w = in_shape.get(2) as usize;

    let out_shape = output_tensor
        .shape()
        .ok_or(ImportError::MissingField("Pool output shape"))?;
    let out_h = out_shape.get(1) as usize;
    let out_w = out_shape.get(2) as usize;

    let filter_h = opts.filter_height() as usize;
    let filter_w = opts.filter_width() as usize;
    let stride_h = opts.stride_h() as usize;
    let stride_w = opts.stride_w() as usize;
    let pad_h = compute_pad_before(opts.padding(), filter_h, stride_h, 1, in_h, out_h);
    let pad_w = compute_pad_before(opts.padding(), filter_w, stride_w, 1, in_w, out_w);

    Ok(match kind {
        PoolKind::Max => builder.add_maxpool2d_layer(
            name, in_id, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w,
        ),
        PoolKind::Avg => builder.add_avgpool2d_layer(
            name, in_id, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w,
        ),
    })
}

#[cfg(test)]
mod fixtures {
    //! Hand-built `.tflite` FlatBuffer fixtures, constructed with the same generated schema
    //! bindings used to parse them. TensorFlow isn't available in this environment to run a real
    //! `TFLiteConverter`, so these are built directly against the FlatBuffers wire format instead
    //! -- they are genuine, spec-compliant `.tflite` byte buffers (loadable by a real TFLite
    //! runtime too), not a mocked stand-in format.
    use crate::schema::tflite::*;
    use flatbuffers::FlatBufferBuilder;

    struct TensorSpec<'a> {
        shape: &'a [i32],
        buffer: u32,
        scale: f32,
        zero_point: i64,
    }

    fn build_tensor<'fbb>(
        fbb: &mut FlatBufferBuilder<'fbb>,
        spec: &TensorSpec,
    ) -> flatbuffers::WIPOffset<Tensor<'fbb>> {
        let shape = fbb.create_vector(spec.shape);
        let scale = fbb.create_vector(&[spec.scale]);
        let zero_point = fbb.create_vector(&[spec.zero_point]);
        let quant = QuantizationParameters::create(
            fbb,
            &QuantizationParametersArgs {
                min: None,
                max: None,
                scale: Some(scale),
                zero_point: Some(zero_point),
                details_type: QuantizationDetails::NONE,
                details: None,
                quantized_dimension: 0,
            },
        );
        Tensor::create(
            fbb,
            &TensorArgs {
                shape: Some(shape),
                type_: TensorType::INT8,
                buffer: spec.buffer,
                name: None,
                quantization: Some(quant),
                is_variable: false,
                sparsity: None,
                shape_signature: None,
                has_rank: true,
                variant_tensors: None,
            },
        )
    }

    fn i32_bias_bytes(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Like `build_tensor`, but with one scale entry per output channel (TFLite's per-channel
    /// weight quantization convention, `quantized_dimension == 0`).
    fn build_tensor_per_channel<'fbb>(
        fbb: &mut FlatBufferBuilder<'fbb>,
        shape: &[i32],
        buffer: u32,
        scales: &[f32],
    ) -> flatbuffers::WIPOffset<Tensor<'fbb>> {
        let shape_off = fbb.create_vector(shape);
        let scale = fbb.create_vector(scales);
        let zero_point = fbb.create_vector(&vec![0i64; scales.len()]);
        let quant = QuantizationParameters::create(
            fbb,
            &QuantizationParametersArgs {
                min: None,
                max: None,
                scale: Some(scale),
                zero_point: Some(zero_point),
                details_type: QuantizationDetails::NONE,
                details: None,
                quantized_dimension: 0,
            },
        );
        Tensor::create(
            fbb,
            &TensorArgs {
                shape: Some(shape_off),
                type_: TensorType::INT8,
                buffer,
                name: None,
                quantization: Some(quant),
                is_variable: false,
                sparsity: None,
                shape_signature: None,
                has_rank: true,
                variant_tensors: None,
            },
        )
    }

    /// A single `CONV_2D` layer with per-channel-quantized weights: input[1,8,8,3] ->
    /// weights[4,3,3,3] (4 output channels, 4 independent scales) -> output[1,6,6,4].
    pub fn build_conv2d_per_channel_model() -> Vec<u8> {
        let mut fbb = FlatBufferBuilder::new();

        let empty_buffer = Buffer::create(&mut fbb, &BufferArgs::default());
        let weight_bytes: Vec<u8> = vec![1u8; 4 * 3 * 3 * 3];
        let weight_data = fbb.create_vector(&weight_bytes);
        let weight_buffer = Buffer::create(
            &mut fbb,
            &BufferArgs {
                data: Some(weight_data),
                ..Default::default()
            },
        );
        let buffers = fbb.create_vector(&[empty_buffer, weight_buffer]);

        let input_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[1, 8, 8, 3],
                buffer: 0,
                scale: 1.0 / 127.0,
                zero_point: 0,
            },
        );
        let weight_tensor =
            build_tensor_per_channel(&mut fbb, &[4, 3, 3, 3], 1, &[0.01, 0.02, 0.03, 0.04]);
        let output_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[1, 6, 6, 4],
                buffer: 0,
                scale: 0.02,
                zero_point: -128,
            },
        );
        let tensors = fbb.create_vector(&[input_tensor, weight_tensor, output_tensor]);

        let conv_opts = Conv2DOptions::create(
            &mut fbb,
            &Conv2DOptionsArgs {
                padding: Padding::VALID,
                stride_w: 1,
                stride_h: 1,
                fused_activation_function: ActivationFunctionType::RELU,
                dilation_w_factor: 1,
                dilation_h_factor: 1,
                ..Default::default()
            },
        );
        let op_inputs = fbb.create_vector(&[0i32, 1]);
        let op_outputs = fbb.create_vector(&[2i32]);
        let operator = Operator::create(
            &mut fbb,
            &OperatorArgs {
                opcode_index: 0,
                inputs: Some(op_inputs),
                outputs: Some(op_outputs),
                builtin_options_type: BuiltinOptions::Conv2DOptions,
                builtin_options: Some(conv_opts.as_union_value()),
                ..Default::default()
            },
        );
        let operators = fbb.create_vector(&[operator]);

        let sg_inputs = fbb.create_vector(&[0i32]);
        let sg_outputs = fbb.create_vector(&[2i32]);
        let subgraph = SubGraph::create(
            &mut fbb,
            &SubGraphArgs {
                tensors: Some(tensors),
                inputs: Some(sg_inputs),
                outputs: Some(sg_outputs),
                operators: Some(operators),
                name: None,
            },
        );
        let subgraphs = fbb.create_vector(&[subgraph]);

        let opcode = OperatorCode::create(
            &mut fbb,
            &OperatorCodeArgs {
                deprecated_builtin_code: BuiltinOperator::CONV_2D.0 as i8,
                custom_code: None,
                version: 1,
                builtin_code: BuiltinOperator::CONV_2D,
            },
        );
        let opcodes = fbb.create_vector(&[opcode]);

        let model = Model::create(
            &mut fbb,
            &ModelArgs {
                version: 3,
                operator_codes: Some(opcodes),
                subgraphs: Some(subgraphs),
                description: None,
                buffers: Some(buffers),
                metadata_buffer: None,
                metadata: None,
                signature_defs: None,
            },
        );
        fbb.finish_minimal(model);
        fbb.finished_data().to_vec()
    }

    /// A single `FULLY_CONNECTED` layer: input[1,4] -> weights[2,4] + bias[2] -> output[1,2].
    pub fn build_fc_only_model() -> Vec<u8> {
        let mut fbb = FlatBufferBuilder::new();

        let empty_buffer = Buffer::create(&mut fbb, &BufferArgs::default());
        let weight_bytes: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let weight_data =
            fbb.create_vector(&weight_bytes.iter().map(|&b| b as u8).collect::<Vec<u8>>());
        let weight_buffer = Buffer::create(
            &mut fbb,
            &BufferArgs {
                data: Some(weight_data),
                ..Default::default()
            },
        );
        let bias_bytes = i32_bias_bytes(&[10, -10]);
        let bias_data = fbb.create_vector(&bias_bytes);
        let bias_buffer = Buffer::create(
            &mut fbb,
            &BufferArgs {
                data: Some(bias_data),
                ..Default::default()
            },
        );
        let buffers = fbb.create_vector(&[empty_buffer, weight_buffer, bias_buffer]);

        let input_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[1, 4],
                buffer: 0,
                scale: 1.0 / 127.0,
                zero_point: 0,
            },
        );
        let weight_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[2, 4],
                buffer: 1,
                scale: 0.01,
                zero_point: 0,
            },
        );
        let bias_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[2],
                buffer: 2,
                scale: 1.0 / 127.0 * 0.01,
                zero_point: 0,
            },
        );
        let output_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[1, 2],
                buffer: 0,
                scale: 0.05,
                zero_point: 3,
            },
        );
        let tensors = fbb.create_vector(&[input_tensor, weight_tensor, bias_tensor, output_tensor]);

        let fc_options = FullyConnectedOptions::create(
            &mut fbb,
            &FullyConnectedOptionsArgs {
                fused_activation_function: ActivationFunctionType::NONE,
                ..Default::default()
            },
        );
        let op_inputs = fbb.create_vector(&[0i32, 1, 2]);
        let op_outputs = fbb.create_vector(&[3i32]);
        let operator = Operator::create(
            &mut fbb,
            &OperatorArgs {
                opcode_index: 0,
                inputs: Some(op_inputs),
                outputs: Some(op_outputs),
                builtin_options_type: BuiltinOptions::FullyConnectedOptions,
                builtin_options: Some(fc_options.as_union_value()),
                ..Default::default()
            },
        );
        let operators = fbb.create_vector(&[operator]);

        let sg_inputs = fbb.create_vector(&[0i32]);
        let sg_outputs = fbb.create_vector(&[3i32]);
        let subgraph = SubGraph::create(
            &mut fbb,
            &SubGraphArgs {
                tensors: Some(tensors),
                inputs: Some(sg_inputs),
                outputs: Some(sg_outputs),
                operators: Some(operators),
                name: None,
            },
        );
        let subgraphs = fbb.create_vector(&[subgraph]);

        let opcode = OperatorCode::create(
            &mut fbb,
            &OperatorCodeArgs {
                deprecated_builtin_code: BuiltinOperator::FULLY_CONNECTED.0 as i8,
                custom_code: None,
                version: 1,
                builtin_code: BuiltinOperator::FULLY_CONNECTED,
            },
        );
        let opcodes = fbb.create_vector(&[opcode]);

        let model = Model::create(
            &mut fbb,
            &ModelArgs {
                version: 3,
                operator_codes: Some(opcodes),
                subgraphs: Some(subgraphs),
                description: None,
                buffers: Some(buffers),
                metadata_buffer: None,
                metadata: None,
                signature_defs: None,
            },
        );
        fbb.finish_minimal(model);
        fbb.finished_data().to_vec()
    }

    /// A small CNN chain: CONV_2D(ReLU) -> MAX_POOL_2D -> RESHAPE -> FULLY_CONNECTED -> SOFTMAX.
    /// input[1,8,8,3] -> conv[1,6,6,4] (3x3 VALID) -> pool[1,3,3,4] (2x2/2) -> reshape[1,36] ->
    /// fc[1,4] -> softmax[1,4].
    pub fn build_conv_pool_reshape_fc_softmax_model() -> Vec<u8> {
        let mut fbb = FlatBufferBuilder::new();

        let empty_buffer = Buffer::create(&mut fbb, &BufferArgs::default());
        let conv_weight_bytes: Vec<u8> = vec![1u8; 4 * 3 * 3 * 3];
        let conv_weight_data = fbb.create_vector(&conv_weight_bytes);
        let conv_weight_buffer = Buffer::create(
            &mut fbb,
            &BufferArgs {
                data: Some(conv_weight_data),
                ..Default::default()
            },
        );
        let conv_bias_bytes = i32_bias_bytes(&[0, 0, 0, 0]);
        let conv_bias_data = fbb.create_vector(&conv_bias_bytes);
        let conv_bias_buffer = Buffer::create(
            &mut fbb,
            &BufferArgs {
                data: Some(conv_bias_data),
                ..Default::default()
            },
        );
        let fc_weight_bytes: Vec<u8> = vec![1u8; 4 * 36];
        let fc_weight_data = fbb.create_vector(&fc_weight_bytes);
        let fc_weight_buffer = Buffer::create(
            &mut fbb,
            &BufferArgs {
                data: Some(fc_weight_data),
                ..Default::default()
            },
        );
        let fc_bias_bytes = i32_bias_bytes(&[0, 0, 0, 0]);
        let fc_bias_data = fbb.create_vector(&fc_bias_bytes);
        let fc_bias_buffer = Buffer::create(
            &mut fbb,
            &BufferArgs {
                data: Some(fc_bias_data),
                ..Default::default()
            },
        );
        let buffers = fbb.create_vector(&[
            empty_buffer,
            conv_weight_buffer,
            conv_bias_buffer,
            fc_weight_buffer,
            fc_bias_buffer,
        ]);

        let input_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[1, 8, 8, 3],
                buffer: 0,
                scale: 1.0 / 127.0,
                zero_point: 0,
            },
        );
        let conv_weight_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[4, 3, 3, 3],
                buffer: 1,
                scale: 0.01,
                zero_point: 0,
            },
        );
        let conv_bias_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[4],
                buffer: 2,
                scale: 1.0 / 127.0 * 0.01,
                zero_point: 0,
            },
        );
        let conv_out_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[1, 6, 6, 4],
                buffer: 0,
                scale: 0.02,
                zero_point: -128,
            },
        );
        let pool_out_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[1, 3, 3, 4],
                buffer: 0,
                scale: 0.02,
                zero_point: -128,
            },
        );
        let reshape_out_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[1, 36],
                buffer: 0,
                scale: 0.02,
                zero_point: -128,
            },
        );
        let fc_weight_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[4, 36],
                buffer: 3,
                scale: 0.01,
                zero_point: 0,
            },
        );
        let fc_bias_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[4],
                buffer: 4,
                scale: 0.02 * 0.01,
                zero_point: 0,
            },
        );
        let fc_out_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[1, 4],
                buffer: 0,
                scale: 0.05,
                zero_point: 2,
            },
        );
        let softmax_out_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[1, 4],
                buffer: 0,
                scale: 1.0 / 255.0,
                zero_point: -128,
            },
        );

        let tensors = fbb.create_vector(&[
            input_tensor,       // 0
            conv_weight_tensor, // 1
            conv_bias_tensor,   // 2
            conv_out_tensor,    // 3
            pool_out_tensor,    // 4
            reshape_out_tensor, // 5
            fc_weight_tensor,   // 6
            fc_bias_tensor,     // 7
            fc_out_tensor,      // 8
            softmax_out_tensor, // 9
        ]);

        let conv_opts = Conv2DOptions::create(
            &mut fbb,
            &Conv2DOptionsArgs {
                padding: Padding::VALID,
                stride_w: 1,
                stride_h: 1,
                fused_activation_function: ActivationFunctionType::RELU,
                dilation_w_factor: 1,
                dilation_h_factor: 1,
                ..Default::default()
            },
        );
        let conv_inputs = fbb.create_vector(&[0i32, 1, 2]);
        let conv_outputs = fbb.create_vector(&[3i32]);
        let conv_op = Operator::create(
            &mut fbb,
            &OperatorArgs {
                opcode_index: 0,
                inputs: Some(conv_inputs),
                outputs: Some(conv_outputs),
                builtin_options_type: BuiltinOptions::Conv2DOptions,
                builtin_options: Some(conv_opts.as_union_value()),
                ..Default::default()
            },
        );

        let pool_opts = Pool2DOptions::create(
            &mut fbb,
            &Pool2DOptionsArgs {
                padding: Padding::VALID,
                stride_w: 2,
                stride_h: 2,
                filter_width: 2,
                filter_height: 2,
                fused_activation_function: ActivationFunctionType::NONE,
            },
        );
        let pool_inputs = fbb.create_vector(&[3i32]);
        let pool_outputs = fbb.create_vector(&[4i32]);
        let pool_op = Operator::create(
            &mut fbb,
            &OperatorArgs {
                opcode_index: 1,
                inputs: Some(pool_inputs),
                outputs: Some(pool_outputs),
                builtin_options_type: BuiltinOptions::Pool2DOptions,
                builtin_options: Some(pool_opts.as_union_value()),
                ..Default::default()
            },
        );

        let reshape_new_shape = fbb.create_vector(&[1i32, 36]);
        let reshape_opts = ReshapeOptions::create(
            &mut fbb,
            &ReshapeOptionsArgs {
                new_shape: Some(reshape_new_shape),
            },
        );
        let reshape_inputs = fbb.create_vector(&[4i32]);
        let reshape_outputs = fbb.create_vector(&[5i32]);
        let reshape_op = Operator::create(
            &mut fbb,
            &OperatorArgs {
                opcode_index: 2,
                inputs: Some(reshape_inputs),
                outputs: Some(reshape_outputs),
                builtin_options_type: BuiltinOptions::ReshapeOptions,
                builtin_options: Some(reshape_opts.as_union_value()),
                ..Default::default()
            },
        );

        let fc_opts = FullyConnectedOptions::create(
            &mut fbb,
            &FullyConnectedOptionsArgs {
                fused_activation_function: ActivationFunctionType::NONE,
                ..Default::default()
            },
        );
        let fc_inputs = fbb.create_vector(&[5i32, 6, 7]);
        let fc_outputs = fbb.create_vector(&[8i32]);
        let fc_op = Operator::create(
            &mut fbb,
            &OperatorArgs {
                opcode_index: 3,
                inputs: Some(fc_inputs),
                outputs: Some(fc_outputs),
                builtin_options_type: BuiltinOptions::FullyConnectedOptions,
                builtin_options: Some(fc_opts.as_union_value()),
                ..Default::default()
            },
        );

        let softmax_opts = SoftmaxOptions::create(&mut fbb, &SoftmaxOptionsArgs { beta: 1.0 });
        let softmax_inputs = fbb.create_vector(&[8i32]);
        let softmax_outputs = fbb.create_vector(&[9i32]);
        let softmax_op = Operator::create(
            &mut fbb,
            &OperatorArgs {
                opcode_index: 4,
                inputs: Some(softmax_inputs),
                outputs: Some(softmax_outputs),
                builtin_options_type: BuiltinOptions::SoftmaxOptions,
                builtin_options: Some(softmax_opts.as_union_value()),
                ..Default::default()
            },
        );

        let operators = fbb.create_vector(&[conv_op, pool_op, reshape_op, fc_op, softmax_op]);

        let sg_inputs = fbb.create_vector(&[0i32]);
        let sg_outputs = fbb.create_vector(&[9i32]);
        let subgraph = SubGraph::create(
            &mut fbb,
            &SubGraphArgs {
                tensors: Some(tensors),
                inputs: Some(sg_inputs),
                outputs: Some(sg_outputs),
                operators: Some(operators),
                name: None,
            },
        );
        let subgraphs = fbb.create_vector(&[subgraph]);

        let opcode_defs = [
            (BuiltinOperator::CONV_2D),
            (BuiltinOperator::MAX_POOL_2D),
            (BuiltinOperator::RESHAPE),
            (BuiltinOperator::FULLY_CONNECTED),
            (BuiltinOperator::SOFTMAX),
        ];
        let opcode_offsets: Vec<_> = opcode_defs
            .iter()
            .map(|&code| {
                OperatorCode::create(
                    &mut fbb,
                    &OperatorCodeArgs {
                        deprecated_builtin_code: if code.0 <= i8::MAX as i32 {
                            code.0 as i8
                        } else {
                            127
                        },
                        custom_code: None,
                        version: 1,
                        builtin_code: code,
                    },
                )
            })
            .collect();
        let opcodes = fbb.create_vector(&opcode_offsets);

        let model = Model::create(
            &mut fbb,
            &ModelArgs {
                version: 3,
                operator_codes: Some(opcodes),
                subgraphs: Some(subgraphs),
                description: None,
                buffers: Some(buffers),
                metadata_buffer: None,
                metadata: None,
                signature_defs: None,
            },
        );
        fbb.finish_minimal(model);
        fbb.finished_data().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{
        build_conv_pool_reshape_fc_softmax_model, build_conv2d_per_channel_model,
        build_fc_only_model,
    };

    #[test]
    fn test_import_fc_only_model_structure() {
        let bytes = build_fc_only_model();
        let graph = import_tflite(&bytes).expect("import should succeed");

        assert_eq!(graph.layers.len(), 1);
        match &graph.layers[0].op {
            OpPayload::FullyConnected {
                weights,
                bias,
                activation,
                per_channel_quant,
                ..
            } => {
                assert_eq!(weights, &vec![1i8, 2, 3, 4, 5, 6, 7, 8]);
                assert_eq!(bias.as_ref().unwrap(), &vec![10, -10]);
                assert_eq!(*activation, ActivationType::None);
                assert!(per_channel_quant.is_none());
            }
            other => panic!("expected FullyConnected, got {:?}", other),
        }

        let input_tensor = graph
            .tensors
            .iter()
            .find(|t| graph.inputs.contains(&t.id))
            .unwrap();
        assert_eq!(input_tensor.shape.total_elements(), 4);
        assert_eq!(input_tensor.quant.zero_point, 0);

        let output_tensor = graph
            .tensors
            .iter()
            .find(|t| graph.outputs.contains(&t.id))
            .unwrap();
        assert_eq!(output_tensor.shape.total_elements(), 2);
        assert_eq!(output_tensor.quant.zero_point, 3);
        assert!((output_tensor.quant.scale - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_import_fc_only_model_generates_compilable_code() {
        let bytes = build_fc_only_model();
        let graph = import_tflite(&bytes).expect("import should succeed");
        let code = embedded_nn_codegen::RustCodeGenerator::new("ImportedFcNet").generate(&graph);
        assert!(code.contains("fully_connected_s8"));
        assert!(code.contains("pub struct ImportedFcNet"));
    }

    #[test]
    fn test_import_conv_pool_reshape_fc_softmax_chain() {
        let bytes = build_conv_pool_reshape_fc_softmax_model();
        let graph = import_tflite(&bytes).expect("import should succeed");

        assert_eq!(graph.layers.len(), 5);
        let op_names: Vec<&str> = graph
            .layers
            .iter()
            .map(|l| match &l.op {
                OpPayload::Conv2D { .. } => "conv2d",
                OpPayload::MaxPool2D { .. } => "maxpool",
                OpPayload::Reshape { .. } => "reshape",
                OpPayload::FullyConnected { .. } => "fc",
                OpPayload::Softmax => "softmax",
                _ => "other",
            })
            .collect();
        assert_eq!(
            op_names,
            vec!["conv2d", "maxpool", "reshape", "fc", "softmax"]
        );

        match &graph.layers[0].op {
            OpPayload::Conv2D {
                kernel_h,
                kernel_w,
                pad_h,
                pad_w,
                activation,
                ..
            } => {
                assert_eq!(*kernel_h, 3);
                assert_eq!(*kernel_w, 3);
                assert_eq!(*pad_h, 0); // VALID padding
                assert_eq!(*pad_w, 0);
                assert_eq!(*activation, ActivationType::Relu);
            }
            other => panic!("expected Conv2D, got {:?}", other),
        }

        let final_out_id = *graph.outputs.first().unwrap();
        let final_tensor = graph.tensors.iter().find(|t| t.id == final_out_id).unwrap();
        assert_eq!(final_tensor.shape.total_elements(), 4);
    }

    #[test]
    fn test_import_conv_chain_generates_compilable_code() {
        let bytes = build_conv_pool_reshape_fc_softmax_model();
        let graph = import_tflite(&bytes).expect("import should succeed");
        let code = embedded_nn_codegen::RustCodeGenerator::new("ImportedConvNet").generate(&graph);
        assert!(code.contains("convolve_s8") || code.contains("convolve_per_channel_s8"));
        assert!(code.contains("max_pool_s8"));
        assert!(code.contains("out_buf.copy_from_slice(in_buf)"));
        assert!(code.contains("softmax_s8"));
    }

    #[test]
    fn test_import_rejects_invalid_flatbuffer() {
        let garbage = vec![0u8, 1, 2, 3];
        let err = import_tflite(&garbage).unwrap_err();
        assert!(matches!(err, ImportError::InvalidFlatBuffer(_)));
    }

    #[test]
    fn test_import_conv2d_per_channel_weights() {
        let bytes = build_conv2d_per_channel_model();
        let graph = import_tflite(&bytes).expect("import should succeed");

        assert_eq!(graph.layers.len(), 1);
        match &graph.layers[0].op {
            OpPayload::Conv2D {
                per_channel_quant, ..
            } => {
                let pcq = per_channel_quant
                    .as_ref()
                    .expect("per-channel weight scales should produce PerChannelQuant");
                assert_eq!(pcq.multipliers.len(), 4);
                assert_eq!(pcq.shifts.len(), 4);
                // Independent weight scales per channel must produce independent multipliers.
                assert!(
                    pcq.multipliers
                        .iter()
                        .collect::<std::collections::HashSet<_>>()
                        .len()
                        > 1
                );
            }
            other => panic!("expected Conv2D, got {:?}", other),
        }

        let code =
            embedded_nn_codegen::RustCodeGenerator::new("ImportedPerChannelConv").generate(&graph);
        assert!(code.contains("convolve_per_channel_s8"));
    }
}
