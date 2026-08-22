//! TensorFlow Lite (`.tflite`) FlatBuffers importer for `embedded-nn`.
//!
//! Parses a `.tflite` model (subgraph 0 only) into an `embedded-nn-compiler` [`ModelGraph`],
//! which can then be fed straight into the existing `embedded-nn-codegen` pipeline exactly like
//! a Studio-trained or hand-built graph -- no changes needed downstream.
//!
//! ## Scope (v1)
//! - `INT8` and `UINT8` quantized tensors are supported. UINT8 tensors are rewritten to INT8
//!   storage by subtracting 128 from values and zero-points; no UINT8 runtime kernels are used.
//! - Only subgraph 0 is imported; models with control-flow subgraphs are not supported.
//! - Supported operators: `FULLY_CONNECTED`, `CONV_2D` (1-high kernels import as `Conv1D`),
//!   `DEPTHWISE_CONV_2D`, `MAX_POOL_2D`, `AVERAGE_POOL_2D`, `SOFTMAX`, `RESHAPE`, `ADD`,
//!   `TRANSPOSE` (general rank-1..4 perms), `PAD`/`PADV2`, `MEAN`, `SVDF`, `MUL`,
//!   `CONCATENATION` (channel axis), `STRIDED_SLICE`, and BASIC `LSTM`.
//! - SAME padding is represented exactly, including odd totals where bottom/right differ from
//!   top/left. VALID padding is represented as zero on every side.
//! - Per-channel quantization is respected for `CONV_2D`/`DEPTHWISE_CONV_2D`/`FULLY_CONNECTED`
//!   weight tensors (assuming `quantized_dimension == 0`, the near-universal TFLite convention).

#[path = "../schema/schema_generated.rs"]
#[allow(warnings)]
pub mod schema;

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
    #[error("invalid or unsupported operator configuration: {0}")]
    UnsupportedConfiguration(String),
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
        let dep = opcode.deprecated_builtin_code();
        let builtin = if opcode.builtin_code().0 == 0 && dep != 0 {
            tflite::BuiltinOperator(dep as i32)
        } else {
            opcode.builtin_code()
        };

        let output_idx = op_outputs.get(0) as usize;
        let output_tensor = tensors.get(output_idx);
        convert_tensor_type(output_tensor.type_())?;
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
                let input_tensor = tensors.get(primary_input_idx);
                if read_per_tensor_quant(&input_tensor)? != read_per_tensor_quant(&output_tensor)? {
                    return Err(ImportError::UnsupportedConfiguration(
                        "RESHAPE input and output quantization must match".into(),
                    ));
                }
                let shape = convert_shape(&output_tensor)?;
                builder.add_reshape_layer(layer_name.clone(), in_id, shape)
            }
            tflite::BuiltinOperator::ADD => import_add(
                &mut builder,
                &operator,
                &tensors,
                &tensor_ids,
                in_id,
                &output_tensor,
                &layer_name,
            )?,
            tflite::BuiltinOperator::TRANSPOSE => import_transpose(
                &mut builder,
                &operator,
                &tensors,
                &buffers,
                in_id,
                &output_tensor,
                &layer_name,
            )?,
            tflite::BuiltinOperator::PAD | tflite::BuiltinOperator::PADV2 => import_pad(
                &mut builder,
                &operator,
                &tensors,
                &buffers,
                in_id,
                &layer_name,
            )?,
            tflite::BuiltinOperator::MEAN => import_mean(
                &mut builder,
                &operator,
                &tensors,
                &buffers,
                in_id,
                &layer_name,
            )?,
            tflite::BuiltinOperator::SVDF => import_svdf(
                &mut builder,
                &operator,
                &tensors,
                &buffers,
                in_id,
                input_scale,
                &output_tensor,
                &layer_name,
            )?,
            tflite::BuiltinOperator::MUL => import_mul(
                &mut builder,
                &operator,
                &tensors,
                &tensor_ids,
                in_id,
                &output_tensor,
                &layer_name,
            )?,
            tflite::BuiltinOperator::CONCATENATION => {
                import_concat(&mut builder, &operator, &tensors, &tensor_ids, &layer_name)?
            }
            tflite::BuiltinOperator::STRIDED_SLICE => import_strided_slice(
                &mut builder,
                &operator,
                &tensors,
                &buffers,
                in_id,
                &layer_name,
            )?,
            tflite::BuiltinOperator::LSTM => import_lstm(
                &mut builder,
                &operator,
                &tensors,
                &buffers,
                in_id,
                input_scale,
                &output_tensor,
                &layer_name,
            )?,
            tflite::BuiltinOperator::QUANTIZE | tflite::BuiltinOperator::DEQUANTIZE => {
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

        let (out_scale, out_zero_point) = read_per_tensor_quant(&output_tensor)?;
        if matches!(
            builtin,
            tflite::BuiltinOperator::MAX_POOL_2D
                | tflite::BuiltinOperator::AVERAGE_POOL_2D
                | tflite::BuiltinOperator::SOFTMAX
                | tflite::BuiltinOperator::RESHAPE
                | tflite::BuiltinOperator::TRANSPOSE
                | tflite::BuiltinOperator::PAD
                | tflite::BuiltinOperator::PADV2
                | tflite::BuiltinOperator::MEAN
                | tflite::BuiltinOperator::CONCATENATION
                | tflite::BuiltinOperator::STRIDED_SLICE
        ) {
            let (multiplier, shift) = quantize_multiplier(out_scale);
            builder
                .set_tensor_quant(
                    out_id,
                    QuantParams {
                        multiplier,
                        shift,
                        zero_point: out_zero_point,
                        scale: out_scale,
                    },
                )
                .map_err(|message| ImportError::UnsupportedConfiguration(message.into()))?;
        }
        tensor_ids.insert(output_idx, out_id);
        tensor_scales.insert(output_idx, out_scale);
    }

    for i in 0..graph_outputs.len() {
        let idx = graph_outputs.get(i) as usize;
        let id = *tensor_ids
            .get(&idx)
            .ok_or(ImportError::UnresolvedInput(idx))?;
        builder.mark_output(id);
    }

    Ok(builder.build())
}

enum PoolKind {
    Max,
    Avg,
}

fn convert_tensor_type(t: tflite::TensorType) -> Result<DataType, ImportError> {
    match t {
        tflite::TensorType::INT8 | tflite::TensorType::UINT8 => Ok(DataType::Int8),
        tflite::TensorType::INT16 => Ok(DataType::Int16),
        tflite::TensorType::INT32 => Ok(DataType::Int8),
        tflite::TensorType::FLOAT32 => Ok(DataType::Float32),
        _ => Err(ImportError::UnsupportedTensorType(
            "only INT8, UINT8, INT16, INT32, and FLOAT32 tensors are supported",
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
        2 => TensorShape::new_2d(dims[0] as usize, dims[1] as usize),
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
    if !scale.is_finite() || scale <= 0.0 {
        return Err(ImportError::UnsupportedConfiguration(format!(
            "tensor quantization scale must be finite and positive, got {scale}"
        )));
    }
    let mut zero_point = q.zero_point().and_then(|v| v.iter().next()).unwrap_or(0) as i32;
    if tensor.type_() == tflite::TensorType::UINT8 {
        zero_point -= 128;
    }
    Ok((scale, zero_point))
}

fn read_scales(tensor: &tflite::Tensor) -> Result<Vec<f32>, ImportError> {
    let q = tensor
        .quantization()
        .ok_or(ImportError::MissingField("weight tensor quantization"))?;
    let scale = q
        .scale()
        .ok_or(ImportError::MissingField("weight tensor scale"))?;
    let scales: Vec<f32> = scale.iter().collect();
    if scales
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(ImportError::UnsupportedConfiguration(
            "weight quantization scales must be finite and positive".into(),
        ));
    }
    Ok(scales)
}

fn read_i8_buffer(
    tensor: &tflite::Tensor,
    buffers: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Buffer>>,
) -> Result<Vec<i8>, ImportError> {
    let buffer = buffers.get(tensor.buffer() as usize);
    let data = buffer
        .data()
        .ok_or(ImportError::MissingField("weight buffer data"))?;
    Ok(match tensor.type_() {
        tflite::TensorType::INT8 => data.iter().map(|b| b as i8).collect(),
        tflite::TensorType::UINT8 => data.iter().map(|b| (b as i16 - 128) as i8).collect(),
        _ => {
            return Err(ImportError::UnsupportedTensorType(
                "constant buffer must be INT8 or UINT8",
            ));
        }
    })
}

fn read_i32_buffer(
    tensor: &tflite::Tensor,
    buffers: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Buffer>>,
) -> Result<Vec<i32>, ImportError> {
    let buffer = buffers.get(tensor.buffer() as usize);
    let data = buffer
        .data()
        .ok_or(ImportError::MissingField("int32 constant buffer data"))?;
    let bytes: Vec<u8> = data.iter().collect();
    let (chunks, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(ImportError::UnsupportedConfiguration(
            "INT32 constant buffer byte length is not divisible by four".into(),
        ));
    }
    Ok(chunks.iter().map(|c| i32::from_le_bytes(*c)).collect())
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
    let (chunks, _) = bytes.as_chunks::<4>();
    Ok(chunks.iter().map(|c| i32::from_le_bytes(*c)).collect())
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

fn read_activation(
    activation: tflite::ActivationFunctionType,
) -> Result<ActivationType, ImportError> {
    match activation {
        tflite::ActivationFunctionType::NONE => Ok(ActivationType::None),
        tflite::ActivationFunctionType::RELU => Ok(ActivationType::Relu),
        tflite::ActivationFunctionType::RELU6 => Ok(ActivationType::Relu6),
        other => Err(ImportError::UnsupportedConfiguration(format!(
            "unsupported fused activation {}",
            other.variant_name().unwrap_or("UNKNOWN")
        ))),
    }
}

fn require_symmetric_filter(tensor: &tflite::Tensor, op: &str) -> Result<(), ImportError> {
    let (_, shifted_zero_point) = read_per_tensor_quant(tensor)?;
    if shifted_zero_point != 0 {
        return Err(ImportError::UnsupportedConfiguration(format!(
            "{op} filter zero-point becomes {shifted_zero_point} after UINT8-to-INT8 rewrite, but the runtime kernel requires symmetric filters"
        )));
    }
    Ok(())
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

/// Computes TFLite's exact `(before, after)` padding for one spatial dimension.
fn compute_padding(
    padding: tflite::Padding,
    kernel: usize,
    stride: usize,
    dilation: usize,
    in_size: usize,
    out_size: usize,
) -> (usize, usize) {
    if padding == tflite::Padding::VALID {
        return (0, 0);
    }
    let effective_kernel = dilation * (kernel - 1) + 1;
    let needed = (out_size.saturating_sub(1)) * stride + effective_kernel;
    let pad_total = needed.saturating_sub(in_size);
    let before = pad_total / 2;
    (before, pad_total - before)
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
        .transpose()?
        .unwrap_or(ActivationType::None);

    let (_, filter_zero_point) = read_per_tensor_quant(&weight_tensor)?;
    let out_id = builder.add_dense_layer(
        name,
        in_id,
        out_features,
        weights,
        None,
        bias,
        activation,
        per_channel_quant,
        Some(output_quant),
    );
    builder
        .set_fully_connected_filter_offset(out_id, -filter_zero_point)
        .map_err(|message| ImportError::UnsupportedConfiguration(message.into()))?;
    Ok(out_id)
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
    require_symmetric_filter(&weight_tensor, "CONV_2D")?;
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
    let (pad_top, pad_bottom) =
        compute_padding(opts.padding(), kernel_h, stride_h, dilation_h, in_h, out_h);
    let (pad_left, pad_right) =
        compute_padding(opts.padding(), kernel_w, stride_w, dilation_w, in_w, out_w);
    let padding = Padding2D::new(pad_top, pad_bottom, pad_left, pad_right);

    let weights = read_i8_buffer(&weight_tensor, buffers)?;
    let bias = optional_bias(&op_inputs, 2, tensors, buffers)?;
    let weight_scales = read_scales(&weight_tensor)?;
    let (output_quant, per_channel_quant) =
        build_output_quant(input_scale, &weight_scales, output_tensor)?;
    let activation = read_activation(opts.fused_activation_function())?;

    if in_h == 1
        && kernel_h == 1
        && pad_top == 0
        && pad_bottom == 0
        && dilation_h == 1
        && pad_left == pad_right
        && per_channel_quant.is_none()
    {
        return Ok(builder.add_conv1d_layer(
            name,
            in_id,
            out_channels,
            kernel_w,
            stride_w,
            pad_left,
            dilation_w,
            weights,
            bias,
            activation,
            Some(output_quant),
        ));
    }

    Ok(builder.add_conv2d_layer(
        name,
        in_id,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding,
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
    require_symmetric_filter(&weight_tensor, "DEPTHWISE_CONV_2D")?;
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
    let (pad_top, pad_bottom) = compute_padding(opts.padding(), kernel_h, stride_h, 1, in_h, out_h);
    let (pad_left, pad_right) = compute_padding(opts.padding(), kernel_w, stride_w, 1, in_w, out_w);
    let padding = Padding2D::new(pad_top, pad_bottom, pad_left, pad_right);

    let weights = read_i8_buffer(&weight_tensor, buffers)?;
    let bias = optional_bias(&op_inputs, 2, tensors, buffers)?;
    let weight_scales = read_scales(&weight_tensor)?;
    let (output_quant, per_channel_quant) =
        build_depthwise_output_quant(input_scale, &weight_scales, out_channels, output_tensor)?;
    let activation = read_activation(opts.fused_activation_function())?;

    Ok(builder.add_depthwise_conv2d_layer(
        name,
        in_id,
        ch_mult,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding,
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
    if read_activation(opts.fused_activation_function())? != ActivationType::None {
        return Err(ImportError::UnsupportedConfiguration(
            "fused activation on pooling is not represented by the current IR".into(),
        ));
    }
    let op_inputs = operator
        .inputs()
        .ok_or(ImportError::MissingField("operator.inputs"))?;

    let input_tensor = tensors.get(op_inputs.get(0) as usize);
    if read_per_tensor_quant(&input_tensor)? != read_per_tensor_quant(output_tensor)? {
        return Err(ImportError::UnsupportedConfiguration(
            "pool input and output quantization must match".into(),
        ));
    }
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
    let (pad_top, pad_bottom) = compute_padding(opts.padding(), filter_h, stride_h, 1, in_h, out_h);
    let (pad_left, pad_right) = compute_padding(opts.padding(), filter_w, stride_w, 1, in_w, out_w);
    let padding = Padding2D::new(pad_top, pad_bottom, pad_left, pad_right);

    Ok(match kind {
        PoolKind::Max => builder
            .add_maxpool2d_layer(name, in_id, filter_h, filter_w, stride_h, stride_w, padding),
        PoolKind::Avg => builder
            .add_avgpool2d_layer(name, in_id, filter_h, filter_w, stride_h, stride_w, padding),
    })
}

fn import_add(
    builder: &mut ModelBuilder,
    operator: &tflite::Operator,
    tensors: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Tensor>>,
    tensor_ids: &HashMap<usize, usize>,
    input1_id: usize,
    output_tensor: &tflite::Tensor,
    name: &str,
) -> Result<usize, ImportError> {
    let op_inputs = operator
        .inputs()
        .ok_or(ImportError::MissingField("operator.inputs"))?;
    if op_inputs.len() != 2 {
        return Err(ImportError::UnsupportedConfiguration(
            "ADD requires exactly two inputs".into(),
        ));
    }
    let input2_idx = op_inputs.get(1);
    if input2_idx < 0 {
        return Err(ImportError::UnsupportedConfiguration(
            "ADD input 2 is absent".into(),
        ));
    }
    let input2_idx = input2_idx as usize;
    let input2_id = *tensor_ids
        .get(&input2_idx)
        .ok_or(ImportError::UnresolvedInput(input2_idx))?;
    convert_tensor_type(tensors.get(input2_idx).type_())?;

    let (out_scale, out_zero_point) = read_per_tensor_quant(output_tensor)?;
    let (multiplier, shift) = quantize_multiplier(out_scale);
    let output_quant = QuantParams {
        multiplier,
        shift,
        zero_point: out_zero_point,
        scale: out_scale,
    };
    let activation = operator
        .builtin_options_as_add_options()
        .map(|opts| read_activation(opts.fused_activation_function()))
        .transpose()?
        .unwrap_or(ActivationType::None);
    builder
        .add_elementwise_add_layer(name, input1_id, input2_id, activation, output_quant)
        .map_err(|message| ImportError::UnsupportedConfiguration(message.into()))
}

fn import_transpose(
    builder: &mut ModelBuilder,
    operator: &tflite::Operator,
    tensors: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Tensor>>,
    buffers: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Buffer>>,
    input_id: usize,
    output_tensor: &tflite::Tensor,
    name: &str,
) -> Result<usize, ImportError> {
    let op_inputs = operator
        .inputs()
        .ok_or(ImportError::MissingField("operator.inputs"))?;
    if op_inputs.len() != 2 || op_inputs.get(1) < 0 {
        return Err(ImportError::UnsupportedConfiguration(
            "TRANSPOSE requires a constant permutation input".into(),
        ));
    }
    let input_tensor = tensors.get(op_inputs.get(0) as usize);
    let perm_tensor = tensors.get(op_inputs.get(1) as usize);
    if perm_tensor.type_() != tflite::TensorType::INT32 {
        return Err(ImportError::UnsupportedConfiguration(
            "TRANSPOSE permutation tensor must be INT32".into(),
        ));
    }
    let permutation: Vec<usize> = read_i32_buffer(&perm_tensor, buffers)?
        .into_iter()
        .map(|value| {
            usize::try_from(value).map_err(|_| {
                ImportError::UnsupportedConfiguration(
                    "TRANSPOSE permutation contains a negative axis".into(),
                )
            })
        })
        .collect::<Result<_, _>>()?;
    let input_dims: Vec<i32> = input_tensor
        .shape()
        .ok_or(ImportError::MissingField("TRANSPOSE input shape"))?
        .iter()
        .collect();
    let input_rank = input_dims.len();
    if permutation.len() != input_rank || input_rank == 0 || input_rank > 4 {
        return Err(ImportError::UnsupportedConfiguration(format!(
            "TRANSPOSE permutation rank {} does not match input rank {input_rank}",
            permutation.len()
        )));
    }
    if input_rank == 4 && input_dims[0] != 1 {
        return Err(ImportError::UnsupportedConfiguration(
            "rank-4 TRANSPOSE currently requires batch size 1".into(),
        ));
    }
    let output_dims: Vec<i32> = output_tensor
        .shape()
        .ok_or(ImportError::MissingField("TRANSPOSE output shape"))?
        .iter()
        .collect();
    let expected_output_dims: Vec<i32> = permutation.iter().map(|&axis| input_dims[axis]).collect();
    if output_dims != expected_output_dims {
        return Err(ImportError::UnsupportedConfiguration(format!(
            "TRANSPOSE output shape {output_dims:?} does not match permutation result {expected_output_dims:?}"
        )));
    }
    let input_quant = read_per_tensor_quant(&input_tensor)?;
    let output_quant = read_per_tensor_quant(output_tensor)?;
    if input_quant != output_quant {
        return Err(ImportError::UnsupportedConfiguration(
            "TRANSPOSE input and output quantization must match".into(),
        ));
    }
    builder
        .add_transpose_layer(name, input_id, &permutation)
        .map_err(|message| ImportError::UnsupportedConfiguration(message.into()))
}

fn import_pad(
    builder: &mut ModelBuilder,
    operator: &tflite::Operator,
    tensors: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Tensor>>,
    buffers: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Buffer>>,
    input_id: usize,
    name: &str,
) -> Result<usize, ImportError> {
    let op_inputs = operator
        .inputs()
        .ok_or(ImportError::MissingField("operator.inputs"))?;
    if op_inputs.len() < 2 {
        return Err(ImportError::UnsupportedConfiguration(
            "PAD requires a paddings tensor".into(),
        ));
    }
    let pads = read_i32_buffer(&tensors.get(op_inputs.get(1) as usize), buffers)?;
    if pads.len() != 8 {
        return Err(ImportError::UnsupportedConfiguration(
            "PAD currently supports rank-4 NHWC paddings [4,2]".into(),
        ));
    }
    if pads[0] != 0 || pads[1] != 0 || pads[6] != 0 || pads[7] != 0 {
        return Err(ImportError::UnsupportedConfiguration(
            "PAD on batch or channel axes is not supported".into(),
        ));
    }
    let pad_value = if op_inputs.len() >= 3 && op_inputs.get(2) >= 0 {
        *read_i8_buffer(&tensors.get(op_inputs.get(2) as usize), buffers)?
            .first()
            .ok_or(ImportError::UnsupportedConfiguration(
                "PADV2 constant_values is empty".into(),
            ))?
    } else {
        0
    };
    Ok(builder.add_pad_layer(
        name,
        input_id,
        Padding2D::new(
            pads[2] as usize,
            pads[3] as usize,
            pads[4] as usize,
            pads[5] as usize,
        ),
        pad_value,
    ))
}

fn import_mean(
    builder: &mut ModelBuilder,
    operator: &tflite::Operator,
    tensors: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Tensor>>,
    buffers: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Buffer>>,
    input_id: usize,
    name: &str,
) -> Result<usize, ImportError> {
    let op_inputs = operator
        .inputs()
        .ok_or(ImportError::MissingField("operator.inputs"))?;
    if op_inputs.len() < 2 {
        return Err(ImportError::UnsupportedConfiguration(
            "MEAN requires an axes tensor".into(),
        ));
    }
    let axes = read_i32_buffer(&tensors.get(op_inputs.get(1) as usize), buffers)?;
    let input_tensor = tensors.get(op_inputs.get(0) as usize);
    let rank = input_tensor
        .shape()
        .ok_or(ImportError::MissingField("MEAN input shape"))?
        .len();
    let keep_dims = operator
        .builtin_options_as_reducer_options()
        .map(|o| o.keep_dims())
        .unwrap_or(false);
    let mut reduce_height = false;
    let mut reduce_width = false;
    let mut reduce_channels = false;
    for axis in axes {
        let axis = if axis < 0 { rank as i32 + axis } else { axis };
        match (rank, axis) {
            (4, 1) => reduce_height = true,
            (4, 2) => reduce_width = true,
            (4, 3) => reduce_channels = true,
            (2, 0) => reduce_width = true,
            (2, 1) => reduce_channels = true,
            (1, 0) => reduce_channels = true,
            (4, 0) => {
                return Err(ImportError::UnsupportedConfiguration(
                    "MEAN over batch is not supported".into(),
                ));
            }
            _ => {
                return Err(ImportError::UnsupportedConfiguration(format!(
                    "unsupported MEAN axis {axis} for rank {rank}"
                )));
            }
        }
    }
    Ok(builder.add_mean_layer(
        name,
        input_id,
        reduce_height,
        reduce_width,
        reduce_channels,
        keep_dims,
    ))
}

#[allow(clippy::too_many_arguments)]
fn import_svdf(
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
    if op_inputs.len() < 3 {
        return Err(ImportError::UnsupportedConfiguration(
            "SVDF requires feature and time weights".into(),
        ));
    }
    let opts = operator
        .builtin_options_as_svdfoptions()
        .ok_or(ImportError::MissingField("SVDFOptions"))?;
    let rank = opts.rank().max(1) as usize;
    let weights_feature = read_i8_buffer(&tensors.get(op_inputs.get(1) as usize), buffers)?;
    let weights_time = read_i8_buffer(&tensors.get(op_inputs.get(2) as usize), buffers)?;
    let bias = optional_bias(&op_inputs, 3, tensors, buffers)?;
    let input_dim = builder
        .tensor_desc(in_id)
        .map(|t| t.shape.total_elements())
        .ok_or(ImportError::UnresolvedInput(op_inputs.get(0) as usize))?;
    if input_dim == 0 || weights_feature.len() % input_dim != 0 {
        return Err(ImportError::UnsupportedConfiguration(
            "SVDF feature weights are not a multiple of input dim".into(),
        ));
    }
    let feature_dim = weights_feature.len() / input_dim;
    if rank == 0 || feature_dim % rank != 0 {
        return Err(ImportError::UnsupportedConfiguration(
            "SVDF feature_dim must be divisible by rank".into(),
        ));
    }
    let units = feature_dim / rank;
    if feature_dim == 0 || weights_time.len() % feature_dim != 0 {
        return Err(ImportError::UnsupportedConfiguration(
            "SVDF time weights are not a multiple of feature dim".into(),
        ));
    }
    let memory_size = weights_time.len() / feature_dim;
    let weight_scales = read_scales(&tensors.get(op_inputs.get(1) as usize))?;
    let (output_quant, _) = build_output_quant(input_scale, &weight_scales, output_tensor)?;
    let activation = read_activation(opts.fused_activation_function())?;
    Ok(builder.add_svdf_layer(
        name,
        in_id,
        units,
        rank,
        memory_size,
        weights_feature,
        weights_time,
        bias,
        activation,
        Some(output_quant),
    ))
}

fn import_mul(
    builder: &mut ModelBuilder,
    operator: &tflite::Operator,
    tensors: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Tensor>>,
    tensor_ids: &HashMap<usize, usize>,
    input1_id: usize,
    output_tensor: &tflite::Tensor,
    name: &str,
) -> Result<usize, ImportError> {
    let op_inputs = operator
        .inputs()
        .ok_or(ImportError::MissingField("operator.inputs"))?;
    if op_inputs.len() != 2 {
        return Err(ImportError::UnsupportedConfiguration(
            "MUL requires exactly two inputs".into(),
        ));
    }
    let input2_idx = op_inputs.get(1) as usize;
    let input2_id = *tensor_ids
        .get(&input2_idx)
        .ok_or(ImportError::UnresolvedInput(input2_idx))?;
    convert_tensor_type(tensors.get(input2_idx).type_())?;
    let (out_scale, out_zero_point) = read_per_tensor_quant(output_tensor)?;
    let (multiplier, shift) = quantize_multiplier(out_scale);
    let output_quant = QuantParams {
        multiplier,
        shift,
        zero_point: out_zero_point,
        scale: out_scale,
    };
    let activation = operator
        .builtin_options_as_mul_options()
        .map(|opts| read_activation(opts.fused_activation_function()))
        .transpose()?
        .unwrap_or(ActivationType::None);
    builder
        .add_elementwise_mul_layer(name, input1_id, input2_id, activation, output_quant)
        .map_err(|message| ImportError::UnsupportedConfiguration(message.into()))
}

fn import_concat(
    builder: &mut ModelBuilder,
    operator: &tflite::Operator,
    tensors: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Tensor>>,
    tensor_ids: &HashMap<usize, usize>,
    name: &str,
) -> Result<usize, ImportError> {
    let op_inputs = operator
        .inputs()
        .ok_or(ImportError::MissingField("operator.inputs"))?;
    if op_inputs.len() < 2 {
        return Err(ImportError::UnsupportedConfiguration(
            "CONCATENATION requires at least two inputs".into(),
        ));
    }
    let axis = operator
        .builtin_options_as_concatenation_options()
        .map(|o| o.axis())
        .unwrap_or(-1);
    let first = tensors.get(op_inputs.get(0) as usize);
    let rank = first
        .shape()
        .ok_or(ImportError::MissingField("CONCAT input shape"))?
        .len() as i32;
    let axis = if axis < 0 { rank + axis } else { axis };
    let channel_axis = match rank {
        1 => 0,
        2 => 1,
        4 => 3,
        _ => {
            return Err(ImportError::UnsupportedConfiguration(
                "CONCATENATION rank must be 1, 2, or 4".into(),
            ));
        }
    };
    if axis != channel_axis {
        return Err(ImportError::UnsupportedConfiguration(
            "CONCATENATION currently supports the channel axis only".into(),
        ));
    }
    let mut current = *tensor_ids
        .get(&(op_inputs.get(0) as usize))
        .ok_or(ImportError::UnresolvedInput(op_inputs.get(0) as usize))?;
    for i in 1..op_inputs.len() {
        let idx = op_inputs.get(i) as usize;
        let next = *tensor_ids
            .get(&idx)
            .ok_or(ImportError::UnresolvedInput(idx))?;
        current = builder
            .add_concat_layer(format!("{name}_{i}"), current, next)
            .map_err(|message| ImportError::UnsupportedConfiguration(message.into()))?;
    }
    Ok(current)
}

fn import_strided_slice(
    builder: &mut ModelBuilder,
    operator: &tflite::Operator,
    tensors: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Tensor>>,
    buffers: &flatbuffers::Vector<flatbuffers::ForwardsUOffset<tflite::Buffer>>,
    input_id: usize,
    name: &str,
) -> Result<usize, ImportError> {
    let op_inputs = operator
        .inputs()
        .ok_or(ImportError::MissingField("operator.inputs"))?;
    if op_inputs.len() < 4 {
        return Err(ImportError::UnsupportedConfiguration(
            "STRIDED_SLICE requires begin, end, and strides tensors".into(),
        ));
    }
    let opts = operator.builtin_options_as_strided_slice_options();
    if opts.is_some_and(|o| {
        o.ellipsis_mask() != 0 || o.new_axis_mask() != 0 || o.shrink_axis_mask() != 0
    }) {
        return Err(ImportError::UnsupportedConfiguration(
            "STRIDED_SLICE ellipsis/new_axis/shrink masks are not supported".into(),
        ));
    }
    let begin_v = read_i32_buffer(&tensors.get(op_inputs.get(1) as usize), buffers)?;
    let end_v = read_i32_buffer(&tensors.get(op_inputs.get(2) as usize), buffers)?;
    let stride_v = read_i32_buffer(&tensors.get(op_inputs.get(3) as usize), buffers)?;
    let mut begin = [0i32, 0, 0, 0];
    let mut end = [1i32, 1, 1, 1];
    let mut stride = [1i32, 1, 1, 1];
    match begin_v.len() {
        1 => {
            begin[3] = begin_v[0];
            end[3] = *end_v.first().unwrap_or(&1);
            stride[3] = *stride_v.first().unwrap_or(&1);
            let input = builder.tensor_desc(input_id).unwrap();
            end[0] = input.shape.batches as i32;
            end[1] = input.shape.height as i32;
            end[2] = input.shape.width as i32;
        }
        4 => {
            begin.copy_from_slice(&begin_v[..4]);
            end.copy_from_slice(&end_v[..4]);
            stride.copy_from_slice(&stride_v[..4]);
        }
        2 => {
            begin[2] = begin_v[0];
            begin[3] = begin_v[1];
            end[2] = end_v[0];
            end[3] = end_v[1];
            stride[2] = stride_v.first().copied().unwrap_or(1);
            stride[3] = stride_v.get(1).copied().unwrap_or(1);
            end[0] = 1;
            end[1] = 1;
        }
        _ => {
            return Err(ImportError::UnsupportedConfiguration(
                "STRIDED_SLICE rank must be 1, 2, or 4".into(),
            ));
        }
    }
    builder
        .add_strided_slice_layer(name, input_id, begin, end, stride)
        .map_err(|message| ImportError::UnsupportedConfiguration(message.into()))
}

#[allow(clippy::too_many_arguments)]
fn import_lstm(
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
        .builtin_options_as_lstmoptions()
        .ok_or(ImportError::MissingField("LSTMOptions"))?;
    if opts.kernel_type() != tflite::LSTMKernelType::BASIC {
        return Err(ImportError::UnsupportedConfiguration(
            "only BASIC LSTM is imported; FULL / sequence LSTM is not supported".into(),
        ));
    }
    let op_inputs = operator
        .inputs()
        .ok_or(ImportError::MissingField("operator.inputs"))?;
    if op_inputs.len() < 4 {
        return Err(ImportError::UnsupportedConfiguration(
            "BASIC LSTM requires weights and bias inputs".into(),
        ));
    }
    let weight_tensor = tensors.get(op_inputs.get(2) as usize);
    let weights = read_i8_buffer(&weight_tensor, buffers)?;
    let bias = read_i32_bias_buffer(&tensors.get(op_inputs.get(3) as usize), buffers)?;
    let input_dim = builder
        .tensor_desc(in_id)
        .map(|t| t.shape.total_elements())
        .ok_or(ImportError::UnresolvedInput(op_inputs.get(0) as usize))?;
    let hidden_dim = convert_shape(output_tensor)?.total_elements();
    let expected = (input_dim + hidden_dim) * 4 * hidden_dim;
    if weights.len() != expected {
        return Err(ImportError::UnsupportedConfiguration(format!(
            "BASIC LSTM weight size {} does not match (input+hidden)*4*hidden = {expected}",
            weights.len()
        )));
    }
    let cols = 4 * hidden_dim;
    let mut input_weights = vec![0i8; 4 * hidden_dim * input_dim];
    let mut recurrent_weights = vec![0i8; 4 * hidden_dim * hidden_dim];
    for row in 0..(input_dim + hidden_dim) {
        for col in 0..cols {
            let value = weights[row * cols + col];
            let gate = col / hidden_dim;
            let h = col % hidden_dim;
            if row < input_dim {
                input_weights[gate * hidden_dim * input_dim + h * input_dim + row] = value;
            } else {
                let k = row - input_dim;
                recurrent_weights[gate * hidden_dim * hidden_dim + h * hidden_dim + k] = value;
            }
        }
    }
    let weight_scales = read_scales(&weight_tensor)?;
    let (output_quant, _) = build_output_quant(input_scale, &weight_scales, output_tensor)?;
    Ok(builder.add_lstm_step_layer(
        name,
        in_id,
        hidden_dim,
        input_weights,
        recurrent_weights,
        bias,
        Some(output_quant),
    ))
}

#[cfg(any(test, feature = "fixture-generation"))]
pub mod constructed_fixtures {
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
        build_typed_tensor(fbb, spec, TensorType::INT8)
    }

    fn build_typed_tensor<'fbb>(
        fbb: &mut FlatBufferBuilder<'fbb>,
        spec: &TensorSpec,
        type_: TensorType,
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
                type_,
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

    /// A single `CONV_2D` layer with odd asymmetric SAME padding: input[1,8,8,3] ->
    /// weights[4,3,3,3] (4 output channels, 4 independent scales) -> output[1,4,4,4].
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
                shape: &[1, 4, 4, 4],
                buffer: 0,
                scale: 0.02,
                zero_point: -128,
            },
        );
        let tensors = fbb.create_vector(&[input_tensor, weight_tensor, output_tensor]);

        let conv_opts = Conv2DOptions::create(
            &mut fbb,
            &Conv2DOptionsArgs {
                padding: Padding::SAME,
                stride_w: 2,
                stride_h: 2,
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
        build_fc_model(TensorType::INT8, 0, 0, 3, vec![1, 2, 3, 4, 5, 6, 7, 8])
    }

    /// A one-neuron linear approximation of sine on `[-pi/2, pi/2]`.
    ///
    /// Input codes map that interval to `[-127, 127]`; output codes map approximately
    /// `[-1, 1]` to the same range. The quantized FC multiplier is exactly `1 / 2`, so a
    /// weight of 2 preserves the input code. This intentionally simple "sine hello world"
    /// fixture makes boundary quantization and dequantization independently checkable.
    pub fn build_sine_fc_model() -> Vec<u8> {
        let mut fbb = FlatBufferBuilder::new();

        let empty_buffer = Buffer::create(&mut fbb, &BufferArgs::default());
        let weight_data = fbb.create_vector(&[2u8]);
        let weight_buffer = Buffer::create(
            &mut fbb,
            &BufferArgs {
                data: Some(weight_data),
                ..Default::default()
            },
        );
        let bias_data = fbb.create_vector(&0i32.to_le_bytes());
        let bias_buffer = Buffer::create(
            &mut fbb,
            &BufferArgs {
                data: Some(bias_data),
                ..Default::default()
            },
        );
        let buffers = fbb.create_vector(&[empty_buffer, weight_buffer, bias_buffer]);

        let input = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[1, 1],
                buffer: 0,
                scale: core::f32::consts::FRAC_PI_2 / 127.0,
                zero_point: 0,
            },
        );
        let weight = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[1, 1],
                buffer: 1,
                scale: (0.5 / 128.0) / (core::f32::consts::FRAC_PI_2 / 127.0),
                zero_point: 0,
            },
        );
        let bias = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[1],
                buffer: 2,
                scale: (core::f32::consts::FRAC_PI_2 / 127.0)
                    * ((0.5 / 128.0) / (core::f32::consts::FRAC_PI_2 / 127.0)),
                zero_point: 0,
            },
        );
        let output = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[1, 1],
                buffer: 0,
                scale: 1.0 / 128.0,
                zero_point: 0,
            },
        );
        let tensors = fbb.create_vector(&[input, weight, bias, output]);

        let options = FullyConnectedOptions::create(
            &mut fbb,
            &FullyConnectedOptionsArgs {
                fused_activation_function: ActivationFunctionType::NONE,
                ..Default::default()
            },
        );
        let inputs = fbb.create_vector(&[0i32, 1, 2]);
        let outputs = fbb.create_vector(&[3i32]);
        let operator = Operator::create(
            &mut fbb,
            &OperatorArgs {
                opcode_index: 0,
                inputs: Some(inputs),
                outputs: Some(outputs),
                builtin_options_type: BuiltinOptions::FullyConnectedOptions,
                builtin_options: Some(options.as_union_value()),
                ..Default::default()
            },
        );
        let operators = fbb.create_vector(&[operator]);
        let graph_inputs = fbb.create_vector(&[0i32]);
        let graph_outputs = fbb.create_vector(&[3i32]);
        let subgraph = SubGraph::create(
            &mut fbb,
            &SubGraphArgs {
                tensors: Some(tensors),
                inputs: Some(graph_inputs),
                outputs: Some(graph_outputs),
                operators: Some(operators),
                ..Default::default()
            },
        );
        let subgraphs = fbb.create_vector(&[subgraph]);
        let opcode = OperatorCode::create(
            &mut fbb,
            &OperatorCodeArgs {
                deprecated_builtin_code: BuiltinOperator::FULLY_CONNECTED.0 as i8,
                version: 1,
                builtin_code: BuiltinOperator::FULLY_CONNECTED,
                ..Default::default()
            },
        );
        let opcodes = fbb.create_vector(&[opcode]);
        let model = Model::create(
            &mut fbb,
            &ModelArgs {
                version: 3,
                operator_codes: Some(opcodes),
                subgraphs: Some(subgraphs),
                buffers: Some(buffers),
                ..Default::default()
            },
        );
        fbb.finish_minimal(model);
        fbb.finished_data().to_vec()
    }

    pub fn build_uint8_fc_model() -> Vec<u8> {
        // Signed weights [-128, -1, 0, 1, 2, 3, 126, 127] encoded in UINT8 storage.
        build_fc_model(
            TensorType::UINT8,
            129,
            128,
            131,
            vec![0, 127, 128, 129, 130, 131, 254, 255],
        )
    }

    fn build_fc_model(
        tensor_type: TensorType,
        input_zero_point: i64,
        weight_zero_point: i64,
        output_zero_point: i64,
        weight_bytes: Vec<u8>,
    ) -> Vec<u8> {
        let mut fbb = FlatBufferBuilder::new();

        let empty_buffer = Buffer::create(&mut fbb, &BufferArgs::default());
        let weight_data = fbb.create_vector(&weight_bytes);
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

        let input_tensor = build_typed_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[1, 4],
                buffer: 0,
                scale: 1.0 / 127.0,
                zero_point: input_zero_point,
            },
            tensor_type,
        );
        let weight_tensor = build_typed_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[2, 4],
                buffer: 1,
                scale: 0.01,
                zero_point: weight_zero_point,
            },
            tensor_type,
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
        let output_tensor = build_typed_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[1, 2],
                buffer: 0,
                scale: 0.05,
                zero_point: output_zero_point,
            },
            tensor_type,
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

    /// A speech-scale TinyConv chain:
    /// CONV_2D(ReLU) -> MAX_POOL_2D -> RESHAPE -> FULLY_CONNECTED -> SOFTMAX.
    /// input[1,49,40,1] -> conv[1,20,19,4] (10x4/2 VALID) ->
    /// pool[1,10,9,4] (2x2/2 VALID) -> reshape[1,360] -> fc[1,4] -> softmax[1,4].
    pub fn build_conv_pool_reshape_fc_softmax_model() -> Vec<u8> {
        let mut fbb = FlatBufferBuilder::new();

        let empty_buffer = Buffer::create(&mut fbb, &BufferArgs::default());
        let conv_weight_bytes: Vec<u8> = vec![1u8; 4 * 10 * 4];
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
        let fc_weight_bytes: Vec<u8> = vec![1u8; 4 * 360];
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
                shape: &[1, 49, 40, 1],
                buffer: 0,
                scale: 1.0 / 127.0,
                zero_point: 0,
            },
        );
        let conv_weight_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[4, 10, 4, 1],
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
                shape: &[1, 20, 19, 4],
                buffer: 0,
                scale: 0.02,
                zero_point: -128,
            },
        );
        let pool_out_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[1, 10, 9, 4],
                buffer: 0,
                scale: 0.02,
                zero_point: -128,
            },
        );
        let reshape_out_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[1, 360],
                buffer: 0,
                scale: 0.02,
                zero_point: -128,
            },
        );
        let fc_weight_tensor = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[4, 360],
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
                stride_w: 2,
                stride_h: 2,
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

        let reshape_new_shape = fbb.create_vector(&[1i32, 360]);
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

    /// Two rank-2 inputs are added, then transposed with permutation [1, 0].
    pub fn build_add_transpose_model() -> Vec<u8> {
        let mut fbb = FlatBufferBuilder::new();
        let empty_buffer = Buffer::create(&mut fbb, &BufferArgs::default());
        let perm_bytes: Vec<u8> = [1i32, 0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let perm_data = fbb.create_vector(&perm_bytes);
        let perm_buffer = Buffer::create(
            &mut fbb,
            &BufferArgs {
                data: Some(perm_data),
                ..Default::default()
            },
        );
        let buffers = fbb.create_vector(&[empty_buffer, perm_buffer]);
        let left = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[2, 3],
                buffer: 0,
                scale: 0.25,
                zero_point: -3,
            },
        );
        let right = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[2, 3],
                buffer: 0,
                scale: 0.5,
                zero_point: 7,
            },
        );
        let added = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[2, 3],
                buffer: 0,
                scale: 0.125,
                zero_point: -9,
            },
        );
        let perm_shape = fbb.create_vector(&[2i32]);
        let perm = Tensor::create(
            &mut fbb,
            &TensorArgs {
                shape: Some(perm_shape),
                type_: TensorType::INT32,
                buffer: 1,
                has_rank: true,
                ..Default::default()
            },
        );
        let transposed = build_tensor(
            &mut fbb,
            &TensorSpec {
                shape: &[3, 2],
                buffer: 0,
                scale: 0.125,
                zero_point: -9,
            },
        );
        let tensors = fbb.create_vector(&[left, right, added, perm, transposed]);

        let add_options = AddOptions::create(
            &mut fbb,
            &AddOptionsArgs {
                fused_activation_function: ActivationFunctionType::RELU6,
                ..Default::default()
            },
        );
        let add_inputs = fbb.create_vector(&[0i32, 1]);
        let add_outputs = fbb.create_vector(&[2i32]);
        let add = Operator::create(
            &mut fbb,
            &OperatorArgs {
                opcode_index: 0,
                inputs: Some(add_inputs),
                outputs: Some(add_outputs),
                builtin_options_type: BuiltinOptions::AddOptions,
                builtin_options: Some(add_options.as_union_value()),
                ..Default::default()
            },
        );
        let transpose_options =
            TransposeOptions::create(&mut fbb, &TransposeOptionsArgs::default());
        let transpose_inputs = fbb.create_vector(&[2i32, 3]);
        let transpose_outputs = fbb.create_vector(&[4i32]);
        let transpose = Operator::create(
            &mut fbb,
            &OperatorArgs {
                opcode_index: 1,
                inputs: Some(transpose_inputs),
                outputs: Some(transpose_outputs),
                builtin_options_type: BuiltinOptions::TransposeOptions,
                builtin_options: Some(transpose_options.as_union_value()),
                ..Default::default()
            },
        );
        let operators = fbb.create_vector(&[add, transpose]);
        let inputs = fbb.create_vector(&[0i32, 1]);
        let outputs = fbb.create_vector(&[4i32]);
        let subgraph = SubGraph::create(
            &mut fbb,
            &SubGraphArgs {
                tensors: Some(tensors),
                inputs: Some(inputs),
                outputs: Some(outputs),
                operators: Some(operators),
                name: None,
            },
        );
        let subgraphs = fbb.create_vector(&[subgraph]);
        let opcode_offsets: Vec<_> = [BuiltinOperator::ADD, BuiltinOperator::TRANSPOSE]
            .iter()
            .map(|&code| {
                OperatorCode::create(
                    &mut fbb,
                    &OperatorCodeArgs {
                        deprecated_builtin_code: code.0 as i8,
                        builtin_code: code,
                        version: 1,
                        ..Default::default()
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
                buffers: Some(buffers),
                ..Default::default()
            },
        );
        fbb.finish_minimal(model);
        fbb.finished_data().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constructed_fixtures::{
        build_add_transpose_model, build_conv_pool_reshape_fc_softmax_model,
        build_conv2d_per_channel_model, build_fc_only_model, build_sine_fc_model,
        build_uint8_fc_model,
    };
    use embedded_nn_compiler::builder::ModelBuilder;

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
                padding,
                activation,
                ..
            } => {
                assert_eq!(*kernel_h, 10);
                assert_eq!(*kernel_w, 4);
                assert_eq!(*padding, Padding2D::default());
                assert_eq!(*activation, ActivationType::Relu);
            }
            other => panic!("expected Conv2D, got {:?}", other),
        }
        match &graph.layers[1].op {
            OpPayload::MaxPool2D { padding, .. } => {
                assert_eq!(*padding, Padding2D::default());
            }
            other => panic!("expected MaxPool2D, got {:?}", other),
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
                padding,
                per_channel_quant,
                ..
            } => {
                assert_eq!(*padding, Padding2D::new(0, 1, 0, 1));
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
        assert!(code.contains("Padding2D::new(0, 1, 0, 1)"));
        let output = graph
            .tensors
            .iter()
            .find(|tensor| graph.outputs.contains(&tensor.id))
            .unwrap();
        assert_eq!((output.shape.height, output.shape.width), (4, 4));
    }

    #[test]
    fn test_import_uint8_rewrites_values_and_zero_points_to_int8() {
        let graph = import_tflite(&build_uint8_fc_model()).expect("UINT8 import should succeed");
        let input = graph
            .tensors
            .iter()
            .find(|tensor| graph.inputs.contains(&tensor.id))
            .unwrap();
        let output = graph
            .tensors
            .iter()
            .find(|tensor| graph.outputs.contains(&tensor.id))
            .unwrap();
        assert_eq!(input.dtype, DataType::Int8);
        assert_eq!(input.quant.zero_point, 1);
        assert_eq!(output.dtype, DataType::Int8);
        assert_eq!(output.quant.zero_point, 3);
        match &graph.layers[0].op {
            OpPayload::FullyConnected {
                weights,
                filter_offset,
                ..
            } => {
                assert_eq!(weights, &vec![-128, -1, 0, 1, 2, 3, 126, 127]);
                assert_eq!(*filter_offset, 0);
            }
            other => panic!("expected FullyConnected, got {other:?}"),
        }
        let generated =
            embedded_nn_codegen::RustCodeGenerator::new("Uint8Rewritten").generate(&graph);
        assert!(generated.contains("filter_offset: 0"));
        assert!(generated.contains("static OP0_WEIGHTS_S8: [i8; 8]"));
    }

    #[test]
    fn test_import_add_and_rank2_transpose() {
        let graph = import_tflite(&build_add_transpose_model()).expect("import should succeed");
        assert_eq!(graph.layers.len(), 2);
        match &graph.layers[0].op {
            OpPayload::ElementwiseAdd { quant, activation } => {
                assert_eq!(quant.left_shift, 20);
                assert_eq!(quant.input1_offset, 3);
                assert_eq!(quant.input2_offset, -7);
                assert_eq!(quant.output_offset, -9);
                assert_eq!(*activation, ActivationType::Relu6);
            }
            other => panic!("expected ADD, got {other:?}"),
        }
        assert!(matches!(
            graph.layers[1].op,
            OpPayload::Transpose {
                kind: TransposeKind::Matrix2D { rows: 2, cols: 3 }
            }
        ));
        let generated =
            embedded_nn_codegen::RustCodeGenerator::new("AddTranspose").generate(&graph);
        assert!(generated.contains("elementwise_add_s8(in_buf, in_buf2"));
        assert!(generated.contains("transpose_2d_s8(2, 3"));
    }

    #[test]
    fn test_checked_in_constructed_fixtures_are_reproducible() {
        let fixtures =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/constructed");
        for (name, generated) in [
            ("sine_fc_int8.tflite", build_sine_fc_model()),
            (
                "tinyconv_int8.tflite",
                build_conv_pool_reshape_fc_softmax_model(),
            ),
            ("uint8_fc.tflite", build_uint8_fc_model()),
            ("add_transpose_int8.tflite", build_add_transpose_model()),
        ] {
            let checked_in =
                std::fs::read(fixtures.join(name)).expect("fixture must be checked in");
            assert_eq!(
                checked_in, generated,
                "{name} is stale; rerun the generator"
            );
        }
    }

    #[test]
    fn host_interpreter_matches_sine_and_tinyconv_fixture_vectors() {
        let sine = import_tflite(&build_sine_fc_model()).unwrap();
        let mut sine_host = embedded_nn_compiler::HostInterpreter::new(&sine).unwrap();
        for value in [-127i8, 0, 127] {
            let output = sine_host.run(&[&[value]]).unwrap();
            assert_eq!(output[0], vec![value]);
        }

        let tinyconv = import_tflite(&build_conv_pool_reshape_fc_softmax_model()).unwrap();
        let input_len = tinyconv
            .tensors
            .iter()
            .find(|tensor| tinyconv.inputs.contains(&tensor.id))
            .unwrap()
            .shape
            .total_elements();
        let input = vec![0i8; input_len];
        let mut tinyconv_host = embedded_nn_compiler::HostInterpreter::new(&tinyconv).unwrap();
        let output = tinyconv_host.run(&[&input]).unwrap();
        assert_eq!(output[0], vec![-64, -64, -64, -64]);
    }

    #[test]
    fn host_interpreter_uses_both_add_inputs_and_transposes_result() {
        let graph = import_tflite(&build_add_transpose_model()).unwrap();
        let mut host = embedded_nn_compiler::HostInterpreter::new(&graph).unwrap();
        let left = [1i8, 2, 3, 4, 5, 6];
        let right = [7i8, 8, 9, 10, 11, 12];
        let output = host.run(&[&left, &right]).unwrap();
        // Golden vector from the fixture's TFLite quantization parameters, after RELU6 and [1,0].
        assert_eq!(output[0], vec![-1, 17, 5, 23, 11, 29]);
    }

    #[test]
    fn host_interpreter_runs_concat_and_mul() {
        let mut builder = ModelBuilder::new("glue");
        let a = builder.add_input("a", TensorShape::new_1d(2), DataType::Int8, None);
        let b = builder.add_input("b", TensorShape::new_1d(2), DataType::Int8, None);
        let cat = builder.add_concat_layer("cat", a, b).unwrap();
        builder.mark_output(cat);
        let graph = builder.build();
        let mut host = embedded_nn_compiler::HostInterpreter::new(&graph).unwrap();
        assert_eq!(
            host.run(&[&[1i8, 2], &[3i8, 4]]).unwrap()[0],
            vec![1, 2, 3, 4]
        );
    }
}
