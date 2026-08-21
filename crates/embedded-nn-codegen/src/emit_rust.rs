use embedded_nn_compiler::arena::ArenaScheduler;
use embedded_nn_compiler::ir::*;

pub struct RustCodeGenerator {
    struct_name: String,
}

fn emit_i8_array(out: &mut String, name: &str, data: &[i8]) {
    out.push_str(&format!("static {}: [i8; {}] = [\n    ", name, data.len()));
    for (i, byte) in data.iter().enumerate() {
        out.push_str(&format!("{}, ", byte));
        if (i + 1) % 12 == 0 {
            out.push_str("\n    ");
        }
    }
    out.push_str("\n];\n\n");
}

fn emit_i32_array(out: &mut String, name: &str, data: &[i32]) {
    out.push_str(&format!("static {}: [i32; {}] = [\n    ", name, data.len()));
    for (i, val) in data.iter().enumerate() {
        out.push_str(&format!("{}, ", val));
        if (i + 1) % 8 == 0 {
            out.push_str("\n    ");
        }
    }
    out.push_str("\n];\n\n");
}

/// Folds the input zero-point correction into an output-channel bias.
///
/// The returned bias is valid only when the generated call also uses `input_offset: 0`.
/// Accumulating in i64 lets codegen decline the optimization instead of changing i32 behavior
/// when the static correction itself is not representable.
fn fold_row_major_bias(
    weights: &[i8],
    output_channels: usize,
    input_offset: i32,
    bias: Option<&[i32]>,
) -> Option<Vec<i32>> {
    if output_channels == 0
        || !weights.len().is_multiple_of(output_channels)
        || bias.is_some_and(|values| values.len() != output_channels)
    {
        return None;
    }

    let row_len = weights.len() / output_channels;
    (0..output_channels)
        .map(|output_channel| {
            let weight_sum: i64 = weights[output_channel * row_len..(output_channel + 1) * row_len]
                .iter()
                .map(|&weight| i64::from(weight))
                .sum();
            let original_bias = bias.map_or(0, |values| i64::from(values[output_channel]));
            i32::try_from(original_bias + i64::from(input_offset) * weight_sum).ok()
        })
        .collect()
}

fn fold_depthwise_bias(
    weights: &[i8],
    output_channels: usize,
    input_offset: i32,
    bias: Option<&[i32]>,
) -> Option<Vec<i32>> {
    if output_channels == 0
        || !weights.len().is_multiple_of(output_channels)
        || bias.is_some_and(|values| values.len() != output_channels)
    {
        return None;
    }

    (0..output_channels)
        .map(|output_channel| {
            let weight_sum: i64 = weights
                .iter()
                .skip(output_channel)
                .step_by(output_channels)
                .map(|&weight| i64::from(weight))
                .sum();
            let original_bias = bias.map_or(0, |values| i64::from(values[output_channel]));
            i32::try_from(original_bias + i64::from(input_offset) * weight_sum).ok()
        })
        .collect()
}

/// Returns a folded bias only for layouts and kernel calls where replacing the runtime
/// `input_offset` with zero is mathematically exact.
fn folded_bias(graph: &ModelGraph, layer: &LayerNode) -> Option<Vec<i32>> {
    let input = graph
        .tensors
        .iter()
        .find(|tensor| tensor.id == layer.inputs[0])?;
    let output = graph
        .tensors
        .iter()
        .find(|tensor| tensor.id == layer.outputs[0])?;
    let input_offset = -input.quant.zero_point;
    if input_offset == 0 {
        return None;
    }

    match &layer.op {
        OpPayload::FullyConnected {
            weights,
            packed_s4: None,
            bias,
            filter_offset: 0,
            ..
        } => fold_row_major_bias(
            weights,
            output.shape.channels,
            input_offset,
            bias.as_deref(),
        ),
        OpPayload::Conv2D {
            weights,
            packed_s4: None,
            bias,
            padding,
            ..
        } if *padding == Padding2D::default() => fold_row_major_bias(
            weights,
            output.shape.channels,
            input_offset,
            bias.as_deref(),
        ),
        OpPayload::DepthwiseConv2D {
            weights,
            bias,
            padding,
            ..
        } if *padding == Padding2D::default() => fold_depthwise_bias(
            weights,
            output.shape.channels,
            input_offset,
            bias.as_deref(),
        ),
        _ => None,
    }
}

/// Maps an IR [`ActivationType`] to a generated `Activation::new(min, max)` expression.
///
/// The FC/Conv/SVDF kernels add the output tensor's zero-point *before* applying the
/// activation clamp (see `fully_connected_per_channel_s8` and friends), so "real zero" in
/// the clamped, offset-added domain is `output_zero_point`, not the literal `0` that would
/// be correct only for symmetric (zero_point == 0) quantization.
/// `LeakyRelu`/`Sigmoid`/`Tanh` aren't clamp-representable, so they fall back to unconstrained.
fn activation_expr(activation: &ActivationType, output_quant: &QuantParams) -> String {
    let zp = output_quant.zero_point;
    match activation {
        ActivationType::None => "Activation::int8_unconstrained()".to_string(),
        ActivationType::Relu => format!("Activation::new({}, i8::MAX as i32)", zp),
        ActivationType::Relu6 => {
            let six_q = zp + (6.0f32 / output_quant.scale).round() as i32;
            format!("Activation::new({}, {})", zp, six_q.min(i8::MAX as i32))
        }
        ActivationType::LeakyRelu { .. } | ActivationType::Sigmoid | ActivationType::Tanh => {
            "Activation::int8_unconstrained()".to_string()
        }
    }
}

impl RustCodeGenerator {
    pub fn new(struct_name: impl Into<String>) -> Self {
        Self {
            struct_name: struct_name.into(),
        }
    }

    pub fn generate(&self, graph: &ModelGraph) -> String {
        let arena_plan = ArenaScheduler::schedule(graph);
        let mut out = String::new();

        let has_conv1d = graph
            .layers
            .iter()
            .any(|l| matches!(l.op, OpPayload::Conv1D { .. }));
        let has_svdf = graph
            .layers
            .iter()
            .any(|l| matches!(l.op, OpPayload::Svdf { .. }));
        let has_per_channel_fc = graph.layers.iter().any(|l| {
            matches!(
                &l.op,
                OpPayload::FullyConnected {
                    per_channel_quant: Some(_),
                    ..
                }
            )
        });
        let has_conv2d_s4 = graph.layers.iter().any(|l| {
            matches!(
                &l.op,
                OpPayload::Conv2D {
                    packed_s4: Some(_),
                    ..
                }
            )
        });
        let has_conv2d_plain = graph.layers.iter().any(|l| {
            matches!(
                &l.op,
                OpPayload::Conv2D {
                    packed_s4: None,
                    per_channel_quant: None,
                    ..
                }
            )
        });
        let has_conv2d_per_channel = graph.layers.iter().any(|l| {
            matches!(
                &l.op,
                OpPayload::Conv2D {
                    packed_s4: None,
                    per_channel_quant: Some(_),
                    ..
                }
            )
        });
        let has_conv2d = has_conv2d_s4 || has_conv2d_plain || has_conv2d_per_channel;
        let has_depthwise = graph
            .layers
            .iter()
            .any(|l| matches!(l.op, OpPayload::DepthwiseConv2D { .. }));
        let has_maxpool = graph
            .layers
            .iter()
            .any(|l| matches!(l.op, OpPayload::MaxPool2D { .. }));
        let has_avgpool = graph
            .layers
            .iter()
            .any(|l| matches!(l.op, OpPayload::AvgPool2D { .. }));
        let has_pad = graph
            .layers
            .iter()
            .any(|l| matches!(l.op, OpPayload::Pad { .. }));
        let has_mean = graph
            .layers
            .iter()
            .any(|l| matches!(l.op, OpPayload::Mean { .. }));
        let has_add = graph
            .layers
            .iter()
            .any(|l| matches!(l.op, OpPayload::ElementwiseAdd { .. }));
        let has_transpose_2d = graph.layers.iter().any(|l| {
            matches!(
                l.op,
                OpPayload::Transpose {
                    kind: TransposeKind::Matrix2D { .. }
                }
            )
        });
        let has_transpose_spatial = graph.layers.iter().any(|l| {
            matches!(
                l.op,
                OpPayload::Transpose {
                    kind: TransposeKind::Spatial4D
                }
            )
        });
        let needs_per_channel_quant_params =
            has_per_channel_fc || has_conv2d_per_channel || has_depthwise;
        let needs_tile =
            has_conv1d || has_conv2d || has_depthwise || has_maxpool || has_avgpool || has_pad;

        // Plain comments remain valid both at a generated file's root and when the generated
        // tokens replace an item through `#[embedded_nn_model(...)]`.
        out.push_str("// Auto-generated by embedded-nn-codegen. DO NOT EDIT MANUALLY.\n");
        out.push_str("// Zero-allocation no_std neural network inference pipeline.\n\n");
        out.push_str("use embedded_nn::{\n");
        out.push_str("    Activation, Dims, FcParams, PerTensorQuantParams,\n");
        out.push_str("    fully_connected_s8, fully_connected_s4, softmax_s8,\n");
        if has_conv1d {
            out.push_str("    convolve_1_x_n_s8,\n");
        }
        if has_svdf {
            out.push_str("    svdf_s8,\n");
        }
        if needs_per_channel_quant_params {
            out.push_str("    PerChannelQuantParams,\n");
        }
        if has_per_channel_fc {
            out.push_str("    fully_connected_per_channel_s8,\n");
        }
        if has_conv2d {
            out.push_str("    ConvParams,\n");
        }
        if has_conv2d_s4 {
            out.push_str("    convolve_s4,\n");
        }
        if has_conv2d_plain {
            out.push_str("    convolve_s8,\n");
        }
        if has_conv2d_per_channel {
            out.push_str("    convolve_per_channel_s8,\n");
        }
        if has_depthwise {
            out.push_str("    depthwise_conv_per_channel_s8, DwConvParams,\n");
        }
        if has_maxpool || has_avgpool {
            out.push_str("    PoolParams,\n");
        }
        if has_maxpool {
            out.push_str("    max_pool_s8,\n");
        }
        if has_avgpool {
            out.push_str("    avg_pool_s8,\n");
        }
        if has_pad {
            out.push_str("    pad_s8,\n");
        }
        if has_mean {
            out.push_str("    reduce_mean_s8,\n");
        }
        if has_add {
            out.push_str("    elementwise_add_s8, ElementwiseAddParams,\n");
        }
        if has_transpose_2d {
            out.push_str("    transpose_2d_s8,\n");
        }
        if has_transpose_spatial {
            out.push_str("    transpose_spatial_s8,\n");
        }
        if needs_tile {
            out.push_str("    Padding2D, Tile,\n");
        }
        out.push_str("};\n\n");

        let arena_size = arena_plan.total_arena_bytes;
        let weights_size = graph.total_weights_size_bytes();

        out.push_str(&format!(
            "pub const ARENA_SIZE_BYTES: usize = {};\n",
            arena_size
        ));
        out.push_str(&format!(
            "pub const FLASH_WEIGHTS_BYTES: usize = {};\n\n",
            weights_size
        ));

        // Compute per-layer SVDF delay-line state offsets: state must persist across
        // separate `predict()` calls, so it lives in a caller-owned buffer, not the arena.
        let mut svdf_state_offsets: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut svdf_state_total = 0usize;
        for layer in &graph.layers {
            if let OpPayload::Svdf {
                weights_feature,
                memory_size,
                ..
            } = &layer.op
            {
                let in_id = layer.inputs[0];
                let in_t = graph.tensors.iter().find(|t| t.id == in_id).unwrap();
                let input_dim = in_t.shape.total_elements();
                let feature_dim = weights_feature.len() / input_dim;
                let state_size = feature_dim * memory_size;
                svdf_state_offsets.insert(layer.id, svdf_state_total);
                svdf_state_total += state_size;
            }
        }
        if has_svdf {
            out.push_str(&format!(
                "pub const SVDF_STATE_BYTES: usize = {};\n\n",
                svdf_state_total
            ));
        }

        // Emit static weight constants
        for layer in &graph.layers {
            let prefix = layer.name.to_uppercase();
            match &layer.op {
                OpPayload::FullyConnected {
                    weights,
                    packed_s4,
                    bias,
                    per_channel_quant,
                    ..
                } => {
                    if let Some(s4) = packed_s4 {
                        emit_i8_array(&mut out, &format!("{}_WEIGHTS_S4", prefix), s4);
                    } else {
                        emit_i8_array(&mut out, &format!("{}_WEIGHTS_S8", prefix), weights);
                    }
                    if let Some(folded) = folded_bias(graph, layer) {
                        emit_i32_array(&mut out, &format!("{}_BIAS_S32", prefix), &folded);
                    } else if let Some(b) = bias {
                        emit_i32_array(&mut out, &format!("{}_BIAS_S32", prefix), b);
                    }
                    if let Some(pcq) = per_channel_quant {
                        emit_i32_array(&mut out, &format!("{}_MULT_S32", prefix), &pcq.multipliers);
                        emit_i32_array(&mut out, &format!("{}_SHIFT_S32", prefix), &pcq.shifts);
                    }
                }
                OpPayload::Conv1D { weights, bias, .. } => {
                    emit_i8_array(&mut out, &format!("{}_WEIGHTS_S8", prefix), weights);
                    if let Some(b) = bias {
                        emit_i32_array(&mut out, &format!("{}_BIAS_S32", prefix), b);
                    }
                }
                OpPayload::Conv2D {
                    weights,
                    packed_s4,
                    bias,
                    per_channel_quant,
                    ..
                } => {
                    if let Some(s4) = packed_s4 {
                        emit_i8_array(&mut out, &format!("{}_WEIGHTS_S4", prefix), s4);
                    } else {
                        emit_i8_array(&mut out, &format!("{}_WEIGHTS_S8", prefix), weights);
                    }
                    if let Some(folded) = folded_bias(graph, layer) {
                        emit_i32_array(&mut out, &format!("{}_BIAS_S32", prefix), &folded);
                    } else if let Some(b) = bias {
                        emit_i32_array(&mut out, &format!("{}_BIAS_S32", prefix), b);
                    }
                    if let Some(pcq) = per_channel_quant {
                        emit_i32_array(&mut out, &format!("{}_MULT_S32", prefix), &pcq.multipliers);
                        emit_i32_array(&mut out, &format!("{}_SHIFT_S32", prefix), &pcq.shifts);
                    }
                }
                OpPayload::DepthwiseConv2D {
                    weights,
                    bias,
                    per_channel_quant,
                    ..
                } => {
                    emit_i8_array(&mut out, &format!("{}_WEIGHTS_S8", prefix), weights);
                    if let Some(folded) = folded_bias(graph, layer) {
                        emit_i32_array(&mut out, &format!("{}_BIAS_S32", prefix), &folded);
                    } else if let Some(b) = bias {
                        emit_i32_array(&mut out, &format!("{}_BIAS_S32", prefix), b);
                    }
                    if let Some(pcq) = per_channel_quant {
                        emit_i32_array(&mut out, &format!("{}_MULT_S32", prefix), &pcq.multipliers);
                        emit_i32_array(&mut out, &format!("{}_SHIFT_S32", prefix), &pcq.shifts);
                    }
                }
                OpPayload::Svdf {
                    weights_feature,
                    weights_time,
                    bias,
                    ..
                } => {
                    emit_i8_array(
                        &mut out,
                        &format!("{}_WEIGHTS_FEATURE_S8", prefix),
                        weights_feature,
                    );
                    emit_i8_array(
                        &mut out,
                        &format!("{}_WEIGHTS_TIME_S8", prefix),
                        weights_time,
                    );
                    if let Some(b) = bias {
                        emit_i32_array(&mut out, &format!("{}_BIAS_S32", prefix), b);
                    }
                }
                OpPayload::MaxPool2D { .. }
                | OpPayload::AvgPool2D { .. }
                | OpPayload::Softmax
                | OpPayload::ElementwiseAdd { .. }
                | OpPayload::Transpose { .. }
                | OpPayload::Reshape { .. }
                | OpPayload::Pad { .. }
                | OpPayload::Mean { .. } => {}
                OpPayload::LstmStep { .. } => {
                    out.push_str(
                        "compile_error!(\"LstmStep code generation is not supported\");\n",
                    );
                }
            }
        }

        // Emit struct definition
        let input_tensor_id = graph.inputs.first().copied().unwrap_or(0);
        let input_tensor = graph
            .tensors
            .iter()
            .find(|t| t.id == input_tensor_id)
            .unwrap();
        let input_len: usize = graph
            .inputs
            .iter()
            .map(|id| {
                graph
                    .tensors
                    .iter()
                    .find(|tensor| tensor.id == *id)
                    .unwrap()
                    .shape
                    .total_elements()
            })
            .sum();

        let output_tensor_id = graph.outputs.first().copied().unwrap_or(0);
        let output_tensor = graph
            .tensors
            .iter()
            .find(|t| t.id == output_tensor_id)
            .unwrap();
        let output_len = output_tensor.shape.total_elements();

        out.push_str(&format!("pub struct {};\n\n", self.struct_name));
        out.push_str(&format!("impl {} {{\n", self.struct_name));
        out.push_str(&format!(
            "    pub const INPUT_DIM: usize = {};\n",
            input_len
        ));
        out.push_str(&format!(
            "    pub const OUTPUT_DIM: usize = {};\n",
            output_len
        ));
        out.push_str(&format!(
            "    pub const INPUT_SCALE: f32 = {:?}f32;\n",
            input_tensor.quant.scale
        ));
        out.push_str(&format!(
            "    pub const INPUT_ZERO_POINT: i32 = {};\n",
            input_tensor.quant.zero_point
        ));
        out.push_str(&format!(
            "    pub const OUTPUT_SCALE: f32 = {:?}f32;\n",
            output_tensor.quant.scale
        ));
        out.push_str(&format!(
            "    pub const OUTPUT_ZERO_POINT: i32 = {};\n",
            output_tensor.quant.zero_point
        ));
        out.push_str(&format!(
            "    pub const ARENA_SIZE: usize = ARENA_SIZE_BYTES;\n\n"
        ));

        // Predict method
        out.push_str("    pub fn predict<'a>(\n");
        out.push_str("        input: &[i8],\n");
        out.push_str("        arena: &'a mut [u8; ARENA_SIZE_BYTES],\n");
        if has_svdf {
            out.push_str("        svdf_state: &mut [i8; SVDF_STATE_BYTES],\n");
        }
        out.push_str("    ) -> Result<&'a [i8], &'static str> {\n");
        out.push_str(&format!("        if input.len() != {} {{\n", input_len));
        out.push_str("            return Err(\"Invalid input length\");\n");
        out.push_str("        }\n\n");

        // Copy the flattened caller input into each graph input's independently scheduled arena
        // allocation. The one-input API remains source-compatible.
        let mut input_cursor = 0usize;
        for &graph_input_id in &graph.inputs {
            let graph_input = graph
                .tensors
                .iter()
                .find(|tensor| tensor.id == graph_input_id)
                .unwrap();
            let graph_input_len = graph_input.shape.total_elements();
            let input_offset = arena_plan.offset_of(graph_input_id).unwrap_or(0);
            out.push_str(&format!(
                "        let input_slice = unsafe {{ core::slice::from_raw_parts_mut(arena.as_mut_ptr().add({}) as *mut i8, {}) }};\n",
                input_offset, graph_input_len
            ));
            out.push_str(&format!(
                "        input_slice.copy_from_slice(&input[{}..{}]);\n",
                input_cursor,
                input_cursor + graph_input_len
            ));
            input_cursor += graph_input_len;
        }
        out.push('\n');

        // Emit layer execution calls
        for layer in &graph.layers {
            let in_id = layer.inputs[0];
            let out_id = layer.outputs[0];
            let in_offset = arena_plan.offset_of(in_id).unwrap_or(0);
            let out_offset = arena_plan.offset_of(out_id).unwrap_or(0);

            let in_t = graph.tensors.iter().find(|t| t.id == in_id).unwrap();
            let out_t = graph.tensors.iter().find(|t| t.id == out_id).unwrap();
            let in_len = in_t.shape.total_elements();
            let out_len = out_t.shape.total_elements();
            let prefix = layer.name.to_uppercase();

            out.push_str(&format!("        // Layer: {}\n", layer.name));
            out.push_str(&format!(
                "        let in_buf = unsafe {{ core::slice::from_raw_parts(arena.as_ptr().add({}) as *const i8, {}) }};\n",
                in_offset, in_len
            ));
            if layer.inputs.len() > 1 {
                let input2_id = layer.inputs[1];
                let input2_offset = arena_plan.offset_of(input2_id).unwrap_or(0);
                let input2_len = graph
                    .tensors
                    .iter()
                    .find(|tensor| tensor.id == input2_id)
                    .unwrap()
                    .shape
                    .total_elements();
                out.push_str(&format!(
                    "        let in_buf2 = unsafe {{ core::slice::from_raw_parts(arena.as_ptr().add({}) as *const i8, {}) }};\n",
                    input2_offset, input2_len
                ));
            }
            out.push_str(&format!(
                "        let out_buf = unsafe {{ core::slice::from_raw_parts_mut(arena.as_mut_ptr().add({}) as *mut i8, {}) }};\n",
                out_offset, out_len
            ));

            match &layer.op {
                OpPayload::FullyConnected {
                    packed_s4,
                    bias,
                    filter_offset,
                    activation,
                    per_channel_quant,
                    ..
                } => {
                    let has_folded_bias = folded_bias(graph, layer).is_some();
                    let bias_ref = if has_folded_bias || bias.is_some() {
                        format!("Some(&{}_BIAS_S32)", prefix)
                    } else {
                        "None".into()
                    };

                    out.push_str("        let fc_params = FcParams {\n");
                    out.push_str(&format!(
                        "            input_offset: {},\n",
                        if has_folded_bias {
                            0
                        } else {
                            -in_t.quant.zero_point
                        }
                    ));
                    out.push_str(&format!("            filter_offset: {},\n", filter_offset));
                    out.push_str(&format!(
                        "            output_offset: {},\n",
                        out_t.quant.zero_point
                    ));
                    out.push_str(&format!(
                        "            activation: {},\n",
                        activation_expr(activation, &out_t.quant)
                    ));
                    out.push_str("        };\n");

                    if packed_s4.is_some() {
                        out.push_str(&format!(
                            "        let quant_params = PerTensorQuantParams::new({}, {});\n",
                            out_t.quant.multiplier, out_t.quant.shift
                        ));
                        out.push_str(&format!(
                            "        fully_connected_s4(\n            &fc_params,\n            &quant_params,\n            &Dims::new(1, 1, 1, {}),\n            in_buf,\n            &Dims::new({}, 1, 1, {}),\n            &{}_WEIGHTS_S4,\n            {},\n            &Dims::new(1, 1, 1, {}),\n            out_buf,\n        ).map_err(|_| \"FC s4 execution failed\")?;\n\n",
                            in_len, in_len, out_len, prefix, bias_ref, out_len
                        ));
                    } else if per_channel_quant.is_some() {
                        out.push_str(&format!(
                            "        let quant_params = PerChannelQuantParams::new(&{}_MULT_S32, &{}_SHIFT_S32);\n",
                            prefix, prefix
                        ));
                        out.push_str(&format!(
                            "        fully_connected_per_channel_s8(\n            &fc_params,\n            &quant_params,\n            &Dims::new(1, 1, 1, {}),\n            in_buf,\n            &Dims::new({}, 1, 1, {}),\n            &{}_WEIGHTS_S8,\n            {},\n            &Dims::new(1, 1, 1, {}),\n            out_buf,\n        ).map_err(|_| \"FC per-channel s8 execution failed\")?;\n\n",
                            in_len, in_len, out_len, prefix, bias_ref, out_len
                        ));
                    } else {
                        out.push_str(&format!(
                            "        let quant_params = PerTensorQuantParams::new({}, {});\n",
                            out_t.quant.multiplier, out_t.quant.shift
                        ));
                        out.push_str(&format!(
                            "        fully_connected_s8(\n            &fc_params,\n            &quant_params,\n            &Dims::new(1, 1, 1, {}),\n            in_buf,\n            &Dims::new({}, 1, 1, {}),\n            &{}_WEIGHTS_S8,\n            {},\n            &Dims::new(1, 1, 1, {}),\n            out_buf,\n        ).map_err(|_| \"FC s8 execution failed\")?;\n\n",
                            in_len, in_len, out_len, prefix, bias_ref, out_len
                        ));
                    }
                }
                OpPayload::Conv1D {
                    kernel_w,
                    stride_w,
                    pad_w,
                    dilation_w,
                    bias,
                    activation,
                    ..
                } => {
                    let bias_ref = if bias.is_some() {
                        format!("Some(&{}_BIAS_S32)", prefix)
                    } else {
                        "None".into()
                    };
                    let in_channels = in_t.shape.channels;
                    let in_width = in_t.shape.width;
                    let out_channels = out_t.shape.channels;
                    let out_width = out_t.shape.width;

                    out.push_str(&format!(
                        "        let conv_params = ConvParams {{\n            input_offset: {},\n            output_offset: {},\n            stride: Tile::new({}, 1),\n            padding: Padding2D::symmetric({}, 0),\n            dilation: Tile::new({}, 1),\n            activation: {},\n        }};\n",
                        -in_t.quant.zero_point, out_t.quant.zero_point, stride_w, pad_w, dilation_w, activation_expr(activation, &out_t.quant)
                    ));
                    out.push_str(&format!(
                        "        let quant_params = PerTensorQuantParams::new({}, {});\n",
                        out_t.quant.multiplier, out_t.quant.shift
                    ));
                    out.push_str(&format!(
                        "        convolve_1_x_n_s8(\n            &conv_params,\n            &quant_params,\n            &Dims::new(1, 1, {}, {}),\n            in_buf,\n            &Dims::new({}, 1, {}, {}),\n            &{}_WEIGHTS_S8,\n            {},\n            &Dims::new(1, 1, {}, {}),\n            out_buf,\n        ).map_err(|_| \"Conv1D s8 execution failed\")?;\n\n",
                        in_width, in_channels, out_channels, kernel_w, in_channels, prefix, bias_ref, out_width, out_channels
                    ));
                }
                OpPayload::Conv2D {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    padding,
                    dilation_h,
                    dilation_w,
                    packed_s4,
                    bias,
                    activation,
                    per_channel_quant,
                    ..
                } => {
                    let has_folded_bias = folded_bias(graph, layer).is_some();
                    let bias_ref = if has_folded_bias || bias.is_some() {
                        format!("Some(&{}_BIAS_S32)", prefix)
                    } else {
                        "None".into()
                    };
                    let in_channels = in_t.shape.channels;
                    let in_h = in_t.shape.height;
                    let in_w = in_t.shape.width;
                    let out_channels = out_t.shape.channels;
                    let out_h = out_t.shape.height;
                    let out_w = out_t.shape.width;
                    let weights_name = if packed_s4.is_some() {
                        format!("{}_WEIGHTS_S4", prefix)
                    } else {
                        format!("{}_WEIGHTS_S8", prefix)
                    };

                    out.push_str(&format!(
                        "        let conv_params = ConvParams {{\n            input_offset: {},\n            output_offset: {},\n            stride: Tile::new({}, {}),\n            padding: Padding2D::new({}, {}, {}, {}),\n            dilation: Tile::new({}, {}),\n            activation: {},\n        }};\n",
                        if has_folded_bias { 0 } else { -in_t.quant.zero_point }, out_t.quant.zero_point, stride_w, stride_h, padding.top, padding.bottom, padding.left, padding.right, dilation_w, dilation_h, activation_expr(activation, &out_t.quant)
                    ));

                    if packed_s4.is_some() {
                        out.push_str(&format!(
                            "        let quant_params = PerTensorQuantParams::new({}, {});\n",
                            out_t.quant.multiplier, out_t.quant.shift
                        ));
                        out.push_str(&format!(
                            "        convolve_s4(\n            &conv_params,\n            &quant_params,\n            &Dims::new(1, {}, {}, {}),\n            in_buf,\n            &Dims::new({}, {}, {}, {}),\n            &{},\n            {},\n            &Dims::new(1, {}, {}, {}),\n            out_buf,\n        ).map_err(|_| \"Conv2D s4 execution failed\")?;\n\n",
                            in_h, in_w, in_channels, out_channels, kernel_h, kernel_w, in_channels, weights_name, bias_ref, out_h, out_w, out_channels
                        ));
                    } else if per_channel_quant.is_some() {
                        out.push_str(&format!(
                            "        let quant_params = PerChannelQuantParams::new(&{}_MULT_S32, &{}_SHIFT_S32);\n",
                            prefix, prefix
                        ));
                        out.push_str(&format!(
                            "        convolve_per_channel_s8(\n            &conv_params,\n            &quant_params,\n            &Dims::new(1, {}, {}, {}),\n            in_buf,\n            &Dims::new({}, {}, {}, {}),\n            &{},\n            {},\n            &Dims::new(1, {}, {}, {}),\n            out_buf,\n        ).map_err(|_| \"Conv2D per-channel s8 execution failed\")?;\n\n",
                            in_h, in_w, in_channels, out_channels, kernel_h, kernel_w, in_channels, weights_name, bias_ref, out_h, out_w, out_channels
                        ));
                    } else {
                        out.push_str(&format!(
                            "        let quant_params = PerTensorQuantParams::new({}, {});\n",
                            out_t.quant.multiplier, out_t.quant.shift
                        ));
                        out.push_str(&format!(
                            "        convolve_s8(\n            &conv_params,\n            &quant_params,\n            &Dims::new(1, {}, {}, {}),\n            in_buf,\n            &Dims::new({}, {}, {}, {}),\n            &{},\n            {},\n            &Dims::new(1, {}, {}, {}),\n            out_buf,\n        ).map_err(|_| \"Conv2D s8 execution failed\")?;\n\n",
                            in_h, in_w, in_channels, out_channels, kernel_h, kernel_w, in_channels, weights_name, bias_ref, out_h, out_w, out_channels
                        ));
                    }
                }
                OpPayload::DepthwiseConv2D {
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    padding,
                    ch_mult,
                    bias,
                    activation,
                    ..
                } => {
                    let has_folded_bias = folded_bias(graph, layer).is_some();
                    let bias_ref = if has_folded_bias || bias.is_some() {
                        format!("Some(&{}_BIAS_S32)", prefix)
                    } else {
                        "None".into()
                    };
                    let in_channels = in_t.shape.channels;
                    let in_h = in_t.shape.height;
                    let in_w = in_t.shape.width;
                    let out_channels = out_t.shape.channels;
                    let out_h = out_t.shape.height;
                    let out_w = out_t.shape.width;

                    out.push_str(&format!(
                        "        let dw_params = DwConvParams {{\n            input_offset: {},\n            output_offset: {},\n            ch_mult: {},\n            stride: Tile::new({}, {}),\n            padding: Padding2D::new({}, {}, {}, {}),\n            dilation: Tile::new(1, 1),\n            activation: {},\n        }};\n",
                        if has_folded_bias { 0 } else { -in_t.quant.zero_point }, out_t.quant.zero_point, ch_mult, stride_w, stride_h, padding.top, padding.bottom, padding.left, padding.right, activation_expr(activation, &out_t.quant)
                    ));
                    out.push_str(&format!(
                        "        let quant_params = PerChannelQuantParams::new(&{}_MULT_S32, &{}_SHIFT_S32);\n",
                        prefix, prefix
                    ));
                    out.push_str(&format!(
                        "        depthwise_conv_per_channel_s8(\n            &dw_params,\n            &quant_params,\n            &Dims::new(1, {}, {}, {}),\n            in_buf,\n            &Dims::new(1, {}, {}, {}),\n            &{}_WEIGHTS_S8,\n            {},\n            &Dims::new(1, {}, {}, {}),\n            out_buf,\n        ).map_err(|_| \"DepthwiseConv2D per-channel s8 execution failed\")?;\n\n",
                        in_h, in_w, in_channels, kernel_h, kernel_w, out_channels, prefix, bias_ref, out_h, out_w, out_channels
                    ));
                }
                OpPayload::MaxPool2D {
                    pool_h,
                    pool_w,
                    stride_h,
                    stride_w,
                    padding,
                } => {
                    let in_h = in_t.shape.height;
                    let in_w = in_t.shape.width;
                    let channels = in_t.shape.channels;
                    let out_h = out_t.shape.height;
                    let out_w = out_t.shape.width;

                    out.push_str(&format!(
                        "        let pool_params = PoolParams {{\n            stride: Tile::new({}, {}),\n            padding: Padding2D::new({}, {}, {}, {}),\n            activation: {},\n        }};\n",
                        stride_w, stride_h, padding.top, padding.bottom, padding.left, padding.right, activation_expr(&ActivationType::None, &out_t.quant)
                    ));
                    out.push_str(&format!(
                        "        max_pool_s8(\n            &pool_params,\n            &Tile::new({}, {}),\n            &Dims::new(1, {}, {}, {}),\n            in_buf,\n            &Dims::new(1, {}, {}, {}),\n            out_buf,\n        ).map_err(|_| \"MaxPool2D s8 execution failed\")?;\n\n",
                        pool_w, pool_h, in_h, in_w, channels, out_h, out_w, channels
                    ));
                }
                OpPayload::AvgPool2D {
                    pool_h,
                    pool_w,
                    stride_h,
                    stride_w,
                    padding,
                } => {
                    let in_h = in_t.shape.height;
                    let in_w = in_t.shape.width;
                    let channels = in_t.shape.channels;
                    let out_h = out_t.shape.height;
                    let out_w = out_t.shape.width;

                    out.push_str(&format!(
                        "        let pool_params = PoolParams {{\n            stride: Tile::new({}, {}),\n            padding: Padding2D::new({}, {}, {}, {}),\n            activation: {},\n        }};\n",
                        stride_w, stride_h, padding.top, padding.bottom, padding.left, padding.right, activation_expr(&ActivationType::None, &out_t.quant)
                    ));
                    out.push_str(&format!(
                        "        avg_pool_s8(\n            &pool_params,\n            &Tile::new({}, {}),\n            &Dims::new(1, {}, {}, {}),\n            in_buf,\n            &Dims::new(1, {}, {}, {}),\n            out_buf,\n        ).map_err(|_| \"AvgPool2D s8 execution failed\")?;\n\n",
                        pool_w, pool_h, in_h, in_w, channels, out_h, out_w, channels
                    ));
                }
                OpPayload::Svdf {
                    rank,
                    bias,
                    activation,
                    ..
                } => {
                    let bias_ref = if bias.is_some() {
                        format!("Some(&{}_BIAS_S32)", prefix)
                    } else {
                        "None".into()
                    };
                    let state_offset = *svdf_state_offsets.get(&layer.id).unwrap_or(&0);

                    out.push_str(&format!(
                        "        let svdf_input_quant = PerTensorQuantParams::new({}, {});\n",
                        in_t.quant.multiplier, in_t.quant.shift
                    ));
                    out.push_str(&format!(
                        "        let svdf_output_quant = PerTensorQuantParams::new({}, {});\n",
                        out_t.quant.multiplier, out_t.quant.shift
                    ));
                    out.push_str(&format!(
                        "        let svdf_activation = {};\n",
                        activation_expr(activation, &out_t.quant)
                    ));
                    out.push_str(&format!(
                        "        svdf_s8(\n            {},\n            {},\n            {},\n            in_buf,\n            &mut svdf_state[{}..{} + {}],\n            &{}_WEIGHTS_FEATURE_S8,\n            &{}_WEIGHTS_TIME_S8,\n            {},\n            &svdf_input_quant,\n            &svdf_output_quant,\n            &svdf_activation,\n            out_buf,\n        ).map_err(|_| \"SVDF s8 execution failed\")?;\n\n",
                        -in_t.quant.zero_point, out_t.quant.zero_point, rank, state_offset, state_offset, out_len, prefix, prefix, bias_ref
                    ));
                    let _ = out_len;
                }
                OpPayload::Softmax => {
                    out.push_str(&format!(
                        "        softmax_s8(\n            in_buf,\n            1,\n            {},\n            1073741824,\n            20,\n            -256,\n            out_buf,\n        ).map_err(|_| \"Softmax s8 execution failed\")?;\n\n",
                        in_len
                    ));
                }
                OpPayload::Reshape { .. } => {
                    // Reshape only changes shape metadata; the underlying row-major data is
                    // identical, so this is a straight copy (in_len == out_len by definition).
                    out.push_str("        out_buf.copy_from_slice(in_buf);\n\n");
                }
                OpPayload::Pad {
                    padding,
                    pad_value,
                } => {
                    out.push_str(&format!(
                        "        pad_s8(\n            &Dims::new(1, {}, {}, {}),\n            in_buf,\n            &Tile::new({}, {}),\n            &Tile::new({}, {}),\n            {},\n            &Dims::new(1, {}, {}, {}),\n            out_buf,\n        ).map_err(|_| \"Pad s8 execution failed\")?;\n\n",
                        in_t.shape.height, in_t.shape.width, in_t.shape.channels,
                        padding.left, padding.top, padding.right, padding.bottom, pad_value,
                        out_t.shape.height, out_t.shape.width, out_t.shape.channels
                    ));
                }
                OpPayload::Mean {
                    reduce_height,
                    reduce_width,
                    reduce_channels,
                    ..
                } => {
                    out.push_str(&format!(
                        "        reduce_mean_s8(\n            {}, {}, {}, {},\n            {}, {}, {},\n            in_buf,\n            out_buf,\n        ).map_err(|_| \"Mean s8 execution failed\")?;\n\n",
                        in_t.shape.batches, in_t.shape.height, in_t.shape.width, in_t.shape.channels,
                        reduce_height, reduce_width, reduce_channels
                    ));
                }
                OpPayload::ElementwiseAdd { quant, activation } => {
                    out.push_str(&format!(
                        "        let add_params = ElementwiseAddParams {{\n            input1_offset: {},\n            input1_mult: {},\n            input1_shift: {},\n            input2_offset: {},\n            input2_mult: {},\n            input2_shift: {},\n            left_shift: {},\n            output_offset: {},\n            output_mult: {},\n            output_shift: {},\n            activation: {},\n        }};\n",
                        quant.input1_offset,
                        quant.input1_multiplier,
                        quant.input1_shift,
                        quant.input2_offset,
                        quant.input2_multiplier,
                        quant.input2_shift,
                        quant.left_shift,
                        quant.output_offset,
                        quant.output_multiplier,
                        quant.output_shift,
                        activation_expr(activation, &out_t.quant)
                    ));
                    out.push_str("        elementwise_add_s8(in_buf, in_buf2, out_buf, &add_params)\n            .map_err(|_| \"ElementwiseAdd s8 execution failed\")?;\n\n");
                }
                OpPayload::Transpose { kind } => match kind {
                    TransposeKind::Matrix2D { rows, cols } => {
                        out.push_str(&format!(
                            "        transpose_2d_s8({}, {}, in_buf, out_buf)\n            .map_err(|_| \"Transpose2D s8 execution failed\")?;\n\n",
                            rows, cols
                        ));
                    }
                    TransposeKind::Spatial4D => {
                        out.push_str(&format!(
                            "        transpose_spatial_s8(&Dims::new({}, {}, {}, {}), in_buf, out_buf)\n            .map_err(|_| \"TransposeSpatial s8 execution failed\")?;\n\n",
                            in_t.shape.batches,
                            in_t.shape.height,
                            in_t.shape.width,
                            in_t.shape.channels
                        ));
                    }
                },
                OpPayload::LstmStep { .. } => {
                    out.push_str(
                        "        compile_error!(\"LstmStep code generation is not supported\");\n",
                    );
                }
            }
        }

        // Return slice pointing to output buffer inside arena
        let final_out_offset = arena_plan.offset_of(output_tensor_id).unwrap_or(0);
        out.push_str(&format!(
            "        let result = unsafe {{ core::slice::from_raw_parts(arena.as_ptr().add({}) as *const i8, {}) }};\n",
            final_out_offset, output_len
        ));
        out.push_str("        Ok(result)\n");
        out.push_str("    }\n");

        // Optional floating-point boundary convenience. All model operators still execute through
        // `predict` using int8 buffers; callers own both conversion buffers so no arena-backed
        // result escapes while the output is being written.
        out.push_str("\n    pub fn predict_f32(\n");
        out.push_str("        input: &[f32],\n");
        out.push_str("        quantized_input: &mut [i8],\n");
        out.push_str("        arena: &mut [u8; ARENA_SIZE_BYTES],\n");
        if has_svdf {
            out.push_str("        svdf_state: &mut [i8; SVDF_STATE_BYTES],\n");
        }
        out.push_str("        output: &mut [f32],\n");
        out.push_str("    ) -> Result<(), &'static str> {\n");
        out.push_str("        if input.len() != Self::INPUT_DIM {\n");
        out.push_str("            return Err(\"Invalid input length\");\n");
        out.push_str("        }\n");
        out.push_str("        if quantized_input.len() != Self::INPUT_DIM {\n");
        out.push_str("            return Err(\"Invalid quantized input length\");\n");
        out.push_str("        }\n");
        out.push_str("        if output.len() != Self::OUTPUT_DIM {\n");
        out.push_str("            return Err(\"Invalid output length\");\n");
        out.push_str("        }\n\n");
        let mut input_cursor = 0usize;
        for &graph_input_id in &graph.inputs {
            let graph_input = graph
                .tensors
                .iter()
                .find(|tensor| tensor.id == graph_input_id)
                .unwrap();
            let end = input_cursor + graph_input.shape.total_elements();
            out.push_str(&format!(
                "        for (quantized, value) in quantized_input[{}..{}].iter_mut().zip(&input[{}..{}]) {{\n",
                input_cursor, end, input_cursor, end
            ));
            out.push_str(&format!(
                "            *quantized = embedded_nn::quantize_f32_to_s8(*value, {:?}f32, {});\n",
                graph_input.quant.scale, graph_input.quant.zero_point
            ));
            out.push_str("        }\n");
            input_cursor = end;
        }
        out.push('\n');
        out.push_str("        let quantized_output = Self::predict(quantized_input, arena");
        if has_svdf {
            out.push_str(", svdf_state");
        }
        out.push_str(")?;\n");
        out.push_str(
            "        for (value, quantized) in output.iter_mut().zip(quantized_output) {\n",
        );
        out.push_str("            *value = (*quantized as i32 - Self::OUTPUT_ZERO_POINT) as f32\n");
        out.push_str("                * Self::OUTPUT_SCALE;\n");
        out.push_str("        }\n");
        out.push_str("        Ok(())\n");
        out.push_str("    }\n");
        out.push_str("}\n");

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use embedded_nn_compiler::builder::ModelBuilder;

    #[test]
    fn test_generate_conv1d_graph() {
        let mut builder = ModelBuilder::new("Conv1DNet");
        let in_id = builder.add_input(
            "sensor_features",
            TensorShape::new_4d(1, 1, 16, 1),
            DataType::Int8,
            None,
        );
        let conv_id = builder.add_conv1d_layer(
            "conv1",
            in_id,
            8,
            3,
            1,
            0,
            1,
            vec![1; 8 * 3 * 1],
            Some(vec![0; 8]),
            ActivationType::Relu,
            None,
        );
        let softmax_id = builder.add_softmax("softmax_out", conv_id);
        builder.mark_output(softmax_id);
        let graph = builder.build();

        let code = RustCodeGenerator::new("Conv1DNet").generate(&graph);
        assert!(code.contains("convolve_1_x_n_s8"));
        assert!(code.contains("Activation::new(0, i8::MAX as i32)"));
        assert!(code.contains("CONV1_WEIGHTS_S8"));
    }

    #[test]
    fn test_relu_activation_clamp_uses_output_zero_point_not_literal_zero() {
        // Regression test: a real TFLite-imported model with an asymmetrically-quantized
        // (nonzero zero-point) ReLU output produced wrong numeric results because the
        // generated activation clamp ignored the output tensor's zero-point, clamping ReLU's
        // "zero" to the literal `0` instead of `output_zero_point`. Caught via
        // `crates/embedded-nn-tflite/tests/real_model_accuracy.rs` diffing against real
        // TFLite reference output.
        let mut builder = ModelBuilder::new("AsymmetricReluNet");
        let in_id = builder.add_input("features", TensorShape::new_1d(4), DataType::Int8, None);
        let dense_id = builder.add_dense_layer(
            "dense1",
            in_id,
            4,
            vec![1; 4 * 4],
            None,
            Some(vec![0; 4]),
            ActivationType::Relu,
            None,
            Some(QuantParams {
                multiplier: 1073741824,
                shift: 0,
                zero_point: -128,
                scale: 0.01,
            }),
        );
        builder.mark_output(dense_id);
        let graph = builder.build();

        let code = RustCodeGenerator::new("AsymmetricReluNet").generate(&graph);
        assert!(
            code.contains("Activation::new(-128, i8::MAX as i32)"),
            "expected ReLU clamp to use the output zero-point (-128), got:\n{code}"
        );
        assert!(!code.contains("Activation::new(0, i8::MAX as i32)"));
    }

    #[test]
    fn test_generate_per_channel_fc_graph() {
        let mut builder = ModelBuilder::new("PerChannelNet");
        let in_id = builder.add_input("input", TensorShape::new_1d(4), DataType::Int8, None);
        let fc_id = builder.add_dense_layer(
            "dense1",
            in_id,
            2,
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            None,
            Some(vec![0, 0]),
            ActivationType::None,
            Some(PerChannelQuant {
                multipliers: vec![1073741824, 1073741824],
                shifts: vec![0, 0],
            }),
            None,
        );
        let softmax_id = builder.add_softmax("softmax_out", fc_id);
        builder.mark_output(softmax_id);
        let graph = builder.build();

        let code = RustCodeGenerator::new("PerChannelNet").generate(&graph);
        assert!(code.contains("fully_connected_per_channel_s8"));
        assert!(code.contains("PerChannelQuantParams"));
        assert!(code.contains("DENSE1_MULT_S32"));
        assert!(code.contains("DENSE1_SHIFT_S32"));
    }

    #[test]
    fn test_generate_fc_graph_threads_asymmetric_zero_points() {
        let mut builder = ModelBuilder::new("AsymmetricNet");
        let asymmetric_input = QuantParams {
            multiplier: 1073741824,
            shift: 0,
            zero_point: -50,
            scale: 0.01,
        };
        let in_id = builder.add_input(
            "input",
            TensorShape::new_1d(4),
            DataType::Int8,
            Some(asymmetric_input),
        );
        let asymmetric_output = QuantParams {
            multiplier: 1073741824,
            shift: 0,
            zero_point: 30,
            scale: 0.02,
        };
        let fc_id = builder.add_dense_layer(
            "dense1",
            in_id,
            2,
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            None,
            Some(vec![0, 0]),
            ActivationType::None,
            None,
            Some(asymmetric_output),
        );
        let softmax_id = builder.add_softmax("softmax_out", fc_id);
        builder.mark_output(softmax_id);
        let graph = builder.build();

        let code = RustCodeGenerator::new("AsymmetricNet").generate(&graph);
        // The symmetric s8 FC correction is folded into bias: [50*10, 50*26].
        assert!(code.contains("static DENSE1_BIAS_S32: [i32; 2] = [\n    500, 1300,"));
        assert!(code.contains("input_offset: 0,"));
        assert!(code.contains("output_offset: 30,"));
    }

    fn asymmetric_quant(zero_point: i32) -> QuantParams {
        QuantParams {
            multiplier: 1073741824,
            shift: 0,
            zero_point,
            scale: 0.25,
        }
    }

    #[test]
    fn test_folded_fc_bias_handles_bias_and_no_bias() {
        for (bias, expected) in [(Some(vec![5, -5]), [45, -45]), (None, [40, -40])] {
            let mut builder = ModelBuilder::new("FoldedFc");
            let input = builder.add_input(
                "input",
                TensorShape::new_1d(4),
                DataType::Int8,
                Some(asymmetric_quant(-4)),
            );
            let output = builder.add_dense_layer(
                "fc",
                input,
                2,
                vec![1, 2, 3, 4, -1, -2, -3, -4],
                None,
                bias,
                ActivationType::None,
                None,
                None,
            );
            builder.mark_output(output);

            let code = RustCodeGenerator::new("FoldedFc").generate(&builder.build());
            assert!(code.contains(&format!(
                "static FC_BIAS_S32: [i32; 2] = [\n    {}, {},",
                expected[0], expected[1]
            )));
            assert!(code.contains("input_offset: 0,"));
            assert!(code.contains("Some(&FC_BIAS_S32)"));
        }
    }

    #[test]
    fn test_folded_conv_and_depthwise_bias_follow_weight_layouts() {
        let mut conv_builder = ModelBuilder::new("FoldedConv");
        let conv_input = conv_builder.add_input(
            "input",
            TensorShape::new_4d(1, 3, 3, 1),
            DataType::Int8,
            Some(asymmetric_quant(-7)),
        );
        let conv_output = conv_builder.add_conv2d_layer(
            "conv",
            conv_input,
            2,
            2,
            2,
            1,
            1,
            Padding2D::default(),
            1,
            1,
            vec![1, 2, 3, 4, -1, -2, -3, -4],
            None,
            Some(vec![5, -5]),
            ActivationType::None,
            None,
            None,
        );
        conv_builder.mark_output(conv_output);
        let conv_code = RustCodeGenerator::new("FoldedConv").generate(&conv_builder.build());
        assert!(conv_code.contains("static CONV_BIAS_S32: [i32; 2] = [\n    75, -75,"));
        assert!(conv_code.contains("input_offset: 0,"));

        let mut depthwise_builder = ModelBuilder::new("FoldedDepthwise");
        let depthwise_input = depthwise_builder.add_input(
            "input",
            TensorShape::new_4d(1, 2, 2, 2),
            DataType::Int8,
            Some(asymmetric_quant(3)),
        );
        let depthwise_output = depthwise_builder.add_depthwise_conv2d_layer(
            "depthwise",
            depthwise_input,
            1,
            2,
            2,
            1,
            1,
            Padding2D::default(),
            // Depthwise layout is [kernel_h, kernel_w, output_channel].
            vec![1, 10, 2, 20, 3, 30, 4, 40],
            None,
            ActivationType::None,
            Some(PerChannelQuant {
                multipliers: vec![1073741824; 2],
                shifts: vec![0; 2],
            }),
            None,
        );
        depthwise_builder.mark_output(depthwise_output);
        let depthwise_code =
            RustCodeGenerator::new("FoldedDepthwise").generate(&depthwise_builder.build());
        assert!(depthwise_code.contains("static DEPTHWISE_BIAS_S32: [i32; 2] = [\n    -30, -300,"));
        assert!(depthwise_code.contains("input_offset: 0,"));
    }

    #[test]
    fn test_folded_parameters_execute_identically_to_runtime_offset_path() {
        use embedded_nn::Padding2D as RuntimePadding2D;
        use embedded_nn::{
            Activation, ConvParams, Dims, DwConvParams, FcParams, PerChannelQuantParams,
            PerTensorQuantParams, Tile, convolve_s8, depthwise_conv_per_channel_s8,
            fully_connected_s8,
        };

        let input = [3, -2, 5, 1];
        let weights = [1, 2, 3, 4, -1, -2, -3, -4];
        let quant = PerTensorQuantParams::new(1073741824, 0);
        let original_fc = FcParams {
            input_offset: 4,
            filter_offset: 0,
            output_offset: 0,
            activation: Activation::int8_unconstrained(),
        };
        let folded_fc = FcParams {
            input_offset: 0,
            ..original_fc
        };
        let mut reference = [0; 2];
        let mut optimized = [0; 2];
        fully_connected_s8(
            &original_fc,
            &quant,
            &Dims::new(1, 1, 1, 4),
            &input,
            &Dims::new(4, 1, 1, 2),
            &weights,
            Some(&[5, -5]),
            &Dims::new(1, 1, 1, 2),
            &mut reference,
        )
        .unwrap();
        fully_connected_s8(
            &folded_fc,
            &quant,
            &Dims::new(1, 1, 1, 4),
            &input,
            &Dims::new(4, 1, 1, 2),
            &weights,
            Some(&[45, -45]),
            &Dims::new(1, 1, 1, 2),
            &mut optimized,
        )
        .unwrap();
        assert_eq!(optimized, reference);

        let conv_input = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let original_conv = ConvParams {
            input_offset: 7,
            output_offset: 0,
            stride: Tile::new(1, 1),
            padding: RuntimePadding2D::default(),
            dilation: Tile::new(1, 1),
            activation: Activation::int8_unconstrained(),
        };
        let folded_conv = ConvParams {
            input_offset: 0,
            ..original_conv
        };
        let mut conv_reference = [0; 8];
        let mut conv_optimized = [0; 8];
        convolve_s8(
            &original_conv,
            &quant,
            &Dims::new(1, 3, 3, 1),
            &conv_input,
            &Dims::new(2, 2, 2, 1),
            &weights,
            Some(&[5, -5]),
            &Dims::new(1, 2, 2, 2),
            &mut conv_reference,
        )
        .unwrap();
        convolve_s8(
            &folded_conv,
            &quant,
            &Dims::new(1, 3, 3, 1),
            &conv_input,
            &Dims::new(2, 2, 2, 1),
            &weights,
            Some(&[75, -75]),
            &Dims::new(1, 2, 2, 2),
            &mut conv_optimized,
        )
        .unwrap();
        assert_eq!(conv_optimized, conv_reference);

        let depthwise_weights = [1, 10, 2, 20, 3, 30, 4, 40];
        let depthwise_input = [3, -2, 5, 1, -4, 2, 6, -1];
        let depthwise_quant = PerChannelQuantParams::new(&[1073741824, 1073741824], &[0, 0]);
        let original_depthwise = DwConvParams {
            input_offset: -3,
            output_offset: 0,
            ch_mult: 1,
            stride: Tile::new(1, 1),
            padding: RuntimePadding2D::default(),
            dilation: Tile::new(1, 1),
            activation: Activation::int8_unconstrained(),
        };
        let folded_depthwise = DwConvParams {
            input_offset: 0,
            ..original_depthwise
        };
        let mut depthwise_reference = [0; 2];
        let mut depthwise_optimized = [0; 2];
        depthwise_conv_per_channel_s8(
            &original_depthwise,
            &depthwise_quant,
            &Dims::new(1, 2, 2, 2),
            &depthwise_input,
            &Dims::new(1, 2, 2, 2),
            &depthwise_weights,
            None,
            &Dims::new(1, 1, 1, 2),
            &mut depthwise_reference,
        )
        .unwrap();
        depthwise_conv_per_channel_s8(
            &folded_depthwise,
            &depthwise_quant,
            &Dims::new(1, 2, 2, 2),
            &depthwise_input,
            &Dims::new(1, 2, 2, 2),
            &depthwise_weights,
            Some(&[-30, -300]),
            &Dims::new(1, 1, 1, 2),
            &mut depthwise_optimized,
        )
        .unwrap();
        assert_eq!(depthwise_optimized, depthwise_reference);
    }

    #[test]
    fn test_padding_asymmetric_filter_and_s4_skip_folding() {
        let mut padded_builder = ModelBuilder::new("PaddedConv");
        let input = padded_builder.add_input(
            "input",
            TensorShape::new_4d(1, 2, 2, 1),
            DataType::Int8,
            Some(asymmetric_quant(-6)),
        );
        let output = padded_builder.add_conv2d_layer(
            "padded",
            input,
            1,
            3,
            3,
            1,
            1,
            Padding2D::symmetric(1, 1),
            1,
            1,
            vec![1; 9],
            None,
            Some(vec![11]),
            ActivationType::None,
            None,
            None,
        );
        padded_builder.mark_output(output);
        let padded_code = RustCodeGenerator::new("PaddedConv").generate(&padded_builder.build());
        assert!(padded_code.contains("static PADDED_BIAS_S32: [i32; 1] = [\n    11,"));
        assert!(padded_code.contains("input_offset: 6,"));

        let mut asymmetric_filter_builder = ModelBuilder::new("AsymmetricFilterFc");
        let input = asymmetric_filter_builder.add_input(
            "input",
            TensorShape::new_1d(2),
            DataType::Int8,
            Some(asymmetric_quant(-6)),
        );
        let output = asymmetric_filter_builder.add_dense_layer(
            "fc",
            input,
            1,
            vec![1, 2],
            None,
            Some(vec![11]),
            ActivationType::None,
            None,
            None,
        );
        asymmetric_filter_builder
            .set_fully_connected_filter_offset(output, 2)
            .unwrap();
        asymmetric_filter_builder.mark_output(output);
        let asymmetric_filter_code = RustCodeGenerator::new("AsymmetricFilterFc")
            .generate(&asymmetric_filter_builder.build());
        assert!(asymmetric_filter_code.contains("static FC_BIAS_S32: [i32; 1] = [\n    11,"));
        assert!(asymmetric_filter_code.contains("input_offset: 6,"));
        assert!(asymmetric_filter_code.contains("filter_offset: 2,"));

        let mut s4_builder = ModelBuilder::new("S4Fc");
        let input = s4_builder.add_input(
            "input",
            TensorShape::new_1d(2),
            DataType::Int8,
            Some(asymmetric_quant(-6)),
        );
        let output = s4_builder.add_dense_layer(
            "fc",
            input,
            1,
            vec![1, 2],
            Some(vec![0x21]),
            None,
            ActivationType::None,
            None,
            None,
        );
        s4_builder.mark_output(output);
        let s4_code = RustCodeGenerator::new("S4Fc").generate(&s4_builder.build());
        assert!(!s4_code.contains("FC_BIAS_S32"));
        assert!(s4_code.contains("input_offset: 6,"));
        assert!(s4_code.contains("fully_connected_s4("));
    }

    #[test]
    fn test_generate_quantization_constants_and_f32_boundary_api() {
        let input_quant = QuantParams {
            multiplier: 1073741824,
            shift: 0,
            zero_point: -12,
            scale: 0.25,
        };
        let output_quant = QuantParams {
            multiplier: 1073741824,
            shift: 0,
            zero_point: 7,
            scale: 0.5,
        };
        let mut builder = ModelBuilder::new("FloatBoundaryNet");
        let in_id = builder.add_input(
            "input",
            TensorShape::new_1d(2),
            DataType::Int8,
            Some(input_quant),
        );
        let out_id = builder.add_dense_layer(
            "dense",
            in_id,
            1,
            vec![1, 1],
            None,
            Some(vec![0]),
            ActivationType::None,
            None,
            Some(output_quant),
        );
        builder.mark_output(out_id);

        let code = RustCodeGenerator::new("FloatBoundaryNet").generate(&builder.build());
        assert!(code.contains("pub const INPUT_SCALE: f32 = 0.25f32;"));
        assert!(code.contains("pub const INPUT_ZERO_POINT: i32 = -12;"));
        assert!(code.contains("pub const OUTPUT_SCALE: f32 = 0.5f32;"));
        assert!(code.contains("pub const OUTPUT_ZERO_POINT: i32 = 7;"));
        assert!(code.contains("pub fn predict_f32("));
        assert!(code.contains("quantized_input: &mut [i8]"));
        assert!(code.contains("let quantized_output = Self::predict("));
        assert!(code.contains("* Self::OUTPUT_SCALE;"));
    }

    #[test]
    fn test_generate_svdf_graph() {
        let mut builder = ModelBuilder::new("SvdfNet");
        let in_id = builder.add_input(
            "sensor_features",
            TensorShape::new_1d(4),
            DataType::Int8,
            None,
        );
        let svdf_id = builder.add_svdf_layer(
            "svdf1",
            in_id,
            16,
            1,
            4,
            vec![1; 16 * 4],
            vec![1; 16 * 4],
            Some(vec![0; 16]),
            ActivationType::None,
            None,
        );
        let softmax_id = builder.add_softmax("softmax_out", svdf_id);
        builder.mark_output(softmax_id);
        let graph = builder.build();

        let code = RustCodeGenerator::new("SvdfNet").generate(&graph);
        assert!(code.contains("svdf_s8"));
        assert!(code.contains("SVDF_STATE_BYTES"));
        assert!(code.contains("svdf_state: &mut [i8; SVDF_STATE_BYTES]"));
    }

    #[test]
    fn test_generate_conv2d_plain_graph() {
        let mut builder = ModelBuilder::new("Conv2DNet");
        let in_id = builder.add_input(
            "input",
            TensorShape::new_4d(1, 8, 8, 3),
            DataType::Int8,
            None,
        );
        let conv_id = builder.add_conv2d_layer(
            "conv2d_1",
            in_id,
            4,
            3,
            3,
            1,
            1,
            Padding2D::default(),
            1,
            1,
            vec![1; 4 * 3 * 3 * 3],
            None,
            Some(vec![0; 4]),
            ActivationType::Relu,
            None,
            None,
        );
        let softmax_id = builder.add_softmax("softmax_out", conv_id);
        builder.mark_output(softmax_id);
        let graph = builder.build();

        let code = RustCodeGenerator::new("Conv2DNet").generate(&graph);
        assert!(code.contains("convolve_s8("));
        assert!(!code.contains("convolve_per_channel_s8"));
        assert!(code.contains("CONV2D_1_WEIGHTS_S8"));
    }

    #[test]
    fn test_generate_conv2d_per_channel_graph() {
        let mut builder = ModelBuilder::new("Conv2DPerChannelNet");
        let in_id = builder.add_input(
            "input",
            TensorShape::new_4d(1, 8, 8, 3),
            DataType::Int8,
            None,
        );
        let conv_id = builder.add_conv2d_layer(
            "conv2d_1",
            in_id,
            4,
            3,
            3,
            1,
            1,
            Padding2D::default(),
            1,
            1,
            vec![1; 4 * 3 * 3 * 3],
            None,
            Some(vec![0; 4]),
            ActivationType::Relu,
            Some(PerChannelQuant {
                multipliers: vec![1073741824; 4],
                shifts: vec![0; 4],
            }),
            None,
        );
        let softmax_id = builder.add_softmax("softmax_out", conv_id);
        builder.mark_output(softmax_id);
        let graph = builder.build();

        let code = RustCodeGenerator::new("Conv2DPerChannelNet").generate(&graph);
        assert!(code.contains("convolve_per_channel_s8"));
        assert!(code.contains("CONV2D_1_MULT_S32"));
        assert!(code.contains("CONV2D_1_SHIFT_S32"));
    }

    #[test]
    fn test_generate_depthwise_conv2d_graph() {
        let mut builder = ModelBuilder::new("DwConvNet");
        let in_id = builder.add_input(
            "input",
            TensorShape::new_4d(1, 8, 8, 4),
            DataType::Int8,
            None,
        );
        let dw_id = builder.add_depthwise_conv2d_layer(
            "dwconv1",
            in_id,
            2,
            3,
            3,
            1,
            1,
            Padding2D::default(),
            vec![1; 4 * 2 * 3 * 3],
            Some(vec![0; 8]),
            ActivationType::Relu,
            Some(PerChannelQuant {
                multipliers: vec![1073741824; 8],
                shifts: vec![0; 8],
            }),
            None,
        );
        let softmax_id = builder.add_softmax("softmax_out", dw_id);
        builder.mark_output(softmax_id);
        let graph = builder.build();

        let code = RustCodeGenerator::new("DwConvNet").generate(&graph);
        assert!(code.contains("depthwise_conv_per_channel_s8"));
        assert!(code.contains("DwConvParams"));
        assert!(code.contains("DWCONV1_WEIGHTS_S8"));
        assert!(code.contains("DWCONV1_MULT_S32"));
        // Regression: DepthwiseConv2D's own params construct `Tile::new(...)` directly (stride/
        // padding/dilation), so `Tile` must be imported even when there's no Conv1D/Conv2D/pool
        // layer in the graph to otherwise trigger it.
        assert!(code.contains("Tile,") || code.contains("Tile\n"));
        assert!(code.contains("Tile::new"));
    }

    #[test]
    fn test_generate_conv2d_maxpool_avgpool_chain_compiles_markers() {
        let mut builder = ModelBuilder::new("PoolNet");
        let in_id = builder.add_input(
            "input",
            TensorShape::new_4d(1, 8, 8, 3),
            DataType::Int8,
            None,
        );
        let conv_id = builder.add_conv2d_layer(
            "conv2d_1",
            in_id,
            4,
            3,
            3,
            1,
            1,
            Padding2D::default(),
            1,
            1,
            vec![1; 4 * 3 * 3 * 3],
            None,
            Some(vec![0; 4]),
            ActivationType::Relu,
            None,
            None,
        );
        let pool_id =
            builder.add_maxpool2d_layer("maxpool1", conv_id, 2, 2, 2, 2, Padding2D::default());
        let avg_id =
            builder.add_avgpool2d_layer("avgpool1", pool_id, 2, 2, 2, 2, Padding2D::default());
        builder.mark_output(avg_id);
        let graph = builder.build();

        let code = RustCodeGenerator::new("PoolNet").generate(&graph);
        assert!(code.contains("convolve_s8("));
        assert!(code.contains("max_pool_s8"));
        assert!(code.contains("avg_pool_s8"));
        assert!(code.contains("PoolParams"));
    }

    #[test]
    fn test_generate_reshape_graph_copies_buffer() {
        let mut builder = ModelBuilder::new("ReshapeNet");
        let in_id = builder.add_input(
            "input",
            TensorShape::new_4d(1, 2, 2, 4),
            DataType::Int8,
            None,
        );
        let reshape_id = builder.add_reshape_layer("reshape1", in_id, TensorShape::new_1d(16));
        let fc_id = builder.add_dense_layer(
            "dense1",
            reshape_id,
            4,
            vec![1; 16 * 4],
            None,
            Some(vec![0; 4]),
            ActivationType::None,
            None,
            None,
        );
        let softmax_id = builder.add_softmax("softmax_out", fc_id);
        builder.mark_output(softmax_id);
        let graph = builder.build();

        let code = RustCodeGenerator::new("ReshapeNet").generate(&graph);
        assert!(code.contains("out_buf.copy_from_slice(in_buf)"));
    }

    #[test]
    fn test_generate_binary_add_uses_second_scheduled_buffer() {
        let mut builder = ModelBuilder::new("AddNet");
        let left = builder.add_input("left", TensorShape::new_1d(4), DataType::Int8, None);
        let right = builder.add_input("right", TensorShape::new_1d(4), DataType::Int8, None);
        let output = builder
            .add_elementwise_add_layer(
                "add",
                left,
                right,
                ActivationType::Relu,
                QuantParams::default(),
            )
            .unwrap();
        builder.mark_output(output);
        let graph = builder.build();
        let plan = ArenaScheduler::schedule(&graph);
        let code = RustCodeGenerator::new("AddNet").generate(&graph);

        assert!(code.contains("elementwise_add_s8(in_buf, in_buf2, out_buf"));
        assert!(code.contains("ElementwiseAddParams"));
        assert!(code.contains("left_shift: 20"));
        assert!(code.contains("pub const INPUT_DIM: usize = 8;"));
        assert_ne!(plan.offset_of(left), plan.offset_of(right));
        assert_ne!(plan.offset_of(left), plan.offset_of(output));
        assert_ne!(plan.offset_of(right), plan.offset_of(output));
    }

    #[test]
    fn test_generate_both_supported_transpose_kernels() {
        let mut matrix_builder = ModelBuilder::new("MatrixTranspose");
        let matrix =
            matrix_builder.add_input("input", TensorShape::new_2d(2, 3), DataType::Int8, None);
        let matrix_out = matrix_builder
            .add_transpose_layer("transpose", matrix, &[1, 0])
            .unwrap();
        matrix_builder.mark_output(matrix_out);
        let matrix_code =
            RustCodeGenerator::new("MatrixTranspose").generate(&matrix_builder.build());
        assert!(matrix_code.contains("transpose_2d_s8(2, 3"));

        let mut spatial_builder = ModelBuilder::new("SpatialTranspose");
        let spatial = spatial_builder.add_input(
            "input",
            TensorShape::new_4d(1, 2, 3, 4),
            DataType::Int8,
            None,
        );
        let spatial_out = spatial_builder
            .add_transpose_layer("transpose", spatial, &[0, 2, 1, 3])
            .unwrap();
        spatial_builder.mark_output(spatial_out);
        let spatial_code =
            RustCodeGenerator::new("SpatialTranspose").generate(&spatial_builder.build());
        assert!(spatial_code.contains("transpose_spatial_s8"));
        assert!(spatial_code.contains("Dims::new(1, 2, 3, 4)"));
    }
}
