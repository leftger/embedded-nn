use embedded_nn_compiler::builder::ModelBuilder;
use embedded_nn_compiler::ir::{ActivationType, DataType, ModelGraph, QuantParams, TensorShape};
use embedded_nn_compiler::quant::{
    calculate_asymmetric_quant_s8, calculate_output_requant_multiplier, calculate_symmetric_quant_s8,
    quantize_weights_s8,
};

const INPUT_SCALE: f32 = 1.0 / 127.0;

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn dense(x: &[f32], weights: &[f32], bias: &[f32], out: usize, inn: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out];
    for o in 0..out {
        let mut sum = bias[o];
        for i in 0..inn {
            sum += x[i] * weights[o * inn + i];
        }
        y[o] = sum;
    }
    y
}

fn output_quant(input_scale: f32, weight_scale: f32, min: f32, max: f32) -> QuantParams {
    let range = calculate_asymmetric_quant_s8(min, max);
    let (multiplier, shift) =
        calculate_output_requant_multiplier(input_scale, weight_scale, range.scale);
    QuantParams {
        multiplier,
        shift,
        zero_point: range.zero_point,
        scale: range.scale,
    }
}

fn quantize_bias(bias: &[f32], input_scale: f32, weight_scale: f32) -> Vec<i32> {
    let denom = (input_scale * weight_scale).max(1e-12);
    bias.iter()
        .map(|&b| (b / denom).round().clamp(i32::MIN as f32, i32::MAX as f32) as i32)
        .collect()
}

fn quantize_input(x: &[f32]) -> Vec<i8> {
    x.iter()
        .map(|&v| ((v / INPUT_SCALE).round().clamp(-128.0, 127.0)) as i8)
        .collect()
}

/// Post-training quantization of a two-layer ReLU MLP into a `ModelGraph`.
pub fn ptq_dense_mlp(
    name: &str,
    weights_fc1: &[f32],
    bias_fc1: &[f32],
    weights_fc2: &[f32],
    bias_fc2: &[f32],
    features: &[Vec<f32>],
) -> ModelGraph {
    let num_inputs = if weights_fc1.is_empty() {
        0
    } else {
        weights_fc1.len() / bias_fc1.len()
    };
    let hidden = bias_fc1.len();
    let classes = bias_fc2.len();

    let mut hidden_lo = f32::INFINITY;
    let mut hidden_hi = f32::NEG_INFINITY;
    let mut logit_lo = f32::INFINITY;
    let mut logit_hi = f32::NEG_INFINITY;
    for x in features {
        let h = dense(x, weights_fc1, bias_fc1, hidden, num_inputs);
        let h: Vec<f32> = h.into_iter().map(relu).collect();
        for &v in &h {
            hidden_lo = hidden_lo.min(v);
            hidden_hi = hidden_hi.max(v);
        }
        let logits = dense(&h, weights_fc2, bias_fc2, classes, hidden);
        for &v in &logits {
            logit_lo = logit_lo.min(v);
            logit_hi = logit_hi.max(v);
        }
    }
    if !hidden_lo.is_finite() {
        hidden_lo = 0.0;
        hidden_hi = 1.0;
    }
    if !logit_lo.is_finite() {
        logit_lo = -1.0;
        logit_hi = 1.0;
    }

    let w1_abs = weights_fc1.iter().fold(0.1f32, |a, w| a.max(w.abs()));
    let w2_abs = weights_fc2.iter().fold(0.1f32, |a, w| a.max(w.abs()));
    let q1 = calculate_symmetric_quant_s8(w1_abs);
    let q2 = calculate_symmetric_quant_s8(w2_abs);
    let w1_s8 = quantize_weights_s8(weights_fc1, q1.scale);
    let w2_s8 = quantize_weights_s8(weights_fc2, q2.scale);
    let b1 = quantize_bias(bias_fc1, INPUT_SCALE, q1.scale);
    let hidden_quant = output_quant(INPUT_SCALE, q1.scale, hidden_lo, hidden_hi);
    let b2 = quantize_bias(bias_fc2, hidden_quant.scale, q2.scale);
    let out_quant = output_quant(hidden_quant.scale, q2.scale, logit_lo, logit_hi);

    let mut builder = ModelBuilder::new(name);
    let in_id = builder.add_input(
        "features",
        TensorShape::new_1d(num_inputs),
        DataType::Int8,
        Some(QuantParams {
            multiplier: 1_073_741_824,
            shift: 0,
            zero_point: 0,
            scale: INPUT_SCALE,
        }),
    );
    let hidden_id = builder.add_dense_layer(
        "fc1",
        in_id,
        hidden,
        w1_s8,
        None,
        Some(b1),
        ActivationType::Relu,
        None,
        Some(hidden_quant),
    );
    let out_id = builder.add_dense_layer(
        "fc2",
        hidden_id,
        classes,
        w2_s8,
        None,
        Some(b2),
        ActivationType::None,
        None,
        Some(out_quant),
    );
    builder.mark_output(out_id);
    builder.build()
}

pub fn quantize_features(features: &[Vec<f32>]) -> Vec<Vec<i8>> {
    features.iter().map(|x| quantize_input(x)).collect()
}
