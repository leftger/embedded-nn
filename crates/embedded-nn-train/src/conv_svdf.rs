use burn::grad_clipping::GradientClippingConfig;
use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use embedded_nn_compiler::HostInterpreter;
use embedded_nn_compiler::builder::ModelBuilder;
use embedded_nn_compiler::ir::{ActivationType, DataType, QuantParams, TensorShape};
use embedded_nn_compiler::quant::{
    calculate_asymmetric_quant_s8, calculate_output_requant_multiplier,
    calculate_symmetric_quant_s8, quantize_weights_s8,
};

use crate::mlp::{
    QuantCompare, TrainArch, TrainB, TrainConfig, TrainMode, TrainReport, flatten_linear,
    maybe_fake_quant, maybe_fake_quant_act, relu, train_dense_mlp,
};
use crate::quantize::{INPUT_SCALE, quantize_features};

#[derive(Module, Debug, Clone)]
struct ConvNet {
    conv: Conv1d<TrainB>,
    fc1: Linear<TrainB>,
    fc2: Linear<TrainB>,
}

impl ConvNet {
    fn new(
        mel: usize,
        frames: usize,
        kernel_w: usize,
        out_ch: usize,
        hidden: usize,
        classes: usize,
        device: &<TrainB as Backend>::Device,
    ) -> Self {
        let out_w = frames - kernel_w + 1;
        Self {
            conv: Conv1dConfig::new(mel, out_ch, kernel_w).init(device),
            fc1: LinearConfig::new(out_ch * out_w, hidden).init(device),
            fc2: LinearConfig::new(hidden, classes).init(device),
        }
    }

    fn forward(&self, x: Tensor<TrainB, 3>, fake_quant: bool) -> Tensor<TrainB, 2> {
        let y = relu(self.conv.forward(x).swap_dims(1, 2).flatten(1, 2));
        let y = maybe_fake_quant_act(y, fake_quant);
        let h = relu(self.fc1.forward(y));
        let h = maybe_fake_quant_act(h, fake_quant);
        let w2 = maybe_fake_quant(self.fc2.weight.val(), fake_quant);
        h.matmul(w2)
            + self
                .fc2
                .bias
                .as_ref()
                .map(|b| b.val().unsqueeze())
                .expect("bias")
    }
}

fn flatten_conv1d(
    conv: &Conv1d<TrainB>,
    out_ch: usize,
    inn: usize,
    k: usize,
) -> (Vec<f32>, Vec<f32>) {
    let w: Vec<f32> = conv.weight.val().into_data().to_vec().expect("conv w");
    let mut ir = vec![0.0f32; out_ch * k * inn];
    for oc in 0..out_ch {
        for ic in 0..inn {
            for ki in 0..k {
                ir[(oc * k + ki) * inn + ic] = w[oc * inn * k + ic * k + ki];
            }
        }
    }
    let b = conv
        .bias
        .as_ref()
        .map(|b| b.val().into_data().to_vec().expect("conv b"))
        .unwrap_or_else(|| vec![0.0; out_ch]);
    (ir, b)
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

fn relu_hi(values: impl Iterator<Item = f32>) -> f32 {
    values.fold(0.0f32, f32::max).max(0.1)
}

fn logits_range(lo: f32, hi: f32) -> (f32, f32) {
    if lo.is_finite() && hi > lo {
        (lo, hi)
    } else {
        (-1.0, 1.0)
    }
}

/// Float Conv1D + ReLU + Dense head matching the exported integer graph.
/// `x` is channel-major `[mel, frames]` as produced by Studio for Burn Conv1d.
#[allow(clippy::too_many_arguments)]
fn forward_conv1d_float(
    x: &[f32],
    weights_conv: &[f32],
    bias_conv: &[f32],
    weights_fc1: &[f32],
    bias_fc1: &[f32],
    weights_fc2: &[f32],
    bias_fc2: &[f32],
    num_frames: usize,
    kernel_w: usize,
    out_ch: usize,
    mel: usize,
    hidden: usize,
    classes: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let out_w = num_frames - kernel_w + 1;
    let conv_len = out_ch * out_w;
    let mut conv_act = vec![0.0f32; conv_len];
    for oc in 0..out_ch {
        for ow in 0..out_w {
            let mut sum = bias_conv[oc];
            for k in 0..kernel_w {
                let t = ow + k;
                for ic in 0..mel {
                    sum += x[ic * num_frames + t] * weights_conv[(oc * kernel_w + k) * mel + ic];
                }
            }
            conv_act[ow * out_ch + oc] = sum.max(0.0);
        }
    }
    let mut hidden_act = vec![0.0f32; hidden];
    for h in 0..hidden {
        let mut sum = bias_fc1[h];
        for i in 0..conv_len {
            sum += conv_act[i] * weights_fc1[h * conv_len + i];
        }
        hidden_act[h] = sum.max(0.0);
    }
    let mut logits = vec![0.0f32; classes];
    for c in 0..classes {
        let mut sum = bias_fc2[c];
        for h in 0..hidden {
            sum += hidden_act[h] * weights_fc2[c * hidden + h];
        }
        logits[c] = sum;
    }
    (conv_act, hidden_act, logits)
}

#[allow(clippy::too_many_arguments)]
fn calibrate_conv1d(
    features: &[Vec<f32>],
    weights_conv: &[f32],
    bias_conv: &[f32],
    weights_fc1: &[f32],
    bias_fc1: &[f32],
    weights_fc2: &[f32],
    bias_fc2: &[f32],
    num_frames: usize,
    kernel_w: usize,
    out_ch: usize,
    mel: usize,
    hidden: usize,
    classes: usize,
) -> ((f32, f32), (f32, f32), (f32, f32)) {
    let mut conv_hi = 0.0f32;
    let mut hidden_hi = 0.0f32;
    let mut logit_lo = f32::INFINITY;
    let mut logit_hi = f32::NEG_INFINITY;
    let expect_len = mel * num_frames;
    for x in features {
        if x.len() != expect_len {
            continue;
        }
        let (conv_act, hidden_act, logits) = forward_conv1d_float(
            x,
            weights_conv,
            bias_conv,
            weights_fc1,
            bias_fc1,
            weights_fc2,
            bias_fc2,
            num_frames,
            kernel_w,
            out_ch,
            mel,
            hidden,
            classes,
        );
        conv_hi = conv_hi.max(relu_hi(conv_act.iter().copied()));
        hidden_hi = hidden_hi.max(relu_hi(hidden_act.iter().copied()));
        for &l in &logits {
            logit_lo = logit_lo.min(l);
            logit_hi = logit_hi.max(l);
        }
    }
    (
        (0.0, conv_hi.max(0.1)),
        (0.0, hidden_hi.max(0.1)),
        logits_range(logit_lo, logit_hi),
    )
}

#[allow(clippy::too_many_arguments)]
fn forward_svdf_float(
    x: &[f32],
    wf: &[f32],
    wt: &[f32],
    sb: &[f32],
    w2: &[f32],
    b2: &[f32],
    units: usize,
    rank: usize,
    memory: usize,
    mel: usize,
    classes: usize,
    num_frames: usize,
) -> (Vec<f32>, Vec<f32>) {
    let feature_dim = units * rank;
    let lookback_start = num_frames.saturating_sub(memory);
    let lookback_len = num_frames - lookback_start;
    let mut raw_feature = vec![vec![0.0f32; feature_dim]; lookback_len];
    for (li, t) in (lookback_start..num_frames).enumerate() {
        for f in 0..feature_dim {
            let mut acc = 0.0f32;
            for i in 0..mel {
                acc += x[t * mel + i] * wf[f * mel + i];
            }
            raw_feature[li][f] = acc;
        }
    }
    let mut svdf_out = vec![0.0f32; units];
    for u in 0..units {
        let mut acc = sb[u];
        for r in 0..rank {
            let f = u * rank + r;
            for m in 0..memory {
                let t = num_frames as isize - memory as isize + m as isize;
                if t >= lookback_start as isize {
                    let li = (t - lookback_start as isize) as usize;
                    acc += raw_feature[li][f] * wt[f * memory + m];
                }
            }
        }
        svdf_out[u] = acc;
    }
    let mut logits = vec![0.0f32; classes];
    for c in 0..classes {
        let mut sum = b2[c];
        for u in 0..units {
            sum += svdf_out[u] * w2[c * units + u];
        }
        logits[c] = sum;
    }
    (svdf_out, logits)
}

#[allow(clippy::too_many_arguments)]
fn calibrate_svdf(
    features: &[Vec<f32>],
    wf: &[f32],
    wt: &[f32],
    sb: &[f32],
    w2: &[f32],
    b2: &[f32],
    units: usize,
    rank: usize,
    memory: usize,
    mel: usize,
    classes: usize,
    num_frames: usize,
) -> ((f32, f32), (f32, f32)) {
    let mut svdf_lo = f32::INFINITY;
    let mut svdf_hi = f32::NEG_INFINITY;
    let mut logit_lo = f32::INFINITY;
    let mut logit_hi = f32::NEG_INFINITY;
    let expect_len = mel * num_frames;
    for x in features {
        if x.len() != expect_len {
            continue;
        }
        let (svdf_out, logits) = forward_svdf_float(
            x, wf, wt, sb, w2, b2, units, rank, memory, mel, classes, num_frames,
        );
        for &v in &svdf_out {
            svdf_lo = svdf_lo.min(v);
            svdf_hi = svdf_hi.max(v);
        }
        for &l in &logits {
            logit_lo = logit_lo.min(l);
            logit_hi = logit_hi.max(l);
        }
    }
    (
        logits_range(svdf_lo, svdf_hi),
        logits_range(logit_lo, logit_hi),
    )
}

#[allow(clippy::too_many_arguments)]
fn ptq_conv1d(
    weights_conv: &[f32],
    bias_conv: &[f32],
    weights_fc1: &[f32],
    bias_fc1: &[f32],
    weights_fc2: &[f32],
    bias_fc2: &[f32],
    num_frames: usize,
    kernel_w: usize,
    out_ch: usize,
    mel: usize,
    hidden: usize,
    classes: usize,
    features: &[Vec<f32>],
) -> embedded_nn_compiler::ir::ModelGraph {
    let out_w = num_frames - kernel_w + 1;
    let conv_len = out_ch * out_w;
    let qc = calculate_symmetric_quant_s8(weights_conv.iter().fold(0.1f32, |a, w| a.max(w.abs())));
    let q1 = calculate_symmetric_quant_s8(weights_fc1.iter().fold(0.1f32, |a, w| a.max(w.abs())));
    let q2 = calculate_symmetric_quant_s8(weights_fc2.iter().fold(0.1f32, |a, w| a.max(w.abs())));
    let (conv_range, hidden_range, logits_range) = calibrate_conv1d(
        features,
        weights_conv,
        bias_conv,
        weights_fc1,
        bias_fc1,
        weights_fc2,
        bias_fc2,
        num_frames,
        kernel_w,
        out_ch,
        mel,
        hidden,
        classes,
    );
    let conv_q = output_quant(INPUT_SCALE, qc.scale, conv_range.0, conv_range.1);
    let hidden_q = output_quant(conv_q.scale, q1.scale, hidden_range.0, hidden_range.1);
    let out_q = output_quant(hidden_q.scale, q2.scale, logits_range.0, logits_range.1);
    let mut builder = ModelBuilder::new("BurnConv1d");
    let in_id = builder.add_input(
        "frames",
        TensorShape::new_4d(1, 1, num_frames, mel),
        DataType::Int8,
        Some(QuantParams {
            multiplier: 1_073_741_824,
            shift: 0,
            zero_point: 0,
            scale: INPUT_SCALE,
        }),
    );
    let conv_id = builder.add_conv1d_layer(
        "conv1",
        in_id,
        out_ch,
        kernel_w,
        1,
        0,
        1,
        quantize_weights_s8(weights_conv, qc.scale),
        Some(quantize_bias(bias_conv, INPUT_SCALE, qc.scale)),
        ActivationType::Relu,
        Some(conv_q.clone()),
    );
    let flat = builder.add_reshape_layer("flat", conv_id, TensorShape::new_1d(conv_len));
    let h = builder.add_dense_layer(
        "fc1",
        flat,
        hidden,
        quantize_weights_s8(weights_fc1, q1.scale),
        None,
        Some(quantize_bias(bias_fc1, conv_q.scale, q1.scale)),
        ActivationType::Relu,
        None,
        Some(hidden_q.clone()),
    );
    let out = builder.add_dense_layer(
        "fc2",
        h,
        classes,
        quantize_weights_s8(weights_fc2, q2.scale),
        None,
        Some(quantize_bias(bias_fc2, hidden_q.scale, q2.scale)),
        ActivationType::None,
        None,
        Some(out_q),
    );
    builder.mark_output(out);
    builder.build()
}

#[allow(clippy::too_many_arguments)]
fn ptq_svdf(
    wf: &[f32],
    wt: &[f32],
    sb: &[f32],
    w2: &[f32],
    b2: &[f32],
    units: usize,
    rank: usize,
    memory: usize,
    mel: usize,
    classes: usize,
    num_frames: usize,
    features: &[Vec<f32>],
) -> embedded_nn_compiler::ir::ModelGraph {
    let qf = calculate_symmetric_quant_s8(wf.iter().fold(0.1f32, |a, w| a.max(w.abs())));
    let qt = calculate_symmetric_quant_s8(wt.iter().fold(0.1f32, |a, w| a.max(w.abs())));
    let q2 = calculate_symmetric_quant_s8(w2.iter().fold(0.1f32, |a, w| a.max(w.abs())));
    let state_scale = 0.1f32;
    let (svdf_range, logits_range) = calibrate_svdf(
        features, wf, wt, sb, w2, b2, units, rank, memory, mel, classes, num_frames,
    );
    let svdf_q = output_quant(state_scale, qt.scale, svdf_range.0, svdf_range.1);
    let out_q = output_quant(svdf_q.scale, q2.scale, logits_range.0, logits_range.1);
    let in_q = QuantParams {
        multiplier: calculate_output_requant_multiplier(INPUT_SCALE, qf.scale, state_scale).0,
        shift: calculate_output_requant_multiplier(INPUT_SCALE, qf.scale, state_scale).1,
        zero_point: 0,
        scale: state_scale,
    };
    let mut builder = ModelBuilder::new("BurnSvdf");
    let in_id = builder.add_input(
        "features",
        TensorShape::new_1d(mel),
        DataType::Int8,
        Some(in_q),
    );
    let svdf = builder.add_svdf_layer(
        "svdf1",
        in_id,
        units,
        rank,
        memory,
        quantize_weights_s8(wf, qf.scale),
        quantize_weights_s8(wt, qt.scale),
        Some(quantize_bias(sb, state_scale, qt.scale)),
        ActivationType::None,
        Some(svdf_q.clone()),
    );
    let out = builder.add_dense_layer(
        "fc2",
        svdf,
        classes,
        quantize_weights_s8(w2, q2.scale),
        None,
        Some(quantize_bias(b2, svdf_q.scale, q2.scale)),
        ActivationType::None,
        None,
        Some(out_q),
    );
    builder.mark_output(out);
    builder.build()
}

fn train_conv1d(features: &[Vec<f32>], labels: &[usize], config: &TrainConfig) -> TrainReport {
    let TrainArch::Conv1d {
        num_frames,
        kernel_w,
        out_channels,
    } = config.arch
    else {
        panic!("expected Conv1d arch");
    };
    let n_samples = features.len();
    if n_samples == 0 {
        return TrainReport {
            weights_fc1: Vec::new(),
            bias_fc1: Vec::new(),
            weights_fc2: Vec::new(),
            bias_fc2: Vec::new(),
            conv1d_weights: Vec::new(),
            conv1d_bias: Vec::new(),
            svdf_weights_feature: Vec::new(),
            svdf_weights_time: Vec::new(),
            svdf_bias: Vec::new(),
            graph: embedded_nn_compiler::ir::ModelGraph::new("Empty"),
            final_loss: 0.0,
        };
    }
    let device = Default::default();
    let mut model = ConvNet::new(
        config.num_inputs,
        num_frames,
        kernel_w,
        out_channels,
        config.hidden,
        config.num_classes,
        &device,
    );
    let mut optim = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init::<TrainB, ConvNet>();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let mut final_loss = 0.0f32;

    let flat_features: Vec<f32> = features.iter().flat_map(|f| f.iter().copied()).collect();
    let flat_labels: Vec<i64> = labels.iter().map(|&l| l as i64).collect();

    let xt = Tensor::<TrainB, 3>::from_data(
        TensorData::new(flat_features, [n_samples, config.num_inputs, num_frames]),
        &device,
    );
    let yt =
        Tensor::<TrainB, 1, Int>::from_data(TensorData::new(flat_labels, [n_samples]), &device);

    let total_epochs = config.epochs.max(1);
    let warmup_epochs = if config.mode == TrainMode::Qat {
        (total_epochs / 4).max(5)
    } else {
        0
    };

    for epoch in 0..total_epochs {
        let use_fake = config.mode == TrainMode::Qat && epoch >= warmup_epochs;
        let logits = model.forward(xt.clone(), use_fake);
        let loss = loss_fn.forward(logits, yt.clone());
        let loss_val: f32 = loss.clone().into_data().to_vec::<f32>().expect("loss")[0];
        final_loss = loss_val;
        let grads = GradientsParams::from_grads::<TrainB, ConvNet>(loss.backward(), &model);
        model = optim.step(config.learning_rate, model, grads);
    }
    let (cw, cb) = flatten_conv1d(&model.conv, out_channels, config.num_inputs, kernel_w);
    let out_w = num_frames - kernel_w + 1;
    let (w1, b1) = flatten_linear(&model.fc1, config.hidden, out_channels * out_w);
    let (w2, b2) = flatten_linear(&model.fc2, config.num_classes, config.hidden);
    let graph = ptq_conv1d(
        &cw,
        &cb,
        &w1,
        &b1,
        &w2,
        &b2,
        num_frames,
        kernel_w,
        out_channels,
        config.num_inputs,
        config.hidden,
        config.num_classes,
        features,
    );
    TrainReport {
        weights_fc1: w1,
        bias_fc1: b1,
        weights_fc2: w2,
        bias_fc2: b2,
        conv1d_weights: cw,
        conv1d_bias: cb,
        svdf_weights_feature: Vec::new(),
        svdf_weights_time: Vec::new(),
        svdf_bias: Vec::new(),
        graph,
        final_loss,
    }
}

#[derive(Module, Debug, Clone)]
struct SvdfNet {
    feature: Linear<TrainB>,
    time: Param<Tensor<TrainB, 2>>,
    head: Linear<TrainB>,
}

impl SvdfNet {
    fn new(
        mel: usize,
        feature_dim: usize,
        memory: usize,
        units: usize,
        classes: usize,
        device: &<TrainB as Backend>::Device,
    ) -> Self {
        Self {
            feature: LinearConfig::new(mel, feature_dim).init(device),
            time: Param::from_tensor(Tensor::<TrainB, 2>::zeros([feature_dim, memory], device)),
            head: LinearConfig::new(units, classes).init(device),
        }
    }
}

fn train_svdf(features: &[Vec<f32>], labels: &[usize], config: &TrainConfig) -> TrainReport {
    let TrainArch::Svdf {
        num_frames,
        rank,
        memory_size,
    } = config.arch
    else {
        panic!("expected Svdf arch");
    };
    let units = config.hidden;
    let feature_dim = units * rank;
    let device = Default::default();
    let mut model = SvdfNet::new(
        config.num_inputs,
        feature_dim,
        memory_size,
        units,
        config.num_classes,
        &device,
    );
    let mut optim = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init::<TrainB, SvdfNet>();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let fake = config.mode == TrainMode::Qat;
    let mut final_loss = 0.0f32;
    for _ in 0..config.epochs.max(1) {
        let mut epoch_loss = 0.0;
        for (x, &y) in features.iter().zip(labels.iter()) {
            let frames = Tensor::<TrainB, 2>::from_data(
                TensorData::new(x.clone(), [num_frames, config.num_inputs]),
                &device,
            );
            let feat = relu(model.feature.forward(frames));
            let feat = maybe_fake_quant_act(feat, fake);
            let start = num_frames.saturating_sub(memory_size);
            #[allow(clippy::single_range_in_vec_init)]
            let window = feat.slice([start..num_frames]);
            let pad = memory_size.saturating_sub(num_frames.min(memory_size));
            let taps = if pad > 0 {
                let zeros = Tensor::<TrainB, 2>::zeros([pad, feature_dim], &device);
                Tensor::cat(vec![zeros, window], 0)
            } else {
                window
            };
            let mixed = taps.swap_dims(0, 1) * model.time.val();
            let units_t = mixed
                .sum_dim(1)
                .reshape([1, units, rank])
                .sum_dim(2)
                .reshape([1, units]);
            let h = maybe_fake_quant_act(units_t, fake);
            let w2 = maybe_fake_quant(model.head.weight.val(), fake);
            let logits = h.matmul(w2)
                + model
                    .head
                    .bias
                    .as_ref()
                    .map(|b| b.val().unsqueeze())
                    .expect("bias");
            let yt =
                Tensor::<TrainB, 1, Int>::from_data(TensorData::new(vec![y as i64], [1]), &device);
            let loss = loss_fn.forward(logits, yt);
            epoch_loss += loss.clone().into_data().to_vec::<f32>().expect("loss")[0];
            let grads = GradientsParams::from_grads::<TrainB, SvdfNet>(loss.backward(), &model);
            model = optim.step(config.learning_rate, model, grads);
        }
        final_loss = epoch_loss / features.len().max(1) as f32;
    }
    let (wf, _) = flatten_linear(&model.feature, feature_dim, config.num_inputs);
    let wt_raw: Vec<f32> = model.time.val().into_data().to_vec().expect("svdf time");
    let sb = vec![0.0f32; units];
    let (w2, b2) = flatten_linear(&model.head, config.num_classes, units);
    let graph = ptq_svdf(
        &wf,
        &wt_raw,
        &sb,
        &w2,
        &b2,
        units,
        rank,
        memory_size,
        config.num_inputs,
        config.num_classes,
        num_frames,
        features,
    );
    TrainReport {
        weights_fc1: Vec::new(),
        bias_fc1: Vec::new(),
        weights_fc2: w2,
        bias_fc2: b2,
        conv1d_weights: Vec::new(),
        conv1d_bias: Vec::new(),
        svdf_weights_feature: wf,
        svdf_weights_time: wt_raw,
        svdf_bias: sb,
        graph,
        final_loss,
    }
}

/// Dispatches Burn training for the configured architecture.
/// Dispatches training across supported TinyML architectures (`DenseMlp`, `Conv1d`, `Svdf`)
/// using Burn with post-training or quantization-aware quantization.
pub fn train_model(features: &[Vec<f32>], labels: &[usize], config: &TrainConfig) -> TrainReport {
    match config.arch {
        TrainArch::DenseMlp => train_dense_mlp(features, labels, config),
        TrainArch::Conv1d { .. } => train_conv1d(features, labels, config),
        TrainArch::Svdf { .. } => train_svdf(features, labels, config),
    }
}

fn argmax(v: &[i8]) -> usize {
    v.iter()
        .enumerate()
        .max_by_key(|(_, x)| **x)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn float_argmax(weights: &[f32], bias: &[f32], x: &[f32], out: usize, inn: usize) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for o in 0..out {
        let mut s = bias[o];
        for i in 0..inn {
            s += x[i] * weights[o * inn + i];
        }
        if s > best_v {
            best_v = s;
            best = o;
        }
    }
    best
}

/// Trains PTQ and QAT copies and reports host-integer vs float argmax agreement.
pub fn compare_quant_paths(
    features: &[Vec<f32>],
    labels: &[usize],
    config: &TrainConfig,
) -> QuantCompare {
    let mut ptq_cfg = config.clone();
    ptq_cfg.mode = TrainMode::Ptq;
    let mut qat_cfg = config.clone();
    qat_cfg.mode = TrainMode::Qat;
    let ptq = train_model(features, labels, &ptq_cfg);
    let qat = train_model(features, labels, &qat_cfg);
    let q_in = quantize_features(features);
    let mut ptq_host = HostInterpreter::new(&ptq.graph).unwrap();
    let mut qat_host = HostInterpreter::new(&qat.graph).unwrap();
    let mut compare = QuantCompare {
        n: features.len(),
        ptq_agrees_float: 0,
        qat_agrees_float: 0,
        qat_agrees_ptq: 0,
    };
    for (i, x) in features.iter().enumerate() {
        if matches!(config.arch, TrainArch::DenseMlp) {
            let float_c = float_argmax(
                &ptq.weights_fc2,
                &ptq.bias_fc2,
                x,
                config.num_classes,
                ptq.weights_fc2.len() / config.num_classes.max(1),
            );
            let _ = labels;
            let ptq_c = argmax(&ptq_host.run(&[&q_in[i]]).unwrap()[0]);
            qat_host.reset_external_state();
            let qat_c = argmax(&qat_host.run(&[&q_in[i]]).unwrap()[0]);
            if ptq_c == float_c {
                compare.ptq_agrees_float += 1;
            }
            if qat_c == float_c {
                compare.qat_agrees_float += 1;
            }
            if qat_c == ptq_c {
                compare.qat_agrees_ptq += 1;
            }
        } else {
            let ptq_c = argmax(&ptq_host.run(&[&q_in[i]]).unwrap()[0]);
            qat_host.reset_external_state();
            let qat_c = argmax(&qat_host.run(&[&q_in[i]]).unwrap()[0]);
            if qat_c == ptq_c {
                compare.qat_agrees_ptq += 1;
            }
            compare.ptq_agrees_float += usize::from(ptq_c == labels[i]);
            compare.qat_agrees_float += usize::from(qat_c == labels[i]);
        }
    }
    compare
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv1d_ptq_runs_interpreter() {
        let frames = 4;
        let mel = 3;
        let x = vec![vec![0.1f32; frames * mel]; 4];
        let y = vec![0usize, 1, 0, 1];
        let report = train_model(
            &x,
            &y,
            &TrainConfig {
                num_inputs: mel,
                hidden: 4,
                num_classes: 2,
                learning_rate: 0.05,
                epochs: 2,
                mode: TrainMode::Ptq,
                arch: TrainArch::Conv1d {
                    num_frames: frames,
                    kernel_w: 2,
                    out_channels: 2,
                },
            },
        );
        let mut host = HostInterpreter::new(&report.graph).unwrap();
        let q = quantize_features(&x);
        assert_eq!(host.run(&[&q[0]]).unwrap()[0].len(), 2);
    }

    #[test]
    fn conv1d_ptq_calibrates_unbounded_relu_range() {
        let frames = 4;
        let mel = 2;
        let kernel_w = 2;
        let out_ch = 1;
        let hidden = 1;
        let classes = 2;
        let out_w = frames - kernel_w + 1;
        let conv_len = out_ch * out_w;
        let x = vec![1.0f32; mel * frames];
        let weights_conv = vec![5.0f32; out_ch * kernel_w * mel];
        let bias_conv = vec![0.0f32; out_ch];
        let weights_fc1 = vec![0.0f32; hidden * conv_len];
        let bias_fc1 = vec![0.0f32; hidden];
        let weights_fc2 = vec![0.0f32; classes * hidden];
        let bias_fc2 = vec![0.0f32; classes];
        let graph = ptq_conv1d(
            &weights_conv,
            &bias_conv,
            &weights_fc1,
            &bias_fc1,
            &weights_fc2,
            &bias_fc2,
            frames,
            kernel_w,
            out_ch,
            mel,
            hidden,
            classes,
            &[x],
        );
        let conv = graph
            .tensors
            .iter()
            .find(|t| t.name.contains("conv"))
            .expect("conv tensor");
        // Peak conv activation is 5 * 2 * 2 = 20, so scale must exceed the old [0, 1] width.
        assert!(conv.quant.scale > 1.0 / 255.0 + 0.01);
    }
}
