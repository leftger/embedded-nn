use burn::backend::{Autodiff, NdArray};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use embedded_nn_compiler::ir::ModelGraph;

use crate::quantize::{ptq_dense_mlp, quantize_features};

pub(crate) type InnerB = NdArray<f32>;
pub(crate) type TrainB = Autodiff<InnerB>;

/// PTQ after float training, or fake-quant QAT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainMode {
    /// Float Adam, then existing `quant.rs` PTQ.
    Ptq,
    /// Fake-quant s8 weights and activations (STE) during Adam, then PTQ.
    Qat,
}

/// Architecture trained on the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainArch {
    /// Mean-pooled Mel vector MLP.
    DenseMlp,
    /// Temporal conv over frames, then MLP head.
    Conv1d {
        /// Analysis frames.
        num_frames: usize,
        /// Kernel width in frames.
        kernel_w: usize,
        /// Conv output channels.
        out_channels: usize,
    },
    /// SVDF delay-line over frames, then linear head.
    Svdf {
        /// Analysis frames.
        num_frames: usize,
        /// Rank.
        rank: usize,
        /// Delay-line depth.
        memory_size: usize,
    },
}

/// Argmax agreement between float, PTQ, and QAT host-integer graphs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantCompare {
    /// Samples compared.
    pub n: usize,
    /// PTQ host-int argmax matches float.
    pub ptq_agrees_float: usize,
    /// QAT host-int argmax matches float.
    pub qat_agrees_float: usize,
    /// QAT host-int argmax matches PTQ.
    pub qat_agrees_ptq: usize,
}

/// Hyperparameters for the host trainer.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Input feature count.
    pub num_inputs: usize,
    /// Hidden units.
    pub hidden: usize,
    /// Output classes.
    pub num_classes: usize,
    /// Adam learning rate.
    pub learning_rate: f64,
    /// Epochs.
    pub epochs: usize,
    /// PTQ or QAT.
    pub mode: TrainMode,
    /// Network family.
    pub arch: TrainArch,
}

/// Trained float weights plus the quantized graph.
#[derive(Debug, Clone)]
pub struct TrainReport {
    /// Flattened `[out, in]` FC1 weights.
    pub weights_fc1: Vec<f32>,
    /// FC1 bias.
    pub bias_fc1: Vec<f32>,
    /// Flattened `[out, in]` FC2 weights.
    pub weights_fc2: Vec<f32>,
    /// FC2 bias.
    pub bias_fc2: Vec<f32>,
    /// Conv1D weights `[out, k, in]` when trained.
    pub conv1d_weights: Vec<f32>,
    /// Conv1D bias.
    pub conv1d_bias: Vec<f32>,
    /// SVDF feature weights.
    pub svdf_weights_feature: Vec<f32>,
    /// SVDF time weights.
    pub svdf_weights_time: Vec<f32>,
    /// SVDF bias.
    pub svdf_bias: Vec<f32>,
    /// Integer graph after PTQ.
    pub graph: ModelGraph,
    /// Last-epoch mean cross-entropy.
    pub final_loss: f32,
}

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl Mlp<TrainB> {
    fn new(
        num_inputs: usize,
        hidden: usize,
        num_classes: usize,
        device: &<TrainB as Backend>::Device,
    ) -> Self {
        Self {
            fc1: LinearConfig::new(num_inputs, hidden).init(device),
            fc2: LinearConfig::new(hidden, num_classes).init(device),
        }
    }

    fn forward(&self, x: Tensor<TrainB, 2>, fake_quant: bool) -> Tensor<TrainB, 2> {
        let w1 = maybe_fake_quant(self.fc1.weight.val(), fake_quant);
        let w2 = maybe_fake_quant(self.fc2.weight.val(), fake_quant);
        let h = relu(x.matmul(w1) + reshape_bias(self.fc1.bias.as_ref().map(|b| b.val())));
        let h = maybe_fake_quant_act(h, fake_quant);
        h.matmul(w2) + reshape_bias(self.fc2.bias.as_ref().map(|b| b.val()))
    }
}

pub(crate) fn relu(x: Tensor<TrainB, 2>) -> Tensor<TrainB, 2> {
    x.clamp_min(0)
}

fn reshape_bias(bias: Option<Tensor<TrainB, 1>>) -> Tensor<TrainB, 2> {
    match bias {
        Some(b) => b.unsqueeze(),
        None => panic!("linear bias is required"),
    }
}

pub(crate) fn maybe_fake_quant(w: Tensor<TrainB, 2>, enable: bool) -> Tensor<TrainB, 2> {
    if enable { fake_quant_s8(w) } else { w }
}

pub(crate) fn maybe_fake_quant_act(x: Tensor<TrainB, 2>, enable: bool) -> Tensor<TrainB, 2> {
    if enable { fake_quant_asymmetric(x) } else { x }
}

fn fake_quant_asymmetric(x: Tensor<TrainB, 2>) -> Tensor<TrainB, 2> {
    let vals: Vec<f32> = x.to_data().to_vec().expect("act data");
    let min = vals.iter().copied().fold(0.0f32, f32::min).min(0.0);
    let max = vals.iter().copied().fold(0.0f32, f32::max).max(0.0);
    let scale = ((max - min) / 255.0).max(1e-7);
    let zp = (-128.0 - min / scale).round().clamp(-128.0, 127.0);
    let q = ((x.clone() / scale + zp).round().clamp(-128.0, 127.0) - zp) * scale;
    ste(x, q)
}

fn fake_quant_s8(w: Tensor<TrainB, 2>) -> Tensor<TrainB, 2> {
    let vals: Vec<f32> = w.to_data().to_vec().expect("weight data");
    let absmax = vals.iter().fold(0.1f32, |a, v| a.max(v.abs()));
    let scale = absmax / 127.0;
    let q = (w.clone() / scale).round().clamp(-128.0, 127.0) * scale;
    ste(w, q)
}

fn ste(x: Tensor<TrainB, 2>, quantized: Tensor<TrainB, 2>) -> Tensor<TrainB, 2> {
    x.clone().add(quantized.sub(x).detach())
}

pub(crate) fn flatten_linear<B: Backend>(
    linear: &Linear<B>,
    out: usize,
    inn: usize,
) -> (Vec<f32>, Vec<f32>) {
    let w = linear.weight.val().into_data();
    let w_vec: Vec<f32> = w.to_vec().expect("linear weights f32");
    // Burn Linear uses `y = x @ W` with W shaped [in, out]. IR/FC is [out, in].
    let mut ir = vec![0.0f32; out * inn];
    for i in 0..inn {
        for o in 0..out {
            ir[o * inn + i] = w_vec[i * out + o];
        }
    }
    let b = linear
        .bias
        .as_ref()
        .map(|b| b.val().into_data().to_vec().expect("bias f32"))
        .unwrap_or_else(|| vec![0.0; out]);
    (ir, b)
}

/// Trains a Dense MLP with Burn Adam and returns PTQ'd [`ModelGraph`].
pub fn train_dense_mlp(
    features: &[Vec<f32>],
    labels: &[usize],
    config: &TrainConfig,
) -> TrainReport {
    assert_eq!(features.len(), labels.len());
    let device = Default::default();
    let mut model = Mlp::<TrainB>::new(
        config.num_inputs,
        config.hidden,
        config.num_classes,
        &device,
    );
    let mut optim = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let fake_quant = config.mode == TrainMode::Qat;
    let mut final_loss = 0.0f32;

    for _ in 0..config.epochs.max(1) {
        let mut epoch_loss = 0.0f32;
        for (x, &y) in features.iter().zip(labels.iter()) {
            let xt = Tensor::<TrainB, 2>::from_data(
                TensorData::new(x.clone(), [1, config.num_inputs]),
                &device,
            );
            let yt =
                Tensor::<TrainB, 1, Int>::from_data(TensorData::new(vec![y as i64], [1]), &device);
            let logits = model.forward(xt, fake_quant);
            let loss = loss_fn.forward(logits, yt);
            let value: f32 = loss.clone().into_data().to_vec::<f32>().expect("loss")[0];
            epoch_loss += value;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(config.learning_rate, model, grads);
        }
        final_loss = epoch_loss / features.len().max(1) as f32;
    }

    let (weights_fc1, bias_fc1) = flatten_linear(&model.fc1, config.hidden, config.num_inputs);
    let (weights_fc2, bias_fc2) = flatten_linear(&model.fc2, config.num_classes, config.hidden);
    let graph = ptq_dense_mlp(
        "BurnMlp",
        &weights_fc1,
        &bias_fc1,
        &weights_fc2,
        &bias_fc2,
        features,
    );
    let _ = quantize_features(features);
    TrainReport {
        weights_fc1,
        bias_fc1,
        weights_fc2,
        bias_fc2,
        conv1d_weights: Vec::new(),
        conv1d_bias: Vec::new(),
        svdf_weights_feature: Vec::new(),
        svdf_weights_time: Vec::new(),
        svdf_bias: Vec::new(),
        graph,
        final_loss,
    }
}

/// Host-interpreter check used by tests: PTQ logits vs integer kernels.
pub fn dequant_outputs(graph: &ModelGraph, outputs: &[i8]) -> Vec<f32> {
    let scale = graph
        .tensors
        .iter()
        .find(|t| graph.outputs.contains(&t.id))
        .map(|t| t.quant.scale)
        .unwrap_or(1.0);
    let zp = graph
        .tensors
        .iter()
        .find(|t| graph.outputs.contains(&t.id))
        .map(|t| t.quant.zero_point)
        .unwrap_or(0);
    outputs
        .iter()
        .map(|&q| (i32::from(q) - zp) as f32 * scale)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use embedded_nn_compiler::HostInterpreter;

    fn xor_set() -> (Vec<Vec<f32>>, Vec<usize>) {
        (
            vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
            vec![0, 1, 1, 0],
        )
    }

    #[test]
    fn ptq_graph_runs_on_host_interpreter() {
        let (x, y) = xor_set();
        let report = train_dense_mlp(
            &x,
            &y,
            &TrainConfig {
                num_inputs: 2,
                hidden: 8,
                num_classes: 2,
                learning_rate: 0.05,
                epochs: 40,
                mode: TrainMode::Ptq,
                arch: TrainArch::DenseMlp,
            },
        );
        let mut host = HostInterpreter::new(&report.graph).unwrap();
        let q = quantize_features(&x);
        for sample in &q {
            let out = host.run(&[sample.as_slice()]).unwrap();
            assert_eq!(out[0].len(), 2);
            let _ = dequant_outputs(&report.graph, &out[0]);
        }
        assert!(report.final_loss.is_finite());
    }

    #[test]
    fn qat_graph_runs_on_host_interpreter() {
        let (x, y) = xor_set();
        let report = train_dense_mlp(
            &x,
            &y,
            &TrainConfig {
                num_inputs: 2,
                hidden: 8,
                num_classes: 2,
                learning_rate: 0.05,
                epochs: 20,
                mode: TrainMode::Qat,
                arch: TrainArch::DenseMlp,
            },
        );
        let mut host = HostInterpreter::new(&report.graph).unwrap();
        let q = quantize_features(&x);
        let _ = host.run(&[&q[0]]).unwrap();
    }
}
