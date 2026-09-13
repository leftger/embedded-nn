//! Burn trainer for a compact, single-block transformer encoder.

use burn::backend::{Autodiff, NdArray};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{gelu_approximate, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::ops::Device;
use burn::tensor::{Int, Tensor, TensorData};
use embedded_nn_compiler::builder::ModelBuilder;
use embedded_nn_compiler::ir::{ActivationType, DataType, ModelGraph, QuantParams, TensorShape};
use embedded_nn_compiler::quant::{
    calculate_asymmetric_quant_s8, calculate_output_requant_multiplier,
    calculate_symmetric_quant_s8, quantize_multiplier, quantize_weights_s8,
};

use crate::mlp::{TrainMode, flatten_linear};

type InnerB = NdArray<f32>;
type TrainB = Autodiff<InnerB>;

/// Hyperparameters for a single-block tiny transformer classifier.
#[derive(Debug, Clone)]
pub struct TinyTransformerConfig {
    /// Tokens per sample.
    pub seq_len: usize,
    /// Features per token.
    pub d_model: usize,
    /// Attention heads; must divide `d_model`.
    pub num_heads: usize,
    /// Feed-forward hidden width.
    pub ff_dim: usize,
    /// Output classes.
    pub num_classes: usize,
    /// Adam learning rate.
    pub learning_rate: f64,
    /// Training epochs.
    pub epochs: usize,
    /// Float PTQ or activation/weight fake-quant QAT.
    pub mode: TrainMode,
}

/// Result of Burn training and integer graph export.
#[derive(Debug, Clone)]
pub struct TinyTransformerReport {
    /// Quantized, executable transformer graph.
    pub graph: ModelGraph,
    /// Last-epoch mean cross-entropy.
    pub final_loss: f32,
}

#[derive(Module, Debug)]
struct TinyTransformer<B: Backend> {
    norm1: RmsNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    norm2: RmsNorm<B>,
    ff1: Linear<B>,
    ff2: Linear<B>,
    classifier: Linear<B>,
    num_heads: usize,
}

impl TinyTransformer<TrainB> {
    fn new(config: &TinyTransformerConfig, device: &Device<TrainB>) -> Self {
        Self {
            norm1: RmsNormConfig::new(config.d_model).init(device),
            q: LinearConfig::new(config.d_model, config.d_model).init(device),
            k: LinearConfig::new(config.d_model, config.d_model).init(device),
            v: LinearConfig::new(config.d_model, config.d_model).init(device),
            out: LinearConfig::new(config.d_model, config.d_model).init(device),
            norm2: RmsNormConfig::new(config.d_model).init(device),
            ff1: LinearConfig::new(config.d_model, config.ff_dim).init(device),
            ff2: LinearConfig::new(config.ff_dim, config.d_model).init(device),
            classifier: LinearConfig::new(config.d_model, config.num_classes).init(device),
            num_heads: config.num_heads,
        }
    }

    fn linear<const D: usize>(
        layer: &Linear<TrainB>,
        input: Tensor<TrainB, D>,
        fake_quant: bool,
    ) -> Tensor<TrainB, D> {
        let weight = fake_quant_tensor(layer.weight.val(), fake_quant);
        burn::tensor::module::linear(input, weight, layer.bias.as_ref().map(|bias| bias.val()))
    }

    fn encode(
        &self,
        input: Tensor<TrainB, 3>,
        fake_quant: bool,
    ) -> (Tensor<TrainB, 3>, Tensor<TrainB, 4>) {
        let [batch, tokens, d_model] = input.dims();
        let head_dim = d_model / self.num_heads;
        let normalized = fake_quant_tensor(self.norm1.forward(input.clone()), fake_quant);
        let q = Self::linear(&self.q, normalized.clone(), fake_quant);
        let k = Self::linear(&self.k, normalized.clone(), fake_quant);
        let v = Self::linear(&self.v, normalized, fake_quant);
        let qh = q
            .reshape([batch, tokens, self.num_heads, head_dim])
            .swap_dims(1, 2);
        let kh = k
            .reshape([batch, tokens, self.num_heads, head_dim])
            .swap_dims(1, 2)
            .swap_dims(2, 3);
        let vh = v
            .reshape([batch, tokens, self.num_heads, head_dim])
            .swap_dims(1, 2);
        let scores = qh.matmul(kh) / (head_dim as f32).sqrt();
        let probabilities = softmax(scores.clone(), 3);
        let context = probabilities
            .matmul(vh)
            .swap_dims(1, 2)
            .reshape([batch, tokens, d_model]);
        let projected = Self::linear(&self.out, context, fake_quant);
        let residual = fake_quant_tensor(input + projected, fake_quant);
        let normalized = fake_quant_tensor(self.norm2.forward(residual.clone()), fake_quant);
        let hidden = gelu_approximate(Self::linear(&self.ff1, normalized, fake_quant));
        let hidden = fake_quant_tensor(hidden, fake_quant);
        let feed_forward = Self::linear(&self.ff2, hidden, fake_quant);
        let encoded = fake_quant_tensor(residual + feed_forward, fake_quant);
        (encoded, scores)
    }

    fn forward(&self, input: Tensor<TrainB, 3>, fake_quant: bool) -> Tensor<TrainB, 2> {
        let [batch, _, d_model] = input.dims();
        let (encoded, _) = self.encode(input, fake_quant);
        let pooled = encoded.mean_dim(1).reshape([batch, d_model]);
        Self::linear(&self.classifier, pooled, fake_quant)
    }
}

fn fake_quant_tensor<const D: usize>(
    tensor: Tensor<TrainB, D>,
    enabled: bool,
) -> Tensor<TrainB, D> {
    if !enabled {
        return tensor;
    }
    let values: Vec<f32> = tensor.to_data().to_vec().expect("fake-quant tensor data");
    let abs_max = values
        .iter()
        .fold(0.1f32, |max, value| max.max(value.abs()));
    let scale = abs_max / 127.0;
    let quantized = (tensor.clone() / scale).round().clamp(-128.0, 127.0) * scale;
    tensor.clone().add(quantized.sub(tensor).detach())
}

fn values<const D: usize>(tensor: Tensor<TrainB, D>) -> Vec<f32> {
    tensor.into_data().to_vec().expect("float tensor data")
}

fn activation_quant(values: &[f32]) -> QuantParams {
    let min = values.iter().copied().fold(0.0f32, f32::min);
    let max = values.iter().copied().fold(0.0f32, f32::max);
    calculate_asymmetric_quant_s8(min, max)
}

fn quantized_linear(
    linear: &Linear<TrainB>,
    out: usize,
    input: usize,
    input_scale: f32,
    observed: &[f32],
) -> (Vec<i8>, Vec<i32>, QuantParams) {
    let (weights, bias) = flatten_linear(linear, out, input);
    let abs_max = weights
        .iter()
        .fold(0.1f32, |max, weight| max.max(weight.abs()));
    let weight_quant = calculate_symmetric_quant_s8(abs_max);
    let weights = quantize_weights_s8(&weights, weight_quant.scale);
    let bias_scale = (input_scale * weight_quant.scale).max(1e-12);
    let bias = bias
        .iter()
        .map(|value| {
            (value / bias_scale)
                .round()
                .clamp(i32::MIN as f32, i32::MAX as f32) as i32
        })
        .collect();
    let mut output_quant = activation_quant(observed);
    (output_quant.multiplier, output_quant.shift) =
        calculate_output_requant_multiplier(input_scale, weight_quant.scale, output_quant.scale);
    (weights, bias, output_quant)
}

fn quantized_gamma(norm: &RmsNorm<TrainB>) -> Vec<i8> {
    values(norm.gamma.val())
        .into_iter()
        .map(|value| (value * 127.0).round().clamp(-128.0, 127.0) as i8)
        .collect()
}

fn export_graph(
    model: &TinyTransformer<TrainB>,
    config: &TinyTransformerConfig,
    features: &[Vec<f32>],
    device: &Device<TrainB>,
) -> ModelGraph {
    let sample_count = features.len();
    let flat: Vec<f32> = features.iter().flatten().copied().collect();
    let input = Tensor::<TrainB, 3>::from_data(
        TensorData::new(flat, [sample_count, config.seq_len, config.d_model]),
        device,
    );
    let normalized1 = model.norm1.forward(input.clone());
    let q = TinyTransformer::linear(&model.q, normalized1.clone(), false);
    let k = TinyTransformer::linear(&model.k, normalized1.clone(), false);
    let v = TinyTransformer::linear(&model.v, normalized1, false);
    let [batch, tokens, d_model] = q.dims();
    let head_dim = d_model / config.num_heads;
    let qh = q
        .clone()
        .reshape([batch, tokens, config.num_heads, head_dim])
        .swap_dims(1, 2);
    let kh = k
        .clone()
        .reshape([batch, tokens, config.num_heads, head_dim])
        .swap_dims(1, 2)
        .swap_dims(2, 3);
    let vh = v
        .clone()
        .reshape([batch, tokens, config.num_heads, head_dim])
        .swap_dims(1, 2);
    let scores = qh.matmul(kh) / (head_dim as f32).sqrt();
    let context = softmax(scores.clone(), 3)
        .matmul(vh)
        .swap_dims(1, 2)
        .reshape([batch, tokens, d_model]);
    let projected = TinyTransformer::linear(&model.out, context.clone(), false);
    let residual1 = input.clone() + projected.clone();
    let normalized2 = model.norm2.forward(residual1.clone());
    let ff1 = TinyTransformer::linear(&model.ff1, normalized2.clone(), false);
    let gelu = gelu_approximate(ff1.clone());
    let feed_forward = TinyTransformer::linear(&model.ff2, gelu.clone(), false);
    let encoded = residual1.clone() + feed_forward.clone();
    let pooled = encoded.clone().mean_dim(1).reshape([batch, d_model]);
    let logits = TinyTransformer::linear(&model.classifier, pooled, false);

    let input_values: Vec<f32> = features.iter().flatten().copied().collect();
    let input_abs = input_values
        .iter()
        .fold(0.1f32, |max, value| max.max(value.abs()));
    let input_quant = calculate_symmetric_quant_s8(input_abs);
    let mut norm1_quant = activation_quant(&values(model.norm1.forward(input)));
    (norm1_quant.multiplier, norm1_quant.shift) =
        quantize_multiplier(1.0 / (32_768.0 * norm1_quant.scale));

    let q_values = values(q);
    let k_values = values(k);
    let v_values = values(v);
    let (qw, qb, q_quant) =
        quantized_linear(&model.q, d_model, d_model, norm1_quant.scale, &q_values);
    let (kw, kb, k_quant) =
        quantized_linear(&model.k, d_model, d_model, norm1_quant.scale, &k_values);
    let (vw, vb, v_quant) =
        quantized_linear(&model.v, d_model, d_model, norm1_quant.scale, &v_values);
    let score_quant = activation_quant(&values(scores));
    let (logits_multiplier, logits_shift) = quantize_multiplier(
        q_quant.scale * k_quant.scale / ((head_dim as f32).sqrt() * score_quant.scale),
    );
    let context_values = values(context);
    let mut context_quant = activation_quant(&context_values);
    (context_quant.multiplier, context_quant.shift) =
        quantize_multiplier(v_quant.scale / (256.0 * context_quant.scale));
    let projected_values = values(projected);
    let (ow, ob, out_quant) = quantized_linear(
        &model.out,
        d_model,
        d_model,
        context_quant.scale,
        &projected_values,
    );
    let residual1_quant = activation_quant(&values(residual1));
    let mut norm2_quant = activation_quant(&values(normalized2));
    (norm2_quant.multiplier, norm2_quant.shift) =
        quantize_multiplier(1.0 / (32_768.0 * norm2_quant.scale));
    let ff1_values = values(ff1);
    let (ff1w, ff1b, ff1_quant) = quantized_linear(
        &model.ff1,
        config.ff_dim,
        d_model,
        norm2_quant.scale,
        &ff1_values,
    );
    let gelu_values = values(gelu);
    let gelu_quant = activation_quant(&gelu_values);
    let feed_forward_values = values(feed_forward);
    let (ff2w, ff2b, feed_forward_quant) = quantized_linear(
        &model.ff2,
        d_model,
        config.ff_dim,
        gelu_quant.scale,
        &feed_forward_values,
    );
    let encoded_quant = activation_quant(&values(encoded));
    let logits_values = values(logits);
    let (cw, cb, classifier_quant) = quantized_linear(
        &model.classifier,
        config.num_classes,
        d_model,
        encoded_quant.scale,
        &logits_values,
    );

    let mut builder = ModelBuilder::new("BurnTinyTransformer");
    let input_id = builder.add_input(
        "tokens",
        TensorShape::new_4d(1, config.seq_len, 1, config.d_model),
        DataType::Int8,
        Some(input_quant),
    );
    let norm1 = builder
        .add_rms_norm_layer(
            "norm1",
            input_id,
            Some(quantized_gamma(&model.norm1)),
            1,
            Some(norm1_quant),
        )
        .expect("valid RMSNorm");
    let q = builder.add_channel_dense_layer(
        "q",
        norm1,
        d_model,
        qw,
        Some(qb),
        ActivationType::None,
        None,
        Some(q_quant),
    );
    let k = builder.add_channel_dense_layer(
        "k",
        norm1,
        d_model,
        kw,
        Some(kb),
        ActivationType::None,
        None,
        Some(k_quant),
    );
    let v = builder.add_channel_dense_layer(
        "v",
        norm1,
        d_model,
        vw,
        Some(vb),
        ActivationType::None,
        None,
        Some(v_quant),
    );
    let attention = builder
        .add_attention_layer(
            "attention",
            q,
            k,
            v,
            config.num_heads,
            logits_multiplier,
            logits_shift,
            Some(context_quant),
        )
        .expect("valid attention");
    let projected = builder.add_channel_dense_layer(
        "out",
        attention,
        d_model,
        ow,
        Some(ob),
        ActivationType::None,
        None,
        Some(out_quant),
    );
    let residual1 = builder
        .add_elementwise_add_layer(
            "attention_residual",
            input_id,
            projected,
            ActivationType::None,
            residual1_quant,
        )
        .expect("valid attention residual");
    let norm2 = builder
        .add_rms_norm_layer(
            "norm2",
            residual1,
            Some(quantized_gamma(&model.norm2)),
            1,
            Some(norm2_quant),
        )
        .expect("valid RMSNorm");
    let ff1 = builder.add_channel_dense_layer(
        "ff1",
        norm2,
        config.ff_dim,
        ff1w,
        Some(ff1b),
        ActivationType::None,
        None,
        Some(ff1_quant),
    );
    let gelu = builder
        .add_gelu_layer("gelu", ff1, true, gelu_quant)
        .expect("valid GELU");
    let ff2 = builder.add_channel_dense_layer(
        "ff2",
        gelu,
        d_model,
        ff2w,
        Some(ff2b),
        ActivationType::None,
        None,
        Some(feed_forward_quant),
    );
    let encoded = builder
        .add_elementwise_add_layer(
            "ff_residual",
            residual1,
            ff2,
            ActivationType::None,
            encoded_quant,
        )
        .expect("valid feed-forward residual");
    let pooled = builder.add_mean_layer("token_mean", encoded, true, false, false, true);
    let classifier = builder.add_dense_layer(
        "classifier",
        pooled,
        config.num_classes,
        cw,
        None,
        Some(cb),
        ActivationType::None,
        None,
        Some(classifier_quant),
    );
    builder.mark_output(classifier);
    builder.build()
}

/// Trains a single-block transformer and exports an executable int8 graph.
pub fn train_tiny_transformer(
    features: &[Vec<f32>],
    labels: &[usize],
    config: &TinyTransformerConfig,
) -> TinyTransformerReport {
    assert_eq!(features.len(), labels.len());
    assert!(config.seq_len > 0);
    assert!(config.d_model > 0);
    assert!(config.num_heads > 0 && config.d_model.is_multiple_of(config.num_heads));
    assert!(config.ff_dim > 0 && config.num_classes > 0);
    assert!(
        features
            .iter()
            .all(|sample| sample.len() == config.seq_len * config.d_model)
    );
    if features.is_empty() {
        return TinyTransformerReport {
            graph: ModelGraph::new("EmptyTinyTransformer"),
            final_loss: 0.0,
        };
    }

    let device = Default::default();
    let mut model = TinyTransformer::new(config, &device);
    let mut optimizer = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let sample_count = features.len();
    let flat: Vec<f32> = features.iter().flatten().copied().collect();
    let input = Tensor::<TrainB, 3>::from_data(
        TensorData::new(flat, [sample_count, config.seq_len, config.d_model]),
        &device,
    );
    let labels: Vec<i64> = labels.iter().map(|label| *label as i64).collect();
    let targets =
        Tensor::<TrainB, 1, Int>::from_data(TensorData::new(labels, [sample_count]), &device);
    let total_epochs = config.epochs.max(1);
    let warmup = if config.mode == TrainMode::Qat {
        (total_epochs / 4).max(1)
    } else {
        total_epochs
    };
    let mut final_loss = 0.0;
    for epoch in 0..total_epochs {
        let fake_quant = config.mode == TrainMode::Qat && epoch >= warmup;
        let logits = model.forward(input.clone(), fake_quant);
        let loss = loss_fn.forward(logits, targets.clone());
        final_loss = loss.clone().into_data().to_vec::<f32>().expect("loss data")[0];
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(config.learning_rate, model, gradients);
    }
    let graph = export_graph(&model, config, features, &device);
    TinyTransformerReport { graph, final_loss }
}

#[cfg(test)]
mod tests {
    use super::*;
    use embedded_nn_compiler::HostInterpreter;

    #[test]
    fn trainer_exports_executable_gelu_transformer() {
        let features = vec![
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            vec![0.9, 0.1, 1.0, 0.0, 0.9, 0.1, 1.0, 0.0],
            vec![0.1, 0.9, 0.0, 1.0, 0.1, 0.9, 0.0, 1.0],
        ];
        let labels = vec![0, 1, 0, 1];
        let config = TinyTransformerConfig {
            seq_len: 2,
            d_model: 4,
            num_heads: 2,
            ff_dim: 8,
            num_classes: 2,
            learning_rate: 0.02,
            epochs: 3,
            mode: TrainMode::Ptq,
        };
        let report = train_tiny_transformer(&features, &labels, &config);
        assert!(report.final_loss.is_finite());
        assert!(report.graph.layers.iter().any(|layer| {
            matches!(
                layer.op,
                embedded_nn_compiler::ir::OpPayload::ScaledDotProductAttention { .. }
            )
        }));
        assert!(
            report.graph.layers.iter().any(|layer| {
                matches!(layer.op, embedded_nn_compiler::ir::OpPayload::Gelu { .. })
            })
        );
        let input_quant = &report.graph.tensors[report.graph.inputs[0]].quant;
        let quantized: Vec<i8> = features[0]
            .iter()
            .map(|value| (value / input_quant.scale).round().clamp(-128.0, 127.0) as i8)
            .collect();
        let mut interpreter = HostInterpreter::new(&report.graph).unwrap();
        let output = interpreter.run(&[&quantized]).unwrap();
        assert_eq!(output[0].len(), 2);
    }

    #[test]
    fn qat_transformer_training_path_runs() {
        let features = vec![
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ];
        let report = train_tiny_transformer(
            &features,
            &[0, 1],
            &TinyTransformerConfig {
                seq_len: 2,
                d_model: 4,
                num_heads: 2,
                ff_dim: 4,
                num_classes: 2,
                learning_rate: 0.01,
                epochs: 2,
                mode: TrainMode::Qat,
            },
        );
        assert!(report.final_loss.is_finite());
        assert!(!report.graph.layers.is_empty());
    }

    #[test]
    fn empty_dataset_returns_empty_graph() {
        let report = train_tiny_transformer(
            &[],
            &[],
            &TinyTransformerConfig {
                seq_len: 2,
                d_model: 4,
                num_heads: 2,
                ff_dim: 4,
                num_classes: 2,
                learning_rate: 0.01,
                epochs: 1,
                mode: TrainMode::Ptq,
            },
        );
        assert!(report.graph.layers.is_empty());
        assert_eq!(report.final_loss, 0.0);
    }
}
