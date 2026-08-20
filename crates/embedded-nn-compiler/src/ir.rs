use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    Int8,
    Int16,
    Int4,
    Float32,
}

impl DataType {
    pub fn size_bytes(&self) -> f32 {
        match self {
            DataType::Int4 => 0.5,
            DataType::Int8 => 1.0,
            DataType::Int16 => 2.0,
            DataType::Float32 => 4.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShape {
    pub batches: usize,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
}

impl TensorShape {
    pub const fn new_1d(len: usize) -> Self {
        Self {
            batches: 1,
            height: 1,
            width: 1,
            channels: len,
        }
    }

    pub const fn new_2d(rows: usize, cols: usize) -> Self {
        Self {
            batches: 1,
            height: 1,
            width: rows,
            channels: cols,
        }
    }

    pub const fn new_4d(b: usize, h: usize, w: usize, c: usize) -> Self {
        Self {
            batches: b,
            height: h,
            width: w,
            channels: c,
        }
    }

    pub fn total_elements(&self) -> usize {
        self.batches * self.height * self.width * self.channels
    }

    pub fn byte_size(&self, dtype: DataType) -> usize {
        let total = self.total_elements();
        match dtype {
            DataType::Int4 => (total + 1) / 2,
            DataType::Int8 => total,
            DataType::Int16 => total * 2,
            DataType::Float32 => total * 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantParams {
    pub multiplier: i32,
    pub shift: i32,
    pub zero_point: i32,
    pub scale: f32,
}

impl Default for QuantParams {
    fn default() -> Self {
        Self {
            multiplier: 1073741824, // 0.5 in Q31
            shift: 0,
            zero_point: 0,
            scale: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActivationType {
    None,
    Relu,
    Relu6,
    LeakyRelu { alpha_mult: i32, alpha_shift: i32 },
    Sigmoid,
    Tanh,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorDesc {
    pub id: usize,
    pub name: String,
    pub shape: TensorShape,
    pub dtype: DataType,
    pub quant: QuantParams,
}

/// Per-channel (per-output-channel) requantization parameters for a weight tensor, as an
/// alternative to `TensorDesc.quant`'s single per-tensor multiplier/shift. One entry per
/// output channel, matching the runtime's `PerChannelQuantParams`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerChannelQuant {
    pub multipliers: Vec<i32>,
    pub shifts: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OpPayload {
    FullyConnected {
        weights: Vec<i8>,
        packed_s4: Option<Vec<i8>>,
        bias: Option<Vec<i32>>,
        activation: ActivationType,
        per_channel_quant: Option<PerChannelQuant>,
    },
    Conv2D {
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        weights: Vec<i8>,
        packed_s4: Option<Vec<i8>>,
        bias: Option<Vec<i32>>,
        activation: ActivationType,
        per_channel_quant: Option<PerChannelQuant>,
    },
    DepthwiseConv2D {
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        /// Channel multiplier: `output_channels = input_channels * ch_mult`, matching the
        /// runtime's `DwConvParams.ch_mult`.
        ch_mult: usize,
        weights: Vec<i8>,
        bias: Option<Vec<i32>>,
        activation: ActivationType,
        per_channel_quant: Option<PerChannelQuant>,
    },
    MaxPool2D {
        pool_h: usize,
        pool_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    },
    AvgPool2D {
        pool_h: usize,
        pool_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    },
    Softmax,
    ElementwiseAdd {
        activation: ActivationType,
    },
    Reshape {
        new_shape: TensorShape,
    },
    LstmStep {
        hidden_dim: usize,
        input_weights: Vec<i8>,
        recurrent_weights: Vec<i8>,
        bias: Vec<i32>,
    },
    Conv1D {
        kernel_w: usize,
        stride_w: usize,
        pad_w: usize,
        dilation_w: usize,
        weights: Vec<i8>,
        bias: Option<Vec<i32>>,
        activation: ActivationType,
    },
    Svdf {
        rank: usize,
        memory_size: usize,
        weights_feature: Vec<i8>,
        weights_time: Vec<i8>,
        bias: Option<Vec<i32>>,
        activation: ActivationType,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerNode {
    pub id: usize,
    pub name: String,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub op: OpPayload,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelGraph {
    pub name: String,
    pub tensors: Vec<TensorDesc>,
    pub layers: Vec<LayerNode>,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
}

impl ModelGraph {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tensors: Vec::new(),
            layers: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn total_weights_size_bytes(&self) -> usize {
        let mut total = 0;
        for layer in &self.layers {
            match &layer.op {
                OpPayload::FullyConnected {
                    weights,
                    packed_s4,
                    bias,
                    per_channel_quant,
                    ..
                } => {
                    if let Some(s4) = packed_s4 {
                        total += s4.len();
                    } else {
                        total += weights.len();
                    }
                    if let Some(b) = bias {
                        total += b.len() * 4;
                    }
                    if let Some(pcq) = per_channel_quant {
                        total += (pcq.multipliers.len() + pcq.shifts.len()) * 4;
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
                        total += s4.len();
                    } else {
                        total += weights.len();
                    }
                    if let Some(b) = bias {
                        total += b.len() * 4;
                    }
                    if let Some(pcq) = per_channel_quant {
                        total += (pcq.multipliers.len() + pcq.shifts.len()) * 4;
                    }
                }
                OpPayload::DepthwiseConv2D {
                    weights,
                    bias,
                    per_channel_quant,
                    ..
                } => {
                    total += weights.len();
                    if let Some(b) = bias {
                        total += b.len() * 4;
                    }
                    if let Some(pcq) = per_channel_quant {
                        total += (pcq.multipliers.len() + pcq.shifts.len()) * 4;
                    }
                }
                OpPayload::LstmStep {
                    input_weights,
                    recurrent_weights,
                    bias,
                    ..
                } => {
                    total += input_weights.len() + recurrent_weights.len() + bias.len() * 4;
                }
                OpPayload::Conv1D { weights, bias, .. } => {
                    total += weights.len();
                    if let Some(b) = bias {
                        total += b.len() * 4;
                    }
                }
                OpPayload::Svdf {
                    weights_feature,
                    weights_time,
                    bias,
                    ..
                } => {
                    total += weights_feature.len() + weights_time.len();
                    if let Some(b) = bias {
                        total += b.len() * 4;
                    }
                }
                _ => {}
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_total_weights_size_bytes_conv1d() {
        let mut graph = ModelGraph::new("conv1d_net");
        graph.layers.push(LayerNode {
            id: 0,
            name: "conv1".into(),
            inputs: vec![0],
            outputs: vec![1],
            op: OpPayload::Conv1D {
                kernel_w: 3,
                stride_w: 1,
                pad_w: 0,
                dilation_w: 1,
                weights: vec![0; 24], // 8 out_channels * 3 kernel_w * 1 in_channel
                bias: Some(vec![0; 8]),
                activation: ActivationType::Relu,
            },
        });

        assert_eq!(graph.total_weights_size_bytes(), 24 + 8 * 4);
    }

    #[test]
    fn test_total_weights_size_bytes_svdf() {
        let mut graph = ModelGraph::new("svdf_net");
        graph.layers.push(LayerNode {
            id: 0,
            name: "svdf1".into(),
            inputs: vec![0],
            outputs: vec![1],
            op: OpPayload::Svdf {
                rank: 1,
                memory_size: 4,
                weights_feature: vec![0; 16], // feature_dim(16) * input_dim(1)
                weights_time: vec![0; 64],    // feature_dim(16) * memory_size(4)
                bias: Some(vec![0; 16]),
                activation: ActivationType::None,
            },
        });

        assert_eq!(graph.total_weights_size_bytes(), 16 + 64 + 16 * 4);
    }
}
