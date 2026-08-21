use crate::ir::*;
use crate::quant::calculate_elementwise_add_quant;

pub struct ModelBuilder {
    graph: ModelGraph,
    next_tensor_id: usize,
    next_layer_id: usize,
}

impl ModelBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            graph: ModelGraph::new(name),
            next_tensor_id: 0,
            next_layer_id: 0,
        }
    }

    pub fn add_input(
        &mut self,
        name: impl Into<String>,
        shape: TensorShape,
        dtype: DataType,
        input_quant: Option<QuantParams>,
    ) -> usize {
        let id = self.next_tensor_id;
        self.next_tensor_id += 1;
        self.graph.tensors.push(TensorDesc {
            id,
            name: name.into(),
            shape,
            dtype,
            quant: input_quant.unwrap_or_default(),
        });
        self.graph.inputs.push(id);
        id
    }

    pub fn tensor_desc(&self, id: usize) -> Option<&TensorDesc> {
        self.graph.tensors.iter().find(|t| t.id == id)
    }

    pub fn add_dense_layer(
        &mut self,
        name: impl Into<String>,
        input_id: usize,
        out_features: usize,
        weights: Vec<i8>,
        packed_s4: Option<Vec<i8>>,
        bias: Option<Vec<i32>>,
        activation: ActivationType,
        per_channel_quant: Option<PerChannelQuant>,
        output_quant: Option<QuantParams>,
    ) -> usize {
        let input_tensor = self
            .graph
            .tensors
            .iter()
            .find(|t| t.id == input_id)
            .expect("Input tensor not found");
        let _in_features = input_tensor.shape.channels;

        let out_dtype = if packed_s4.is_some() {
            DataType::Int8
        } else {
            input_tensor.dtype
        };
        let out_id = self.next_tensor_id;
        self.next_tensor_id += 1;

        let layer_name = name.into();
        self.graph.tensors.push(TensorDesc {
            id: out_id,
            name: format!("{}_out", layer_name),
            shape: TensorShape::new_1d(out_features),
            dtype: out_dtype,
            quant: output_quant.unwrap_or_default(),
        });

        let layer_id = self.next_layer_id;
        self.next_layer_id += 1;

        self.graph.layers.push(LayerNode {
            id: layer_id,
            name: layer_name,
            inputs: vec![input_id],
            outputs: vec![out_id],
            op: OpPayload::FullyConnected {
                weights,
                packed_s4,
                bias,
                filter_offset: 0,
                activation,
                per_channel_quant,
            },
        });

        out_id
    }

    pub fn add_conv1d_layer(
        &mut self,
        name: impl Into<String>,
        input_id: usize,
        out_channels: usize,
        kernel_w: usize,
        stride_w: usize,
        pad_w: usize,
        dilation_w: usize,
        weights: Vec<i8>,
        bias: Option<Vec<i32>>,
        activation: ActivationType,
        output_quant: Option<QuantParams>,
    ) -> usize {
        let input_tensor = self
            .graph
            .tensors
            .iter()
            .find(|t| t.id == input_id)
            .expect("Input tensor not found");
        let in_width = input_tensor.shape.width;
        let out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;
        let out_dtype = input_tensor.dtype;

        let out_id = self.next_tensor_id;
        self.next_tensor_id += 1;

        let layer_name = name.into();
        self.graph.tensors.push(TensorDesc {
            id: out_id,
            name: format!("{}_out", layer_name),
            shape: TensorShape::new_4d(1, 1, out_width, out_channels),
            dtype: out_dtype,
            quant: output_quant.unwrap_or_default(),
        });

        let layer_id = self.next_layer_id;
        self.next_layer_id += 1;

        self.graph.layers.push(LayerNode {
            id: layer_id,
            name: layer_name,
            inputs: vec![input_id],
            outputs: vec![out_id],
            op: OpPayload::Conv1D {
                kernel_w,
                stride_w,
                pad_w,
                dilation_w,
                weights,
                bias,
                activation,
            },
        });

        out_id
    }

    pub fn add_svdf_layer(
        &mut self,
        name: impl Into<String>,
        input_id: usize,
        units: usize,
        rank: usize,
        memory_size: usize,
        weights_feature: Vec<i8>,
        weights_time: Vec<i8>,
        bias: Option<Vec<i32>>,
        activation: ActivationType,
        output_quant: Option<QuantParams>,
    ) -> usize {
        let input_tensor = self
            .graph
            .tensors
            .iter()
            .find(|t| t.id == input_id)
            .expect("Input tensor not found");
        let out_dtype = input_tensor.dtype;

        let out_id = self.next_tensor_id;
        self.next_tensor_id += 1;

        let layer_name = name.into();
        self.graph.tensors.push(TensorDesc {
            id: out_id,
            name: format!("{}_out", layer_name),
            shape: TensorShape::new_1d(units),
            dtype: out_dtype,
            quant: output_quant.unwrap_or_default(),
        });

        let layer_id = self.next_layer_id;
        self.next_layer_id += 1;

        self.graph.layers.push(LayerNode {
            id: layer_id,
            name: layer_name,
            inputs: vec![input_id],
            outputs: vec![out_id],
            op: OpPayload::Svdf {
                rank,
                memory_size,
                weights_feature,
                weights_time,
                bias,
                activation,
            },
        });

        out_id
    }

    /// Standard 2D dilated-conv output spatial size.
    fn conv2d_output_dim(
        in_size: usize,
        kernel: usize,
        stride: usize,
        pad_before: usize,
        pad_after: usize,
        dilation: usize,
    ) -> usize {
        (in_size + pad_before + pad_after - dilation * (kernel - 1) - 1) / stride + 1
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_conv2d_layer(
        &mut self,
        name: impl Into<String>,
        input_id: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding: Padding2D,
        dilation_h: usize,
        dilation_w: usize,
        weights: Vec<i8>,
        packed_s4: Option<Vec<i8>>,
        bias: Option<Vec<i32>>,
        activation: ActivationType,
        per_channel_quant: Option<PerChannelQuant>,
        output_quant: Option<QuantParams>,
    ) -> usize {
        let input_tensor = self
            .graph
            .tensors
            .iter()
            .find(|t| t.id == input_id)
            .expect("Input tensor not found");
        let out_h = Self::conv2d_output_dim(
            input_tensor.shape.height,
            kernel_h,
            stride_h,
            padding.top,
            padding.bottom,
            dilation_h,
        );
        let out_w = Self::conv2d_output_dim(
            input_tensor.shape.width,
            kernel_w,
            stride_w,
            padding.left,
            padding.right,
            dilation_w,
        );
        let out_dtype = if packed_s4.is_some() {
            DataType::Int8
        } else {
            input_tensor.dtype
        };

        let out_id = self.next_tensor_id;
        self.next_tensor_id += 1;

        let layer_name = name.into();
        self.graph.tensors.push(TensorDesc {
            id: out_id,
            name: format!("{}_out", layer_name),
            shape: TensorShape::new_4d(1, out_h, out_w, out_channels),
            dtype: out_dtype,
            quant: output_quant.unwrap_or_default(),
        });

        let layer_id = self.next_layer_id;
        self.next_layer_id += 1;

        self.graph.layers.push(LayerNode {
            id: layer_id,
            name: layer_name,
            inputs: vec![input_id],
            outputs: vec![out_id],
            op: OpPayload::Conv2D {
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
                activation,
                per_channel_quant,
            },
        });

        out_id
    }

    /// `per_channel_quant` should always be `Some` in practice: the runtime has no per-tensor
    /// depthwise kernel (`depthwise_conv_per_channel_s8` is the only one), so codegen always
    /// emits a call to it, referencing `{PREFIX}_MULT_S32`/`{PREFIX}_SHIFT_S32` statics that are
    /// only emitted when `per_channel_quant` is set.
    #[allow(clippy::too_many_arguments)]
    pub fn add_depthwise_conv2d_layer(
        &mut self,
        name: impl Into<String>,
        input_id: usize,
        ch_mult: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding: Padding2D,
        weights: Vec<i8>,
        bias: Option<Vec<i32>>,
        activation: ActivationType,
        per_channel_quant: Option<PerChannelQuant>,
        output_quant: Option<QuantParams>,
    ) -> usize {
        let input_tensor = self
            .graph
            .tensors
            .iter()
            .find(|t| t.id == input_id)
            .expect("Input tensor not found");
        let out_h = Self::conv2d_output_dim(
            input_tensor.shape.height,
            kernel_h,
            stride_h,
            padding.top,
            padding.bottom,
            1,
        );
        let out_w = Self::conv2d_output_dim(
            input_tensor.shape.width,
            kernel_w,
            stride_w,
            padding.left,
            padding.right,
            1,
        );
        let out_channels = input_tensor.shape.channels * ch_mult;
        let out_dtype = input_tensor.dtype;

        let out_id = self.next_tensor_id;
        self.next_tensor_id += 1;

        let layer_name = name.into();
        self.graph.tensors.push(TensorDesc {
            id: out_id,
            name: format!("{}_out", layer_name),
            shape: TensorShape::new_4d(1, out_h, out_w, out_channels),
            dtype: out_dtype,
            quant: output_quant.unwrap_or_default(),
        });

        let layer_id = self.next_layer_id;
        self.next_layer_id += 1;

        self.graph.layers.push(LayerNode {
            id: layer_id,
            name: layer_name,
            inputs: vec![input_id],
            outputs: vec![out_id],
            op: OpPayload::DepthwiseConv2D {
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                padding,
                ch_mult,
                weights,
                bias,
                activation,
                per_channel_quant,
            },
        });

        out_id
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_maxpool2d_layer(
        &mut self,
        name: impl Into<String>,
        input_id: usize,
        pool_h: usize,
        pool_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding: Padding2D,
    ) -> usize {
        let input_tensor = self
            .graph
            .tensors
            .iter()
            .find(|t| t.id == input_id)
            .expect("Input tensor not found");
        let out_h = Self::conv2d_output_dim(
            input_tensor.shape.height,
            pool_h,
            stride_h,
            padding.top,
            padding.bottom,
            1,
        );
        let out_w = Self::conv2d_output_dim(
            input_tensor.shape.width,
            pool_w,
            stride_w,
            padding.left,
            padding.right,
            1,
        );
        let out_channels = input_tensor.shape.channels;
        let out_dtype = input_tensor.dtype;
        let out_quant = input_tensor.quant.clone();

        let out_id = self.next_tensor_id;
        self.next_tensor_id += 1;

        let layer_name = name.into();
        self.graph.tensors.push(TensorDesc {
            id: out_id,
            name: format!("{}_out", layer_name),
            shape: TensorShape::new_4d(1, out_h, out_w, out_channels),
            dtype: out_dtype,
            // Pooling doesn't requantize -- it passes int8 values through unchanged, so its
            // output shares the input tensor's quantization exactly.
            quant: out_quant,
        });

        let layer_id = self.next_layer_id;
        self.next_layer_id += 1;

        self.graph.layers.push(LayerNode {
            id: layer_id,
            name: layer_name,
            inputs: vec![input_id],
            outputs: vec![out_id],
            op: OpPayload::MaxPool2D {
                pool_h,
                pool_w,
                stride_h,
                stride_w,
                padding,
            },
        });

        out_id
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_avgpool2d_layer(
        &mut self,
        name: impl Into<String>,
        input_id: usize,
        pool_h: usize,
        pool_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding: Padding2D,
    ) -> usize {
        let input_tensor = self
            .graph
            .tensors
            .iter()
            .find(|t| t.id == input_id)
            .expect("Input tensor not found");
        let out_h = Self::conv2d_output_dim(
            input_tensor.shape.height,
            pool_h,
            stride_h,
            padding.top,
            padding.bottom,
            1,
        );
        let out_w = Self::conv2d_output_dim(
            input_tensor.shape.width,
            pool_w,
            stride_w,
            padding.left,
            padding.right,
            1,
        );
        let out_channels = input_tensor.shape.channels;
        let out_dtype = input_tensor.dtype;
        let out_quant = input_tensor.quant.clone();

        let out_id = self.next_tensor_id;
        self.next_tensor_id += 1;

        let layer_name = name.into();
        self.graph.tensors.push(TensorDesc {
            id: out_id,
            name: format!("{}_out", layer_name),
            shape: TensorShape::new_4d(1, out_h, out_w, out_channels),
            dtype: out_dtype,
            quant: out_quant,
        });

        let layer_id = self.next_layer_id;
        self.next_layer_id += 1;

        self.graph.layers.push(LayerNode {
            id: layer_id,
            name: layer_name,
            inputs: vec![input_id],
            outputs: vec![out_id],
            op: OpPayload::AvgPool2D {
                pool_h,
                pool_w,
                stride_h,
                stride_w,
                padding,
            },
        });

        out_id
    }

    /// Reshapes a tensor without changing its data (row-major layout is preserved), so codegen
    /// emits a straight buffer copy. `new_shape` must have the same total element count as the
    /// input tensor's shape.
    pub fn add_reshape_layer(
        &mut self,
        name: impl Into<String>,
        input_id: usize,
        new_shape: TensorShape,
    ) -> usize {
        let input_tensor = self
            .graph
            .tensors
            .iter()
            .find(|t| t.id == input_id)
            .expect("Input tensor not found");
        let out_dtype = input_tensor.dtype;
        let out_quant = input_tensor.quant.clone();

        let out_id = self.next_tensor_id;
        self.next_tensor_id += 1;

        let layer_name = name.into();
        self.graph.tensors.push(TensorDesc {
            id: out_id,
            name: format!("{}_out", layer_name),
            shape: new_shape,
            dtype: out_dtype,
            quant: out_quant,
        });

        let layer_id = self.next_layer_id;
        self.next_layer_id += 1;

        self.graph.layers.push(LayerNode {
            id: layer_id,
            name: layer_name,
            inputs: vec![input_id],
            outputs: vec![out_id],
            op: OpPayload::Reshape { new_shape },
        });

        out_id
    }

    pub fn add_pad_layer(
        &mut self,
        name: impl Into<String>,
        input_id: usize,
        padding: Padding2D,
        pad_value: i8,
    ) -> usize {
        let input_tensor = self
            .graph
            .tensors
            .iter()
            .find(|t| t.id == input_id)
            .expect("Input tensor not found");
        let out_id = self.next_tensor_id;
        self.next_tensor_id += 1;
        let layer_name = name.into();
        let shape = TensorShape::new_4d(
            input_tensor.shape.batches,
            input_tensor.shape.height + padding.top + padding.bottom,
            input_tensor.shape.width + padding.left + padding.right,
            input_tensor.shape.channels,
        );
        self.graph.tensors.push(TensorDesc {
            id: out_id,
            name: format!("{}_out", layer_name),
            shape,
            dtype: input_tensor.dtype,
            quant: input_tensor.quant.clone(),
        });
        let layer_id = self.next_layer_id;
        self.next_layer_id += 1;
        self.graph.layers.push(LayerNode {
            id: layer_id,
            name: layer_name,
            inputs: vec![input_id],
            outputs: vec![out_id],
            op: OpPayload::Pad {
                padding,
                pad_value,
            },
        });
        out_id
    }

    pub fn add_mean_layer(
        &mut self,
        name: impl Into<String>,
        input_id: usize,
        reduce_height: bool,
        reduce_width: bool,
        reduce_channels: bool,
        keep_dims: bool,
    ) -> usize {
        let input_tensor = self
            .graph
            .tensors
            .iter()
            .find(|t| t.id == input_id)
            .expect("Input tensor not found");
        let mut h = input_tensor.shape.height;
        let mut w = input_tensor.shape.width;
        let mut c = input_tensor.shape.channels;
        if reduce_height {
            h = 1;
        }
        if reduce_width {
            w = 1;
        }
        if reduce_channels {
            c = 1;
        }
        let shape = if keep_dims {
            TensorShape::new_4d(input_tensor.shape.batches, h, w, c)
        } else if reduce_height && reduce_width && !reduce_channels {
            TensorShape::new_1d(input_tensor.shape.channels)
        } else {
            TensorShape::new_4d(input_tensor.shape.batches, h, w, c)
        };
        let out_id = self.next_tensor_id;
        self.next_tensor_id += 1;
        let layer_name = name.into();
        self.graph.tensors.push(TensorDesc {
            id: out_id,
            name: format!("{}_out", layer_name),
            shape,
            dtype: input_tensor.dtype,
            quant: input_tensor.quant.clone(),
        });
        let layer_id = self.next_layer_id;
        self.next_layer_id += 1;
        self.graph.layers.push(LayerNode {
            id: layer_id,
            name: layer_name,
            inputs: vec![input_id],
            outputs: vec![out_id],
            op: OpPayload::Mean {
                reduce_height,
                reduce_width,
                reduce_channels,
                keep_dims,
            },
        });
        out_id
    }

    pub fn add_elementwise_add_layer(
        &mut self,
        name: impl Into<String>,
        input1_id: usize,
        input2_id: usize,
        activation: ActivationType,
        output_quant: QuantParams,
    ) -> Result<usize, &'static str> {
        let input1 = self
            .graph
            .tensors
            .iter()
            .find(|t| t.id == input1_id)
            .ok_or("ElementwiseAdd input 1 tensor not found")?;
        let input2 = self
            .graph
            .tensors
            .iter()
            .find(|t| t.id == input2_id)
            .ok_or("ElementwiseAdd input 2 tensor not found")?;
        if input1.shape != input2.shape {
            return Err("ElementwiseAdd broadcasting is not supported");
        }
        if input1.dtype != DataType::Int8 || input2.dtype != DataType::Int8 {
            return Err("ElementwiseAdd only supports int8 tensors");
        }

        let shape = input1.shape;
        let quant = calculate_elementwise_add_quant(&input1.quant, &input2.quant, &output_quant);
        let out_id = self.next_tensor_id;
        self.next_tensor_id += 1;
        let layer_name = name.into();
        self.graph.tensors.push(TensorDesc {
            id: out_id,
            name: format!("{}_out", layer_name),
            shape,
            dtype: DataType::Int8,
            quant: output_quant,
        });
        let layer_id = self.next_layer_id;
        self.next_layer_id += 1;
        self.graph.layers.push(LayerNode {
            id: layer_id,
            name: layer_name,
            inputs: vec![input1_id, input2_id],
            outputs: vec![out_id],
            op: OpPayload::ElementwiseAdd { quant, activation },
        });
        Ok(out_id)
    }

    /// Adds one of the transpose forms implemented by the runtime.
    pub fn add_transpose_layer(
        &mut self,
        name: impl Into<String>,
        input_id: usize,
        permutation: &[usize],
    ) -> Result<usize, &'static str> {
        let input = self
            .graph
            .tensors
            .iter()
            .find(|t| t.id == input_id)
            .ok_or("Transpose input tensor not found")?;
        if input.dtype != DataType::Int8 {
            return Err("Transpose only supports int8 tensors");
        }

        let (kind, shape) = match permutation {
            [1, 0] => (
                TransposeKind::Matrix2D {
                    rows: input.shape.width,
                    cols: input.shape.channels,
                },
                TensorShape::new_2d(input.shape.channels, input.shape.width),
            ),
            [0, 2, 1, 3] => (
                TransposeKind::Spatial4D,
                TensorShape::new_4d(
                    input.shape.batches,
                    input.shape.width,
                    input.shape.height,
                    input.shape.channels,
                ),
            ),
            _ => return Err("unsupported transpose rank or permutation"),
        };

        let out_id = self.next_tensor_id;
        self.next_tensor_id += 1;
        let layer_name = name.into();
        self.graph.tensors.push(TensorDesc {
            id: out_id,
            name: format!("{}_out", layer_name),
            shape,
            dtype: input.dtype,
            quant: input.quant.clone(),
        });
        let layer_id = self.next_layer_id;
        self.next_layer_id += 1;
        self.graph.layers.push(LayerNode {
            id: layer_id,
            name: layer_name,
            inputs: vec![input_id],
            outputs: vec![out_id],
            op: OpPayload::Transpose { kind },
        });
        Ok(out_id)
    }

    /// Sets the filter offset on a just-created fully-connected layer.
    pub fn set_fully_connected_filter_offset(
        &mut self,
        layer_output_id: usize,
        filter_offset: i32,
    ) -> Result<(), &'static str> {
        let layer = self
            .graph
            .layers
            .iter_mut()
            .find(|layer| layer.outputs == [layer_output_id])
            .ok_or("FullyConnected layer not found")?;
        match &mut layer.op {
            OpPayload::FullyConnected {
                filter_offset: offset,
                ..
            } => {
                *offset = filter_offset;
                Ok(())
            }
            _ => Err("layer is not FullyConnected"),
        }
    }

    pub fn add_softmax(&mut self, name: impl Into<String>, input_id: usize) -> usize {
        let input_tensor = self
            .graph
            .tensors
            .iter()
            .find(|t| t.id == input_id)
            .expect("Input tensor not found");
        let out_id = self.next_tensor_id;
        self.next_tensor_id += 1;

        let layer_name = name.into();
        self.graph.tensors.push(TensorDesc {
            id: out_id,
            name: format!("{}_out", layer_name),
            shape: input_tensor.shape,
            dtype: input_tensor.dtype,
            quant: QuantParams::default(),
        });

        let layer_id = self.next_layer_id;
        self.next_layer_id += 1;

        self.graph.layers.push(LayerNode {
            id: layer_id,
            name: layer_name,
            inputs: vec![input_id],
            outputs: vec![out_id],
            op: OpPayload::Softmax,
        });

        out_id
    }

    pub fn mark_output(&mut self, tensor_id: usize) {
        if !self.graph.outputs.contains(&tensor_id) {
            self.graph.outputs.push(tensor_id);
        }
    }

    pub fn set_tensor_quant(
        &mut self,
        tensor_id: usize,
        quant: QuantParams,
    ) -> Result<(), &'static str> {
        let tensor = self
            .graph
            .tensors
            .iter_mut()
            .find(|tensor| tensor.id == tensor_id)
            .ok_or("tensor not found")?;
        tensor.quant = quant;
        Ok(())
    }

    pub fn build(mut self) -> ModelGraph {
        if self.graph.outputs.is_empty() && !self.graph.tensors.is_empty() {
            let last_id = self.graph.tensors.last().unwrap().id;
            self.graph.outputs.push(last_id);
        }
        self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_conv1d_layer_output_shape() {
        let mut builder = ModelBuilder::new("conv1d_test");
        let in_id = builder.add_input(
            "input",
            TensorShape::new_4d(1, 1, 16, 1),
            DataType::Int8,
            None,
        );
        let out_id = builder.add_conv1d_layer(
            "conv1",
            in_id,
            8,
            3,
            1,
            0,
            1,
            vec![0; 8 * 3 * 1],
            Some(vec![0; 8]),
            ActivationType::Relu,
            None,
        );

        let out_tensor = builder
            .graph
            .tensors
            .iter()
            .find(|t| t.id == out_id)
            .unwrap();
        // (16 + 2*0 - 3) / 1 + 1 = 14
        assert_eq!(out_tensor.shape.width, 14);
        assert_eq!(out_tensor.shape.channels, 8);
    }

    #[test]
    fn test_add_svdf_layer_output_shape() {
        let mut builder = ModelBuilder::new("svdf_test");
        let in_id = builder.add_input("input", TensorShape::new_1d(4), DataType::Int8, None);
        let out_id = builder.add_svdf_layer(
            "svdf1",
            in_id,
            16,
            1,
            4,
            vec![0; 16 * 4], // feature_dim(16) * input_dim(4)
            vec![0; 16 * 4], // feature_dim(16) * memory_size(4)
            Some(vec![0; 16]),
            ActivationType::None,
            None,
        );

        let out_tensor = builder
            .graph
            .tensors
            .iter()
            .find(|t| t.id == out_id)
            .unwrap();
        assert_eq!(out_tensor.shape.total_elements(), 16);
    }

    #[test]
    fn test_add_dense_layer_with_per_channel_quant() {
        let mut builder = ModelBuilder::new("per_channel_test");
        let in_id = builder.add_input("input", TensorShape::new_1d(4), DataType::Int8, None);
        builder.add_dense_layer(
            "dense1",
            in_id,
            2,
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            None,
            Some(vec![0, 0]),
            ActivationType::None,
            Some(PerChannelQuant {
                multipliers: vec![10, 20],
                shifts: vec![1, 2],
            }),
            None,
        );

        let layer = builder
            .graph
            .layers
            .iter()
            .find(|l| l.name == "dense1")
            .unwrap();
        match &layer.op {
            OpPayload::FullyConnected {
                per_channel_quant, ..
            } => {
                let pcq = per_channel_quant.as_ref().expect("per_channel_quant set");
                assert_eq!(pcq.multipliers, vec![10, 20]);
                assert_eq!(pcq.shifts, vec![1, 2]);
            }
            _ => panic!("expected FullyConnected payload"),
        }
    }

    #[test]
    fn test_add_conv2d_layer_valid_padding_output_shape() {
        let mut builder = ModelBuilder::new("conv2d_test");
        let in_id = builder.add_input(
            "input",
            TensorShape::new_4d(1, 8, 8, 3),
            DataType::Int8,
            None,
        );
        let out_id = builder.add_conv2d_layer(
            "conv2d_1",
            in_id,
            16,
            3,
            3,
            1,
            1,
            Padding2D::default(),
            1,
            1,
            vec![0; 16 * 3 * 3 * 3],
            None,
            Some(vec![0; 16]),
            ActivationType::Relu,
            None,
            None,
        );

        let out_tensor = builder
            .graph
            .tensors
            .iter()
            .find(|t| t.id == out_id)
            .unwrap();
        // (8 - 3) / 1 + 1 = 6
        assert_eq!(out_tensor.shape.height, 6);
        assert_eq!(out_tensor.shape.width, 6);
        assert_eq!(out_tensor.shape.channels, 16);
    }

    #[test]
    fn test_add_conv2d_layer_same_padding_preserves_spatial_dims() {
        let mut builder = ModelBuilder::new("conv2d_same_test");
        let in_id = builder.add_input(
            "input",
            TensorShape::new_4d(1, 8, 8, 3),
            DataType::Int8,
            None,
        );
        // 3x3 kernel, stride 1, pad 1 on each side == TFLite SAME padding for this config.
        let out_id = builder.add_conv2d_layer(
            "conv2d_1",
            in_id,
            8,
            3,
            3,
            1,
            1,
            Padding2D::symmetric(1, 1),
            1,
            1,
            vec![0; 8 * 3 * 3 * 3],
            None,
            None,
            ActivationType::None,
            None,
            None,
        );

        let out_tensor = builder
            .graph
            .tensors
            .iter()
            .find(|t| t.id == out_id)
            .unwrap();
        assert_eq!(out_tensor.shape.height, 8);
        assert_eq!(out_tensor.shape.width, 8);
    }

    #[test]
    fn test_add_depthwise_conv2d_layer_output_shape() {
        let mut builder = ModelBuilder::new("dw_conv2d_test");
        let in_id = builder.add_input(
            "input",
            TensorShape::new_4d(1, 8, 8, 4),
            DataType::Int8,
            None,
        );
        let out_id = builder.add_depthwise_conv2d_layer(
            "dwconv1",
            in_id,
            2,
            3,
            3,
            1,
            1,
            Padding2D::default(),
            vec![0; 4 * 2 * 3 * 3],
            Some(vec![0; 8]),
            ActivationType::Relu,
            None,
            None,
        );

        let out_tensor = builder
            .graph
            .tensors
            .iter()
            .find(|t| t.id == out_id)
            .unwrap();
        // out_channels = in_channels(4) * ch_mult(2)
        assert_eq!(out_tensor.shape.channels, 8);
        assert_eq!(out_tensor.shape.height, 6);
        assert_eq!(out_tensor.shape.width, 6);
    }

    #[test]
    fn test_add_maxpool2d_layer_output_shape_and_quant_passthrough() {
        let mut builder = ModelBuilder::new("maxpool_test");
        let input_quant = QuantParams {
            multiplier: 123,
            shift: 4,
            zero_point: -5,
            scale: 0.02,
        };
        let in_id = builder.add_input(
            "input",
            TensorShape::new_4d(1, 8, 8, 4),
            DataType::Int8,
            Some(input_quant.clone()),
        );
        let out_id = builder.add_maxpool2d_layer("pool1", in_id, 2, 2, 2, 2, Padding2D::default());

        let out_tensor = builder
            .graph
            .tensors
            .iter()
            .find(|t| t.id == out_id)
            .unwrap();
        assert_eq!(out_tensor.shape.height, 4);
        assert_eq!(out_tensor.shape.width, 4);
        assert_eq!(out_tensor.shape.channels, 4);
        // Pooling doesn't requantize -- output shares the input's quant exactly.
        assert_eq!(out_tensor.quant, input_quant);
    }

    #[test]
    fn test_add_avgpool2d_layer_output_shape() {
        let mut builder = ModelBuilder::new("avgpool_test");
        let in_id = builder.add_input(
            "input",
            TensorShape::new_4d(1, 8, 8, 4),
            DataType::Int8,
            None,
        );
        let out_id = builder.add_avgpool2d_layer("pool1", in_id, 2, 2, 2, 2, Padding2D::default());

        let out_tensor = builder
            .graph
            .tensors
            .iter()
            .find(|t| t.id == out_id)
            .unwrap();
        assert_eq!(out_tensor.shape.height, 4);
        assert_eq!(out_tensor.shape.width, 4);
        assert_eq!(out_tensor.shape.channels, 4);
    }

    #[test]
    fn test_add_reshape_layer_preserves_dtype_and_quant() {
        let mut builder = ModelBuilder::new("reshape_test");
        let input_quant = QuantParams {
            multiplier: 111,
            shift: 2,
            zero_point: -7,
            scale: 0.03,
        };
        let in_id = builder.add_input(
            "input",
            TensorShape::new_4d(1, 2, 2, 4),
            DataType::Int8,
            Some(input_quant.clone()),
        );
        let out_id = builder.add_reshape_layer("reshape1", in_id, TensorShape::new_1d(16));

        let out_tensor = builder
            .graph
            .tensors
            .iter()
            .find(|t| t.id == out_id)
            .unwrap();
        assert_eq!(out_tensor.shape.total_elements(), 16);
        assert_eq!(out_tensor.dtype, DataType::Int8);
        assert_eq!(out_tensor.quant, input_quant);

        let layer = builder
            .graph
            .layers
            .iter()
            .find(|l| l.name == "reshape1")
            .unwrap();
        assert!(matches!(layer.op, OpPayload::Reshape { .. }));
    }

    #[test]
    fn test_add_elementwise_add_tracks_both_inputs_and_tflite_quantization() {
        let mut builder = ModelBuilder::new("add_test");
        let q1 = QuantParams {
            scale: 0.25,
            zero_point: -3,
            ..QuantParams::default()
        };
        let q2 = QuantParams {
            scale: 0.5,
            zero_point: 7,
            ..QuantParams::default()
        };
        let out_q = QuantParams {
            scale: 0.125,
            zero_point: -9,
            ..QuantParams::default()
        };
        let in1 = builder.add_input("in1", TensorShape::new_1d(4), DataType::Int8, Some(q1));
        let in2 = builder.add_input("in2", TensorShape::new_1d(4), DataType::Int8, Some(q2));
        let out = builder
            .add_elementwise_add_layer("add", in1, in2, ActivationType::Relu6, out_q)
            .unwrap();
        builder.mark_output(out);
        let graph = builder.build();

        assert_eq!(graph.layers[0].inputs, vec![in1, in2]);
        match &graph.layers[0].op {
            OpPayload::ElementwiseAdd { quant, activation } => {
                assert_eq!(quant.input1_offset, 3);
                assert_eq!(quant.input1_multiplier, 1_073_741_824);
                assert_eq!(quant.input1_shift, -1);
                assert_eq!(quant.input2_offset, -7);
                assert_eq!(quant.input2_multiplier, 1_073_741_824);
                assert_eq!(quant.input2_shift, 0);
                assert_eq!(quant.output_offset, -9);
                assert_eq!(quant.output_multiplier, 1_073_741_824);
                assert_eq!(quant.output_shift, -16);
                assert_eq!(quant.left_shift, 20);
                assert_eq!(*activation, ActivationType::Relu6);
            }
            other => panic!("expected ElementwiseAdd, got {other:?}"),
        }
    }

    #[test]
    fn test_transpose_builder_accepts_only_runtime_forms() {
        let mut builder = ModelBuilder::new("transpose_test");
        let matrix = builder.add_input("matrix", TensorShape::new_2d(2, 3), DataType::Int8, None);
        let transposed = builder
            .add_transpose_layer("transpose", matrix, &[1, 0])
            .unwrap();
        assert_eq!(
            builder
                .graph
                .tensors
                .iter()
                .find(|tensor| tensor.id == transposed)
                .unwrap()
                .shape,
            TensorShape::new_2d(3, 2)
        );
        assert!(builder.add_transpose_layer("bad", matrix, &[0, 1]).is_err());
    }
}
