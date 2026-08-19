use crate::ir::*;

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
    ) -> usize {
        let id = self.next_tensor_id;
        self.next_tensor_id += 1;
        self.graph.tensors.push(TensorDesc {
            id,
            name: name.into(),
            shape,
            dtype,
            quant: QuantParams::default(),
        });
        self.graph.inputs.push(id);
        id
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
            quant: QuantParams::default(),
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
                activation,
            },
        });

        out_id
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

    pub fn build(mut self) -> ModelGraph {
        if self.graph.outputs.is_empty() && !self.graph.tensors.is_empty() {
            let last_id = self.graph.tensors.last().unwrap().id;
            self.graph.outputs.push(last_id);
        }
        self.graph
    }
}
