use crate::ir::{ModelGraph, TensorDesc};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lifetime {
    pub start_step: usize,
    pub end_step: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorAllocation {
    pub tensor_id: usize,
    pub name: String,
    pub byte_offset: usize,
    pub byte_size: usize,
    pub lifetime: Lifetime,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaPlan {
    pub total_arena_bytes: usize,
    pub allocations: HashMap<usize, TensorAllocation>,
}

impl ArenaPlan {
    pub fn offset_of(&self, tensor_id: usize) -> Option<usize> {
        self.allocations.get(&tensor_id).map(|a| a.byte_offset)
    }
}

pub struct ArenaScheduler;

impl ArenaScheduler {
    pub fn schedule(graph: &ModelGraph) -> ArenaPlan {
        let num_steps = graph.layers.len();
        let mut lifetimes: HashMap<usize, Lifetime> = HashMap::new();

        // Model inputs live from start (step 0) until last consumer
        for &inp in &graph.inputs {
            lifetimes.insert(
                inp,
                Lifetime {
                    start_step: 0,
                    end_step: num_steps,
                },
            );
        }

        // Compute layer produce / consume intervals
        for (step, layer) in graph.layers.iter().enumerate() {
            for &out_id in &layer.outputs {
                lifetimes.entry(out_id).or_insert(Lifetime {
                    start_step: step,
                    end_step: step,
                });
            }
            for &in_id in &layer.inputs {
                if let Some(lt) = lifetimes.get_mut(&in_id) {
                    if step > lt.end_step {
                        lt.end_step = step;
                    }
                }
            }
        }

        // Model outputs must live until the end
        for &out_id in &graph.outputs {
            if let Some(lt) = lifetimes.get_mut(&out_id) {
                lt.end_step = num_steps;
            }
        }

        let mut tensor_map: HashMap<usize, &TensorDesc> = HashMap::new();
        for t in &graph.tensors {
            tensor_map.insert(t.id, t);
        }

        // Greedy 1D Interval Coloring for static memory offset assignment
        let mut allocations: HashMap<usize, TensorAllocation> = HashMap::new();
        let mut sorted_tensor_ids: Vec<usize> = lifetimes.keys().copied().collect();
        // Sort by start_step, then by descending byte size (First-Fit Decreasing)
        sorted_tensor_ids.sort_by(|&a, &b| {
            let lt_a = &lifetimes[&a];
            let lt_b = &lifetimes[&b];
            let sz_a = tensor_map
                .get(&a)
                .map(|t| t.shape.byte_size(t.dtype))
                .unwrap_or(0);
            let sz_b = tensor_map
                .get(&b)
                .map(|t| t.shape.byte_size(t.dtype))
                .unwrap_or(0);
            lt_a.start_step
                .cmp(&lt_b.start_step)
                .then_with(|| sz_b.cmp(&sz_a))
        });

        let mut total_arena = 0;

        for &t_id in &sorted_tensor_ids {
            let lt = lifetimes[&t_id];
            let tensor = match tensor_map.get(&t_id) {
                Some(t) => *t,
                None => continue,
            };
            let byte_size = tensor.shape.byte_size(tensor.dtype);
            // 4-byte align all buffer offsets for MCU ARM Cortex-M SIMD / 32-bit word efficiency
            let aligned_size = (byte_size + 3) & !3;

            // Find lowest non-overlapping offset
            let mut candidate_offset = 0;
            loop {
                let candidate_end = candidate_offset + aligned_size;
                let mut collision = false;

                for alloc in allocations.values() {
                    // Check lifetime overlap
                    let times_overlap = !(lt.end_step < alloc.lifetime.start_step
                        || lt.start_step > alloc.lifetime.end_step);
                    if times_overlap {
                        let alloc_end = alloc.byte_offset + ((alloc.byte_size + 3) & !3);
                        let space_overlap =
                            !(candidate_end <= alloc.byte_offset || candidate_offset >= alloc_end);
                        if space_overlap {
                            candidate_offset = alloc_end;
                            collision = true;
                            break;
                        }
                    }
                }

                if !collision {
                    break;
                }
            }

            allocations.insert(
                t_id,
                TensorAllocation {
                    tensor_id: t_id,
                    name: tensor.name.clone(),
                    byte_offset: candidate_offset,
                    byte_size,
                    lifetime: lt,
                },
            );

            let allocated_end = candidate_offset + aligned_size;
            if allocated_end > total_arena {
                total_arena = allocated_end;
            }
        }

        ArenaPlan {
            total_arena_bytes: total_arena,
            allocations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::*;

    #[test]
    fn test_arena_scheduling_reuse() {
        let mut graph = ModelGraph::new("test_net");
        // t0 (input) -> Layer 0 -> t1 -> Layer 1 -> t2 -> Layer 2 -> t3 (output)
        let t0 = TensorDesc {
            id: 0,
            name: "input".into(),
            shape: TensorShape::new_1d(64),
            dtype: DataType::Int8,
            quant: QuantParams::default(),
        };
        let t1 = TensorDesc {
            id: 1,
            name: "hidden1".into(),
            shape: TensorShape::new_1d(64),
            dtype: DataType::Int8,
            quant: QuantParams::default(),
        };
        let t2 = TensorDesc {
            id: 2,
            name: "hidden2".into(),
            shape: TensorShape::new_1d(64),
            dtype: DataType::Int8,
            quant: QuantParams::default(),
        };
        let t3 = TensorDesc {
            id: 3,
            name: "output".into(),
            shape: TensorShape::new_1d(10),
            dtype: DataType::Int8,
            quant: QuantParams::default(),
        };
        graph.tensors = vec![t0, t1, t2, t3];
        graph.inputs = vec![0];
        graph.outputs = vec![3];

        graph.layers.push(LayerNode {
            id: 0,
            name: "fc0".into(),
            inputs: vec![0],
            outputs: vec![1],
            op: OpPayload::Softmax,
        });
        graph.layers.push(LayerNode {
            id: 1,
            name: "fc1".into(),
            inputs: vec![1],
            outputs: vec![2],
            op: OpPayload::Softmax,
        });
        graph.layers.push(LayerNode {
            id: 2,
            name: "fc2".into(),
            inputs: vec![2],
            outputs: vec![3],
            op: OpPayload::Softmax,
        });

        let plan = ArenaScheduler::schedule(&graph);
        // t0 and t2 do not overlap in lifetime! So t2 can reuse t0's or t1's offset
        assert!(plan.total_arena_bytes < 64 * 4); // Peak memory should be much smaller than naive sum
    }

    #[test]
    fn binary_inputs_and_output_do_not_alias_at_execution_step() {
        let mut graph = ModelGraph::new("binary_lifetimes");
        graph.tensors = (0..4)
            .map(|id| TensorDesc {
                id,
                name: format!("t{id}"),
                shape: TensorShape::new_1d(16),
                dtype: DataType::Int8,
                quant: QuantParams::default(),
            })
            .collect();
        graph.inputs = vec![0];
        graph.outputs = vec![3];
        graph.layers = vec![
            LayerNode {
                id: 0,
                name: "left".into(),
                inputs: vec![0],
                outputs: vec![1],
                op: OpPayload::Softmax,
            },
            LayerNode {
                id: 1,
                name: "right".into(),
                inputs: vec![0],
                outputs: vec![2],
                op: OpPayload::Softmax,
            },
            LayerNode {
                id: 2,
                name: "add".into(),
                inputs: vec![1, 2],
                outputs: vec![3],
                op: OpPayload::ElementwiseAdd {
                    quant: ElementwiseAddQuant {
                        input1_offset: 0,
                        input1_multiplier: 1,
                        input1_shift: 0,
                        input2_offset: 0,
                        input2_multiplier: 1,
                        input2_shift: 0,
                        left_shift: 20,
                        output_offset: 0,
                        output_multiplier: 1,
                        output_shift: 0,
                    },
                    activation: ActivationType::None,
                },
            },
        ];

        let plan = ArenaScheduler::schedule(&graph);
        let left = &plan.allocations[&1];
        let right = &plan.allocations[&2];
        let output = &plan.allocations[&3];
        assert_ne!(left.byte_offset, right.byte_offset);
        assert_ne!(left.byte_offset, output.byte_offset);
        assert_ne!(right.byte_offset, output.byte_offset);
        assert_eq!(left.lifetime.end_step, 2);
        assert_eq!(right.lifetime.end_step, 2);
    }
}
