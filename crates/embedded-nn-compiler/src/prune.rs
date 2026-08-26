//! L1 Norm Magnitude Structured Neuron Pruning for Neural Network Graphs.
//!
//! Inspired by embedded neural network pruning techniques, this module computes
//! the total input and output L1 weight magnitudes for hidden neurons across adjacent
//! dense layers and performs structured pruning to reduce memory footprint and latency.

use crate::ir::{ModelGraph, OpPayload, TensorShape};
use serde::{Deserialize, Serialize};

/// Detailed summary report of a structured pruning step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PruningReport {
    /// ID of the upstream layer whose output neuron was removed.
    pub fc1_layer_id: usize,
    /// ID of the downstream layer whose input connection was removed.
    pub fc2_layer_id: usize,
    /// Index of the pruned neuron in the hidden layer.
    pub neuron_index: usize,
    /// Sum of absolute incoming and outgoing weights for the pruned neuron.
    pub l1_score: f32,
    /// Hidden dimension before pruning.
    pub old_hidden_dim: usize,
    /// Hidden dimension after pruning.
    pub new_hidden_dim: usize,
}

/// Computes the L1 importance score (sum of absolute incoming + outgoing weights) for each hidden neuron.
pub fn compute_fc_neuron_l1_importances(
    fc1_weights: &[i8],
    in_features: usize,
    hidden_features: usize,
    fc2_weights: &[i8],
    out_features: usize,
) -> Vec<f32> {
    assert_eq!(
        fc1_weights.len(),
        in_features * hidden_features,
        "fc1_weights length mismatch"
    );
    assert_eq!(
        fc2_weights.len(),
        hidden_features * out_features,
        "fc2_weights length mismatch"
    );

    let mut importances = Vec::with_capacity(hidden_features);

    for k in 0..hidden_features {
        // Incoming weights sum for neuron k
        let mut in_sum = 0f32;
        let fc1_row_offset = k * in_features;
        for j in 0..in_features {
            in_sum += (fc1_weights[fc1_row_offset + j] as i32).abs() as f32;
        }

        // Outgoing weights sum for neuron k
        let mut out_sum = 0f32;
        for m in 0..out_features {
            let fc2_idx = m * hidden_features + k;
            out_sum += (fc2_weights[fc2_idx] as i32).abs() as f32;
        }

        importances.push(in_sum + out_sum);
    }

    importances
}

/// Identifies adjacent FullyConnected layer pairs and finds the neuron with the lowest L1 score.
///
/// Returns `(fc1_layer_idx, fc2_layer_idx, neuron_index, l1_score)`.
pub fn find_lightest_fc_neuron(
    graph: &ModelGraph,
) -> Result<(usize, usize, usize, f32), &'static str> {
    let mut best_candidate = None;
    let mut min_score = f32::MAX;

    // Scan adjacent layers
    for i in 0..graph.layers.len() {
        let l1 = &graph.layers[i];
        if let OpPayload::FullyConnected { weights: w1, .. } = &l1.op {
            let l1_out_tensor_id = match l1.outputs.first() {
                Some(&id) => id,
                None => continue,
            };

            // Find layer that consumes l1_out_tensor_id
            for j in 0..graph.layers.len() {
                if i == j {
                    continue;
                }
                let l2 = &graph.layers[j];
                if l2.inputs.contains(&l1_out_tensor_id) {
                    if let OpPayload::FullyConnected { weights: w2, .. } = &l2.op {
                        // Found FC1 -> FC2 pair
                        let in_tensor_id = match l1.inputs.first() {
                            Some(&id) => id,
                            None => continue,
                        };
                        let in_desc = match graph.tensors.iter().find(|t| t.id == in_tensor_id) {
                            Some(t) => t,
                            None => continue,
                        };
                        let out_desc = match graph.tensors.iter().find(|t| t.id == l1_out_tensor_id)
                        {
                            Some(t) => t,
                            None => continue,
                        };
                        let l2_out_tensor_id = match l2.outputs.first() {
                            Some(&id) => id,
                            None => continue,
                        };
                        let l2_out_desc =
                            match graph.tensors.iter().find(|t| t.id == l2_out_tensor_id) {
                                Some(t) => t,
                                None => continue,
                            };

                        let in_features = in_desc.shape.channels;
                        let hidden_features = out_desc.shape.channels;
                        let out_features = l2_out_desc.shape.channels;

                        if hidden_features <= 1 {
                            // Cannot prune single remaining neuron
                            continue;
                        }

                        if w1.len() != in_features * hidden_features
                            || w2.len() != hidden_features * out_features
                        {
                            continue;
                        }

                        let importances = compute_fc_neuron_l1_importances(
                            w1,
                            in_features,
                            hidden_features,
                            w2,
                            out_features,
                        );

                        for (neuron_idx, &score) in importances.iter().enumerate() {
                            if score < min_score {
                                min_score = score;
                                best_candidate = Some((i, j, neuron_idx, score));
                            }
                        }
                    }
                }
            }
        }
    }

    best_candidate.ok_or("No prunable FullyConnected hidden layers found in graph")
}

/// Prunes a specific hidden neuron index between two adjacent FullyConnected layers.
pub fn prune_fc_hidden_neuron(
    graph: &mut ModelGraph,
    fc1_layer_idx: usize,
    fc2_layer_idx: usize,
    neuron_to_remove: usize,
) -> Result<PruningReport, &'static str> {
    if fc1_layer_idx >= graph.layers.len() || fc2_layer_idx >= graph.layers.len() {
        return Err("Layer index out of bounds");
    }

    let l1_out_tensor_id = *graph.layers[fc1_layer_idx]
        .outputs
        .first()
        .ok_or("FC1 has no outputs")?;
    let l1_in_tensor_id = *graph.layers[fc1_layer_idx]
        .inputs
        .first()
        .ok_or("FC1 has no inputs")?;
    let l2_out_tensor_id = *graph.layers[fc2_layer_idx]
        .outputs
        .first()
        .ok_or("FC2 has no outputs")?;

    let in_features = graph
        .tensors
        .iter()
        .find(|t| t.id == l1_in_tensor_id)
        .ok_or("FC1 input tensor not found")?
        .shape
        .channels;
    let hidden_features = graph
        .tensors
        .iter()
        .find(|t| t.id == l1_out_tensor_id)
        .ok_or("FC1 output tensor not found")?
        .shape
        .channels;
    let out_features = graph
        .tensors
        .iter()
        .find(|t| t.id == l2_out_tensor_id)
        .ok_or("FC2 output tensor not found")?
        .shape
        .channels;

    if neuron_to_remove >= hidden_features {
        return Err("neuron_to_remove index out of range");
    }
    if hidden_features <= 1 {
        return Err("Cannot prune layer with <= 1 neuron");
    }

    // 1. Calculate L1 score before mutating
    let l1_score = {
        let (w1, w2) = match (
            &graph.layers[fc1_layer_idx].op,
            &graph.layers[fc2_layer_idx].op,
        ) {
            (
                OpPayload::FullyConnected { weights: w1, .. },
                OpPayload::FullyConnected { weights: w2, .. },
            ) => (w1, w2),
            _ => return Err("Both layers must be FullyConnected"),
        };
        let scores =
            compute_fc_neuron_l1_importances(w1, in_features, hidden_features, w2, out_features);
        scores[neuron_to_remove]
    };

    // 2. Prune FC1 (remove row `neuron_to_remove`)
    if let OpPayload::FullyConnected {
        weights,
        bias,
        per_channel_quant,
        ..
    } = &mut graph.layers[fc1_layer_idx].op
    {
        // Remove row from weights: drain range [k * in_features .. (k + 1) * in_features]
        let start_idx = neuron_to_remove * in_features;
        let end_idx = start_idx + in_features;
        weights.drain(start_idx..end_idx);

        if let Some(b) = bias {
            if neuron_to_remove < b.len() {
                b.remove(neuron_to_remove);
            }
        }

        if let Some(pcq) = per_channel_quant {
            if neuron_to_remove < pcq.multipliers.len() {
                pcq.multipliers.remove(neuron_to_remove);
                pcq.shifts.remove(neuron_to_remove);
            }
        }
    }

    // 3. Prune FC2 (remove column `neuron_to_remove` for each output channel)
    if let OpPayload::FullyConnected { weights, .. } = &mut graph.layers[fc2_layer_idx].op {
        let mut new_weights = Vec::with_capacity(out_features * (hidden_features - 1));
        for m in 0..out_features {
            let row_offset = m * hidden_features;
            for col in 0..hidden_features {
                if col != neuron_to_remove {
                    new_weights.push(weights[row_offset + col]);
                }
            }
        }
        *weights = new_weights;
    }

    // 4. Update tensor shapes in graph
    if let Some(t) = graph.tensors.iter_mut().find(|t| t.id == l1_out_tensor_id) {
        t.shape = TensorShape::new_1d(hidden_features - 1);
    }

    Ok(PruningReport {
        fc1_layer_id: fc1_layer_idx,
        fc2_layer_id: fc2_layer_idx,
        neuron_index: neuron_to_remove,
        l1_score,
        old_hidden_dim: hidden_features,
        new_hidden_dim: hidden_features - 1,
    })
}

/// Iteratively prunes the lightest neuron across the graph until `target_pruned_neurons` are removed.
pub fn prune_graph_l1(graph: &mut ModelGraph, target_pruned_neurons: usize) -> Vec<PruningReport> {
    let mut reports = Vec::new();

    for _ in 0..target_pruned_neurons {
        match find_lightest_fc_neuron(graph) {
            Ok((fc1_id, fc2_id, neuron_idx, _score)) => {
                match prune_fc_hidden_neuron(graph, fc1_id, fc2_id, neuron_idx) {
                    Ok(rep) => reports.push(rep),
                    Err(_) => break,
                }
            }
            Err(_) => break,
        }
    }

    reports
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::ModelBuilder;
    use crate::ir::{ActivationType, DataType, TensorShape};

    #[test]
    fn test_compute_fc_neuron_l1_importances() {
        let fc1_weights = vec![10, 10, 1, 1, 5, 5];
        let fc2_weights = vec![2, 1, 4, 3, 1, 6];

        let scores = compute_fc_neuron_l1_importances(&fc1_weights, 2, 3, &fc2_weights, 2);
        assert_eq!(scores, vec![25.0, 4.0, 20.0]);
    }

    #[test]
    fn test_prune_fc_hidden_neuron_end_to_end() {
        let mut builder = ModelBuilder::new("test_mlp");
        let in_id = builder.add_input("in", TensorShape::new_1d(2), DataType::Int8, None);

        let fc1_weights = vec![10, 10, 1, 1, 5, 5];
        let fc1_bias = vec![0, 0, 0];
        let fc1_out = builder.add_dense_layer(
            "fc1",
            in_id,
            3,
            fc1_weights,
            None,
            Some(fc1_bias),
            ActivationType::Relu,
            None,
            None,
        );

        let fc2_weights = vec![2, 1, 4, 3, 1, 6];
        let fc2_bias = vec![100, 200];
        let _fc2_out = builder.add_dense_layer(
            "fc2",
            fc1_out,
            2,
            fc2_weights,
            None,
            Some(fc2_bias),
            ActivationType::None,
            None,
            None,
        );

        let mut graph = builder.build();
        assert_eq!(graph.layers.len(), 2);

        let (fc1_id, fc2_id, neuron_idx, score) = find_lightest_fc_neuron(&graph).unwrap();
        assert_eq!(fc1_id, 0);
        assert_eq!(fc2_id, 1);
        assert_eq!(neuron_idx, 1);
        assert_eq!(score, 4.0);

        let report = prune_fc_hidden_neuron(&mut graph, fc1_id, fc2_id, neuron_idx).unwrap();
        assert_eq!(report.old_hidden_dim, 3);
        assert_eq!(report.new_hidden_dim, 2);

        if let OpPayload::FullyConnected { weights, bias, .. } = &graph.layers[0].op {
            assert_eq!(weights, &vec![10, 10, 5, 5]);
            assert_eq!(bias.as_ref().unwrap(), &vec![0, 0]);
        }

        if let OpPayload::FullyConnected { weights, .. } = &graph.layers[1].op {
            assert_eq!(weights, &vec![2, 4, 3, 6]);
        }

        let out1 = graph.tensors.iter().find(|t| t.id == fc1_out).unwrap();
        assert_eq!(out1.shape.channels, 2);
    }
}
