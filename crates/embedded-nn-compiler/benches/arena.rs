//! Benchmarks the greedy interval-coloring arena scheduler as graph size grows.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use embedded_nn_compiler::arena::ArenaScheduler;
use embedded_nn_compiler::ir::*;

fn build_chain_graph(num_layers: usize) -> ModelGraph {
    let mut graph = ModelGraph::new("bench_chain");
    let width = 64usize;

    for i in 0..=num_layers {
        graph.tensors.push(TensorDesc {
            id: i,
            name: format!("t{}", i),
            shape: TensorShape::new_1d(width),
            dtype: DataType::Int8,
            quant: QuantParams::default(),
        });
    }
    graph.inputs = vec![0];
    graph.outputs = vec![num_layers];

    for i in 0..num_layers {
        graph.layers.push(LayerNode {
            id: i,
            name: format!("layer{}", i),
            inputs: vec![i],
            outputs: vec![i + 1],
            op: OpPayload::Softmax,
        });
    }

    graph
}

fn bench_arena_schedule(c: &mut Criterion) {
    let mut group = c.benchmark_group("arena_schedule");
    for &num_layers in &[8usize, 32, 128] {
        let graph = build_chain_graph(num_layers);
        group.bench_with_input(
            BenchmarkId::from_parameter(num_layers),
            &graph,
            |b, graph| {
                b.iter(|| black_box(ArenaScheduler::schedule(black_box(graph))));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_arena_schedule);
criterion_main!(benches);
