//! Benchmarks for the hot-path quantized inference kernels, using shapes representative of
//! Studio's default TinyML config (16 mel-bin input, 16 hidden units, 4 output classes).

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use embedded_nn::{
    Activation, ConvParams, Dims, FcParams, PerChannelQuantParams, PerTensorQuantParams, Tile,
    convolve_1_x_n_s8, fully_connected_per_channel_s8, fully_connected_s8, requantize, softmax_s8,
};

fn bench_fully_connected_s8(c: &mut Criterion) {
    let in_features = 16usize;
    let out_features = 16usize;

    let fc_params = FcParams {
        input_offset: 0,
        filter_offset: 0,
        output_offset: 0,
        activation: Activation::int8_unconstrained(),
    };
    let quant_params = PerTensorQuantParams::new(1073741824, 0);
    let input_dims = Dims::new(1, 1, 1, in_features as i32);
    let input: Vec<i8> = (0..in_features).map(|i| (i % 127) as i8).collect();
    let filter_dims = Dims::new(in_features as i32, 1, 1, out_features as i32);
    let kernel: Vec<i8> = (0..in_features * out_features)
        .map(|i| ((i * 7) % 127) as i8)
        .collect();
    let bias: Vec<i32> = vec![0; out_features];
    let output_dims = Dims::new(1, 1, 1, out_features as i32);
    let mut output = vec![0i8; out_features];

    c.bench_function("fully_connected_s8_16x16", |b| {
        b.iter(|| {
            fully_connected_s8(
                black_box(&fc_params),
                black_box(&quant_params),
                black_box(&input_dims),
                black_box(&input),
                black_box(&filter_dims),
                black_box(&kernel),
                black_box(Some(&bias)),
                black_box(&output_dims),
                black_box(&mut output),
            )
            .unwrap();
        });
    });
}

fn bench_fully_connected_per_channel_s8(c: &mut Criterion) {
    let in_features = 16usize;
    let out_features = 16usize;

    let fc_params = FcParams {
        input_offset: 1,
        filter_offset: 0,
        output_offset: -128,
        activation: Activation::new(-128, i8::MAX as i32),
    };
    let multipliers: Vec<i32> = (0..out_features).map(|_| 1_400_000_000).collect();
    let shifts: Vec<i32> = vec![-8; out_features];
    let quant_params = PerChannelQuantParams::new(&multipliers, &shifts);
    let input_dims = Dims::new(1, 1, 1, in_features as i32);
    let input: Vec<i8> = (0..in_features).map(|i| (i % 127) as i8).collect();
    let filter_dims = Dims::new(in_features as i32, 1, 1, out_features as i32);
    let kernel: Vec<i8> = (0..in_features * out_features)
        .map(|i| ((i * 7) % 127) as i8)
        .collect();
    let bias: Vec<i32> = vec![0; out_features];
    let output_dims = Dims::new(1, 1, 1, out_features as i32);
    let mut output = vec![0i8; out_features];

    c.bench_function("fully_connected_per_channel_s8_16x16", |b| {
        b.iter(|| {
            fully_connected_per_channel_s8(
                black_box(&fc_params),
                black_box(&quant_params),
                black_box(&input_dims),
                black_box(&input),
                black_box(&filter_dims),
                black_box(&kernel),
                black_box(Some(&bias)),
                black_box(&output_dims),
                black_box(&mut output),
            )
            .unwrap();
        });
    });
}

fn bench_convolve_1_x_n_s8(c: &mut Criterion) {
    let in_width = 16usize;
    let in_channels = 1usize;
    let out_channels = 8usize;
    let kernel_w = 3usize;
    let out_width = in_width - kernel_w + 1;

    let conv_params = ConvParams {
        input_offset: 0,
        output_offset: 0,
        stride: Tile::new(1, 1),
        padding: Tile::new(0, 0),
        dilation: Tile::new(1, 1),
        activation: Activation::int8_unconstrained(),
    };
    let quant_params = PerTensorQuantParams::new(1073741824, 0);
    let input_dims = Dims::new(1, 1, in_width as i32, in_channels as i32);
    let input: Vec<i8> = (0..in_width * in_channels)
        .map(|i| (i % 127) as i8)
        .collect();
    let filter_dims = Dims::new(out_channels as i32, 1, kernel_w as i32, in_channels as i32);
    let kernel: Vec<i8> = (0..out_channels * kernel_w * in_channels)
        .map(|i| ((i * 5) % 127) as i8)
        .collect();
    let bias: Vec<i32> = vec![0; out_channels];
    let output_dims = Dims::new(1, 1, out_width as i32, out_channels as i32);
    let mut output = vec![0i8; out_width * out_channels];

    c.bench_function("convolve_1_x_n_s8_16w_3k_8c", |b| {
        b.iter(|| {
            convolve_1_x_n_s8(
                black_box(&conv_params),
                black_box(&quant_params),
                black_box(&input_dims),
                black_box(&input),
                black_box(&filter_dims),
                black_box(&kernel),
                black_box(Some(&bias)),
                black_box(&output_dims),
                black_box(&mut output),
            )
            .unwrap();
        });
    });
}

fn bench_softmax_s8(c: &mut Criterion) {
    let num_classes = 4usize;
    let input = [10i8, 20i8, 30i8, 40i8];
    let mut output = [0i8; 4];

    c.bench_function("softmax_s8_4classes", |b| {
        b.iter(|| {
            softmax_s8(
                black_box(&input),
                black_box(1),
                black_box(num_classes),
                black_box(1073741824),
                black_box(20),
                black_box(-256),
                black_box(&mut output),
            )
            .unwrap();
        });
    });
}

fn bench_requantize(c: &mut Criterion) {
    c.bench_function("requantize", |b| {
        b.iter(|| requantize(black_box(123_456), black_box(1073741824), black_box(3)));
    });
}

criterion_group!(
    benches,
    bench_fully_connected_s8,
    bench_fully_connected_per_channel_s8,
    bench_convolve_1_x_n_s8,
    bench_softmax_s8,
    bench_requantize
);
criterion_main!(benches);
